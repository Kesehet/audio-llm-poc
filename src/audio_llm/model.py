from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, WhisperModel


class AudioProjector(nn.Module):
    def __init__(self, audio_dim: int, llm_dim: int, hidden_dim: int, pool_stride: int = 4):
        super().__init__()
        self.pool_stride = pool_stride
        self.pool = nn.AvgPool1d(pool_stride, stride=pool_stride, ceil_mode=True)
        self.net = nn.Sequential(
            nn.LayerNorm(audio_dim),
            nn.Linear(audio_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, llm_dim),
            nn.LayerNorm(llm_dim),
        )

    def forward(self, x):
        x = self.pool(x.transpose(1, 2)).transpose(1, 2)
        return self.net(x)


class AudioLLM(nn.Module):
    """Frozen Whisper encoder + trainable projector + frozen causal LLM."""

    def __init__(
        self,
        audio_encoder_name="openai/whisper-small",
        llm_name="Qwen/Qwen2.5-0.5B-Instruct",
        projector_hidden=1536,
        audio_pool_stride=4,
        system_prompt="You are a concise voice assistant.",
        train_llm=False,
    ):
        super().__init__()
        whisper = WhisperModel.from_pretrained(audio_encoder_name)
        self.audio_encoder = whisper.encoder
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_name)
        self.system_prompt = system_prompt
        self.audio_pool_stride = audio_pool_stride

        audio_dim = whisper.config.d_model
        llm_dim = self.llm.config.hidden_size
        self.projector = AudioProjector(audio_dim, llm_dim, projector_hidden, audio_pool_stride)

        for p in self.audio_encoder.parameters():
            p.requires_grad = False
        if not train_llm:
            for p in self.llm.parameters():
                p.requires_grad = False
        self.audio_encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        self.audio_encoder.eval()
        return self

    def _prompt_ids(self, prompt: str):
        user_text = prompt.strip() or "Respond to the user's spoken request."
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_text},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )[0]

    def encode_audio(self, input_features, num_samples):
        with torch.no_grad():
            enc = self.audio_encoder(input_features=input_features).last_hidden_state
        audio_embeds = self.projector(enc)

        valid = torch.div(num_samples + 319, 320, rounding_mode="floor")
        valid = torch.div(valid + self.audio_pool_stride - 1, self.audio_pool_stride, rounding_mode="floor")
        valid = valid.clamp(max=audio_embeds.shape[1])
        return audio_embeds, valid

    def forward(self, input_features, num_samples, responses: List[str], prompts: Optional[List[str]] = None, max_target_tokens=128):
        device = input_features.device
        prompts = prompts or [""] * len(responses)
        audio_embeds, valid_audio = self.encode_audio(input_features, num_samples.to(device))
        token_embed = self.llm.get_input_embeddings()
        eos = self.tokenizer.eos_token_id

        sequences, labels = [], []
        for i, (response, prompt) in enumerate(zip(responses, prompts)):
            a = audio_embeds[i, : int(valid_audio[i].item())]
            p_ids = self._prompt_ids(prompt).to(device)
            r_ids = self.tokenizer(response, add_special_tokens=False, truncation=True, max_length=max_target_tokens, return_tensors="pt").input_ids[0].to(device)
            r_ids = torch.cat([r_ids, torch.tensor([eos], device=device)])
            seq = torch.cat([a, token_embed(p_ids), token_embed(r_ids)], dim=0)
            lab = torch.cat([
                torch.full((a.shape[0] + p_ids.shape[0],), -100, device=device, dtype=torch.long),
                r_ids,
            ])
            sequences.append(seq)
            labels.append(lab)

        max_len = max(x.shape[0] for x in sequences)
        dim = sequences[0].shape[-1]
        embeds = torch.zeros(len(sequences), max_len, dim, device=device, dtype=sequences[0].dtype)
        mask = torch.zeros(len(sequences), max_len, device=device, dtype=torch.long)
        label_pad = torch.full((len(sequences), max_len), -100, device=device, dtype=torch.long)
        for i, (seq, lab) in enumerate(zip(sequences, labels)):
            n = seq.shape[0]
            embeds[i, :n] = seq
            mask[i, :n] = 1
            label_pad[i, :n] = lab

        return self.llm(inputs_embeds=embeds, attention_mask=mask, labels=label_pad, use_cache=False)

    @torch.no_grad()
    def generate_from_audio(self, input_features, num_samples, prompt="", max_new_tokens=80, temperature=0.0):
        device = input_features.device
        audio_embeds, valid_audio = self.encode_audio(input_features, num_samples.to(device))
        a = audio_embeds[:, : int(valid_audio[0].item())]
        p_ids = self._prompt_ids(prompt).unsqueeze(0).to(device)
        p_embeds = self.llm.get_input_embeddings()(p_ids)
        prefix = torch.cat([a, p_embeds], dim=1)
        attn = torch.ones(prefix.shape[:2], device=device, dtype=torch.long)

        out = self.llm(inputs_embeds=prefix, attention_mask=attn, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]
        generated = []
        eos = self.tokenizer.eos_token_id
        for _ in range(max_new_tokens):
            if temperature and temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                nxt = torch.multinomial(probs, 1)
            else:
                nxt = logits.argmax(dim=-1, keepdim=True)
            token = int(nxt.item())
            if token == eos:
                break
            generated.append(token)
            attn = torch.cat([attn, torch.ones((1, 1), device=device, dtype=torch.long)], dim=1)
            out = self.llm(input_ids=nxt, attention_mask=attn, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
