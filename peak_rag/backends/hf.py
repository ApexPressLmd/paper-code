"""Hugging Face backends used for all reported results (Table "External components").

These classes import torch / transformers / sentence-transformers lazily, so that the rest of the
package (graph mining, retrieval, calibration, evaluation and statistics) runs without them.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import Request, Response


class HFGenerator:
    """Qwen/Qwen2.5-7B-Instruct (revision a09a354), BF16, batch size 1, chat template."""

    def __init__(self, name: str, revision: str | None = None, dtype: str = "bfloat16",
                 max_input_tokens: int = 4096, device: str = "cuda"):
        import torch  # noqa: WPS433
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self.torch = torch
        self.max_input_tokens = max_input_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            name, revision=revision, torch_dtype=getattr(torch, dtype), device_map=device)
        self.model.eval()

    def generate(self, request: Request) -> Response:
        torch = self.torch
        messages = [{"role": "user", "content": request.prompt}]
        ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        if ids.shape[1] > self.max_input_tokens:
            # at most 4,096 input tokens per call: keep the head (instructions, first evidence) and the tail
            # (question and answer cue) of the prompt
            ids = torch.cat([ids[:, : self.max_input_tokens // 2], ids[:, -self.max_input_tokens // 2:]], dim=1)
        ids = ids.to(self.model.device)
        kwargs = {"max_new_tokens": request.max_new_tokens}
        if request.temperature > 0:
            kwargs.update(do_sample=True, temperature=request.temperature, top_p=request.top_p)
            if request.seed is not None:
                torch.manual_seed(request.seed)
        else:
            kwargs.update(do_sample=False)
        with torch.no_grad():
            out = self.model.generate(ids, pad_token_id=self.tokenizer.eos_token_id, **kwargs)
        new = out[0, ids.shape[1]:]
        text = self.tokenizer.decode(new, skip_special_tokens=True)
        return Response(text=text, prompt_tokens=int(ids.shape[1]), completion_tokens=int(new.shape[0]))


class HFEmbedder:
    """BAAI/bge-m3 (revision 5617a9f; 1,024 dimensions; up to 8,192 tokens)."""

    def __init__(self, name: str, revision: str | None = None, max_tokens: int = 8192, device: str = "cuda"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(name, revision=revision, device=device)
        self.model.max_seq_length = max_tokens
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True, batch_size=32), dtype=np.float32)


class HFNLI:
    """MoritzLaurer/mDeBERTa-v3-base-mnli-xnli (revision 8adb042), FP32.

    Labels are read from the model's ``id2label`` mapping. Premises longer than the window
    (512 tokens) are split into windows and the window with the highest entailment is kept.
    """

    def __init__(self, name: str, revision: str | None = None, window_tokens: int = 512, device: str = "cuda",
                 dtype: str = "float32"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.window = window_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            name, revision=revision, torch_dtype=getattr(torch, dtype)).to(device).eval()
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.order = [next(i for i, v in id2label.items() if v.startswith(lbl))
                      for lbl in ("entail", "neutral", "contradict")]

    def _windows(self, premise: str, hypothesis: str) -> list[str]:
        h_len = len(self.tokenizer.tokenize(hypothesis))
        budget = max(32, self.window - h_len - 4)
        toks = self.tokenizer.tokenize(premise)
        if len(toks) <= budget:
            return [premise]
        stride = budget // 2
        return [self.tokenizer.convert_tokens_to_string(toks[i:i + budget]) for i in range(0, len(toks), stride)]

    def entail(self, premises: Sequence[str], hypotheses: Sequence[str]) -> np.ndarray:
        torch = self.torch
        out = np.zeros((len(premises), 3), dtype=np.float32)
        for i, (p, h) in enumerate(zip(premises, hypotheses)):
            wins = self._windows(p, h)
            enc = self.tokenizer(wins, [h] * len(wins), truncation=True, max_length=self.window,
                                 padding=True, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                probs = torch.softmax(self.model(**enc).logits, dim=-1).cpu().numpy()[:, self.order]
            out[i] = probs[int(np.argmax(probs[:, 0]))]
        return out
