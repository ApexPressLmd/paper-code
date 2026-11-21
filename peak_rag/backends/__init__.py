"""Backend factory: ``hf`` loads the models of the external-components table; ``offline`` loads the
CPU stand-ins used by the tests and the toy demo."""
from __future__ import annotations

from .base import CallLog, CountingLLM, Request, Response, clipped_cosine, cosine_matrix
from .offline import HashingEmbedder, LexicalNLI, RuleLLM

__all__ = ["CallLog", "CountingLLM", "Request", "Response", "clipped_cosine", "cosine_matrix",
           "HashingEmbedder", "LexicalNLI", "RuleLLM", "make_backends"]


def make_backends(cfg, kind: str = "offline", device: str = "cuda"):
    """Return (generator, embedder, nli)."""
    if kind == "offline":
        return RuleLLM(), HashingEmbedder(cfg.models.embedder.dim), LexicalNLI()
    if kind == "hf":
        from .hf import HFEmbedder, HFGenerator, HFNLI

        m = cfg.models
        gen = HFGenerator(m.generator.name, m.generator.revision, m.generator.dtype,
                          cfg.retrieval.max_input_tokens, device)
        emb = HFEmbedder(m.embedder.name, m.embedder.revision, m.embedder.max_tokens, device)
        nli = HFNLI(m.nli.name, m.nli.revision, cfg.decoding.nli_window_tokens, device, m.nli.dtype)
        return gen, emb, nli
    raise ValueError(f"unknown backend kind {kind!r}")
