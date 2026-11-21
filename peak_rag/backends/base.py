"""Backend interfaces and the per-question call accountant.

Every language-model interaction goes through :class:`CountingLLM`, which records the call site,
the prompt and output token counts and the wall-clock latency. Call sites and their decoding settings
follow Appendix B (Table "Language-model call sites") and the Decoding row of the external-components
table. NLI scoring and embedding are separate batched inference and are not counted as model calls.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import numpy as np

# Call sites counted per query (Appendix B). Offline construction call sites are counted separately.
ONLINE_CALL_SITES = (
    "decompose_query", "build_reasoning_dag", "classifier", "answer", "path_answer",
    "conflict_scoring", "uncertainty_sample", "regeneration", "regeneration_sample",
)
OFFLINE_CALL_SITES = ("extract_facts", "resolve_conflict")


@dataclass
class Request:
    call_site: str
    prompt: str
    fields: dict[str, Any]
    max_new_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = None


@dataclass
class Response:
    text: str
    prompt_tokens: int
    completion_tokens: int


class LLMBackend(Protocol):
    name: str

    def generate(self, request: Request) -> Response: ...


class Embedder(Protocol):
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:  # rows L2-normalised
        ...


class NLIModel(Protocol):
    def entail(self, premises: Sequence[str], hypotheses: Sequence[str]) -> np.ndarray:
        """Return an (n, 3) array of probabilities [entailment, neutral, contradiction]."""
        ...


@dataclass
class CallLog:
    calls: list[dict] = field(default_factory=list)

    @property
    def n_calls(self) -> int:
        return len(self.calls)

    @property
    def tokens(self) -> int:
        return sum(c["prompt_tokens"] + c["completion_tokens"] for c in self.calls)

    @property
    def latency_s(self) -> float:
        return sum(c["latency_s"] for c in self.calls)

    def by_site(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.calls:
            out[c["call_site"]] = out.get(c["call_site"], 0) + 1
        return out


class CountingLLM:
    """Wraps a backend; one instance per question so that the log is per query."""

    def __init__(self, backend: LLMBackend, log: CallLog | None = None):
        self.backend = backend
        self.log = log or CallLog()

    def __call__(self, request: Request) -> str:
        t0 = time.perf_counter()
        resp = self.backend.generate(request)
        self.log.calls.append({
            "call_site": request.call_site,
            "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
            "latency_s": time.perf_counter() - t0,
            "temperature": request.temperature,
        })
        return resp.text


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float32) @ np.asarray(b, dtype=np.float32).T


def clipped_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """max{0, min[1, cos(a, b)]} for L2-normalised rows (Eq. 1, rel(c, q))."""
    return np.clip(cosine_matrix(a, b), 0.0, 1.0)
