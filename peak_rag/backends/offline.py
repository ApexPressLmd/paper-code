"""CPU-only stand-in backends for tests and the toy demo.

They are NOT the models used for the reported results. They exist so that every stage of the
pipeline (and every call site, with its exact call count) can be exercised on a machine without a
GPU or network access. Their outputs are deterministic given the request seed.

* :class:`HashingEmbedder` - hashed word and character-trigram features, L2-normalised.
* :class:`LexicalNLI` - entailment from content-token coverage, contradiction from negation mismatch.
* :class:`RuleLLM` - answers each call site from the structured request fields with simple rules.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Sequence

import numpy as np

from ..text import content_tokens, normalize_answer, split_sentences, tokens
from .base import Request, Response

_NEG = {"not", "no", "never", "without", "cannot", "neither", "nor"}


def _h(s: str, mod: int) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % mod


class HashingEmbedder:
    def __init__(self, dim: int = 1024):
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            toks = content_tokens(text) or tokens(text)
            for tok in toks:
                out[i, _h("w:" + tok, self.dim)] += 1.0
                stem = tok[:5]
                out[i, _h("s:" + stem, self.dim)] += 0.5
            norm = np.linalg.norm(out[i])
            if norm > 0:
                out[i] /= norm
        return out


class LexicalNLI:
    def entail(self, premises: Sequence[str], hypotheses: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(premises), 3), dtype=np.float32)
        for i, (p, h) in enumerate(zip(premises, hypotheses)):
            hp = set(content_tokens(h))
            pp = set(content_tokens(p))
            stems_p = {t[:5] for t in pp}
            if not hp:
                out[i] = (0.0, 1.0, 0.0)
                continue
            cover = sum(1 for t in hp if t in pp or t[:5] in stems_p) / len(hp)
            neg_mismatch = bool(set(tokens(h)) & _NEG) != bool(set(tokens(p)) & _NEG)
            if neg_mismatch and cover >= 0.6:
                out[i] = (0.05, 0.15, 0.80)
            else:
                e = cover ** 2
                out[i] = (e, (1 - e) * 0.9, (1 - e) * 0.1)
        return out


_PATTERNS = [
    (re.compile(r"^(?P<h>.+?) requires (?P<t>.+?)$", re.I), "requires"),
    (re.compile(r"^(?P<h>.+?) is based on (?P<t>.+?)$", re.I), "based_on"),
    (re.compile(r"^(?P<h>.+?) precedes (?P<t>.+?)$", re.I), "precedes"),
    (re.compile(r"^(?P<h>.+?) causes (?P<t>.+?)$", re.I), "causes"),
    (re.compile(r"^(?P<h>.+?) is measured by (?P<t>.+?)$", re.I), "measured_by"),
    (re.compile(r"^(?P<h>.+?) is part of (?P<t>.+?)$", re.I), "part_of"),
    (re.compile(r"^(?P<h>.+?) reduces (?P<t>.+?)$", re.I), "reduces"),
    (re.compile(r"^(?P<h>.+?) increases (?P<t>.+?)$", re.I), "increases"),
    (re.compile(r"^(?P<h>.+?) is used in (?P<t>.+?)$", re.I), "used_in"),
    (re.compile(r"^(?P<h>.+?) is (?P<t>an? .+?)$", re.I), "is_a"),
]
_CHAIN_CUES = ("why", "how does", "how do", "explain", "cause", "leads to", "result", "first", "then", "before")
_PARALLEL_CUES = ("compare", "both", "whether", "same", "differ", "which of", "are")


class RuleLLM:
    name = "rule-based offline backend"

    def __init__(self, parametric_memory: Sequence[dict] | None = None):
        # optional parametric "self knowledge" returned by the conflict-scoring call
        self.memory = list(parametric_memory or [])

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _best_sentence(question: str, evidence: Sequence[str], rng: np.random.Generator | None = None) -> tuple[str, float]:
        q = set(content_tokens(question))
        scored = []
        for item in evidence:
            for sent in split_sentences(item):
                st = set(content_tokens(sent))
                if not st:
                    continue
                overlap = len(q & st) / max(1, len(q))
                scored.append((overlap, sent))
        if not scored:
            return "", 0.0
        scored.sort(key=lambda x: -x[0])
        if rng is not None and len(scored) > 1:
            top = scored[:3]
            w = np.array([s for s, _ in top]) + 1e-3
            j = int(rng.choice(len(top), p=w / w.sum()))
            return top[j][1], top[j][0]
        return scored[0][1], scored[0][0]

    @staticmethod
    def extract_triples(text: str) -> list[dict]:
        triples = []
        for sent in split_sentences(text):
            s = sent.strip().rstrip(".")
            for pat, rel in _PATTERNS:
                m = pat.match(s)
                if m:
                    triples.append({"head": m.group("h").strip(), "relation": rel, "tail": m.group("t").strip()})
                    break
        return triples

    # ------------------------------------------------------------------ dispatch
    def generate(self, request: Request) -> Response:
        f = request.fields
        rng = np.random.default_rng(request.seed) if request.temperature > 0 else None
        site = request.call_site
        if site == "decompose_query":
            q = f["question"].strip().rstrip("?")
            parts = [p.strip() for p in re.split(r",? and then |; | then ", q) if p.strip()]
            text = json.dumps([p + "?" for p in parts] or [f["question"]])
        elif site == "build_reasoning_dag":
            n = len(f["sub_questions"])
            text = json.dumps([[i, i + 1] for i in range(n - 1)])
        elif site == "classifier":
            low = f["question"].lower()
            chained = any(c in low for c in _CHAIN_CUES) and not low.startswith(("is ", "are ", "does ", "do "))
            text = "CHAINED" if chained else "PARALLEL"
        elif site in ("answer", "path_answer", "uncertainty_sample", "regeneration", "regeneration_sample"):
            evidence = f.get("evidence", [])
            sent, score = self._best_sentence(f["question"], evidence, rng)
            text = sent if score >= 0.34 else "The evidence is insufficient to answer this question."
        elif site == "conflict_scoring":
            claims = split_sentences(f.get("answer", ""))
            qset = set(content_tokens(f["question"]))
            self_facts = [m for m in self.memory if qset & set(content_tokens(m["text"]))]
            text = json.dumps({"atomic_facts": claims,
                               "self_facts": [{"text": m["text"], "confidence": m.get("confidence", 0.8)} for m in self_facts]})
        elif site == "extract_facts":
            text = json.dumps(self.extract_triples(f["passage"]))
        elif site == "resolve_conflict":
            a, b = f["fact_a"], f["fact_b"]
            same = (normalize_answer(a["head"]), a["relation"], normalize_answer(a["tail"])) == (
                normalize_answer(b["head"]), b["relation"], normalize_answer(b["tail"]))
            text = json.dumps({"decision": "merge" if same else "keep_both",
                               "justification": "identical meaning" if same else "both supported by their passages"})
        else:  # pragma: no cover
            raise ValueError(f"unknown call site {site}")
        return Response(text=text, prompt_tokens=len(tokens(request.prompt)), completion_tokens=len(tokens(text)))
