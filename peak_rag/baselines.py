"""Reference systems re-run under the shared backbone, corpus and context budget
(Section 4, Supplementary Table S13) and the thresholded abstention alternatives of Figure 9.

* DenseRAG               top-12 passages by embedding similarity, one answer call at T = 0.
* graph PPR, lambda = 0  PeakRAG with the ``lambda0`` preset (no prerequisite bias).
* SelfConsistency        compute-matched: dense retrieval and ``n`` samples at T = 0.7, majority vote.
* SelfCheckGPT / semantic-entropy thresholds: a confidence score from k = 5 samples, with an
  abstention threshold frozen on development data at the same 10% abstention budget as the gate
  (``runner.thresholded_systems``). ``semantic_entropy_confidence`` adds the same confidence to records
  of an external system that stores sampled answers (PARKA + semantic entropy).

The fifteen published baselines are run from their authors' code; their per-question records enter the
evaluation through the prediction-record schema (see ``peak_rag/eval/tables.py``).
"""
from __future__ import annotations

import math
import time
from collections import Counter

import numpy as np

from . import prompts
from .backends.base import CallLog, CountingLLM, clipped_cosine
from .text import is_insufficient, normalize_answer


class DenseRAG:
    name = "Dense RAG"

    def __init__(self, kb, cfg, generator, embedder, nli):
        self.kb, self.cfg, self.gen, self.emb, self.nli = kb, cfg, generator, embedder, nli

    def retrieve_passages(self, question: str) -> list[dict]:
        q = self.emb.encode([question])
        sims = clipped_cosine(q, self.kb.passage_emb)[0]
        return [self.kb.passages[i] for i in np.argsort(-sims)[: self.cfg.retrieval.context_budget]]

    def retrieve(self, question: str) -> list[str]:
        return [p["text"] for p in self.retrieve_passages(question)]

    def _samples(self, llm, question, ev, n, seed):
        return [llm(prompts.answer(self.cfg, question, ev, call_site="uncertainty_sample",
                                   temperature=self.cfg.gating.sample_temperature, seed=seed * 1000 + j)).strip()
                for j in range(n)]

    def answer(self, question: str, *, seed: int = 17, **_) -> dict:
        log = CallLog()
        llm = CountingLLM(self.gen, log)
        t0 = time.perf_counter()
        ps = self.retrieve_passages(question)
        ev = [p["text"] for p in ps]
        a = llm(prompts.answer(self.cfg, question, ev)).strip()
        rec = {"answer": a, "abstained": is_insufficient(a), "evidence_text": {f"p:{p['pid']}": p["text"] for p in ps}}
        rec.update(self.confidence(llm, question, ev, a, seed))
        rec.update(n_calls=log.n_calls, tokens=log.tokens, latency_s=time.perf_counter() - t0)
        if rec["abstained"]:
            rec["answer"] = ""
        return rec

    def confidence(self, llm, question, ev, a, seed) -> dict:
        return {}


class SelfConsistency(DenseRAG):
    name = "Self-consistency"

    def __init__(self, *args, n_samples: int, **kw):
        super().__init__(*args, **kw)
        self.n = n_samples       # compute-matched: the measured mean number of PEAK-RAG calls per question

    def answer(self, question: str, *, seed: int = 17, **_) -> dict:
        log = CallLog()
        llm = CountingLLM(self.gen, log)
        t0 = time.perf_counter()
        ps = self.retrieve_passages(question)
        ev = [p["text"] for p in ps]
        samples = self._samples(llm, question, ev, self.n, seed)
        votes = Counter(normalize_answer(s) for s in samples)
        best = votes.most_common(1)[0][0]
        a = next(s for s in samples if normalize_answer(s) == best)
        return {"answer": "" if is_insufficient(a) else a, "abstained": is_insufficient(a),
                "evidence_text": {f"p:{p['pid']}": p["text"] for p in ps}, "confidence": votes[best] / len(samples),
                "n_calls": log.n_calls, "tokens": log.tokens, "latency_s": time.perf_counter() - t0}


class DenseSelfCheck(DenseRAG):
    """SelfCheckGPT-style score: mean NLI support of the main answer by each of k samples."""
    name = "Dense + SelfCheckGPT"

    def confidence(self, llm, question, ev, a, seed) -> dict:
        samples = self._samples(llm, question, ev, self.cfg.gating.k_samples, seed)
        if not a:
            return {"confidence": 0.0}
        p = self.nli.entail(samples, [a] * len(samples))[:, 0]
        return {"confidence": float(p.mean())}


def semantic_entropy(samples: list[str], nli) -> float:
    """Entropy of the cluster distribution, clustering by bidirectional entailment."""
    clusters: list[list[str]] = []
    for s in samples:
        for cl in clusters:
            p = nli.entail([cl[0], s], [s, cl[0]])
            if normalize_answer(cl[0]) == normalize_answer(s) or (p[0, 0] >= 0.5 and p[1, 0] >= 0.5):
                cl.append(s)
                break
        else:
            clusters.append([s])
    probs = np.array([len(c) for c in clusters], float) / len(samples)
    return float(-(probs * np.log(probs)).sum())


class DenseSemanticEntropy(DenseRAG):
    name = "Dense + semantic entropy"

    def confidence(self, llm, question, ev, a, seed) -> dict:
        samples = self._samples(llm, question, ev, self.cfg.gating.k_samples, seed)
        return {"confidence": semantic_entropy_confidence(samples, self.nli)}


def freeze_threshold(dev_confidence, budget: float = 0.10) -> float:
    """Abstain below the budget quantile of development confidences."""
    return float(np.quantile(np.asarray(dev_confidence, float), budget))


def apply_threshold(rec: dict, threshold: float) -> dict:
    out = dict(rec)
    if out.get("confidence") is not None and out["confidence"] < threshold:
        out["abstained"], out["answer"] = True, ""
    return out


def semantic_entropy_confidence(samples: list[str], nli) -> float:
    """Confidence 1 - H / log(n) from the semantic entropy of ``samples``."""
    if not samples:
        return 0.0
    return float(1.0 - semantic_entropy(samples, nli) / math.log(max(2, len(samples))))
