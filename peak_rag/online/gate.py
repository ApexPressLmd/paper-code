"""Contribution 2: question-type-aware reasoning and the conflict-aware faithfulness gate
(Section 3.4; Algorithm 3; Eqs. 3-4).

Signals
  v_branch  parallel: mean NLI verification score of the atomic claims of the candidate against C;
            chained: relevance of the top path found by expanding fact edges up to 4 hops, beam B = 4.
  c_conf    = 1/2 c_answer + 1/2 c_param,
            c_answer = 1 - NLIEntail(candidate, C),
            c_param  = mean over SelfFacts(q) of rel * (1 - faithful).
  u_faith   = mean(faith) * consistency over k = 5 samples at T = 0.7.
Score       s = sigmoid(w1 v_branch + w2 (1 - c_conf) + w3 u_faith + b)                      (Eq. 3)
Action      Accept if s >= theta_high; Regenerate if theta_low <= s < theta_high; else Abstain (Eq. 4).
Regeneration (middle band only, at most once): answer from the top fact triples of C at T = 0, recompute
the signals with k new samples, return the regenerated answer only if s' >= theta_high, else abstain.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .. import prompts
from ..backends.base import CountingLLM, clipped_cosine
from ..offline.build import KnowledgeBase, _parse_json
from ..text import normalize_answer, split_sentences
from .retrieval import Evidence, RetrievalResult


@dataclass
class Signals:
    v_branch: float
    c_answer: float
    c_param: float
    c_conf: float
    u_faith: float
    faith: list[float] = field(default_factory=list)
    consistency: float = 0.0


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------------- routing and denoising
def classify(llm: CountingLLM, cfg, question: str) -> str:
    out = llm(prompts.classifier(cfg, question)).strip().upper()
    return "chained" if out.startswith("CHAIN") else "parallel"


def task_aware_denoise(ev: list[Evidence], route: str, ret: RetrievalResult, embedder, question: str, cfg) -> list[Evidence]:
    """top-N = 8. Parallel drops cross-redundant evidence (similarity >= 0.95); chained keeps evidence
    lying on the reasoning chain (items that mention a concept of a sub-question, or share an entity with
    such an item)."""
    rcfg = cfg.reasoning
    if not ev:
        return []
    emb = embedder.encode([e.text for e in ev])
    qrel = clipped_cosine(embedder.encode([question]), emb)[0]
    order = np.argsort(-(qrel + np.array([e.score for e in ev]) / (max(e.score for e in ev) or 1.0)))
    kept: list[int] = []
    if route == "parallel":
        for i in order:
            if any(float(emb[i] @ emb[j]) >= rcfg.parallel_redundancy for j in kept):
                continue
            kept.append(int(i))
            if len(kept) >= rcfg.denoiser_top_n:
                break
    else:
        chain = {c for cs in ret.concepts for c in cs}
        on_chain = [i for i in order if set(ev[i].entities) & chain or any(c.lower() in ev[i].text.lower() for c in chain)]
        linked = {e for i in on_chain for e in ev[i].entities}
        second = [i for i in order if i not in on_chain and set(ev[i].entities) & linked]
        rest = [i for i in order if i not in on_chain and i not in second]
        kept = [int(i) for i in (on_chain + second + rest)[: rcfg.denoiser_top_n]]
    return [ev[i] for i in kept]


# ---------------------------------------------------------------------------- branches
def nli_support(nli, evidence: list[Evidence], hypotheses: list[str]) -> np.ndarray:
    """Entailment of each hypothesis by the evidence: maximum over evidence items."""
    if not evidence or not hypotheses:
        return np.zeros(len(hypotheses))
    prem = [e.text for e in evidence]
    probs = nli.entail([p for _ in hypotheses for p in prem], [h for h in hypotheses for _ in prem])
    return probs[:, 0].reshape(len(hypotheses), len(prem)).max(axis=1)


def best_path(kb: KnowledgeBase, ret: RetrievalResult, embedder, question: str, cfg) -> tuple[list[dict], float]:
    """Expand fact edges from the seed entities up to ``chained_max_hops`` hops, keeping the ``beam_width``
    best partial paths by mean clipped cosine relevance of their facts to q."""
    rc = cfg.reasoning
    seeds = [c for c in (ret.concepts[ret.order[0]] if ret.order else [])] or \
            [c for cs in ret.concepts for c in cs]
    facts = kb.active_facts
    if not seeds or not facts:
        return [], 0.0
    q = embedder.encode([question])[0]
    rel = {f["fid"]: float(np.clip(kb.fact_emb[f["fid"]] @ q, 0, 1)) for f in facts}
    adj: dict[str, list[dict]] = {}
    for f in facts:
        adj.setdefault(f["head"], []).append(f)
        adj.setdefault(f["tail"], []).append(f)
    beam = [([], s, 0.0) for s in seeds]
    best: tuple[list[dict], float] = ([], 0.0)
    for _hop in range(rc.chained_max_hops):
        cand = []
        for path, node, _ in beam:
            used = {f["fid"] for f in path}
            for f in adj.get(node, []):
                if f["fid"] in used:
                    continue
                nxt = f["tail"] if f["head"] == node else f["head"]
                new = path + [f]
                cand.append((new, nxt, float(np.mean([rel[x["fid"]] for x in new]))))
        if not cand:
            break
        cand.sort(key=lambda x: -x[2])
        beam = cand[: rc.beam_width]
        if beam[0][2] > best[1]:
            best = (beam[0][0], beam[0][2])
    return best


def parallel_branch(llm: CountingLLM, cfg, question: str, ev: list[Evidence]) -> str:
    return llm(prompts.answer(cfg, question, [e.text for e in ev])).strip()


def chained_branch(llm: CountingLLM, cfg, kb, ret: RetrievalResult, ev: list[Evidence], embedder, question: str) -> tuple[str, float, list[dict]]:
    """Ordered path answers: one call per step of pi(q), 2-4 steps; the last step answers q."""
    path, relevance = best_path(kb, ret, embedder, question, cfg)
    steps = [ret.sub_questions[i] for i in ret.order]
    if len(steps) < 2:                       # a one-step chain is answered as (sub-question, question)
        steps = steps + [question]
    steps = steps[: cfg.reasoning.max_chain_steps]
    path_text = [kb.fact_text(f) for f in path]
    previous: list[str] = []
    answer = ""
    for j, s in enumerate(steps):
        target = question if j == len(steps) - 1 else s
        answer = llm(prompts.answer(cfg, target, path_text + [e.text for e in ev], call_site="path_answer",
                                    previous=previous)).strip()
        previous.append(f"{s} -> {answer}")
    return answer, relevance, path


# ---------------------------------------------------------------------------- signals
def _cap_tokens(items: list, text_of, limit: int, count_tokens) -> list:
    """Keep items in order while their cumulative length stays within ``limit`` tokens."""
    out, used = [], 0
    for it in items:
        n = count_tokens(text_of(it))
        if used + n > limit:
            break
        out.append(it)
        used += n
    return out


def conflict_and_claims(llm: CountingLLM, cfg, question: str, candidate: str, count_tokens=None) -> tuple[list[str], list[dict]]:
    """One conflict-scoring call returns the atomic claims of the candidate and the parametric self facts.
    Each field has its own limit (384 new tokens each, 768 for the call); output beyond a field's limit is
    discarded."""
    from ..text import approx_token_count
    count_tokens = count_tokens or approx_token_count
    m = cfg.decoding.max_new_tokens
    out = _parse_json(llm(prompts.conflict_scoring(cfg, question, candidate)), {})
    if not isinstance(out, dict):
        out = {}
    claims = [c for c in out.get("atomic_facts", []) if isinstance(c, str) and c.strip()]
    claims = _cap_tokens(claims, lambda c: c, m.conflict_scoring_atomic_facts, count_tokens)
    self_facts = []
    for sf in out.get("self_facts", []):
        if isinstance(sf, dict):
            text = sf.get("text") or " ".join(map(str, sf.get("triple", [])))
            if text:
                self_facts.append({"text": text, "confidence": float(sf.get("confidence", 1.0))})
    self_facts = _cap_tokens(self_facts, lambda s: s["text"], m.conflict_scoring_self_facts, count_tokens)
    return claims or split_sentences(candidate), self_facts


def compute_signals(cfg, nli, embedder, question: str, candidate: str, ev: list[Evidence], claims: list[str],
                    self_facts: list[dict], samples: list[str], route: str, path_relevance: float,
                    claim_verifier=None) -> Signals:
    """``claim_verifier(claims, evidence) -> scores`` replaces NLI claim verification when an independent
    verifier is used (Supplementary Table S18)."""
    ab = cfg.ablation
    if route == "parallel":
        if not claims:
            v = 0.0
        elif claim_verifier is not None:
            v = float(np.mean(claim_verifier(claims, [e.text for e in ev])))
        else:
            v = float(np.mean(nli_support(nli, ev, claims)))
    else:
        v = float(path_relevance)
    c_answer = 1.0 - float(nli_support(nli, ev, [candidate])[0]) if candidate else 1.0
    if self_facts:
        q = embedder.encode([question])
        rel = clipped_cosine(q, embedder.encode([s["text"] for s in self_facts]))[0]
        faithful = nli_support(nli, ev, [s["text"] for s in self_facts])
        c_param = float(np.mean(rel * (1.0 - faithful)))
    else:
        c_param = 0.0
    if not ab.use_c_answer and ab.use_c_param:
        c_conf = c_param
    elif not ab.use_c_param and ab.use_c_answer:
        c_conf = c_answer
    elif not ab.use_c_param and not ab.use_c_answer:
        c_conf = 0.0
    else:
        c_conf = 0.5 * c_answer + 0.5 * c_param
    faith = list(map(float, nli_support(nli, ev, samples))) if samples else []
    consistency = self_consistency(samples, nli)
    u = float(np.mean(faith)) * consistency if samples else 0.0
    return Signals(v, c_answer, c_param, c_conf, u, faith, consistency)


def self_consistency(samples: list[str], nli) -> float:
    """Share of samples in the largest cluster of equivalent answers (normalised string equality or
    bidirectional NLI entailment >= 0.5)."""
    if not samples:
        return 0.0
    clusters: list[list[str]] = []
    for s in samples:
        for cl in clusters:
            rep = cl[0]
            if normalize_answer(rep) == normalize_answer(s):
                cl.append(s)
                break
            p = nli.entail([rep, s], [s, rep])
            if p[0, 0] >= 0.5 and p[1, 0] >= 0.5:
                cl.append(s)
                break
        else:
            clusters.append([s])
    return max(len(c) for c in clusters) / len(samples)


def gate_score(sig: Signals, cfg, weights: dict) -> float:
    ab = cfg.ablation
    terms = [(ab.use_v_branch, weights["w1"], sig.v_branch),
             (ab.use_c_answer or ab.use_c_param, weights["w2"], 1.0 - sig.c_conf),
             (ab.use_u_faith, weights["w3"], sig.u_faith)]
    if ab.uncalibrated_average:
        vals = [x for on, _, x in terms if on]
        return float(np.mean(vals)) if vals else 0.0
    return sigmoid(sum(w * x for on, w, x in terms if on) + weights["b"])


def decide(s: float, cfg, theta_low: float, theta_high: float) -> str:
    """Eq. (4)."""
    if s >= theta_high:
        return "accept"
    if s >= theta_low or not cfg.ablation.threshold_abstention:
        return "regenerate"
    return "abstain"
