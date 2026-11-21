"""End-to-end inference (Section 3.5, Figure 4; Section 3.7 call accounting).

Model calls per query at the default k = 5 (NLI scoring and embedding are not counted):
  decomposition 1 + classifier 1 + answer 1 + conflict scoring 1 + k uncertainty samples = 9 (parallel);
  a chained question adds the reasoning DAG (1) and 2-4 ordered path answers in place of the single
  answer call, i.e. 11-13 calls; one regeneration adds one answer call and k samples (15-19 calls).
"""
from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

import numpy as np

from .. import prompts
from ..backends.base import CallLog, CountingLLM
from ..offline.build import KnowledgeBase
from ..text import is_insufficient, split_sentences
from . import gate as G
from .retrieval import Evidence, retrieve


class PeakRAG:
    def __init__(self, kb: KnowledgeBase, cfg, generator, embedder, nli, weights: dict | None = None,
                 thresholds: tuple[float, float] | None = None, aux_generator=None, claim_verifier=None):
        """``aux_generator`` (self-fact generation) and ``claim_verifier`` (claim verification) replace the
        backbone and the NLI model for the independent-family analysis of Supplementary Table S18."""
        self.kb, self.cfg, self.gen, self.emb, self.nli = kb, cfg, generator, embedder, nli
        self.aux_gen, self.claim_verifier = aux_generator, claim_verifier
        tok = getattr(generator, "tokenizer", None)
        self.count_tokens = (lambda s: len(tok.tokenize(s))) if tok is not None else None
        self.weights = dict(weights or cfg.gating.weights)
        self.theta_low, self.theta_high = thresholds or (cfg.gating.theta_low, cfg.gating.theta_high)

    # ------------------------------------------------------------------ helpers
    def _samples(self, llm, question, texts, k, seed, regen=False) -> list[str]:
        out = []
        for j in range(k):
            s = seed * 1000 + j + (500 if regen else 0)
            if regen:
                req = prompts.regeneration(self.cfg, question, texts, sample=True, seed=s)
            else:
                req = prompts.answer(self.cfg, question, texts, call_site="uncertainty_sample",
                                     temperature=self.cfg.gating.sample_temperature, seed=s)
            out.append(llm(req).strip())
        return out

    def _evidence_path(self, answer: str, ev: list[Evidence]) -> list[str]:
        facts = [e for e in ev if e.kind == "fact"]
        if not facts:
            return []
        # support of the answer by each fact: premise = fact, hypothesis = answer
        support = (self.nli.entail([e.text for e in facts], [answer] * len(facts))[:, 0]
                   if answer else np.zeros(len(facts)))
        order = sorted(range(len(facts)), key=lambda i: (-support[i], -facts[i].score))
        return [facts[i].eid for i in order[: self.cfg.gating.evidence_path_facts]]

    def _branch(self, llm, route, question, ret, ev):
        if route == "parallel":
            return G.parallel_branch(llm, self.cfg, question, ev), 0.0, []
        return G.chained_branch(llm, self.cfg, self.kb, ret, ev, self.emb, question)

    # ------------------------------------------------------------------ main
    def answer(self, question: str, *, seed: int = 17, oracle_route: str | None = None) -> dict[str, Any]:
        cfg, ab = self.cfg, self.cfg.ablation
        log = CallLog()
        llm = CountingLLM(self.gen, log)
        t0 = time.perf_counter()

        router = ab.router
        if router == "predicted":
            route = G.classify(llm, cfg, question)
        elif router == "oracle":
            if oracle_route not in ("parallel", "chained"):
                raise ValueError("oracle routing needs a gold 'route' label for every question")
            route = oracle_route
        elif router in ("parallel", "chained"):
            route = router
        else:  # both: run both branches, keep the one with the higher branch verification
            route = "both"

        ret = retrieve(self.kb, question, llm, self.emb, cfg, route="chained" if route in ("chained", "both") else "parallel")
        rec: dict[str, Any] = {"question": question, "seed": seed, "sub_questions": ret.sub_questions,
                               "order": ret.order, "concepts": ret.concepts}

        if route == "both":
            cands = []
            for r in ("parallel", "chained"):
                ev_r = G.task_aware_denoise(ret.evidence, r, ret, self.emb, question, cfg)
                a_r, rel_r, path_r = self._branch(llm, r, question, ret, ev_r)
                v_r = float(np.mean(G.nli_support(self.nli, ev_r, split_sentences(a_r)))) if r == "parallel" and a_r else rel_r
                cands.append((v_r, r, a_r, rel_r, path_r, ev_r))
            _, route, candidate, path_rel, path, ev = max(cands, key=lambda c: c[0])
        else:
            ev = G.task_aware_denoise(ret.evidence, route, ret, self.emb, question, cfg)
            candidate, path_rel, path = self._branch(llm, route, question, ret, ev)
        rec.update(route=route, candidate=candidate, evidence=[e.eid for e in ev], path=[f["fid"] for f in path])

        if not ab.gate:  # -C2: the candidate is returned; an explicit statement of insufficiency is an abstention
            final = candidate
            action_initial = action_final = "abstain" if is_insufficient(candidate) else "accept"
            sig = None
            s = s_regen = None
        else:
            aux = CountingLLM(self.aux_gen, log) if self.aux_gen is not None else llm
            claims, self_facts = G.conflict_and_claims(aux, cfg, question, candidate, self.count_tokens)
            texts = [e.text for e in ev]
            samples = self._samples(llm, question, texts, cfg.gating.k_samples, seed)
            sig = G.compute_signals(cfg, self.nli, self.emb, question, candidate, ev, claims, self_facts,
                                    samples, route, path_rel, self.claim_verifier)
            s = G.gate_score(sig, cfg, self.weights)
            action_initial = G.decide(s, cfg, self.theta_low, self.theta_high)
            s_regen = None
            final = candidate
            if action_initial == "accept":
                action_final = "accept"
            elif action_initial == "abstain":
                action_final, final = "abstain", ""
            elif not ab.regeneration or cfg.gating.max_regenerations < 1:
                action_final = "accept"            # middle band returned without regeneration
            else:
                top_facts = [e.text for e in sorted(ev, key=lambda e: -e.score) if e.kind == "fact"][: cfg.gating.evidence_path_facts]
                if not top_facts:
                    action_final, final = "abstain", ""
                else:
                    regen = llm(prompts.regeneration(cfg, question, top_facts)).strip()
                    regen_samples = self._samples(llm, question, top_facts, cfg.gating.k_samples, seed, regen=True)
                    sig2 = G.compute_signals(cfg, self.nli, self.emb, question, regen, ev, split_sentences(regen),
                                             self_facts, regen_samples, route, path_rel, self.claim_verifier)
                    s_regen = G.gate_score(sig2, cfg, self.weights)
                    rec["signals_after_regeneration"] = asdict(sig2)
                    if s_regen >= self.theta_high or not ab.threshold_abstention:
                        action_final, final = "regenerate", regen
                    else:
                        action_final, final = "abstain", ""
            if action_final != "abstain" and is_insufficient(final):
                action_final, final = "abstain", ""

        abstained = action_final == "abstain"
        rec.update(
            answer="" if abstained else final, abstained=abstained, action_initial=action_initial,
            action_final=action_final, s=s, s_after_regeneration=s_regen,
            signals=asdict(sig) if sig else None,
            evidence_path=[] if abstained else self._evidence_path(final, ev),
            n_calls=log.n_calls, calls_by_site=log.by_site(), tokens=log.tokens,
            latency_s=time.perf_counter() - t0, evidence_text={e.eid: e.text for e in ev},
            evidence_pids=sorted({e.pid for e in ev if e.pid}),
        )
        return rec
