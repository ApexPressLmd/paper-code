"""Prompt templates (Appendix B, Table "Language-model call sites, decoding settings, and prompt
skeletons"). Each builder returns a :class:`Request` with the decoding settings of its call site.
"""
from __future__ import annotations

import json
from typing import Sequence

from .backends.base import Request

# Six labelled demonstrations for the DualProcess classifier (few-shot = 6).
CLASSIFIER_SHOTS = [
    ("Are both the bolt and the nut made of the same steel grade?", "PARALLEL"),
    ("Which of the two bearings has the higher load rating?", "PARALLEL"),
    ("Is the shaft diameter given in the drawing and in the parts list?", "PARALLEL"),
    ("Why does a rotating shaft with a keyway fail by fatigue?", "CHAINED"),
    ("How does the bending moment determine the required shaft diameter?", "CHAINED"),
    ("What material property follows from the yield stress, and which safety factor does it set?", "CHAINED"),
]


def _evidence_block(evidence: Sequence[str]) -> str:
    return "\n".join(f"[{i + 1}] {e}" for i, e in enumerate(evidence))


def decompose_query(cfg, question: str) -> Request:
    prompt = (f"Split {question!r} into the smallest independent sub-questions needed to answer it. "
              "Return a JSON list of strings.")
    return Request("decompose_query", prompt, {"question": question},
                   cfg.decoding.max_new_tokens.decompose_query, cfg.gating.answer_temperature)


def build_reasoning_dag(cfg, sub_questions: Sequence[str]) -> Request:
    listing = "\n".join(f"{i}: {s}" for i, s in enumerate(sub_questions))
    prompt = (f"Given sub-questions\n{listing}\nlist every pair (i, j) such that j needs the answer of i. "
              "Return JSON edges; the result must be acyclic.")
    return Request("build_reasoning_dag", prompt, {"sub_questions": list(sub_questions)},
                   cfg.decoding.max_new_tokens.build_reasoning_dag, cfg.gating.answer_temperature)


def classifier(cfg, question: str) -> Request:
    shots = "\n".join(f"Q: {q}\nA: {a}" for q, a in CLASSIFIER_SHOTS[: cfg.reasoning.classifier_shots])
    prompt = (f"{shots}\nLabel the question as PARALLEL (independent checks) or CHAINED (each step feeds the next). "
              f"Answer with one word.\nQ: {question}\nA:")
    return Request("classifier", prompt, {"question": question},
                   cfg.decoding.max_new_tokens.classifier, cfg.gating.answer_temperature)


def answer(cfg, question: str, evidence: Sequence[str], *, call_site: str = "answer",
           temperature: float | None = None, seed: int | None = None, previous: Sequence[str] = ()) -> Request:
    prev = ("Intermediate answers:\n" + "\n".join(previous) + "\n") if previous else ""
    prompt = (f"Evidence:\n{_evidence_block(evidence)}\n{prev}Using only the evidence, answer: {question}\n"
              "Cite the fact identifiers you used. If the evidence is insufficient, say so.")
    t = cfg.gating.answer_temperature if temperature is None else temperature
    key = {"answer": "answer", "path_answer": "path_answer", "uncertainty_sample": "uncertainty_sample"}[call_site]
    return Request(call_site, prompt, {"question": question, "evidence": list(evidence), "previous": list(previous)},
                   cfg.decoding.max_new_tokens[key], t, cfg.gating.sample_top_p if t > 0 else 1.0, seed)


def conflict_scoring(cfg, question: str, candidate: str) -> Request:
    """One call returns both the parametric self facts and the atomic claims of the candidate answer."""
    prompt = (f"Without consulting any evidence, state what you know about {question!r} as triples with a "
              "confidence in [0, 1] (field \"self_facts\"). Then split the following answer into atomic "
              f"claims (field \"atomic_facts\"). Answer: {candidate}\nReturn one JSON object.")
    budget = (cfg.decoding.max_new_tokens.conflict_scoring_atomic_facts
              + cfg.decoding.max_new_tokens.conflict_scoring_self_facts)
    return Request("conflict_scoring", prompt, {"question": question, "answer": candidate},
                   budget, cfg.gating.answer_temperature)


def regeneration(cfg, question: str, triples: Sequence[str], *, sample: bool = False, seed: int | None = None) -> Request:
    prompt = (f"Answer {question!r} using only these verified triples:\n{_evidence_block(triples)}\n"
              "Do not use passages and do not add unsupported claims.")
    if sample:
        return Request("regeneration_sample", prompt, {"question": question, "evidence": list(triples)},
                       cfg.decoding.max_new_tokens.uncertainty_sample, cfg.gating.sample_temperature,
                       cfg.gating.sample_top_p, seed)
    return Request("regeneration", prompt, {"question": question, "evidence": list(triples)},
                   cfg.decoding.max_new_tokens.regeneration, cfg.gating.regeneration_temperature)


# ---------------------------------------------------------------- offline construction prompts
def extract_facts(cfg, passage: str) -> Request:
    prompt = ("Extract every factual statement in the passage as a triple (head, relation, tail), with an "
              "optional time and scope. Use the relation names requires, precedes or based_on for "
              f"prerequisite statements. Return a JSON list.\nPassage: {passage}")
    return Request("extract_facts", prompt, {"passage": passage},
                   cfg.decoding.offline_max_new_tokens.extract_facts, 0.0)


def resolve_conflict(fact_a: dict, fact_b: dict, passage_a: str, passage_b: str, nli_contradiction: float,
                     max_new_tokens: int) -> Request:
    prompt = ("Two facts may conflict or be redundant. Using only their evidence passages, decide one of "
              "keep_both, drop_first, drop_second or merge, and give a source-based justification. Keep both "
              "when they differ in time or scope or when the relation can have several values. Return JSON "
              f"{{\"decision\": ..., \"justification\": ...}}.\nFact A: {json.dumps(fact_a)}\nPassage A: {passage_a}\n"
              f"Fact B: {json.dumps(fact_b)}\nPassage B: {passage_b}\n"
              f"Passage-level NLI contradiction score (a review signal, not a reason to delete): {nli_contradiction:.2f}")
    return Request("resolve_conflict", prompt, {"fact_a": fact_a, "fact_b": fact_b, "passage_a": passage_a,
                                                "passage_b": passage_b, "nli_contradiction": nli_contradiction},
                   max_new_tokens, 0.0)
