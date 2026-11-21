"""Per-question metrics (Section 4.1).

Primary metrics follow the source paper of each dataset: MLPQ Hits@1, XMQAs F1, TempRAG (ChronoQA)
context faithfulness, BioASQ-QA accuracy and Open-ViTabQA exact match. Full-coverage scoring: on an
answerable question an abstention counts as incorrect (score 0); on an evidence-removal unanswerable
question an abstention earns full credit (1) and any answer earns none (0).

Auxiliary metrics: atomic claims of an answer are judged by the claim-level evaluator
(Qwen/Qwen2.5-14B-Instruct, distinct from the NLI model inside the gate).
  hallucination        an answered question with >= 1 atomic claim not entailed by the reference
                       evidence (gold supporting passages where provided); absence of entailment is
                       lack of support, not contradiction.
  context faithfulness share of atomic claims entailed by the retrieved evidence C.
  context relevance    share of retrieved items judged relevant to the question.
  retrieval recall     share of gold supporting facts or passages in C.
  path recall          share of gold reasoning-path elements in C (only where gold paths are released).
"""
from __future__ import annotations

from typing import Callable, Sequence

from ..text import best_over, exact_match, split_sentences, token_f1

ClaimJudge = Callable[[str, Sequence[str]], bool]   # (claim, evidence texts) -> supported?
Decomposer = Callable[[str], list]                   # answer -> atomic claims
RelevanceJudge = Callable[[str, str], bool]          # (question, evidence item) -> relevant?


def hits_at_1(prediction: str, golds: Sequence[str]) -> float:
    """Top-1 prediction (first line of the answer) matches any gold entity or alias."""
    top = (prediction or "").strip().splitlines()[0] if prediction else ""
    return best_over(golds, top, exact_match) if top else 0.0


def accuracy(prediction: str, golds: Sequence[str]) -> float:
    """BioASQ-QA: exact label for yes/no items, exact match of any gold for factoid and list items."""
    p = (prediction or "").strip().lower()
    g = [x.strip().lower() for x in golds]
    if g and g[0] in ("yes", "no"):
        first = p.split()[0].strip(".,;:") if p else ""
        return float(first == g[0])
    return best_over(golds, prediction, exact_match)


def f1(prediction: str, golds: Sequence[str]) -> float:
    return best_over(golds, prediction, token_f1)


def em(prediction: str, golds: Sequence[str]) -> float:
    return best_over(golds, prediction, exact_match)


def context_faithfulness(answer: str, evidence: Sequence[str], judge: ClaimJudge,
                         decompose: Decomposer = split_sentences) -> float:
    claims = decompose(answer)
    if not claims:
        return 0.0
    return sum(judge(c, evidence) for c in claims) / len(claims)


def hallucinated(answer: str, reference_evidence: Sequence[str], judge: ClaimJudge,
                 decompose: Decomposer = split_sentences) -> bool:
    return any(not judge(c, reference_evidence) for c in decompose(answer))


def context_relevance(question: str, evidence: Sequence[str], judge: RelevanceJudge) -> float | None:
    if not evidence:
        return None
    return sum(judge(question, e) for e in evidence) / len(evidence)


def answer_score(metric: str, answer: str, golds: Sequence[str]) -> float:
    """Score of an answer under the dataset metric, ignoring abstention (used for the calibration target)."""
    return {"hits_at_1": hits_at_1, "f1": f1, "accuracy": accuracy, "exact_match": em}[metric](answer, golds)


def primary_score(metric: str, record: dict, golds: Sequence[str], judge: ClaimJudge | None = None,
                  decompose: Decomposer = split_sentences) -> float:
    """Full-coverage primary score of one prediction record."""
    if record.get("unanswerable"):
        return 1.0 if record["abstained"] else 0.0
    if record["abstained"]:
        return 0.0
    a = record["answer"]
    if metric == "hits_at_1":
        return hits_at_1(a, golds)
    if metric == "f1":
        return f1(a, golds)
    if metric == "accuracy":
        return accuracy(a, golds)
    if metric == "exact_match":
        return em(a, golds)
    if metric == "context_faithfulness":
        if judge is None:
            raise ValueError("context faithfulness needs a claim judge")
        return context_faithfulness(a, list(record.get("evidence_text", {}).values()), judge, decompose)
    raise ValueError(metric)


def recall(retrieved_ids: Sequence[str], gold_ids: Sequence[str]) -> float | None:
    if not gold_ids:
        return None
    got = set(retrieved_ids)
    return sum(1 for g in gold_ids if g in got) / len(gold_ids)


def nli_claim_judge(nli, threshold: float = 0.5) -> ClaimJudge:
    """Claim judge backed by an NLI model (used by the offline demo; the reported numbers use the
    Qwen2.5-14B-Instruct evaluator through :func:`llm_claim_judge`)."""
    def judge(claim: str, evidence: Sequence[str]) -> bool:
        if not evidence:
            return False
        probs = nli.entail(list(evidence), [claim] * len(evidence))
        return bool(probs[:, 0].max() >= threshold)
    return judge


def lexical_relevance_judge() -> RelevanceJudge:
    """Offline stand-in for the relevance judgement: the item shares a content word with the question."""
    from ..text import content_tokens

    def judge(question: str, item: str) -> bool:
        return bool(set(content_tokens(question)) & set(content_tokens(item)))
    return judge


def llm_relevance_judge(generator, max_new_tokens: int = 4) -> RelevanceJudge:
    from ..backends.base import Request

    def judge(question: str, item: str) -> bool:
        prompt = f"Question: {question}\nEvidence: {item}\nIs the evidence relevant to answering the question? Answer yes or no."
        return generator.generate(Request("evaluator", prompt, {}, max_new_tokens, 0.0)).text.strip().lower().startswith("yes")
    return judge


def llm_decomposer(generator, max_new_tokens: int = 384) -> Decomposer:
    """Atomic-claim decomposition by the evaluator model, following FActScore. If the output is not a
    JSON list, the answer's sentences are used as claims."""
    import json
    from ..backends.base import Request

    def decompose(answer: str) -> list:
        if not answer.strip():
            return []
        prompt = ("Break the following answer into a JSON list of atomic claims, each a short self-contained "
                  f"statement of one fact.\nAnswer: {answer}")
        text = generator.generate(Request("evaluator", prompt, {}, max_new_tokens, 0.0)).text
        try:
            claims = json.loads(text[text.index("["):text.rindex("]") + 1])
            return [c for c in claims if isinstance(c, str) and c.strip()] or split_sentences(answer)
        except ValueError:
            return split_sentences(answer)
    return decompose


def llm_claim_judge(generator, max_new_tokens: int = 4) -> ClaimJudge:
    from ..backends.base import Request

    def judge(claim: str, evidence: Sequence[str]) -> bool:
        ev = "\n".join(f"[{i + 1}] {e}" for i, e in enumerate(evidence))
        prompt = (f"Evidence:\n{ev}\nIs the following claim entailed by the evidence, with the same time and "
                  f"scope? Answer yes or no.\nClaim: {claim}")
        out = generator.generate(Request("evaluator", prompt, {"claim": claim, "evidence": list(evidence)},
                                         max_new_tokens, 0.0))
        return out.text.strip().lower().startswith("yes")
    return judge
