"""Glue used by the command-line scripts and the tests: run a system over questions, score the records
with the claim-level evaluator, and turn them into the prediction and calibration tables consumed by
``peak_rag.eval.tables``."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .baselines import DenseRAG, DenseSelfCheck, DenseSemanticEntropy, SelfConsistency
from .calibration import correctness_label
from .config import BUILD_TIME_PRESETS, ablation_preset, load_config, override
from .eval import metrics as M
from .online.pipeline import PeakRAG
from .text import split_sentences

REFERENCE_SYSTEMS = {"dense": DenseRAG, "self-consistency": SelfConsistency,
                     "dense+selfcheck": DenseSelfCheck, "dense+semantic-entropy": DenseSemanticEntropy}


def system_config(base_cfg, variant: str = "full", *, inference: bool = True):
    """Copy of the configuration with a named ablation preset applied (see config.ablation_preset).
    Mining-signal variants change the graph and must be applied when it is built (build_graph.py)."""
    if inference and variant in BUILD_TIME_PRESETS:
        raise ValueError(f"variant {variant!r} changes the mined graph: build it with "
                         f"`scripts/build_graph.py --variant={variant}` and run inference on that graph")
    cfg = copy.deepcopy(base_cfg)
    return override(cfg, ablation_preset(variant))


def make_system(name: str, kb, cfg, gen, emb, nli, weights=None, thresholds=None, *, sc_samples: int | None = None,
                aux_generator=None, claim_verifier=None):
    if name == "self-consistency":
        if not sc_samples:
            raise ValueError("self-consistency is compute-matched: pass the measured mean number of PEAK-RAG "
                             "calls (run_inference.py --match-calls <PEAK-RAG records> or --sc-samples N)")
        return SelfConsistency(kb, cfg, gen, emb, nli, n_samples=sc_samples)
    if name in REFERENCE_SYSTEMS:
        return REFERENCE_SYSTEMS[name](kb, cfg, gen, emb, nli)
    return PeakRAG(kb, cfg, gen, emb, nli, weights, thresholds, aux_generator, claim_verifier)


def matched_samples(records: Iterable[dict], split: str = "test") -> int:
    """Rounded mean number of model calls of PEAK-RAG records on ``split``."""
    calls = [r["n_calls"] for r in records if r.get("n_calls") and r.get("split", split) == split]
    return max(1, int(round(float(np.mean(calls))))) if calls else 1


def run_questions(system, questions: Iterable[dict], seeds: Iterable[int], system_name: str) -> list[dict]:
    out = []
    questions = list(questions)
    for seed in seeds:
        for q in questions:
            rec = system.answer(q["question"], seed=seed, oracle_route=q.get("route"))
            rec.update(dataset=q["dataset"], qid=q["qid"], split=q["split"], system=system_name, seed=seed,
                       unanswerable=int(q.get("unanswerable", 0)), answers=q["answers"], question=q["question"],
                       gold_support=q.get("gold_support", []), gold_path=q.get("gold_path", []))
            out.append(rec)
    return out


# ---------------------------------------------------------------------------- evaluator
@dataclass
class Evaluator:
    """Claim-level evaluator: atomic-claim decomposition, claim support and item relevance.
    Reported results: Qwen/Qwen2.5-14B-Instruct (models.evaluator), distinct from the gate's NLI model."""
    judge: M.ClaimJudge
    decompose: M.Decomposer
    relevance: M.RelevanceJudge
    name: str


def make_evaluator(cfg, kind: str = "offline", device: str = "cuda") -> Evaluator:
    if kind == "hf":
        from .backends.hf import HFGenerator
        e = cfg.models.evaluator
        gen = HFGenerator(e.name, e.revision, e.dtype, cfg.retrieval.max_input_tokens, device)
        ec = cfg.evaluation.evaluator
        return Evaluator(M.llm_claim_judge(gen, ec.judge_max_new_tokens), M.llm_decomposer(gen, ec.decompose_max_new_tokens),
                         M.llm_relevance_judge(gen, ec.judge_max_new_tokens), e.name)
    from .backends.offline import LexicalNLI      # a separate instance from the gate's NLI model
    return Evaluator(M.nli_claim_judge(LexicalNLI()), split_sentences, M.lexical_relevance_judge(),
                     "offline lexical stand-in")


def score_records(records: list[dict], cfg, evaluator: Evaluator, passage_text: dict[str, str] | None = None,
                  default_metric: str = "f1", passage_rep: dict[str, str] | None = None) -> list[dict]:
    """Primary score (full-coverage), calibration target y, and the auxiliary metrics of Section 4.1.

    ``passage_text`` should cover the original corpus, including the passages removed for the
    evidence-removal subset, so that answered unanswerable questions are checked against their reference
    evidence. ``passage_rep`` maps merged passages to their representative node, so that a gold passage
    merged during structural unification counts as retrieved when its representative is retrieved."""
    passage_text = passage_text or {}
    members: dict[str, set[str]] = {}
    for pid, rep in (passage_rep or {}).items():
        members.setdefault(rep, set()).add(pid)
    metric_of = dict(cfg.evaluation.primary_metric)
    for r in records:
        metric = metric_of.get(r["dataset"], default_metric)
        r["metric"] = metric
        r["primary_score"] = M.primary_score(metric, r, r["answers"], evaluator.judge, evaluator.decompose)
        raw = r.get("candidate", r.get("answer", ""))
        r["y"] = correctness_label(metric, raw, r["answers"], f1_threshold=cfg.gating.calibration_f1_threshold,
                                   unanswerable=bool(r["unanswerable"]))
        ev = r.get("evidence_text") or {}
        ev_ids, ev_text = list(map(str, ev.keys())), list(ev.values())
        answered = not r["abstained"] and bool(r.get("answer"))
        r["context_faithfulness"] = (M.context_faithfulness(r["answer"], ev_text, evaluator.judge, evaluator.decompose)
                                     if answered else None)
        gold = [passage_text[g] for g in r.get("gold_support", []) if g in passage_text]
        reference = gold or ev_text            # identified source passages when no gold support is released
        # every answered test question counts, including answered evidence-removal questions
        r["hallucinated"] = (int(M.hallucinated(r["answer"], reference, evaluator.judge, evaluator.decompose))
                             if answered else None)
        r["context_relevance"] = M.context_relevance(r.get("question", ""), ev_text, evaluator.relevance)
        retrieved = {i[2:] for i in ev_ids if i.startswith("p:")} | set(r.get("evidence_pids", []))
        retrieved |= {m for pid in list(retrieved) for m in members.get(pid, ())}
        r["retrieval_recall"] = M.recall(sorted(retrieved), r.get("gold_support", []))
        if r.get("gold_path"):
            joined = " ".join(ev_text).lower()
            r["path_recall"] = sum(1 for g in r["gold_path"] if str(g).lower() in joined) / len(r["gold_path"])
        else:
            r["path_recall"] = None
    return records


# ---------------------------------------------------------------------------- frames
PRED_COLS = ["dataset", "qid", "system", "seed", "split", "unanswerable", "abstained", "primary_score", "y",
             "confidence", "s", "n_calls", "tokens", "latency_s", "context_faithfulness", "hallucinated",
             "context_relevance", "retrieval_recall", "path_recall"]


def prediction_frame(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame([{c: r.get(c) for c in PRED_COLS} for r in records])
    # PEAK-RAG ranks its answers by the gate score s
    df["confidence"] = pd.to_numeric(df["confidence"]).combine_first(pd.to_numeric(df["s"]))
    return df


def calibration_frame(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        if not r.get("signals"):
            continue
        sg = r["signals"]
        rows.append({"dataset": r["dataset"], "qid": r["qid"], "split": r["split"], "seed": r["seed"],
                     "unanswerable": r["unanswerable"], "v_branch": sg["v_branch"], "c_conf": sg["c_conf"],
                     "u_faith": sg["u_faith"], "s": r["s"], "y": r["y"], "action_initial": r["action_initial"],
                     "action_final": r["action_final"], "s_after_regeneration": r["s_after_regeneration"]})
    return pd.DataFrame(rows)


def thresholded_systems(pred: pd.DataFrame, budget: float, systems: Iterable[str]) -> pd.DataFrame:
    """Thresholded abstention alternatives (Figure 9): for each listed system that records a confidence,
    freeze the abstention threshold at the ``budget`` quantile of its development confidences and abstain
    below it on the test split. Returns prediction rows named '<system> + threshold'."""
    from .baselines import freeze_threshold
    out = []
    for system in systems:
        g = pred[(pred.system == system) & pred.confidence.notna()]
        dev, test = g[g.split == "dev"], g[g.split == "test"].copy()
        if dev.empty or test.empty:
            continue
        th = freeze_threshold(dev.confidence.to_numpy(float), budget)
        low = test.confidence.astype(float) < th
        test.loc[low, "abstained"] = True
        test.loc[low & (test.unanswerable == 0), "primary_score"] = 0.0
        test.loc[low & (test.unanswerable == 1), "primary_score"] = 1.0
        test["system"] = system + " + threshold"
        test["threshold"] = th
        out.append(test)
    return pd.concat(out, ignore_index=True) if out else pred.iloc[0:0]


__all__ = ["load_config", "make_system", "make_evaluator", "run_questions", "score_records", "prediction_frame",
           "calibration_frame", "system_config", "thresholded_systems", "matched_samples"]
