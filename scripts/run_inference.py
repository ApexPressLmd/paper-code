#!/usr/bin/env python3
"""Online stage: answer questions with PEAK-RAG (or an ablation variant / reference system), score each
record with the claim-level evaluator and write per-question records.

python scripts/run_inference.py --kb runs/x/kb.pkl --questions data/x/questions.jsonl \
    --system PEAK-RAG --variant full --seeds 17 29 43 71 101 --out runs/x/peak_full.jsonl \
    [--calibration runs/calibration.json] [--backend hf] [--set retrieval.lambda_mix=0.1]
Reference systems: dense, self-consistency (needs --match-calls or --sc-samples), dense+selfcheck,
dense+semantic-entropy. --verifier independent runs claim verification and self-fact generation with the
independent model family (Supplementary Table S18).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from peak_rag.backends import make_backends  # noqa: E402
from peak_rag.config import load_config, override, parse_sets  # noqa: E402
from peak_rag.data import read_jsonl, write_jsonl  # noqa: E402
from peak_rag.offline.build import KnowledgeBase  # noqa: E402
from peak_rag.runner import (REFERENCE_SYSTEMS, make_evaluator, make_system, matched_samples,  # noqa: E402
                             run_questions, score_records, system_config)


def independent_family(cfg, backend: str):
    """(aux_generator, claim_verifier) from models.independent_verifier."""
    if backend != "hf":
        from peak_rag.backends.offline import LexicalNLI, RuleLLM
        nli = LexicalNLI()
        return RuleLLM(), lambda claims, ev: [max(nli.entail(ev, [c] * len(ev))[:, 0]) if ev else 0.0 for c in claims]
    from peak_rag.backends.hf import HFGenerator
    from peak_rag.eval.metrics import llm_claim_judge
    v = cfg.models.independent_verifier
    gen = HFGenerator(v.name, v.revision, v.dtype, cfg.retrieval.max_input_tokens)
    judge = llm_claim_judge(gen)
    return gen, lambda claims, ev: [float(judge(c, ev)) for c in claims]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--system", default="PEAK-RAG", help="PEAK-RAG or one of " + ", ".join(REFERENCE_SYSTEMS))
    ap.add_argument("--name", default=None, help="system name written to the records")
    ap.add_argument("--variant", default="full")
    ap.add_argument("--split", default=None, choices=[None, "dev", "test"])
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--calibration", default=None, help="JSON written by fit_calibration.py")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key.path=value")
    ap.add_argument("--backend", default="offline", choices=["offline", "hf"])
    ap.add_argument("--evaluator", default=None, choices=[None, "offline", "hf"],
                    help="claim-level evaluator (default: same kind as --backend)")
    ap.add_argument("--match-calls", default=None, help="PEAK-RAG records whose mean call count sets the "
                                                         "self-consistency sample count")
    ap.add_argument("--sc-samples", type=int, default=None)
    ap.add_argument("--verifier", default="default", choices=["default", "independent"])
    ap.add_argument("--reference-corpus", default=None,
                    help="original corpus (before evidence removal), for the reference evidence of hallucination")
    a = ap.parse_args()
    cfg = override(system_config(load_config(a.config), a.variant), parse_sets(a.set))
    gen, emb, nli = make_backends(cfg, a.backend)
    kb = KnowledgeBase.load(a.kb)
    weights = thresholds = None
    if a.calibration:
        with open(a.calibration) as fh:
            cal = json.load(fh)
        weights, thresholds = cal["weights"], (cal["theta_low"], cal["theta_high"])
    sc = a.sc_samples or (matched_samples(read_jsonl(a.match_calls)) if a.match_calls else None)
    aux = verifier = None
    if a.verifier == "independent":
        aux, verifier = independent_family(cfg, a.backend)
    system = make_system(a.system, kb, cfg, gen, emb, nli, weights, thresholds, sc_samples=sc,
                         aux_generator=aux, claim_verifier=verifier)
    questions = [q for q in read_jsonl(a.questions) if a.split is None or q["split"] == a.split]
    seeds = a.seeds or list(cfg.evaluation.seeds)
    name = a.name or (a.system if a.variant == "full" else f"{a.system} [{a.variant}]")
    evaluator = make_evaluator(cfg, a.evaluator or a.backend)
    passage_text = {p["pid"]: p["text"] for p in kb.passages}
    if a.reference_corpus:
        from peak_rag.offline.build import split_passages
        passage_text.update({p["pid"]: p["text"] for p in split_passages(read_jsonl(a.reference_corpus))})
    records = score_records(run_questions(system, questions, seeds, name), cfg, evaluator, passage_text,
                            passage_rep=kb.passage_rep)
    for r in records:
        r["evaluator"] = evaluator.name
        if sc:
            r["sc_samples"] = sc
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    write_jsonl(a.out, records)
    calls = [r["n_calls"] for r in records]
    print(f"{name}: {len(records)} records, mean calls {sum(calls) / len(calls):.2f}, "
          f"mean primary score {100 * sum(r['primary_score'] for r in records) / len(records):.1f}")


if __name__ == "__main__":
    main()
