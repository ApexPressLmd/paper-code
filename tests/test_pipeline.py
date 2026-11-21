"""End-to-end tests on the toy corpus with the CPU stand-in backends: call accounting (Section 3.7),
gate routing, ablation switches and the evaluation tables."""
import copy
import importlib.util
import os

import pytest

from peak_rag.backends import make_backends
from peak_rag.config import ablation_preset, load_config, override
from peak_rag.offline.build import build_knowledge_base
from peak_rag.online.pipeline import PeakRAG
from peak_rag.runner import (calibration_frame, make_evaluator, prediction_frame, run_questions, score_records,
                             system_config, thresholded_systems)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def setup():
    spec = importlib.util.spec_from_file_location("mk", os.path.join(ROOT, "scripts", "make_toy_data.py"))
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    corpus = [{"doc_id": d, "title": d, "text": " ".join(s)} for d, s in mk.DOCS.items()]
    questions = [{"qid": f"toy{i:03d}", "dataset": ds, "split": "dev" if i % 2 == 0 else "test", "question": q,
                  "answers": [a], "route": r, "gold_support": [f"{doc}#0"], "gold_path": mk.GOLD_PATHS.get(q, []),
                  "unanswerable": 0}
                 for i, (ds, q, a, doc, r) in enumerate(mk.QUESTIONS)]
    cfg = load_config()
    gen, emb, nli = make_backends(cfg, "offline")
    kb = build_knowledge_base(corpus, cfg, gen, emb, nli)
    return cfg, gen, emb, nli, kb, questions


def test_offline_build(setup):
    cfg, *_, kb, _ = setup
    assert kb.prereq and kb.active_facts and kb.transition.shape[0] == len(kb.nodes)
    col = kb.transition.sum(axis=0).A.ravel()
    assert ((abs(col - 1) < 1e-9) | kb.dangling).all()                     # column-stochastic
    assert all(e["omega"] in (1.0, 0.5) or e["source"] == "order" for e in kb.prereq.values())


def test_call_counts(setup):
    cfg, gen, emb, nli, kb, qs = setup
    sysm = PeakRAG(kb, cfg, gen, emb, nli)
    for q in qs:
        rec = sysm.answer(q["question"], seed=17)
        n, site = rec["n_calls"], rec["calls_by_site"]
        regen = site.get("regeneration", 0)
        assert regen <= 1
        if rec["route"] == "parallel":
            assert n == 9 + 6 * regen
            assert site == {k: v for k, v in {"decompose_query": 1, "classifier": 1, "answer": 1, "conflict_scoring": 1,
                                              "uncertainty_sample": 5, "regeneration": regen,
                                              "regeneration_sample": 5 * regen}.items() if v}
        else:
            assert 11 <= n - 6 * regen <= 13 and site["build_reasoning_dag"] == 1 and 2 <= site["path_answer"] <= 4
        if regen:
            assert 15 <= n <= 19 and rec["action_initial"] == "regenerate"
            assert rec["action_final"] in ("regenerate", "abstain")
        assert rec["action_final"] != "abstain" or rec["answer"] == ""


def test_call_counts_scale_with_k(setup):
    cfg, gen, emb, nli, kb, qs = setup
    for k in (1, 3, 10):
        c = override(copy.deepcopy(cfg), ablation_preset(f"k={k}"))
        rec = PeakRAG(kb, c, gen, emb, nli, thresholds=(0.0, 0.0)).answer("What reduces pitting?", seed=17)
        assert rec["route"] == "parallel" and rec["n_calls"] == 4 + k


def test_regeneration_once_and_rule(setup):
    cfg, gen, emb, nli, kb, _ = setup
    sysm = PeakRAG(kb, cfg, gen, emb, nli, thresholds=(0.0, 0.999))     # force the middle band
    rec = sysm.answer("What reduces pitting?", seed=17)
    assert rec["action_initial"] == "regenerate" and rec["calls_by_site"]["regeneration"] == 1
    assert rec["s_after_regeneration"] is not None
    assert (rec["action_final"] == "regenerate") == (rec["s_after_regeneration"] >= 0.999)


def test_ablation_switches(setup):
    cfg, gen, emb, nli, kb, _ = setup
    q = "Why does a rotating shaft with a keyway fail by fatigue?"
    no_c2 = PeakRAG(kb, override(copy.deepcopy(cfg), ablation_preset("-C2")), gen, emb, nli).answer(q)
    assert no_c2["signals"] is None and "conflict_scoring" not in no_c2["calls_by_site"]
    no_thr = PeakRAG(kb, override(copy.deepcopy(cfg), ablation_preset("-threshold-abstention")), gen, emb, nli,
                     thresholds=(0.999, 0.9999)).answer(q)
    assert no_thr["action_initial"] == "regenerate"
    lam0 = override(copy.deepcopy(cfg), ablation_preset("lambda0"))
    assert lam0.retrieval.lambda_mix == 0.0


def test_scoring_and_tables(setup):
    from peak_rag.eval import tables as T
    cfg, gen, emb, nli, kb, qs = setup
    recs = run_questions(PeakRAG(kb, cfg, gen, emb, nli), qs, [17, 29], "PEAK-RAG")
    recs += run_questions(PeakRAG(kb, override(copy.deepcopy(cfg), ablation_preset("-C1")), gen, emb, nli),
                          qs, [17, 29], "PEAK-RAG [-C1]")
    recs = score_records(recs, cfg, make_evaluator(cfg), {p["pid"]: p["text"] for p in kb.passages})
    assert all(r["retrieval_recall"] is not None for r in recs)
    assert any(r["path_recall"] is not None for r in recs)
    pred = prediction_frame(recs)
    abl = T.ablation_table(pred[pred.split == "test"])
    assert {"PathRec", "AbstF1", "Hallu"} <= set(abl.columns)
    mean, sd, fmt = T.main_table(pred[pred.split == "test"])
    assert set(mean.index) == {"PEAK-RAG", "PEAK-RAG [-C1]"}
    pt = T.paired_table(pred[pred.split == "test"], {d: "PEAK-RAG [-C1]" for d in mean.columns}, n_boot=200, n_perm=200)
    assert (pt.p_holm >= pt.p_raw - 1e-12).all()
    cal = calibration_frame([r for r in recs if r["system"] == "PEAK-RAG"])
    rep = T.calibration_report(cal, 0.35, 0.70)
    assert 0 <= rep["test_ece"] <= 1 and "abstention" in rep
    lodo = T.lodo_table(cal)
    assert set(lodo.fit) == {"LODO", "pooled"}


def test_thresholded_baselines_and_compute_matching(setup):
    from peak_rag.runner import make_system, matched_samples
    cfg, gen, emb, nli, kb, qs = setup
    sc = make_system("dense+selfcheck", kb, cfg, gen, emb, nli)
    recs = score_records(run_questions(sc, qs, [17], "dense+selfcheck"), cfg, make_evaluator(cfg))
    th = thresholded_systems(prediction_frame(recs), 0.10, ["dense+selfcheck"])
    assert set(th.system) == {"dense+selfcheck + threshold"} and (th.split == "test").all()
    with pytest.raises(ValueError):
        make_system("self-consistency", kb, cfg, gen, emb, nli)
    assert matched_samples([{"n_calls": 9}, {"n_calls": 17}]) == 13


def test_guards(setup):
    cfg, gen, emb, nli, kb, _ = setup
    with pytest.raises(ValueError):
        system_config(cfg, "typed-only")              # build-time variant at inference
    oracle = PeakRAG(kb, override(copy.deepcopy(cfg), ablation_preset("router-oracle")), gen, emb, nli)
    with pytest.raises(ValueError):
        oracle.answer("What reduces pitting?")       # no gold route label
    assert oracle.answer("What reduces pitting?", oracle_route="parallel")["calls_by_site"].get("classifier") is None
