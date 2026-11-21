"""Unit tests of the two contributions and of the gate calibration."""
import networkx as nx
import numpy as np
import pytest

from peak_rag.calibration import fit_gate, score, select_thresholds
from peak_rag.config import load_config
from peak_rag.offline.prereq import foundationality, mine_prerequisites
from peak_rag.online import gate as G
from peak_rag.online.retrieval import kahn_sort, mix_teleport, prerequisite_weights


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_mining_signals_directions_and_max_merge(cfg):
    facts = [{"head": "shaft design", "relation": "requires", "tail": "fatigue"},
             {"head": "statics", "relation": "precedes", "tail": "strength"},
             {"head": "stress", "relation": "based_on", "tail": "strength"}]
    concepts = {"shaft design", "fatigue", "statics", "strength", "stress", "fillet"}
    mentions = {"d1": [("statics", "d1#0", 0), ("strength", "d1#0", 5), ("fillet", "d1#1", 10_000_000)],
                "d2": [("fatigue", "d2#0", 0), ("shaft design", "d2#0", 3)]}
    P = mine_prerequisites(facts, mentions, concepts, cfg)
    assert P[("fatigue", "shaft design")] == {"omega": 1.0, "source": "typed"}   # X requires Y: Y -> X
    assert P[("statics", "strength")] == {"omega": 1.0, "source": "typed"}       # max(1.0, 0.5, 0.3)
    assert P[("strength", "stress")]["omega"] == 1.0                              # X based_on Y: Y -> X
    assert P[("statics", "fillet")] == {"omega": pytest.approx(0.3 / 2), "source": "order"}   # 0.3/(j-i), j-i = 2
    assert P[("strength", "fillet")] == {"omega": pytest.approx(0.3), "source": "order"}
    # ranks over the corpus D: with a global order in which d2's concepts come first, j - i grows
    rank = {"fatigue": 0, "shaft design": 1, "statics": 2, "strength": 3, "fillet": 4}
    Pg = mine_prerequisites(facts, mentions, concepts, cfg, ["order"], rank)
    assert Pg[("statics", "fillet")]["omega"] == pytest.approx(0.3 / 2)
    assert ("fatigue", "statics") not in Pg                                       # never in one document
    only_cooc = mine_prerequisites(facts, mentions, concepts, cfg, ["cooc"])
    assert only_cooc[("statics", "strength")] == {"omega": 0.5, "source": "cooc"}
    assert ("statics", "fillet") not in only_cooc                                 # different passages
    assert foundationality(P)["statics"] == 2


class _KB:
    def __init__(self, prereq):
        self.prereq = prereq


def test_eq1_prerequisite_weights_and_directions():
    kb = _KB({("S", "M"): {"omega": 1.0}, ("M", "F"): {"omega": 0.5}, ("K", "F"): {"omega": 1.0}, ("F", "D"): {"omega": 1.0}})
    rel = {"F": 0.8, "D": 0.6}
    w = prerequisite_weights(kb, ["F", "D"], rel, "forward")
    assert w == pytest.approx({"M": 0.5 * 0.8, "K": 1.0 * 0.8, "F": 1.0 * 0.6})
    assert prerequisite_weights(kb, ["F"], {"F": 1.0}, "reverse") == {"D": 1.0}
    assert prerequisite_weights(kb, ["F"], {"F": 1.0}, "none") == {}


def test_eq2_teleport_mixing():
    t = np.array([2.0, 2.0, 0.0, 0.0])
    w = np.array([0.0, 0.0, 3.0, 1.0])
    tt = mix_teleport(t, w, 0.3)
    assert np.allclose(tt, 0.7 * np.array([0.5, 0.5, 0, 0]) + 0.3 * np.array([0, 0, 0.75, 0.25]))
    assert np.allclose(mix_teleport(t, np.zeros(4), 0.3), t / t.sum())         # fallback to t
    assert np.isclose(tt.sum(), 1.0)


def test_ppr_matches_networkx_with_dangling_teleport():
    from scipy import sparse
    from peak_rag.online.retrieval import personalized_pagerank
    g = nx.DiGraph()
    g.add_weighted_edges_from([(0, 1, 1.0), (1, 2, 0.6), (2, 0, 0.4), (1, 3, 1.0), (4, 1, 1.0)])  # 3 is dangling
    n = 5
    A = sparse.lil_matrix((n, n))
    for u, v, d in g.edges(data=True):
        A[v, u] = d["weight"]
    A = A.tocsr()
    out = np.asarray(A.sum(axis=0)).ravel()
    kb = type("KB", (), {})()
    kb.transition = (A @ sparse.diags(np.where(out > 0, 1 / np.where(out > 0, out, 1), 0))).tocsr()
    kb.dangling = out == 0
    tele = np.array([0.1, 0.2, 0.3, 0.0, 0.4])
    r = personalized_pagerank(kb, tele, 0.15, 100, 1e-12)
    ref = nx.pagerank(g, alpha=0.85, personalization=dict(enumerate(tele)), dangling=dict(enumerate(tele)),
                      max_iter=1000, tol=1e-14, weight="weight")
    assert np.allclose(r, [ref[i] for i in range(n)], atol=1e-8)


def test_kahn_sort_breaks_ties_by_foundationality():
    assert kahn_sort(3, [], [1.0, 5.0, 3.0]) == [1, 2, 0]
    assert kahn_sort(3, [(2, 1)], [1.0, 5.0, 3.0]) == [2, 1, 0]


def test_gate_score_and_decision_bands(cfg):
    sig = G.Signals(v_branch=0.86, c_answer=0.2, c_param=0.16, c_conf=0.18, u_faith=0.88)
    s = G.gate_score(sig, cfg, cfg.gating.weights)
    z = 2.0 * 0.86 + 2.0 * (1 - 0.18) + 1.4254 * 0.88 - 3.0288
    assert s == pytest.approx(1 / (1 + np.exp(-z)))
    assert G.decide(0.70, cfg, 0.35, 0.70) == "accept"
    assert G.decide(0.69, cfg, 0.35, 0.70) == "regenerate"
    assert G.decide(0.35, cfg, 0.35, 0.70) == "regenerate"
    assert G.decide(0.34, cfg, 0.35, 0.70) == "abstain"


def test_calibration_constraints_and_threshold_rules():
    rng = np.random.default_rng(0)
    X = rng.random((5000, 3))
    z = 2 * X[:, 0] + 2 * X[:, 1] + 1.4 * X[:, 2] - 3.0
    y = (rng.random(5000) < 1 / (1 + np.exp(-z))).astype(float)
    X[:, 2] = 1 - X[:, 2]                       # a feature with a negative association must get w = 0
    w = fit_gate(X, y, 0.01)
    assert w["w3"] == 0.0 and w["w1"] > 0 and w["w2"] > 0 and w["converged"]
    s = score(X, w)
    lo, hi = select_thresholds(s, y, 0.10, 0.85)
    assert np.mean(s < lo) == pytest.approx(0.10, abs=0.001)
    assert y[s >= hi].mean() >= 0.85
    smaller = s[s < hi]
    if len(smaller):                            # hi is the smallest score reaching the target
        assert all(y[s >= t].mean() < 0.85 for t in np.sort(smaller)[-20:])


def test_calibration_label_uses_metric_scorer():
    from peak_rag.calibration import correctness_label
    assert correctness_label("accuracy", "Yes, because the bolt is preloaded.", ["yes"]) == 1
    assert correctness_label("hits_at_1", "Statics\nsecond line", ["statics"]) == 1
    assert correctness_label("f1", "shaft design requires fatigue", ["Shaft design requires fatigue failure."]) == 1
    assert correctness_label("exact_match", "The evidence is insufficient.", ["x"]) == 0
    assert correctness_label("exact_match", "x", ["x"], unanswerable=True) == 0


def test_conflict_scoring_field_caps(cfg):
    import json
    from peak_rag.backends.base import CallLog, CountingLLM, Response

    class Big:
        def generate(self, req):
            out = {"atomic_facts": ["word " * 100] * 10, "self_facts": [{"text": "word " * 100}] * 10}
            return Response(json.dumps(out), 1, 1)
    claims, sf = G.conflict_and_claims(CountingLLM(Big(), CallLog()), cfg, "q", "a", lambda s: len(s.split()))
    assert len(claims) == 3 and len(sf) == 3                                    # 3 x 100 <= 384 < 4 x 100
