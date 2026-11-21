"""The default configuration must equal the values stated in the manuscript (configuration table,
external-components table, Sections 3.3-3.6, Appendix D)."""
from peak_rag.config import load_config


def test_construction_and_mining_constants():
    c = load_config()
    assert c.construction.conflict_similarity_delta == 0.83
    assert c.construction.unification_threshold == 0.86
    assert (c.construction.bridge_weights.fact, c.construction.bridge_weights.ontology,
            c.construction.bridge_weights.source) == (1.0, 0.6, 0.4)
    assert c.construction.schema_percentile_tau == 40
    pm = c.prerequisite_mining
    assert (pm.omega_typed, pm.omega_cooc, pm.omega_order_numerator) == (1.0, 0.5, 0.3)
    assert set(pm.typed_relations) == {"requires", "precedes", "based_on"}
    assert pm.duplicate_merge == "max"


def test_retrieval_reasoning_gating_constants():
    c = load_config()
    r, g, rs = c.retrieval, c.gating, c.reasoning
    assert (r.lambda_mix, r.alpha_teleport, r.ppr_max_iter, r.ppr_tol) == (0.3, 0.15, 100, 1e-8)
    assert (r.seed_top_k, r.mmr_dedup, r.context_budget, r.max_input_tokens) == (30, 0.97, 12, 4096)
    assert (rs.classifier_shots, rs.denoiser_top_n, rs.chained_max_hops, rs.beam_width, rs.parallel_redundancy) == (6, 8, 4, 4, 0.95)
    assert (g.k_samples, g.sample_temperature, g.answer_temperature, g.regeneration_temperature) == (5, 0.7, 0.0, 0.0)
    assert (g.theta_low, g.theta_high, g.abstention_budget, g.precision_target) == (0.35, 0.70, 0.10, 0.85)
    assert g.evidence_path_facts == 6 and g.max_regenerations == 1 and g.sample_top_p == 0.9
    assert g.weights == {"w1": 2.0, "w2": 2.0, "w3": 1.4254, "b": -3.0288}
    assert g.l2_eta == 0.01 and g.calibration_f1_threshold == 0.80


def test_decoding_models_evaluation_constants():
    c = load_config()
    m = c.decoding.max_new_tokens
    assert (m.decompose_query, m.build_reasoning_dag, m.classifier, m.answer, m.path_answer) == (256, 256, 8, 512, 512)
    assert (m.conflict_scoring_atomic_facts, m.conflict_scoring_self_facts, m.uncertainty_sample, m.regeneration) == (384, 384, 512, 512)
    assert c.decoding.nli_window_tokens == 512
    assert c.models.generator.name == "Qwen/Qwen2.5-7B-Instruct" and c.models.generator.revision == "a09a354"
    assert c.models.embedder.name == "BAAI/bge-m3" and c.models.embedder.revision == "5617a9f" and c.models.embedder.dim == 1024
    assert c.models.nli.name == "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli" and c.models.nli.revision == "8adb042"
    assert c.models.evaluator.name == "Qwen/Qwen2.5-14B-Instruct"
    ev = c.evaluation
    assert ev.seeds == [17, 29, 43, 71, 101]
    assert (ev.resampling_seed_main, ev.resampling_seed_supp) == (20261003, 20261004)
    assert (ev.n_bootstrap, ev.n_permutations, ev.ece_bins) == (10000, 10000, 10)
    assert (ev.dev_questions_per_dataset, ev.test_questions_per_dataset, ev.unanswerable_per_dataset) == (1000, 1000, 100)
    assert dict(ev.comparators) == {"MLPQ": "Yuan et al.", "XMQAs": "LAMRF", "TempRAG": "MI-RAG",
                                    "BioASQ": "PARKA", "Open-ViTabQA": "Ren et al."}
    assert dict(ev.primary_metric) == {"MLPQ": "hits_at_1", "XMQAs": "f1", "TempRAG": "context_faithfulness",
                                       "BioASQ": "accuracy", "Open-ViTabQA": "exact_match"}
