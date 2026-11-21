"""Configuration loading.

The YAML file ``configs/default.yaml`` holds every constant of the manuscript's configuration table.
``load_config`` returns a nested attribute dictionary; ``override`` applies dotted-key overrides, e.g.
``override(cfg, {"retrieval.lambda_mix": 0.0})`` for the lambda = 0 reference.
"""
from __future__ import annotations

import copy
import os
from typing import Any, Mapping

import yaml

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "default.yaml")


class Cfg(dict):
    """Dictionary with attribute access, recursively applied."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:  # pragma: no cover - explicit error message
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __deepcopy__(self, memo):
        return Cfg({k: copy.deepcopy(v, memo) for k, v in self.items()})


def _wrap(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return Cfg({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def load_config(path: str | None = None, overrides: Mapping[str, Any] | None = None) -> Cfg:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as fh:
        cfg = _wrap(yaml.safe_load(fh))
    cfg.setdefault("ablation", Cfg())
    for key, value in ABLATION_DEFAULTS.items():
        cfg.ablation.setdefault(key, value)
    if overrides:
        override(cfg, overrides)
    return cfg


def override(cfg: Cfg, overrides: Mapping[str, Any]) -> Cfg:
    for dotted, value in overrides.items():
        node = cfg
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, Cfg())
        if parts[-1] not in node and parts[0] != "ablation":
            raise KeyError(f"unknown configuration key: {dotted}")
        node[parts[-1]] = _wrap(value)
    return cfg


def parse_sets(items) -> dict[str, Any]:
    """Command-line overrides ``key.path=value`` (value parsed as YAML), e.g. retrieval.lambda_mix=0.1."""
    out = {}
    for item in items or []:
        key, _, raw = item.partition("=")
        if not key or not _:
            raise ValueError(f"expected key=value, got {item!r}")
        out[key.strip()] = yaml.safe_load(raw)
    return out


# Switches used by the ablations of Section 4.4 and Supplementary Tables S2, S3, S5, S8 and S17.
# All default to the full system.
ABLATION_DEFAULTS: dict[str, Any] = {
    "prereq_teleport": True,        # C1 coupling point (ii): PPR teleport bias (Eq. 2)
    "prereq_tiebreak": True,        # C1 coupling point (i): foundationality tie-breaking in pi(q)
    "propagation": "forward",       # forward | reverse | undirected | none   (Supplementary Table S2)
    "mining_signals": ["typed", "cooc", "order"],   # Supplementary Table S5
    "gate": True,                   # C2: conflict-aware faithfulness gate
    "use_v_branch": True,           # Supplementary Table S17
    "use_c_answer": True,
    "use_c_param": True,
    "use_u_faith": True,
    "uncalibrated_average": False,  # replace Eq. 3 by an unweighted mean of the three signals
    "regeneration": True,
    "threshold_abstention": True,
    "router": "predicted",          # predicted | oracle | parallel | chained | both  (Supplementary Table S8)
}


# presets that change the mined graph P; they are applied by build_graph.py, not at inference
BUILD_TIME_PRESETS = frozenset({"typed-only", "cooc-only", "order-only"})


def ablation_preset(name: str) -> dict[str, Any]:
    """Named variants of Table 6 (ablation) and the supplementary component tables."""
    presets = {
        "full": {},
        "-C1": {"ablation.prereq_teleport": False, "ablation.prereq_tiebreak": False},
        "-C2": {"ablation.gate": False},
        "-both": {"ablation.prereq_teleport": False, "ablation.prereq_tiebreak": False, "ablation.gate": False},
        "-tiebreak": {"ablation.prereq_tiebreak": False},
        "-teleport": {"ablation.prereq_teleport": False},
        "lambda0": {"retrieval.lambda_mix": 0.0},
        "reverse": {"ablation.propagation": "reverse"},
        "undirected": {"ablation.propagation": "undirected"},
        "no-prereq": {"ablation.propagation": "none"},
        "typed-only": {"ablation.mining_signals": ["typed"]},
        "cooc-only": {"ablation.mining_signals": ["cooc"]},
        "order-only": {"ablation.mining_signals": ["order"]},
        "-v_branch": {"ablation.use_v_branch": False},
        "-c_answer": {"ablation.use_c_answer": False},
        "-c_param": {"ablation.use_c_param": False},
        "-u_faith": {"ablation.use_u_faith": False},
        "uncalibrated": {"ablation.uncalibrated_average": True},
        "-regeneration": {"ablation.regeneration": False},
        "-threshold-abstention": {"ablation.threshold_abstention": False},
        "router-oracle": {"ablation.router": "oracle"},
        "router-parallel": {"ablation.router": "parallel"},
        "router-chained": {"ablation.router": "chained"},
        "router-both": {"ablation.router": "both"},
    }
    for k in (1, 3, 10):
        presets[f"k={k}"] = {"gating.k_samples": k}
    if name not in presets:
        raise KeyError(f"unknown ablation preset {name!r}; choose from {sorted(presets)}")
    return presets[name]
