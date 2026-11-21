#!/usr/bin/env bash
# End-to-end smoke run on the toy corpus with the CPU stand-in backends (no GPU, no downloads).
# Exercises every stage: offline build, evidence-removal subset, PEAK-RAG and its ablations, reference
# baselines, calibration fitting on dev, and all evaluation tables. The toy numbers are not results.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=runs/toy_demo
SEEDS="17 29 43 71 101"
rm -rf "$OUT" && mkdir -p "$OUT"

python scripts/make_toy_data.py data/toy
python scripts/make_unanswerable.py --corpus data/toy/corpus.jsonl --questions data/toy/questions.jsonl \
  --out-dir "$OUT/data" --per-dataset 1
python scripts/build_graph.py --corpus "$OUT/data/corpus.jsonl" --out "$OUT/kb.pkl" > /dev/null

# 1) development pass with the released calibration, then refit the gate on these dev records
python scripts/run_inference.py --kb "$OUT/kb.pkl" --questions "$OUT/data/questions.jsonl" --split dev \
  --seeds 17 --out "$OUT/peak_dev.jsonl"
python scripts/fit_calibration.py --records "$OUT/peak_dev.jsonl" --out "$OUT/calibration.json" > /dev/null

# 2) all systems on both splits (dev records are kept for the calibration analyses)
python scripts/run_inference.py --kb "$OUT/kb.pkl" --questions "$OUT/data/questions.jsonl" --seeds $SEEDS \
  --calibration "$OUT/calibration.json" --reference-corpus data/toy/corpus.jsonl --out "$OUT/peak.jsonl"
for v in -C1 -C2 -both; do
  python scripts/run_inference.py --kb "$OUT/kb.pkl" --questions "$OUT/data/questions.jsonl" --split test \
    --seeds $SEEDS --variant="$v" --calibration "$OUT/calibration.json" --out "$OUT/peak_$v.jsonl"
done
# reference systems; the thresholded alternatives need development records to freeze their thresholds
python scripts/run_inference.py --kb "$OUT/kb.pkl" --questions "$OUT/data/questions.jsonl" --split test \
  --seeds $SEEDS --system dense --out "$OUT/dense.jsonl"
python scripts/run_inference.py --kb "$OUT/kb.pkl" --questions "$OUT/data/questions.jsonl" --split test \
  --seeds $SEEDS --system self-consistency --match-calls "$OUT/peak.jsonl" --out "$OUT/self-consistency.jsonl"
for s in dense+selfcheck dense+semantic-entropy; do
  python scripts/run_inference.py --kb "$OUT/kb.pkl" --questions "$OUT/data/questions.jsonl" \
    --seeds $SEEDS --system "$s" --out "$OUT/$s.jsonl"
done
# mining-signal ablation: a separate graph
python scripts/build_graph.py --corpus "$OUT/data/corpus.jsonl" --variant=typed-only --out "$OUT/kb_typed.pkl" > /dev/null
python scripts/run_inference.py --kb "$OUT/kb_typed.pkl" --questions "$OUT/data/questions.jsonl" --split test \
  --seeds $SEEDS --name "PEAK-RAG [typed-only]" --calibration "$OUT/calibration.json" --out "$OUT/peak_typed-only.jsonl"

THETA=$(python -c "import json; c=json.load(open('$OUT/calibration.json')); print(c['theta_low'], c['theta_high'])")
python scripts/evaluate.py --records "$OUT"/peak.jsonl "$OUT"/peak_-*.jsonl "$OUT"/peak_typed-only.jsonl \
  "$OUT"/dense.jsonl "$OUT"/self-consistency.jsonl "$OUT"/dense+*.jsonl \
  --out "$OUT/tables" --theta $THETA \
  --comparators-json '{"Toy-Shaft": "dense", "Toy-Gear": "dense", "Toy-Bearing": "self-consistency"}'
ls "$OUT/tables"
