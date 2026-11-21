#!/usr/bin/env python3
"""Write the toy mechanical-design corpus used by the tests and the demo.

The toy corpus only exercises the code path; it is not one of the five evaluation benchmarks and none
of its outputs is reported in the paper.
Usage: python scripts/make_toy_data.py data/toy
"""
import json
import os
import sys

DOCS = {
    "statics": [
        "Statics precedes strength of materials.",
        "Free body diagrams are part of statics.",
        "Equilibrium of forces is based on Newton's laws.",
        "Statics requires equilibrium of forces.",
    ],
    "strength": [
        "Strength of materials requires statics.",
        "Stress analysis is based on strength of materials.",
        "Bending moment increases bending stress.",
        "Yield strength is measured by a tensile test.",
    ],
    "concentration": [
        "Stress concentration requires stress analysis.",
        "A keyway increases stress concentration.",
        "A shoulder fillet reduces stress concentration.",
        "Stress concentration is measured by the stress concentration factor.",
    ],
    "fatigue": [
        "Fatigue failure requires stress concentration.",
        "Fatigue failure is based on the S-N curve.",
        "Cyclic loading causes fatigue failure.",
        "Surface finish reduces fatigue strength.",
    ],
    "shaft": [
        "Shaft design requires fatigue failure.",
        "Shaft design requires bending stress.",
        "A rotating shaft is used in gearboxes.",
        "Shaft diameter reduces bending stress.",
    ],
    "gears": [
        "Gear design requires shaft design.",
        "Gear tooth bending is based on the Lewis equation.",
        "Gear contact stress causes pitting.",
        "Lubrication reduces pitting.",
    ],
    "bearings": [
        "Bearing selection requires shaft design.",
        "Bearing life is measured by the L10 rating.",
        "Misalignment reduces bearing life.",
        "Preload increases bearing stiffness.",
    ],
    "materials": [
        "Heat treatment increases yield strength.",
        "Alloy steel is used in shafts.",
        "Material selection precedes shaft design.",
        "Corrosion reduces fatigue strength.",
    ],
    # single-question documents: candidates for the evidence-removal unanswerable subset
    "couplings_a": ["A flexible coupling reduces shaft misalignment."],
    "keys_a": ["A parallel key is used in hub connections."],
    "couplings_b": ["Coupling selection requires shaft design."],
    "keys_b": ["Key design requires shear stress."],
    "springs_a": ["Spring rate is measured by a load test."],
    "springs_b": ["Spring design requires shear stress."],
}

GOLD_PATHS = {   # gold reasoning paths for the chained toy questions (path recall)
    "Why does a rotating shaft with a keyway fail by fatigue?": ["keyway", "stress concentration", "fatigue failure"],
    "How does gear design depend on statics, and then on shaft design?": ["statics", "shaft design", "gear design"],
    "Explain why bearing selection depends on fatigue failure": ["fatigue failure", "shaft design", "bearing selection"],
}

QUESTIONS = [
    # (dataset, question, answer, support doc, route)
    ("Toy-Shaft", "What does shaft design require?", "Shaft design requires fatigue failure.", "shaft", "parallel"),
    ("Toy-Shaft", "What reduces bending stress in a shaft?", "Shaft diameter reduces bending stress.", "shaft", "parallel"),
    ("Toy-Shaft", "Why does a rotating shaft with a keyway fail by fatigue?", "Fatigue failure requires stress concentration.", "fatigue", "chained"),
    ("Toy-Shaft", "What causes fatigue failure?", "Cyclic loading causes fatigue failure.", "fatigue", "parallel"),
    ("Toy-Shaft", "What is fatigue failure based on?", "Fatigue failure is based on the S-N curve.", "fatigue", "parallel"),
    ("Toy-Shaft", "Where is a rotating shaft used?", "A rotating shaft is used in gearboxes.", "shaft", "parallel"),
    ("Toy-Shaft", "What increases stress concentration?", "A keyway increases stress concentration.", "concentration", "parallel"),
    ("Toy-Shaft", "What reduces stress concentration?", "A shoulder fillet reduces stress concentration.", "concentration", "parallel"),
    ("Toy-Gear", "What does gear design require?", "Gear design requires shaft design.", "gears", "parallel"),
    ("Toy-Gear", "What is gear tooth bending based on?", "Gear tooth bending is based on the Lewis equation.", "gears", "parallel"),
    ("Toy-Gear", "What causes pitting?", "Gear contact stress causes pitting.", "gears", "parallel"),
    ("Toy-Gear", "What reduces pitting?", "Lubrication reduces pitting.", "gears", "parallel"),
    ("Toy-Gear", "How does gear design depend on statics, and then on shaft design?", "Gear design requires shaft design.", "gears", "chained"),
    ("Toy-Gear", "What does statics precede?", "Statics precedes strength of materials.", "statics", "parallel"),
    ("Toy-Gear", "What is equilibrium of forces based on?", "Equilibrium of forces is based on Newton's laws.", "statics", "parallel"),
    ("Toy-Gear", "What does strength of materials require?", "Strength of materials requires statics.", "strength", "parallel"),
    ("Toy-Bearing", "What does bearing selection require?", "Bearing selection requires shaft design.", "bearings", "parallel"),
    ("Toy-Bearing", "How is bearing life measured?", "Bearing life is measured by the L10 rating.", "bearings", "parallel"),
    ("Toy-Bearing", "What reduces bearing life?", "Misalignment reduces bearing life.", "bearings", "parallel"),
    ("Toy-Bearing", "What increases bearing stiffness?", "Preload increases bearing stiffness.", "bearings", "parallel"),
    ("Toy-Bearing", "What increases yield strength?", "Heat treatment increases yield strength.", "materials", "parallel"),
    ("Toy-Bearing", "How is yield strength measured?", "Yield strength is measured by a tensile test.", "strength", "parallel"),
    ("Toy-Bearing", "Explain why bearing selection depends on fatigue failure", "Bearing selection requires shaft design.", "bearings", "chained"),
    ("Toy-Bearing", "What does corrosion reduce?", "Corrosion reduces fatigue strength.", "materials", "parallel"),
    ("Toy-Shaft", "What reduces shaft misalignment?", "A flexible coupling reduces shaft misalignment.", "couplings_a", "parallel"),
    ("Toy-Shaft", "Where is a parallel key used?", "A parallel key is used in hub connections.", "keys_a", "parallel"),
    ("Toy-Gear", "What does coupling selection require?", "Coupling selection requires shaft design.", "couplings_b", "parallel"),
    ("Toy-Gear", "What does key design require?", "Key design requires shear stress.", "keys_b", "parallel"),
    ("Toy-Bearing", "How is spring rate measured?", "Spring rate is measured by a load test.", "springs_a", "parallel"),
    ("Toy-Bearing", "What does spring design require?", "Spring design requires shear stress.", "springs_b", "parallel"),
]


def main(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "corpus.jsonl"), "w", encoding="utf-8") as fh:
        for doc_id, sents in DOCS.items():
            fh.write(json.dumps({"doc_id": doc_id, "title": doc_id, "text": " ".join(sents)}) + "\n")
    with open(os.path.join(out_dir, "questions.jsonl"), "w", encoding="utf-8") as fh:
        for i, (ds, q, a, doc, route) in enumerate(QUESTIONS):
            # alternate dev / test so that every toy dataset has both splits
            split = "dev" if i % 2 == 0 else "test"
            fh.write(json.dumps({"qid": f"toy{i:03d}", "dataset": ds, "split": split, "question": q,
                                 "answers": [a], "route": route, "gold_support": [f"{doc}#0"],
                                 "gold_path": GOLD_PATHS.get(q, []), "unanswerable": 0}) + "\n")
    print(f"wrote {len(DOCS)} documents and {len(QUESTIONS)} questions to {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/toy")
