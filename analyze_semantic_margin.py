"""
analyze_semantic_margin.py

Extends analyze_feature_space.py (Experiment D) with a semantic-margin analysis:
tests whether feature-space displacement under distortion actually damages
class separability, not just how far features move.

REUSES from your existing analyze_feature_space.py / pipeline (do not re-derive):
  - the same 25 matched (clean, simulated, real) triplets
  - the same M2 / M3 checkpoints and CLIP backbone
  - the same 25 text anchors
  - the already-computed clean->simulated and clean->real cosine distances

Fill in the four `# TODO` blocks below by pasting the exact code/data from your
existing analyze_feature_space.py, so everything lines up with Experiment D
instead of being recomputed from scratch.
"""

import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from pathlib import Path

from analyze_feature_space import load_square, simulate_combined
from eval_recognition import load_head, represent, require
from features import anchor_rows, build_encoders, load_caption_table
from kit import file_in, load_config, pick_device, project_path, read_csv, read_rgb, set_seed

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Reuse existing setup -- paste in from analyze_feature_space.py
# ---------------------------------------------------------------------------

# Experiment D seed, config, and checkpoint paths (analyze_feature_space.load_models).
SEED = 0
set_seed(SEED)
_cfg = load_config(str(Path(__file__).with_name("config.yaml")))
_device = pick_device(_cfg["device"])
_image_size = int(_cfg["clip"]["image_size"])
CHECKPOINTS = {
    "M2": str(file_in(_cfg, "ckpt_dir", "m2_ckpt", SEED)),
    "M3": str(project_path(_cfg, _cfg["paths"]["ckpt_dir"]) / "m3.pt"),
}
for _ckpt in CHECKPOINTS.values():
    require(Path(_ckpt))

# Same caption table and anchor_rows normalization as Experiment D.
_names, _captions = load_caption_table(file_in(_cfg, "results_dir", "captions_json", SEED))
_encoder, _text_encoder = build_encoders(_cfg, _device)
_anchor_matrix = anchor_rows(_text_encoder, _names, _captions)
CLASS_ANCHORS = {name: _anchor_matrix[index].numpy() for index, name in enumerate(_names)}

# Triplets and distances are the rows Experiment D wrote. Simulated views were not
# image files; sim_path is the distort() call recorded in that CSV.
_metric_rows = read_csv(project_path(_cfg, _cfg["paths"]["results_dir"]) / "feature_space_metrics.csv")
TRIPLETS = []
_seen_ids = set()
DIST = {}
for _row in _metric_rows:
    if _row["instance_id"] not in _seen_ids:
        _seen_ids.add(_row["instance_id"])
        TRIPLETS.append({
            "instance_id": _row["instance_id"],
            "class": _row["class"],
            "clean_path": _row["clean_path"],
            "sim_path": _row["simulated_path"],
            "real_path": _row["real_path"],
        })
    DIST[(_row["model"], _row["instance_id"], "simulated")] = float(_row["clean_sim_cosine_distance"])
    DIST[(_row["model"], _row["instance_id"], "real")] = float(_row["clean_real_cosine_distance"])

_dim = int(_anchor_matrix.shape[1])
_HEADS = {name: load_head(Path(path), _dim, _device) for name, path in CHECKPOINTS.items()}
_SIMULATED = None


def _simulated_batch():
    """Same 25-image distort() batch Experiment D encoded (combined, severity 1.0, seed 0)."""
    global _SIMULATED
    if _SIMULATED is None:
        clean_rows = [{"clean_path": Path(trip["clean_path"])} for trip in TRIPLETS]
        clean = load_square(clean_rows, "clean_path", _image_size)
        _SIMULATED = simulate_combined(clean.to(_device), SEED).cpu()
    return _SIMULATED


def encode_image(model_name, image_path):
    """eval_recognition.represent: frozen CLIP, then the Experiment D head."""
    text = str(image_path)
    marker = "batch_index="
    if text.startswith("in-memory invariance_core.distort") and marker in text:
        start = text.index(marker) + len(marker)
        index = int(text[start:text.index(")", start)])
        image = _simulated_batch()[index].unsqueeze(0)
    else:
        image = read_rgb(Path(text), _image_size).unsqueeze(0)
    with torch.no_grad():
        feature = represent(model_name, _encoder, _HEADS[model_name], image.to(_device))
    return feature.cpu().numpy()[0]


# ---------------------------------------------------------------------------
# 2. Semantic margin computation
# ---------------------------------------------------------------------------

def margin_for(embedding, true_class, anchors):
    """
    embedding: (D,) normalized image feature
    anchors: dict[class_name] -> (D,) normalized text anchor
    Returns: correct_sim, nearest_wrong_sim, margin, predicted_class
    """
    sims = {c: float(np.dot(embedding, a)) for c, a in anchors.items()}
    correct_sim = sims[true_class]
    wrong = {c: s for c, s in sims.items() if c != true_class}
    nearest_wrong_class = max(wrong, key=wrong.get)
    nearest_wrong_sim = wrong[nearest_wrong_class]
    predicted_class = max(sims, key=sims.get)
    margin = correct_sim - nearest_wrong_sim
    return correct_sim, nearest_wrong_sim, margin, predicted_class


def run():
    rows = []
    for model_name in ["M2", "M3"]:
        for trip in TRIPLETS:
            for condition, path_key in [("clean", "clean_path"),
                                         ("simulated", "sim_path"),
                                         ("real", "real_path")]:
                emb = encode_image(model_name, trip[path_key])
                correct_sim, nearest_wrong_sim, margin, pred = margin_for(
                    emb, trip["class"], CLASS_ANCHORS
                )
                feature_distance = 0.0 if condition == "clean" else DIST.get(
                    (model_name, trip["instance_id"], condition), float("nan")
                )
                rows.append({
                    "model": model_name,
                    "instance_id": trip["instance_id"],
                    "class": trip["class"],
                    "condition": condition,
                    "feature_distance": feature_distance,
                    "correct_similarity": correct_sim,
                    "nearest_wrong_similarity": nearest_wrong_sim,
                    "semantic_margin": margin,
                    "predicted_class": pred,
                    "correct": int(pred == trip["class"]),
                })

    out_csv = RESULTS_DIR / "semantic_margin_metrics.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[saved] {out_csv}  ({len(rows)} rows)")

    return rows


# ---------------------------------------------------------------------------
# 3. Aggregate margins + drops
# ---------------------------------------------------------------------------

def summarize(rows):
    print("\n=== Mean semantic margin ===")
    means = {}
    for model in ["M2", "M3"]:
        for cond in ["clean", "simulated", "real"]:
            vals = [r["semantic_margin"] for r in rows
                    if r["model"] == model and r["condition"] == cond]
            m = float(np.mean(vals)) if vals else float("nan")
            means[(model, cond)] = m
            print(f"  {model} {cond:10s}: mean margin = {m:.4f}  (n={len(vals)})")

    print("\n=== Margin drop from clean ===")
    for model in ["M2", "M3"]:
        clean = means[(model, "clean")]
        for cond in ["simulated", "real"]:
            drop = clean - means[(model, cond)]
            print(f"  {model} {cond} margin drop: {drop:.4f}")

    return means


# ---------------------------------------------------------------------------
# 4. Sanity check against Experiment C (STOP on mismatch)
# ---------------------------------------------------------------------------

# From your existing Experiment C results -- update if these have changed.
EXPERIMENT_C_REAL_ACCURACY = {
    "M2": 0.76,
    "M3": 0.68,
}

def sanity_check(rows):
    print("\n=== Sanity check vs Experiment C (real recapture accuracy) ===")
    mismatch = False
    for model in ["M2", "M3"]:
        real_rows = [r for r in rows if r["model"] == model and r["condition"] == "real"]
        acc = float(np.mean([r["correct"] for r in real_rows])) if real_rows else float("nan")
        expected = EXPERIMENT_C_REAL_ACCURACY[model]
        ok = abs(acc - expected) < 1e-6
        status = "OK" if ok else "MISMATCH"
        print(f"  {model}: anchor-based accuracy = {acc:.4f}, Experiment C = {expected:.4f}  [{status}]")
        if not ok:
            mismatch = True
    if mismatch:
        print("\n[STOP] Real-recapture accuracy from anchor similarities does not match "
              "Experiment C. Do not proceed until this is resolved -- likely cause is a "
              "different checkpoint, different anchor set, or different preprocessing "
              "between the two scripts.")
        raise SystemExit(1)
    print("  Sanity check passed: anchor-based classification matches Experiment C.")


# ---------------------------------------------------------------------------
# 5. Plots: clean->distorted distance vs margin drop
# ---------------------------------------------------------------------------

def plot_distance_vs_margin(rows, model_name):
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    clean_by_id = {r["instance_id"]: r for r in rows
                   if r["model"] == model_name and r["condition"] == "clean"}

    for condition, color, label in [("simulated", "#1f4e9c", "Simulated"),
                                     ("real", "#c9781f", "Real recapture")]:
        xs, ys = [], []
        for r in rows:
            if r["model"] != model_name or r["condition"] != condition:
                continue
            clean_r = clean_by_id.get(r["instance_id"])
            if clean_r is None:
                continue
            margin_drop = clean_r["semantic_margin"] - r["semantic_margin"]
            xs.append(r["feature_distance"])
            ys.append(margin_drop)
        ax.scatter(xs, ys, c=color, label=label, alpha=0.8,
                   edgecolors="white", linewidths=0.5)

    ax.axhline(0, color="#999999", lw=1, ls="--")
    ax.set_xlabel("clean -> distorted cosine distance")
    ax.set_ylabel("semantic margin drop from clean")
    ax.set_title(f"{model_name}: feature displacement vs. semantic margin drop")
    ax.legend()
    fig.tight_layout()
    out_path = RESULTS_DIR / f"distance_vs_margin_{model_name.lower()}.png"
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    print(f"[saved] {out_path}")


# ---------------------------------------------------------------------------
# 6. Most interesting real-recapture cases: small distance, big margin drop
# ---------------------------------------------------------------------------

def print_interesting_cases(rows, model_name="M3", top_k=5):
    clean_by_id = {r["instance_id"]: r for r in rows
                   if r["model"] == model_name and r["condition"] == "clean"}
    real_rows = [r for r in rows if r["model"] == model_name and r["condition"] == "real"]

    cases = []
    for r in real_rows:
        clean_r = clean_by_id.get(r["instance_id"])
        if clean_r is None:
            continue
        margin_drop = clean_r["semantic_margin"] - r["semantic_margin"]
        cases.append({
            "instance_id": r["instance_id"],
            "true_class": r["class"],
            "clean_margin": clean_r["semantic_margin"],
            "real_margin": r["semantic_margin"],
            "margin_drop": margin_drop,
            "clean_to_real_distance": r["feature_distance"],
            "predicted_class": r["predicted_class"],
        })

    # small displacement, large margin damage -> rank by margin_drop desc,
    # break ties toward smaller distance
    cases.sort(key=lambda c: (-c["margin_drop"], c["clean_to_real_distance"]))

    print(f"\n=== Top {top_k} interesting real-recapture cases ({model_name}) ===")
    print("(large semantic margin drop despite relatively small feature displacement)")
    for c in cases[:top_k]:
        print(f"  id={c['instance_id']:12s} true={c['true_class']:15s} "
              f"clean_margin={c['clean_margin']:.4f} real_margin={c['real_margin']:.4f} "
              f"clean->real_dist={c['clean_to_real_distance']:.4f} pred={c['predicted_class']}")


# ---------------------------------------------------------------------------
# 7. Summary (cautious wording, no proof claims)
# ---------------------------------------------------------------------------

def print_summary(means):
    m2_real_drop = means[("M2", "clean")] - means[("M2", "real")]
    m3_real_drop = means[("M3", "clean")] - means[("M3", "real")]

    print("\n=== Summary (preliminary, n=25) ===")

    if m3_real_drop > m2_real_drop:
        print(f"- M3 loses more semantic margin on real recaptures than M2 "
              f"({m3_real_drop:.4f} vs {m2_real_drop:.4f}), consistent with the earlier "
              f"Experiment C reversal (M2 outperforming M3 on real recapture accuracy).")
    else:
        print(f"- M3's real-recapture margin drop ({m3_real_drop:.4f}) does not exceed "
              f"M2's ({m2_real_drop:.4f}); this does not support a straightforward "
              f"'M3 degrades more semantically' story on its own.")

    print("- Whether small representation shifts can still cause large semantic damage: "
          "see the printed interesting cases above. Any case with a small "
          "clean->real distance and a large margin drop is preliminary evidence "
          "that displacement magnitude alone does not determine classification "
          "outcome, but with n=25 this is suggestive, not conclusive.")

    print("- Feature-distance magnitude alone explaining classification failures: "
          "if margin drop and feature distance are not tightly correlated in the "
          "scatter plots, that suggests distance-based robustness claims from "
          "Experiment D need the margin lens to be interpreted correctly; if they "
          "are tightly correlated, the two framings agree and distance was already "
          "a reasonable proxy.")

    print("\nThese are preliminary, small-sample (n=25) observations. No claim of "
          "overfitting or proof is made; read the actual numbers above before citing "
          "any of this externally.")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Traceability ===")
    print("Checkpoints:", json.dumps(CHECKPOINTS, indent=2))
    print("Num class anchors:", len(CLASS_ANCHORS))
    print("Num triplets:", len(TRIPLETS))
    print("Results dir:", RESULTS_DIR.resolve())

    if not TRIPLETS or not CLASS_ANCHORS:
        raise SystemExit(
            "Fill in TODO #1-#4 (checkpoints, anchors, triplets, cached distances) "
            "and TODO #5 (encode_image) before running."
        )

    rows = run()
    sanity_check(rows)
    means = summarize(rows)
    plot_distance_vs_margin(rows, "M2")
    plot_distance_vs_margin(rows, "M3")
    print_interesting_cases(rows, "M3", top_k=5)
    print_summary(means)
