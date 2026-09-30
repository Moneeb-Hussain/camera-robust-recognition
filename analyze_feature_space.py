"""Compare how simulated distortion and real recapture move M2 and M3 features.

Primary sample: the 25 Experiment C triplets (clean, combined severity 1.0, phone recapture).
Semantic separation uses the recognition test split when that recapture set has one image per class.
"""
import inspect
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import eval_recognition
from eval_recognition import load_head, represent, require
from features import anchor_rows, build_encoders, load_caption_table
from invariance_core import distort
from kit import (
    IMAGE_SUFFIXES,
    announce,
    base_parser,
    file_in,
    load_config,
    pick_device,
    project_path,
    read_csv,
    read_rgb,
    set_seed,
    write_csv,
)
from select_recapture_images import group_by_class, pick_one_each, test_paths
from train_baselines import make_dataset

CONDITIONS = ("clean", "simulated", "real")
CONDITION_COLOR = {"clean": "#1f77b4", "simulated": "#ff7f0e", "real": "#d62728"}
GAP_LABEL_TOLERANCE = 0.05


def require_experiment_a_distort() -> str:
    """Stop if this script is not calling the distortion used by Experiment A."""
    if distort is not eval_recognition.distort:
        raise RuntimeError(
            "invariance_core.distort is not the function eval_recognition uses. "
            "Refusing to substitute another distortion."
        )
    original = inspect.unwrap(distort)
    source = inspect.getsourcefile(original)
    source_path = Path(source) if source else None
    if source_path is None or not source_path.is_file() or source_path.name != "invariance_core.py":
        raise RuntimeError(
            "The Experiment A distortion implementation invariance_core.distort could not be found."
        )
    print(f"distortion function: {original.__module__}.{original.__qualname__}")
    print(f"distortion file: {source_path.resolve()}")
    print("distortion wrapper: torch.no_grad (same object eval_recognition.distort)")
    print("distortion used by: eval_recognition.accumulate")
    return str(source_path.resolve())


def find_recapture(photo_dir: Path, filename: str) -> tuple[Path | None, str | None]:
    """Match a manifest name to the phone photo. The stem must match; the suffix may be .jpeg."""
    exact = photo_dir / filename
    if exact.is_file():
        return exact.resolve(), None
    if not photo_dir.is_dir():
        return None, f"recapture folder missing: {photo_dir}"
    stem = Path(filename).stem.lower()
    hits = []
    seen = set()
    for path in photo_dir.iterdir():
        if not path.is_file() or path.stem.lower() != stem or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        hits.append(resolved)
    if len(hits) == 1:
        return hits[0], None
    if not hits:
        return None, f"recapture cannot be matched to a file: {exact.resolve()}"
    listed = ", ".join(str(path) for path in hits)
    return None, f"multiple recapture files for {Path(filename).stem}: {listed}"


def resolve_triplets(cfg: dict, seed: int) -> list[dict]:
    """Pair each Experiment C recapture with the clean test image select_recapture_images picked."""
    manifest_path = file_in(cfg, "results_dir", "recapture_manifest", seed)
    if not manifest_path.is_file():
        raise RuntimeError(f"Missing Experiment C manifest {manifest_path}")
    image_dir = project_path(cfg, cfg["paths"]["image_dir"])
    photo_dir = project_path(cfg, cfg["paths"]["recapture_photos"])
    splits_path = file_in(cfg, "results_dir", "splits_json", seed)
    print(f"dataset image_dir: {image_dir.resolve()}")
    print(f"recapture_photos: {photo_dir.resolve()}")
    print(f"manifest: {manifest_path.resolve()}")
    print(f"splits: {splits_path.resolve()}")
    manifest = read_csv(manifest_path)
    chosen = pick_one_each(group_by_class(test_paths(cfg)), seed)
    if len(manifest) != 25 or len(chosen) != 25:
        raise RuntimeError(
            f"Expected the 25 Experiment C images, got manifest={len(manifest)} selection={len(chosen)}."
        )
    by_filename = {row["filename"]: row for row in manifest}
    if len(by_filename) != len(manifest):
        raise RuntimeError(f"Duplicate filenames in {manifest_path.name}.")
    matched = []
    missing = []
    expected_names = []
    for index, (class_name, relative) in enumerate(chosen, start=1):
        filename = f"{index:02d}_{class_name.lower()}.jpg"
        expected_names.append(filename)
        row = by_filename.get(filename)
        if row is None:
            raise RuntimeError(
                f"Recapture {filename} (class {class_name}, clean {relative}) "
                f"is not in {manifest_path.name}."
            )
        if row["true_label"] != class_name:
            raise RuntimeError(
                f"Label mismatch for {filename}: manifest={row['true_label']} selection={class_name}."
            )
        clean = (image_dir / relative).resolve()
        real, real_problem = find_recapture(photo_dir, filename)
        shown_real = real if real is not None else (photo_dir / filename).resolve()
        instance_id = Path(filename).stem
        print(f"match id={instance_id} class={class_name} relative={relative}")
        print(f"  clean: {clean}")
        print(f"  real:  {shown_real}")
        if real is not None and real.name != filename:
            print(f"  manifest name {filename} resolved to {real.name}")
        if not clean.is_file():
            missing.append(f"clean counterpart missing for {filename}: {clean}")
        if real_problem:
            missing.append(real_problem)
        matched.append({
            "instance_id": instance_id,
            "class_name": class_name,
            "relative": relative,
            "clean_path": clean,
            "real_path": real,
        })
    extra = sorted(set(by_filename) - set(expected_names))
    if extra:
        missing.append(f"manifest rows are not in the Experiment C selection: {extra}")
    if missing:
        detail = "\n".join(missing)
        raise RuntimeError(f"Experiment C matching failed.\n{detail}")
    print(f"matched samples: {len(matched)}")
    return matched


def load_square(rows: list[dict], key: str, image_size: int) -> torch.Tensor:
    """Read matched images with the same resize used by recognition and recapture eval."""
    return torch.stack([read_rgb(row[key], image_size) for row in rows])


def simulate_combined(clean: torch.Tensor, seed: int) -> torch.Tensor:
    """Combined severity 1.0, one batch, via invariance_core.distort."""
    print(
        "distortion call: invariance_core.distort("
        f"clean_batch, kind='combined', severity=1.0, seed={seed})"
    )
    print(f"distortion batch shape: {tuple(clean.shape)}")
    simulated = distort(clean, "combined", 1.0, seed=seed)
    if simulated.shape != clean.shape:
        raise RuntimeError(f"distort returned {tuple(simulated.shape)}, expected {tuple(clean.shape)}.")
    return simulated


def load_models(cfg: dict, seed: int, dim: int, device: torch.device) -> dict:
    """Load M2 and M3 with eval_recognition.load_head. Missing files abort the run."""
    clip = cfg["clip"]
    specs = (
        ("M2", file_in(cfg, "ckpt_dir", "m2_ckpt", seed)),
        ("M3", project_path(cfg, cfg["paths"]["ckpt_dir"]) / "m3.pt"),
    )
    models = {}
    for name, path in specs:
        require(path)
        head = load_head(path, dim, device)
        print(f"model: {name}")
        print(f"checkpoint path: {path.resolve()}")
        print(f"checkpoint filename: {path.name}")
        print(
            f"backbone: frozen OpenAI CLIP {clip['model_name']} "
            f"pretrained={clip['pretrained']} image_size={clip['image_size']} "
            f"encoder=features.FrozenImageEncoder"
        )
        print(
            f"head: {type(head).__module__}.{type(head).__name__} "
            f"blocks={len(list(head.blocks))} "
            f"proj={head.proj.in_features}->{head.proj.out_features} "
            f"output=L2-normalized training={head.training}"
        )
        print("checkpoint loader: eval_recognition.load_head")
        print("feature path: eval_recognition.represent (encoder, then head; before text-anchor comparison)")
        models[name] = head
    return models


@torch.no_grad()
def embed_conditions(models: dict, encoder, batches: dict[str, torch.Tensor], device) -> tuple[dict, int]:
    """Run represent() on each condition. Returns model -> condition -> (N, D) features."""
    bank = {}
    dim = None
    for model, module in models.items():
        bank[model] = {}
        for condition in CONDITIONS:
            features = represent(model, encoder, module, batches[condition].to(device)).cpu()
            if features.ndim != 2:
                raise RuntimeError(f"{model} {condition} features have shape {tuple(features.shape)}.")
            if dim is None:
                dim = int(features.shape[-1])
            elif int(features.shape[-1]) != dim:
                raise RuntimeError(f"Feature dim changed at {model} {condition}: {features.shape[-1]} vs {dim}.")
            bank[model][condition] = features
            print(f"features model={model} condition={condition} shape={tuple(features.shape)}")
    print(f"feature dimensionality: {dim}")
    return bank, dim


def stack_records(bank: dict, matched: list[dict]) -> list[dict]:
    """One stored vector per model, instance, and condition."""
    records = []
    for model, by_condition in bank.items():
        for condition, matrix in by_condition.items():
            if matrix.shape[0] != len(matched):
                raise RuntimeError(f"{model} {condition} has {matrix.shape[0]} rows, expected {len(matched)}.")
            for index, row in enumerate(matched):
                records.append({
                    "model": model,
                    "class": row["class_name"],
                    "instance_id": row["instance_id"],
                    "condition": condition,
                    "feature": matrix[index],
                })
    return records


def cosine_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """1 - cosine similarity, the complement of the stability score in eval_recognition."""
    return 1.0 - F.cosine_similarity(left, right, dim=-1)


def mean_std(values: torch.Tensor) -> tuple[float, float]:
    """Sample mean and sample standard deviation."""
    if values.numel() < 2:
        raise RuntimeError("Need at least two distances to report a standard deviation.")
    array = values.detach().float().cpu().numpy()
    return float(array.mean()), float(array.std(ddof=1))


def displacement_table(bank: dict) -> dict:
    """Per-model mean cosine distance from each clean feature to its sim and real pair."""
    print("cosine distance definition: 1 - torch.nn.functional.cosine_similarity")
    summary = {}
    for model, by_condition in bank.items():
        clean = by_condition["clean"]
        sim = cosine_distance(clean, by_condition["simulated"])
        real = cosine_distance(clean, by_condition["real"])
        sim_mean, sim_std = mean_std(sim)
        real_mean, real_std = mean_std(real)
        if sim_mean == 0:
            raise RuntimeError(f"{model} mean simulated cosine distance is 0; the real/sim ratio is undefined.")
        ratio = real_mean / sim_mean
        summary[model] = {
            "sim": sim_mean,
            "sim_std": sim_std,
            "real": real_mean,
            "real_std": real_std,
            "ratio": ratio,
        }
        print(f"{model} simulated: {sim_mean:.4f} ± {sim_std:.4f} (sample std, ddof=1)")
        print(f"{model} real: {real_mean:.4f} ± {real_std:.4f} (sample std, ddof=1)")
        print(f"{model} real_distance / simulated_distance: {ratio:.4f} (ratio of means)")
    return summary


def metric_rows(matched: list[dict], bank: dict, seed: int) -> list[dict]:
    """One CSV row per model and matched instance. Simulated images stay in memory."""
    rows = []
    for model, by_condition in bank.items():
        sim_distance = cosine_distance(by_condition["clean"], by_condition["simulated"])
        real_distance = cosine_distance(by_condition["clean"], by_condition["real"])
        for index, row in enumerate(matched):
            simulated_path = (
                "in-memory invariance_core.distort"
                f"(kind=combined, severity=1.0, seed={seed}, batch_index={index})"
                f" of {row['clean_path']}"
            )
            rows.append({
                "model": model,
                "instance_id": row["instance_id"],
                "class": row["class_name"],
                "clean_path": str(row["clean_path"]),
                "simulated_path": simulated_path,
                "real_path": str(row["real_path"]),
                "clean_sim_cosine_distance": float(sim_distance[index]),
                "clean_real_cosine_distance": float(real_distance[index]),
            })
    return rows


def project_pca(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two-component PCA. Explained-variance fractions sum against every component."""
    if matrix.ndim != 2 or matrix.shape[0] < 3 or matrix.shape[1] < 2:
        raise RuntimeError(f"PCA cannot run on feature matrix {matrix.shape}.")
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    variance = singular ** 2
    if variance.size < 2 or float(variance.sum()) == 0.0:
        raise RuntimeError("PCA failed because the collected features have no spread.")
    coords = centered @ vt[:2].T
    explained = variance / variance.sum()
    return coords, explained[:2]


def draw_feature_space(model: str, bank: dict, matched: list[dict], dest: Path) -> None:
    """One PCA plane. Color is condition. Faint segments tie each triplet together."""
    blocks = [bank[model][condition].numpy() for condition in CONDITIONS]
    matrix = np.concatenate(blocks, axis=0)
    coords, explained = project_pca(matrix)
    count = len(matched)
    split = {condition: coords[offset * count:(offset + 1) * count] for offset, condition in enumerate(CONDITIONS)}
    fig, axis = plt.subplots(figsize=(8.2, 7.2))
    clean_xy = split["clean"]
    for condition in ("simulated", "real"):
        other = split[condition]
        for index in range(count):
            axis.plot(
                [clean_xy[index, 0], other[index, 0]],
                [clean_xy[index, 1], other[index, 1]],
                color=CONDITION_COLOR[condition],
                alpha=0.28,
                linewidth=0.8,
                zorder=1,
            )
    for condition in CONDITIONS:
        points = split[condition]
        axis.scatter(
            points[:, 0],
            points[:, 1],
            s=28,
            color=CONDITION_COLOR[condition],
            label=condition,
            zorder=2,
        )
    for index, row in enumerate(matched):
        axis.annotate(
            row["instance_id"],
            clean_xy[index],
            textcoords="offset points",
            xytext=(3, 3),
            fontsize=6,
            color="#333333",
            zorder=3,
        )
    axis.set_xlabel(f"PC1 ({100 * explained[0]:.1f}%)")
    axis.set_ylabel(f"PC2 ({100 * explained[1]:.1f}%)")
    axis.set_title(f"{model} features: clean, simulated combined 1.0, real recapture")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=140)
    plt.close(fig)
    print(f"plot: {dest.resolve()}")
    print(f"  {model} PC1 ({100 * explained[0]:.1f}%) PC2 ({100 * explained[1]:.1f}%)")


def class_counts(labels: list[int], names: list[str]) -> Counter:
    """How many labeled images each class contributes."""
    return Counter(names[label] for label in labels)


def separation(features: torch.Tensor, labels: torch.Tensor) -> dict:
    """Mean within-class and between-class cosine distance on unordered pairs."""
    feats = F.normalize(features.float(), dim=-1)
    distance = 1.0 - feats @ feats.T
    label = labels.view(-1)
    same = label[:, None].eq(label[None, :])
    upper = torch.triu(torch.ones(label.shape[0], label.shape[0], dtype=torch.bool), diagonal=1)
    within = distance[same & upper]
    between = distance[~same & upper]
    if within.numel() == 0:
        raise RuntimeError("No within-class pairs. Refusing to fabricate a within-class distance.")
    within_mean = float(within.mean())
    between_mean = float(between.mean())
    if within_mean == 0.0:
        raise RuntimeError("Within-class cosine distance is 0, so the separation ratio is undefined.")
    return {
        "within": within_mean,
        "between": between_mean,
        "ratio": between_mean / within_mean,
        "n_within_pairs": int(within.numel()),
        "n_between_pairs": int(between.numel()),
        "n": int(label.shape[0]),
    }


@torch.no_grad()
def semantic_on_test(models: dict, encoder, cfg: dict, seed: int, device) -> None:
    """Within/between distances on the test split, which has repeated classes."""
    dataset, names = make_dataset(cfg, "test", seed)
    labels = [dataset.class_to_index[relative.split("/", 1)[0]] for relative in dataset.relative_paths]
    counts = class_counts(labels, names)
    smallest = min(counts.values())
    print(f"semantic subset: recognition test split")
    print(f"semantic subset images: {len(dataset)}")
    print(f"semantic subset classes: {len(counts)} min_per_class={smallest} max_per_class={max(counts.values())}")
    if smallest < 2:
        raise RuntimeError(
            "The test split does not have multiple images per class, so within-class distance cannot be estimated."
        )
    spec = cfg["eval_recognition"]
    loader = DataLoader(dataset, batch_size=int(spec["batch_size"]), shuffle=False)
    print(
        "semantic distortion call: invariance_core.distort("
        "batch, kind='combined', severity=1.0, seed=seed+batch_index)"
    )
    print("semantic feature path: eval_recognition.represent")
    collected = {model: {"clean": [], "simulated": []} for model in models}
    seen = []
    for start, (images, batch_labels) in enumerate(loader):
        images = images.to(device)
        view = distort(images, "combined", 1.0, seed=seed + start)
        for model, module in models.items():
            collected[model]["clean"].append(represent(model, encoder, module, images).cpu())
            collected[model]["simulated"].append(represent(model, encoder, module, view).cpu())
        seen.append(batch_labels)
        if start % 10 == 0:
            print(f"semantic batch {start} images_so_far={sum(part.shape[0] for part in seen)}")
    label_tensor = torch.cat(seen)
    print("semantic separation (test split; clean and simulated only; real photos exist only for the 25 recaptures)")
    for model in models:
        for condition in ("clean", "simulated"):
            stats = separation(torch.cat(collected[model][condition]), label_tensor)
            print(
                f"{model} {condition}: within={stats['within']:.4f} between={stats['between']:.4f} "
                f"separation={stats['ratio']:.4f} "
                f"pairs_within={stats['n_within_pairs']} pairs_between={stats['n_between_pairs']} n={stats['n']}"
            )


def relation(left: float, right: float) -> str:
    """larger, smaller, or comparable. The band is a label, not a significance test."""
    scale = max(abs(left), abs(right), 1e-12)
    if abs(left - right) / scale < GAP_LABEL_TOLERANCE:
        return "comparable"
    if left > right:
        return "larger"
    return "smaller"


def print_interpretation(summary: dict) -> None:
    """Factual displacement summary. No significance test and no causal claim about the augmentor."""
    if summary["M3"]["real"] > summary["M3"]["sim"]:
        m3_line = "M3 real-recapture displacement is larger than its simulated displacement."
        consistency = "consistent"
    elif summary["M3"]["real"] < summary["M3"]["sim"]:
        m3_line = "M3 real-recapture displacement is smaller than its simulated displacement."
        consistency = "not consistent"
    else:
        m3_line = "M3 real-recapture displacement is equal to its simulated displacement."
        consistency = "not consistent"
    m2_vs_m3 = relation(summary["M2"]["ratio"], summary["M3"]["ratio"])
    print(m3_line)
    if m2_vs_m3 == "comparable":
        print("M2 shows a comparable sim-to-real gap to M3.")
    else:
        print(f"M2 shows a {m2_vs_m3} sim-to-real gap than M3.")
    print(
        f"Gap words use a {GAP_LABEL_TOLERANCE:.0%} relative band on the compared numbers. "
        "That band is a descriptive label, not a significance test."
    )
    print(
        "This result is "
        f"{consistency} with the hypothesis that M3 learned invariance more specifically "
        "to the simulated distortion distribution than to real camera variation."
    )
    print("PCA is descriptive and does not by itself establish invariance.")
    print(
        "n=25 is a small sample. Any per-model gap reported here is a descriptive, "
        "preliminary signal, not a statistically confirmed effect."
    )


def semantic_on_matched(bank: dict, matched: list[dict]) -> None:
    """Within/between distances on the recapture subset when a class appears more than once."""
    classes = [row["class_name"] for row in matched]
    index = {name: number for number, name in enumerate(sorted(set(classes)))}
    labels = torch.tensor([index[name] for name in classes])
    print("semantic separation on the matched recapture subset")
    for model, by_condition in bank.items():
        for condition in CONDITIONS:
            stats = separation(by_condition[condition], labels)
            print(
                f"{model} {condition}: within={stats['within']:.4f} between={stats['between']:.4f} "
                f"separation={stats['ratio']:.4f} "
                f"pairs_within={stats['n_within_pairs']} pairs_between={stats['n_between_pairs']} n={stats['n']}"
            )


def recapture_is_one_per_class(matched: list[dict]) -> bool:
    """Experiment C is built as one test photo per class. Confirm before skipping within-class math."""
    counts = Counter(row["class_name"] for row in matched)
    print(f"recapture classes: {len(counts)} max_per_class={max(counts.values())}")
    return max(counts.values()) == 1


def main() -> None:
    args = base_parser("Compare M2 and M3 feature movement on the Experiment C recapture triplets.").parse_args()
    set_seed(args.seed)
    cfg = load_config(args.config)
    device = pick_device(cfg["device"])
    results_dir = project_path(cfg, cfg["paths"]["results_dir"])
    metrics_path = results_dir / "feature_space_metrics.csv"
    plot_paths = {
        "M2": results_dir / "feature_space_m2.png",
        "M3": results_dir / "feature_space_m3.png",
    }
    print(f"device: {device}")
    print(f"seed: {args.seed}")
    print(f"output metrics: {metrics_path.resolve()}")
    print(f"output plot M2: {plot_paths['M2'].resolve()}")
    print(f"output plot M3: {plot_paths['M3'].resolve()}")
    require_experiment_a_distort()
    matched = resolve_triplets(cfg, args.seed)
    image_size = int(cfg["clip"]["image_size"])
    clean = load_square(matched, "clean_path", image_size)
    real = load_square(matched, "real_path", image_size)
    simulated = simulate_combined(clean.to(device), args.seed).cpu()
    for index, row in enumerate(matched):
        print(
            f"simulated id={row['instance_id']} "
            f"source=invariance_core.distort kind=combined severity=1.0 seed={args.seed} batch_index={index}"
        )
    names, captions = load_caption_table(file_in(cfg, "results_dir", "captions_json", args.seed))
    encoder, text_encoder = build_encoders(cfg, device)
    anchors = anchor_rows(text_encoder, names, captions)
    dim = int(anchors.shape[1])
    print(f"anchor rows: {anchors.shape[0]} dim_from_anchors={dim}")
    models = load_models(cfg, args.seed, dim, device)
    bank, feature_dim = embed_conditions(
        models,
        encoder,
        {"clean": clean, "simulated": simulated, "real": real},
        device,
    )
    if feature_dim != dim:
        raise RuntimeError(f"represent() returned dim {feature_dim}, anchor dim is {dim}.")
    records = stack_records(bank, matched)
    print(f"stored feature records: {len(records)} (model, class, instance_id, condition)")
    summary = displacement_table(bank)
    for model, dest in plot_paths.items():
        draw_feature_space(model, bank, matched, dest)
    print(
        "simulated_path in the metrics CSV is not a separate image file. "
        "It names the in-memory invariance_core.distort tensor that represent() consumed."
    )
    rows = metric_rows(matched, bank, args.seed)
    write_csv(metrics_path, rows)
    print(f"metrics: {metrics_path.resolve()} rows={len(rows)}")
    if recapture_is_one_per_class(matched):
        print(
            "Within-class cosine distance cannot be estimated from the 25-image Experiment C set: "
            "it contains only one example per class. "
            "Semantic separation is computed on the recognition test split instead."
        )
        semantic_on_test(models, encoder, cfg, args.seed, device)
    else:
        semantic_on_matched(bank, matched)
    print_interpretation(summary)
    announce(
        f"Matched {len(matched)} clean/simulated/real triplets. Feature dim {feature_dim}.",
        f"Metrics: {metrics_path}",
        f"Plots: {plot_paths['M2'].name}, {plot_paths['M3'].name}.",
    )


if __name__ == "__main__":
    main()
