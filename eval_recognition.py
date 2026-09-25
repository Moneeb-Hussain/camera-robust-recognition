"""Test-set distortion sweeps, leave-one-out, and CLIP-plus-head latency."""
import csv
import statistics
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from features import anchor_rows, build_encoders, load_caption_table
from invariance_core import OPS, InvariantHead, distort
from kit import announce, base_parser, file_in, load_config, pick_device, project_path, set_seed, write_csv
from train_baselines import LinearProbe, make_dataset


def load_blob(path, device):
    """Read a checkpoint. M3 stores a head; M1 and M2 store a state dict."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def require(path) -> None:
    """Stop early when a model file has not been trained yet."""
    if not path.is_file():
        raise RuntimeError(f"Missing checkpoint {path}")


def load_head(path, dim: int, device) -> InvariantHead:
    """Rebuild an invariant head from either train_ours or train_baselines."""
    blob = load_blob(path, device)
    head = InvariantHead(dim, dim).to(device).eval()
    state = blob["head"] if isinstance(blob, dict) and "head" in blob else blob["state"]
    head.load_state_dict(state)
    return head


def load_probe(path, dim: int, n_classes: int, device) -> LinearProbe:
    """Rebuild the linear probe."""
    blob = load_blob(path, device)
    probe = LinearProbe(dim, n_classes).to(device).eval()
    probe.load_state_dict(blob["state"])
    return probe


def represent(model: str, encoder, module, images):
    """Features used for stability. M0 and M1 stay in CLIP space; the heads project."""
    features = encoder(images)
    if model in {"M0", "M1"}:
        return features
    return module(features)


def classify(model: str, features, anchors, module):
    """Top-1 class. The probe uses its own logits; everyone else uses caption anchors."""
    if model == "M1":
        return module(features).argmax(dim=-1)
    return (features @ anchors.t()).argmax(dim=-1)


def fresh_stats(kinds, severities) -> dict:
    """Zero the counters for one model."""
    return {(kind, float(severity)): {"correct": 0, "cosine": 0.0, "n": 0} for kind in kinds for severity in severities}


@torch.no_grad()
def accumulate(stats, images, labels, model, encoder, module, anchors, kinds, severities, seed, device):
    """Score one batch under every requested distortion."""
    images = images.to(device)
    labels = labels.to(device)
    clean = represent(model, encoder, module, images)
    for kind in kinds:
        for severity in severities:
            view = distort(images, kind, float(severity), seed=seed)
            feats = represent(model, encoder, module, view)
            pred = classify(model, feats, anchors, module)
            bucket = stats[(kind, float(severity))]
            bucket["correct"] += int((pred == labels).sum().item())
            bucket["cosine"] += float(F.cosine_similarity(clean, feats, dim=-1).sum().item())
            bucket["n"] += int(labels.numel())


def rows_from(stats, model: str, seed: int, extra: dict | None = None) -> list[dict]:
    """Turn counters into accuracy and mean clean-to-distorted cosine."""
    rows = []
    for (kind, severity), bucket in stats.items():
        row = {
            "model": model,
            "kind": kind,
            "severity": severity,
            "accuracy": bucket["correct"] / bucket["n"],
            "stability": bucket["cosine"] / bucket["n"],
            "n_images": bucket["n"],
            "seed": seed,
        }
        if extra:
            row = {**extra, **row}
        rows.append(row)
    return rows


def sweep(model, encoder, module, anchors, loader, kinds, severities, seed, device) -> list[dict]:
    """Full pass of one model over the test loader."""
    stats = fresh_stats(kinds, severities)
    for start, (images, labels) in enumerate(loader):
        accumulate(stats, images, labels, model, encoder, module, anchors, kinds, severities, seed + start, device)
    return rows_from(stats, model, seed)


def load_leaveout(path, dim: int, device):
    """Load one leave-one-out head, or return None when that file is absent."""
    try:
        require(path)
        return load_head(path, dim, device)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"warning: skipping leave-one-out {path.name}: {exc}")
        return None


def write_leaveout(path, rows: list[dict]) -> None:
    """Write expB even when no leave-one-out checkpoint was found."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "held_out", "model", "kind", "severity", "accuracy", "stability", "n_images", "seed",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def leaveout_path(cfg: dict, held_out: str):
    """Checkpoint trained with every op except the one being tested."""
    ops = [name for name in cfg["augmentor"]["ops"] if name != held_out]
    if held_out not in OPS:
        raise RuntimeError(f"{held_out} is not a learned distortion")
    return project_path(cfg, cfg["paths"]["ckpt_dir"]) / f"m3_{'-'.join(ops)}.pt"


def median_ms(encoder, head, device, image_size: int, warmup: int, runs: int) -> float:
    """Median milliseconds for one 224 image through CLIP and the head."""
    encoder = encoder.to(device).eval()
    head = head.to(device).eval()
    image = torch.rand(1, 3, image_size, image_size, device=device)

    def step():
        with torch.no_grad():
            head(encoder(image))

    for _ in range(warmup):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
    return float(statistics.median(times))


def main() -> None:
    args = base_parser("Evaluate M0-M3 and M3-t0 under camera distortions.").parse_args()
    set_seed(args.seed)
    cfg = load_config(args.config)
    device = pick_device(cfg["device"])
    spec = cfg["eval_recognition"]
    names, captions = load_caption_table(file_in(cfg, "results_dir", "captions_json", args.seed))
    encoder, text_encoder = build_encoders(cfg, device)
    anchors = anchor_rows(text_encoder, names, captions).to(device)
    test_set, _names = make_dataset(cfg, "test", args.seed)
    loader = DataLoader(test_set, batch_size=int(spec["batch_size"]), shuffle=False)
    dim = anchors.shape[1]
    m1_path = file_in(cfg, "ckpt_dir", "m1_ckpt", args.seed)
    m2_path = file_in(cfg, "ckpt_dir", "m2_ckpt", args.seed)
    m3_path = project_path(cfg, cfg["paths"]["ckpt_dir"]) / "m3.pt"
    m3_t0_path = project_path(cfg, cfg["paths"]["ckpt_dir"]) / "m3_t0.pt"
    for path in (m1_path, m2_path, m3_path, m3_t0_path):
        require(path)
    modules = {
        "M0": None,
        "M1": load_probe(m1_path, dim, len(names), device),
        "M2": load_head(m2_path, dim, device),
        "M3": load_head(m3_path, dim, device),
        "M3-t0": load_head(m3_t0_path, dim, device),
    }
    kinds = list(spec["kinds"])
    severities = list(spec["severities"])
    sweep_rows = []
    for model, module in modules.items():
        sweep_rows.extend(sweep(model, encoder, module, anchors, loader, kinds, severities, args.seed, device))
    write_csv(file_in(cfg, "results_dir", "expA_csv", args.seed), sweep_rows)
    leave_rows = []
    skipped = 0
    for held_out in spec["leaveout_kinds"]:
        path = leaveout_path(cfg, held_out)
        dropped = load_leaveout(path, dim, device)
        if dropped is None:
            skipped += 1
            continue
        stats = fresh_stats([held_out], severities)
        for start, (images, labels) in enumerate(loader):
            accumulate(stats, images, labels, "M3", encoder, dropped, anchors, [held_out], severities, args.seed + start, device)
        leave_rows.extend(rows_from(stats, "M3-drop", args.seed, {"held_out": held_out}))
        full = [row for row in sweep_rows if row["model"] == "M3" and row["kind"] == held_out]
        for row in full:
            leave_rows.append({"held_out": held_out, "model": "M3", **{k: v for k, v in row.items() if k != "model"}})
    write_leaveout(file_in(cfg, "results_dir", "expB_csv", args.seed), leave_rows)
    warmup = int(spec["warmup"])
    runs = int(spec["timing_runs"])
    image_size = int(cfg["clip"]["image_size"])
    latency_rows = []
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.insert(0, torch.device("cuda"))
    for timing_device in devices:
        latency_rows.append({
            "device": timing_device.type,
            "median_ms": median_ms(encoder, modules["M3"], timing_device, image_size, warmup, runs),
            "runs": runs,
            "warmup": warmup,
            "image_size": image_size,
            "seed": args.seed,
        })
    latency_path = file_in(cfg, "results_dir", "latency_csv", args.seed)
    write_csv(latency_path, latency_rows)
    latency_text = ", ".join(f"{row['device']} {row['median_ms']:.2f} ms" for row in latency_rows)
    announce(
        f"Swept {len(modules)} models on {len(test_set)} test images, {len(kinds)} kinds, {len(severities)} severities.",
        f"CSV: {file_in(cfg, 'results_dir', 'expA_csv', args.seed).name}, {file_in(cfg, 'results_dir', 'expB_csv', args.seed).name}, {latency_path.name}.",
        f"Median CLIP+head latency: {latency_text}. Leave-one-out skipped: {skipped}.",
    )


if __name__ == "__main__":
    main()
