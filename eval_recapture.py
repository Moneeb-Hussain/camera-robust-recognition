"""Score real recaptured grocery photos with M0 through M3 and the M3-t0 ablation."""
import torch

from features import TextAnchorEncoder, anchor_rows, load_caption_table
from invariance_core import InvariantHead
from kit import (
    announce,
    base_parser,
    build_clip,
    file_in,
    load_checkpoint,
    load_config,
    pick_device,
    project_path,
    read_csv,
    read_rgb,
    set_seed,
    write_csv,
)
from train_baselines import LinearProbe


def read_saved(path, device):
    """Load a checkpoint. A missing file is a warning, not a crash."""
    if not path.is_file():
        print(f"warning: skipping {path.name} because it is missing")
        return None
    sidecar = path.with_suffix(".json")
    try:
        loaded = load_checkpoint(path, sidecar, device)
    except Exception as exc:
        print(f"warning: could not use the sidecar for {path.name}: {exc}")
        loaded = None
    if loaded is not None:
        return loaded[0]
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)
    except Exception as exc:
        print(f"warning: skipping {path.name}: {exc}")
        return None


def load_probe(path, dim: int, n_classes: int, device):
    """Rebuild the linear probe when its file is present."""
    blob = read_saved(path, device)
    if blob is None:
        return None
    probe = LinearProbe(dim, n_classes).to(device).eval()
    probe.load_state_dict(blob["state"])
    return probe


def load_head(path, dim: int, device):
    """Rebuild an invariant head from a baseline file or from train_ours."""
    blob = read_saved(path, device)
    if blob is None:
        return None
    head = InvariantHead(dim, dim).to(device).eval()
    state = blob["head"] if isinstance(blob, dict) and "head" in blob else blob["state"]
    head.load_state_dict(state)
    return head


def load_photos(cfg: dict, rows: list[dict]) -> torch.Tensor:
    """Read recapture_photos with the same resize-to-square step as the rest of the project."""
    folder = project_path(cfg, cfg["paths"]["recapture_photos"])
    size = int(cfg["clip"]["image_size"])
    images = []
    for row in rows:
        path = folder / row["filename"]
        if not path.is_file():
            raise RuntimeError(f"Missing recaptured photo {path}")
        images.append(read_rgb(path, size))
    return torch.stack(images)


def predict_labels(model: str, features, anchors, module, names: list[str]) -> list[str]:
    """Turn one forward pass into class folder names."""
    if model == "M1":
        index = module(features).argmax(dim=-1)
    else:
        index = (features @ anchors.t()).argmax(dim=-1)
    return [names[int(item)] for item in index.cpu()]


@torch.no_grad()
def score_model(model, images, encoder, module, anchors, names):
    """Predict every recaptured photo with one model."""
    features = encoder(images)
    if model not in {"M0", "M1"}:
        features = module(features)
    return predict_labels(model, features, anchors, module, names)


def main() -> None:
    args = base_parser("Score real recaptured photos with the trained recognizers.").parse_args()
    set_seed(args.seed)
    cfg = load_config(args.config)
    device = pick_device(cfg["device"])
    manifest = read_csv(file_in(cfg, "results_dir", "recapture_manifest", args.seed))
    names, captions = load_caption_table(file_in(cfg, "results_dir", "captions_json", args.seed))
    clip_model, encoder, tokenizer = build_clip(cfg, device)
    text_encoder = TextAnchorEncoder(clip_model, tokenizer).to(device)
    anchors = anchor_rows(text_encoder, names, captions).to(device)
    images = load_photos(cfg, manifest).to(device)
    dim = anchors.shape[1]
    ckpt = project_path(cfg, cfg["paths"]["ckpt_dir"])
    modules = {"M0": None}
    probe = load_probe(file_in(cfg, "ckpt_dir", "m1_ckpt", args.seed), dim, len(names), device)
    if probe is not None:
        modules["M1"] = probe
    for model, path in (
        ("M2", file_in(cfg, "ckpt_dir", "m2_ckpt", args.seed)),
        ("M3", ckpt / "m3.pt"),
        ("M3-t0", ckpt / "m3_t0.pt"),
    ):
        head = load_head(path, dim, device)
        if head is not None:
            modules[model] = head
    truth = [row["true_label"] for row in manifest]
    detailed = [{"filename": row["filename"], "true_label": row["true_label"]} for row in manifest]
    summary = []
    for model, module in modules.items():
        guessed = score_model(model, images, encoder, module, anchors, names)
        correct = sum(pred == label for pred, label in zip(guessed, truth))
        for row, pred in zip(detailed, guessed):
            row[model] = pred
        summary.append({
            "model": model,
            "correct": correct,
            "total": len(truth),
            "accuracy": correct / len(truth),
        })
        print(f"{model}: {correct / len(truth):.3f} ({correct}/{len(truth)})")
    summary_path = file_in(cfg, "results_dir", "recapture_csv", args.seed)
    detail_path = file_in(cfg, "results_dir", "recapture_detailed_csv", args.seed)
    write_csv(summary_path, summary)
    write_csv(detail_path, detailed)
    scored = ", ".join(row["model"] for row in summary)
    announce(
        f"Scored {len(truth)} recaptured photos with {scored}.",
        f"Summary: {summary_path}",
        f"Per-photo predictions: {detail_path}",
    )


if __name__ == "__main__":
    main()
