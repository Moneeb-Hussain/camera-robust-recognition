"""Score zero-shot CLIP and the trained head under each camera distortion."""
from invariance_core import OPS, InvariantHead
from kit import (
    announce,
    base_parser,
    build_clip,
    distorted_accuracy,
    file_in,
    load_checkpoint,
    load_config,
    load_dataset,
    make_anchors,
    pick_device,
    project_path,
    set_seed,
    write_csv,
)


def check_kinds(kinds: list[str]) -> None:
    """Reject a distortion name the core does not know."""
    allowed = set(OPS) | {"combined"}
    unknown = [kind for kind in kinds if kind not in allowed]
    if unknown:
        raise RuntimeError(f"Unknown distortion kinds: {unknown}")


def load_head(cfg, names, dim, device, seed):
    """Rebuild the trained head when this seed has a checkpoint."""
    loaded = load_checkpoint(
        file_in(cfg, "checkpoint_dir", "checkpoint", seed),
        file_in(cfg, "checkpoint_dir", "classes_json", seed),
        device,
    )
    if loaded is None:
        return None
    blob, meta = loaded
    if meta["classes"] != names or int(meta["dim"]) != dim:
        raise RuntimeError("Checkpoint classes do not match the image folder. Retrain this seed.")
    head = InvariantHead(dim, dim).to(device).eval()
    head.load_state_dict(blob["head"])
    return head


def score_model(images, labels, encode, anchors, cfg, seed, model_name, device) -> list[dict]:
    """Accuracy of one model at every severity and distortion listed in config."""
    rows = []
    batch_size = int(cfg["eval"]["batch_size"])
    for kind in cfg["eval"]["kinds"]:
        for severity in cfg["eval"]["severities"]:
            accuracy = distorted_accuracy(
                images, labels, encode, anchors, kind, float(severity), seed, batch_size, device,
            )
            rows.append({
                "model": model_name,
                "kind": kind,
                "severity": float(severity),
                "accuracy": accuracy,
                "n_images": len(labels),
                "seed": seed,
            })
    return rows


def main() -> None:
    args = base_parser("Measure accuracy as camera distortions get stronger.").parse_args()
    set_seed(args.seed)
    cfg = load_config(args.config)
    check_kinds(list(cfg["eval"]["kinds"]))
    device = pick_device(cfg["device"])
    images, labels, names = load_dataset(
        project_path(cfg, cfg["paths"]["image_dir"]), int(cfg["clip"]["image_size"]),
    )
    model, enc, tokenizer = build_clip(cfg, device)
    anchors = make_anchors(model, tokenizer, names, cfg["clip"]["prompt"], device)
    enc = enc.to(device).eval()
    rows = score_model(images, labels, enc, anchors.to(device), cfg, args.seed, "clip", device)
    head = load_head(cfg, names, anchors.shape[1], device, args.seed)
    models = "clip"
    if head is not None:
        rows.extend(score_model(
            images, labels, lambda batch: head(enc(batch)), anchors, cfg, args.seed, "invariant", device,
        ))
        models = "clip, invariant"
    csv_path = file_in(cfg, "results_dir", "robustness_csv", args.seed)
    write_csv(csv_path, rows)
    clean = [row for row in rows if float(row["severity"]) == 0.0 and row["kind"] == cfg["eval"]["kinds"][0]]
    best = max(clean, key=lambda row: float(row["accuracy"]))
    announce(
        f"Scored {len(labels)} images at {len(cfg['eval']['severities'])} severities and {len(cfg['eval']['kinds'])} distortions.",
        f"CSV: {csv_path}",
        f"Models: {models}. Best clean accuracy {float(best['accuracy']):.3f} ({best['model']}).",
    )


if __name__ == "__main__":
    main()
