"""Train the invariant head against the learned camera augmentor."""
import torch

from invariance_core import AutoAugmentor, InvariantHead, train_step
from kit import (
    announce,
    base_parser,
    build_clip,
    distorted_accuracy,
    file_in,
    load_config,
    load_dataset,
    make_anchors,
    pick_device,
    project_path,
    save_checkpoint,
    set_seed,
    write_csv,
)


def run_epoch(images, labels, enc, head, aug, anchors, opt_head, opt_aug, cfg, device, seed, epoch):
    """One pass: the augmentor looks for harmful distortions, then the head learns to ignore them."""
    batch_size = int(cfg["train"]["batch_size"])
    if len(labels) < batch_size:
        raise RuntimeError("train.batch_size is larger than the image folder. Lower it in config.yaml.")
    generator = torch.Generator().manual_seed(seed + epoch)
    order = torch.randperm(len(labels), generator=generator).tolist()
    totals = {"A_inv": 0.0, "B_total": 0.0}
    steps = 0
    lam_text = float(cfg["train"]["lam_text"])
    lam_sem = float(cfg["train"]["lam_sem"])
    for start in range(0, len(labels) - batch_size + 1, batch_size):
        idx = order[start:start + batch_size]
        info = train_step(
            images[idx].to(device), labels[idx].to(device), enc, head, aug, anchors,
            opt_head, opt_aug, lam_text=lam_text, lam_sem=lam_sem,
        )
        for key in totals:
            totals[key] += info[key]
        steps += 1
    return {key: value / steps for key, value in totals.items()}


def main() -> None:
    args = base_parser("Train a head that keeps class captions stable under camera distortion.").parse_args()
    set_seed(args.seed)
    cfg = load_config(args.config)
    device = pick_device(cfg["device"])
    image_dir = project_path(cfg, cfg["paths"]["image_dir"])
    images, labels, names = load_dataset(image_dir, int(cfg["clip"]["image_size"]))
    model, enc, tokenizer = build_clip(cfg, device)
    anchors = make_anchors(model, tokenizer, names, cfg["clip"]["prompt"], device)
    dim = anchors.shape[1]
    head = InvariantHead(dim, dim).to(device)
    aug = AutoAugmentor(ops=cfg["augmentor"]["ops"], jitter=float(cfg["augmentor"]["jitter"])).to(device)
    opt_head = torch.optim.AdamW(head.parameters(), lr=float(cfg["train"]["lr_head"]))
    opt_aug = torch.optim.Adam(aug.parameters(), lr=float(cfg["train"]["lr_aug"]))
    rows = []
    epochs = int(cfg["train"]["epochs"])
    for epoch in range(1, epochs + 1):
        losses = run_epoch(images, labels, enc, head, aug, anchors, opt_head, opt_aug, cfg, device, args.seed, epoch)
        head.eval()
        clean = distorted_accuracy(
            images, labels, lambda batch: head(enc(batch)), anchors,
            "combined", 0.0, args.seed, int(cfg["eval"]["batch_size"]), device,
        )
        rows.append({
            "epoch": epoch,
            "a_inv": losses["A_inv"],
            "b_total": losses["B_total"],
            "clean_acc": clean,
            "seed": args.seed,
        })
    csv_path = file_in(cfg, "results_dir", "train_csv", args.seed)
    write_csv(csv_path, rows)
    ckpt = file_in(cfg, "checkpoint_dir", "checkpoint", args.seed)
    classes_path = file_in(cfg, "checkpoint_dir", "classes_json", args.seed)
    save_checkpoint(ckpt, classes_path, head, aug, names, dim)
    last = rows[-1]
    announce(
        f"Trained {epochs} epochs on {len(labels)} images and {len(names)} classes.",
        f"CSV: {csv_path}",
        f"Checkpoint: {ckpt}  clean accuracy {last['clean_acc']:.3f}",
    )


if __name__ == "__main__":
    main()
