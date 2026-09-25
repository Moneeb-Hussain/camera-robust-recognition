"""Train the head and the learned augmentor together. The augmentor step stays in full precision."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from invariance_core import OPS, AutoAugmentor, InvariantHead, distort, train_step

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    """Flags for the text weight, which distortions are learned, and how long to train."""
    parser = argparse.ArgumentParser(description="Train InvariantHead against AutoAugmentor.")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lam_text", type=float, default=1.0, help="0 turns off text anchoring (ckpt/m3_t0).")
    parser.add_argument("--ops", default="", help="Comma list, for example moire,photo,noise,blur.")
    parser.add_argument("--epochs", type=int, default=15)
    return parser.parse_args()


def load_config(path: str) -> dict:
    """Read paths and learning rates from config.yaml."""
    cfg_path = Path(path).resolve()
    with cfg_path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["_config_dir"] = str(cfg_path.parent)
    return cfg


def project_path(cfg: dict, raw: str) -> Path:
    """Resolve a config path next to config.yaml."""
    path = Path(raw)
    if path.is_absolute():
        return path
    return Path(cfg["_config_dir"]) / path


def named(cfg: dict, dir_key: str, file_key: str) -> Path:
    """Join a config folder and a config file name."""
    return project_path(cfg, cfg["paths"][dir_key]) / cfg["files"][file_key]


def set_seed(seed: int) -> None:
    """Make NumPy and PyTorch repeat this run."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name: str) -> torch.device:
    """Use CUDA when config says auto and a GPU is present."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def announce(line1: str, line2: str, line3: str) -> None:
    """Print the three-line summary."""
    print(line1)
    print(line2)
    print(line3)


def read_rgb(path: Path, size: int) -> torch.Tensor:
    """Load one photo as a float tensor in [0, 1], resized when it is read."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"OpenCV could not read {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    array = np.ascontiguousarray(rgb)
    return torch.from_numpy(array).permute(2, 0, 1).float() / 255.0


class GrocerySplit(Dataset):
    """One split from splits.json. Each image is resized at load time."""

    def __init__(self, image_dir: Path, relative_paths: list[str], class_names: list[str], image_size: int):
        self.image_dir = image_dir
        self.relative_paths = list(relative_paths)
        self.class_to_index = {name: index for index, name in enumerate(class_names)}
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.relative_paths)

    def __getitem__(self, index: int):
        relative = self.relative_paths[index]
        image = read_rgb(self.image_dir / relative, self.image_size)
        label = self.class_to_index[relative.split("/", 1)[0]]
        return image, label


def load_split(cfg: dict, split: str, class_names: list[str]) -> GrocerySplit:
    """Build one grocery split. Class order matches the anchor rows."""
    splits_path = named(cfg, "results_dir", "splits_json")
    if not splits_path.is_file():
        raise RuntimeError(f"Missing {splits_path}. Run prep_data.py first.")
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    image_dir = project_path(cfg, cfg["paths"]["image_dir"])
    return GrocerySplit(image_dir, splits[split], class_names, int(cfg["clip"]["image_size"]))


def parse_ops(raw: str, cfg: dict) -> list[str]:
    """Turn a comma list into distortion names. An empty flag uses the config list."""
    chosen = [part.strip() for part in raw.split(",") if part.strip()] or list(cfg["augmentor"]["ops"])
    unknown = [name for name in chosen if name not in OPS]
    if unknown:
        raise RuntimeError(f"Unknown ops {unknown}. Choose from {OPS}.")
    return chosen


class FrozenCLIP(nn.Module):
    """Frozen image tower. A [0, 1] batch is normalized the CLIP way inside forward."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        mean = torch.tensor(model.visual.image_mean, dtype=torch.float32)
        std = torch.tensor(model.visual.image_std, dtype=torch.float32)
        self.register_buffer("mean", mean.view(1, 3, 1, 1))
        self.register_buffer("std", std.view(1, 3, 1, 1))
        for param in self.model.parameters():
            param.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.model.encode_image((images - self.mean) / self.std).float()
        return F.normalize(features, dim=-1)


def build_clip(cfg: dict, device: torch.device) -> tuple[FrozenCLIP, object]:
    """Load the config CLIP once for images and for class captions."""
    name = cfg["clip"]["model_name"]
    model, _, _ = open_clip.create_model_and_transforms(
        name, pretrained=cfg["clip"]["pretrained"], force_quick_gelu=True,
    )
    model = model.float().to(device).eval()
    encoder = FrozenCLIP(model).to(device).eval()
    return encoder, open_clip.get_tokenizer(name)


@torch.no_grad()
def text_anchors(model, tokenizer, captions: list[str], device: torch.device) -> torch.Tensor:
    """Encode one caption per class into a unit-length anchor matrix."""
    tokens = tokenizer(captions).to(device)
    return F.normalize(model.encode_text(tokens).float(), dim=-1)


def class_captions(cfg: dict) -> tuple[list[str], list[str]]:
    """Read captions.json in sorted class order."""
    path = named(cfg, "results_dir", "captions_json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    names = sorted(raw)
    return names, [raw[name] for name in names]


def checkpoint_path(cfg: dict, lam_text: float, ops: list[str]) -> Path:
    """No-text run with the full op list is ckpt/m3_t0. Other ablations keep their own file."""
    folder = project_path(cfg, cfg["paths"]["ckpt_dir"])
    full = list(cfg["augmentor"]["ops"])
    if lam_text == 0.0 and ops == full:
        return folder / "m3_t0.pt"
    if ops == full:
        return folder / "m3.pt"
    tag = "m3_t0" if lam_text == 0.0 else "m3"
    return folder / f"{tag}_{'-'.join(ops)}.pt"


def flatten_raw(aug: AutoAugmentor) -> dict:
    """Copy the augmentor's learned raw numbers into a flat dict that prints easily."""
    flat = {}
    for name, param in aug.raw.items():
        values = param.detach().float().cpu().reshape(-1)
        if values.numel() == 1:
            flat[name] = float(values)
        else:
            for index, value in enumerate(values.tolist()):
                flat[f"{name}_{index}"] = float(value)
    return flat


def pair_stats(images, encoder, head, aug) -> tuple[float, float]:
    """Cosine between clean and distorted head outputs, and how spread out the batch is."""
    head.eval()
    with torch.no_grad():
        clean = head(encoder(images))
        distorted = head(encoder(aug(images)))
        cosine = float(F.cosine_similarity(clean, distorted, dim=-1).mean())
        spread = float(clean.std(dim=0).mean())
    return cosine, spread


def run_epoch(loader, encoder, head, aug, anchors, opt_head, opt_aug, cfg, device, lam_text, epoch) -> dict:
    """One epoch. Step A and step B both run without autocast, because they share train_step."""
    totals = {"A_inv": 0.0, "B_total": 0.0, "cosine": 0.0, "head_std": 0.0}
    steps = 0
    warnings = 0
    limit = float(cfg["train_ours"]["collapse_std"])
    lam_sem = float(cfg["train"]["lam_sem"])
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        if images.shape[0] < 2:
            continue
        with torch.autocast(device_type=device.type, enabled=False):
            info = train_step(
                images, labels, encoder, head, aug, anchors, opt_head, opt_aug,
                lam_text=lam_text, lam_sem=lam_sem,
            )
        cosine, spread = pair_stats(images, encoder, head, aug)
        if spread < limit:
            warnings += 1
            print(f"warning: head output std {spread:.4f} < {limit} at epoch {epoch}")
        totals["A_inv"] += info["A_inv"]
        totals["B_total"] += info["B_total"]
        totals["cosine"] += cosine
        totals["head_std"] += spread
        steps += 1
    if steps == 0:
        raise RuntimeError("No full batches. Lower train_ours.batch_size in config.yaml.")
    return {key: value / steps for key, value in totals.items()} | {"warnings": warnings}


@torch.no_grad()
def val_accuracy(loader, encoder, head, anchors, device, seed: int, severity: float) -> float:
    """Nearest-caption accuracy. severity 0 is clean; 0.5 is combined camera distortion."""
    head.eval()
    correct = 0
    total = 0
    kind = "combined"
    for index, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        view = distort(images, kind, severity, seed=seed + index)
        pred = (head(encoder(view)) @ anchors.t()).argmax(dim=-1)
        correct += int((pred == labels).sum().item())
        total += int(labels.numel())
    return correct / total


def upsert(path: Path, rows: list[dict], key: tuple) -> None:
    """Replace older rows from this same run, then write the CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    old = []
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as handle:
            old = list(csv.DictReader(handle))
    kept = [row for row in old if (row.get("lam_text"), row.get("ops"), row.get("seed")) != key]
    all_rows = kept + rows
    fieldnames = list(dict.fromkeys(name for row in all_rows for name in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)


def save_run(path: Path, head, aug, names, lam_text, ops, raw_history) -> None:
    """Save the head, the augmentor, and every epoch of raw settings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "head": head.state_dict(),
        "aug": aug.state_dict(),
        "classes": names,
        "lam_text": lam_text,
        "ops": ops,
        "raw_history": raw_history,
    }, path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    cfg = load_config(args.config)
    device = pick_device(cfg["device"])
    ops = parse_ops(args.ops, cfg)
    names, captions = class_captions(cfg)
    train_dataset = load_split(cfg, "train", names)
    val_set = load_split(cfg, "val", names)
    batch_size = int(cfg["train_ours"]["batch_size"])
    if len(train_dataset) < batch_size:
        raise RuntimeError(f"Need at least {batch_size} training images.")
    encoder, tokenizer = build_clip(cfg, device)
    anchors = text_anchors(encoder.model, tokenizer, captions, device)
    head = InvariantHead(anchors.shape[1], anchors.shape[1]).to(device)
    aug = AutoAugmentor(ops=ops, jitter=float(cfg["augmentor"]["jitter"])).to(device)
    opt_head = torch.optim.AdamW(head.parameters(), lr=float(cfg["train_ours"]["lr_head"]))
    opt_aug = torch.optim.Adam(aug.parameters(), lr=float(cfg["train_ours"]["lr_aug"]))
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator, drop_last=True)
    print(f"train batches per epoch: {len(train_loader)}, total train images: {len(train_dataset)}")
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    rows = []
    raw_rows = []
    warnings = 0
    for epoch in range(1, args.epochs + 1):
        stats = run_epoch(
            train_loader, encoder, head, aug, anchors, opt_head, opt_aug, cfg, device, args.lam_text, epoch,
        )
        warnings += int(stats["warnings"])
        clean_acc = val_accuracy(val_loader, encoder, head, anchors, device, args.seed, 0.0)
        combined_acc = val_accuracy(val_loader, encoder, head, anchors, device, args.seed, 0.5)
        raw = flatten_raw(aug)
        rows.append({
            "epoch": epoch,
            "A_inv": stats["A_inv"],
            "B_total": stats["B_total"],
            "clean_dist_cosine": stats["cosine"],
            "val_clean_acc": clean_acc,
            "val_combined_acc": combined_acc,
            "head_std": stats["head_std"],
            "lam_text": args.lam_text,
            "ops": ",".join(ops),
            "seed": args.seed,
        })
        raw_rows.append({"epoch": epoch, "lam_text": args.lam_text, "ops": ",".join(ops), "seed": args.seed, **raw})
    key = (str(args.lam_text), ",".join(ops), str(args.seed))
    log_path = named(cfg, "results_dir", "train_ours_csv")
    raw_path = named(cfg, "results_dir", "aug_raw_csv")
    upsert(log_path, rows, key)
    upsert(raw_path, raw_rows, key)
    ckpt = checkpoint_path(cfg, args.lam_text, ops)
    save_run(ckpt, head, aug, names, args.lam_text, ops, raw_rows)
    last = rows[-1]
    announce(
        f"Trained {args.epochs} epochs, lam_text {args.lam_text}, ops {','.join(ops)}.",
        f"CSV: {log_path}  val clean {last['val_clean_acc']:.3f}, combined@0.5 {last['val_combined_acc']:.3f}.",
        f"Checkpoint: {ckpt}  collapse warnings: {warnings}. Raw settings: {raw_path}.",
    )


if __name__ == "__main__":
    main()
