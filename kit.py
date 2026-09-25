"""Shared helpers: config paths, frozen CLIP, image folders, and CSV IO."""
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
from invariance_core import distort

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def base_parser(description: str) -> argparse.ArgumentParser:
    """Flags shared by every script: where config.yaml is, and the seed."""
    parser = argparse.ArgumentParser(description=description)
    default_cfg = str(Path(__file__).with_name("config.yaml"))
    parser.add_argument("--config", default=default_cfg, help="Path to config.yaml")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def load_config(path: str) -> dict:
    """Read the yaml that holds every path and setting."""
    cfg_path = Path(path).resolve()
    with cfg_path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["_config_dir"] = str(cfg_path.parent)
    return cfg


def project_path(cfg: dict, raw: str) -> Path:
    """Turn a path from config into an absolute path beside config.yaml."""
    path = Path(raw)
    if path.is_absolute():
        return path
    return Path(cfg["_config_dir"]) / path


def file_in(cfg: dict, dir_key: str, file_key: str, seed: int) -> Path:
    """Build an output path from a config folder and a seed-stamped file name."""
    folder = project_path(cfg, cfg["paths"][dir_key])
    return folder / cfg["files"][file_key].format(seed=seed)


def set_seed(seed: int) -> None:
    """Make Python, NumPy, and PyTorch repeat the same run."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name: str) -> torch.device:
    """Use the requested device, or CUDA when config says auto."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def announce(line1: str, line2: str, line3: str) -> None:
    """Print the three-line summary every script ends with."""
    print(line1)
    print(line2)
    print(line3)


def write_csv(path: Path, rows: list[dict]) -> None:
    """Store experiment rows so a separate script can plot them."""
    if not rows:
        raise RuntimeError(f"No rows to write for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    """Load a results CSV written by an experiment script."""
    if not path.is_file():
        raise RuntimeError(f"Missing {path}. Run the experiment script for this seed first.")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_rgb(path: Path, size: int) -> torch.Tensor:
    """Load one image as a float tensor in [0, 1], resized to a square."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"OpenCV could not read {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    array = np.ascontiguousarray(rgb)
    return torch.from_numpy(array).permute(2, 0, 1).float() / 255.0


def load_dataset(image_dir: Path, size: int):
    """Read class folders into tensors. The folder name is the class."""
    if not image_dir.is_dir():
        raise RuntimeError(f"Image folder not found: {image_dir}. Run make_toy_images.py first.")
    samples = []
    for folder in sorted(p for p in image_dir.iterdir() if p.is_dir()):
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        samples.extend((file, folder.name) for file in files)
    if len(samples) < 2:
        raise RuntimeError("Need at least two images under class folders.")
    names = sorted({name for _, name in samples})
    index = {name: i for i, name in enumerate(names)}
    images = [read_rgb(path, size) for path, _ in samples]
    labels = [index[name] for _, name in samples]
    return torch.stack(images), torch.tensor(labels, dtype=torch.long), names


class UnitImageEncoder(nn.Module):
    """Frozen CLIP vision tower. A [0, 1] image goes in; a unit feature comes out."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        mean = torch.tensor(self.model.visual.image_mean, dtype=torch.float32)
        std = torch.tensor(self.model.visual.image_std, dtype=torch.float32)
        self.register_buffer("mean", mean.view(1, 3, 1, 1))
        self.register_buffer("std", std.view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.model.encode_image((images - self.mean) / self.std).float()
        return F.normalize(features, dim=-1)


def build_clip(cfg: dict, device: torch.device):
    """Load frozen CLIP plus an image callable that invariance_core can train through."""
    name = cfg["clip"]["model_name"]
    model, _, _ = open_clip.create_model_and_transforms(
        name, pretrained=cfg["clip"]["pretrained"], force_quick_gelu=True,
    )
    model = model.float().to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    encoder = UnitImageEncoder(model).to(device).eval()
    return model, encoder, open_clip.get_tokenizer(name)


@torch.no_grad()
def make_anchors(model, tokenizer, names: list[str], prompt: str, device: torch.device) -> torch.Tensor:
    """Turn each class name into a caption and encode it with CLIP text."""
    tokens = tokenizer([prompt.format(name=name) for name in names]).to(device)
    return F.normalize(model.encode_text(tokens).float(), dim=-1)


def save_checkpoint(path_pt: Path, path_json: Path, head, aug, classes: list[str], dim: int) -> None:
    """Save the head, the learned distortions, and the class list."""
    path_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"head": head.state_dict(), "aug": aug.state_dict()}, path_pt)
    path_json.write_text(json.dumps({"classes": classes, "dim": dim}), encoding="utf-8")


def load_checkpoint(path_pt: Path, path_json: Path, device: torch.device):
    """Load a training run. Returns None when this seed has not been trained."""
    if not path_pt.is_file() or not path_json.is_file():
        return None
    meta = json.loads(path_json.read_text(encoding="utf-8"))
    blob = torch.load(path_pt, map_location=device, weights_only=True)
    return blob, meta


@torch.no_grad()
def distorted_accuracy(images, labels, encode, anchors, kind, severity, seed, batch_size, device) -> float:
    """Apply one camera distortion, then measure nearest-caption accuracy."""
    correct = 0
    for start in range(0, len(labels), batch_size):
        batch = images[start:start + batch_size].to(device)
        target = labels[start:start + batch_size].to(device)
        twisted = distort(batch, kind, severity, seed=seed + start)
        pred = (encode(twisted) @ anchors.t()).argmax(dim=-1)
        correct += int((pred == target).sum().item())
    return correct / len(labels)
