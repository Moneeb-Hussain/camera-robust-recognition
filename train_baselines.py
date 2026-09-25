"""Three baselines: zero-shot CLIP, a linear probe, and a head on fixed camera noise."""
import json
import math

import kornia.augmentation as K
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from features import anchor_rows, build_encoders, load_caption_table
from invariance_core import InvariantHead, loss_text, op_moire
from kit import announce, base_parser, file_in, load_config, pick_device, project_path, set_seed, write_csv
from prep_data import GroceryDataset


class FixedCameraAugment(nn.Module):
    """Same four Kornia transforms every time, plus one random moire stripe. No learned adversary."""

    def __init__(self):
        super().__init__()
        self.perspective = K.RandomPerspective(distortion_scale=0.2, p=1.0)
        self.color = K.ColorJitter(0.4, 0.4, 0.4, 0.1, p=1.0)
        self.blur = K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=1.0)
        self.noise = K.RandomGaussianNoise(mean=0.0, std=0.05, p=1.0)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch = images.shape[0]
        stripe = op_moire(
            images,
            torch.empty(batch, device=images.device).uniform_(0.0, 0.10),
            torch.empty(batch, device=images.device).uniform_(4.0, 40.0),
            torch.empty(batch, device=images.device).uniform_(4.0, 40.0),
            torch.empty(batch, device=images.device).uniform_(0.0, 2 * math.pi),
        )
        twisted = self.noise(self.blur(self.color(self.perspective(stripe))))
        return twisted.clamp(0, 1)


class LinearProbe(nn.Module):
    """One matrix from a cached CLIP vector to class logits."""

    def __init__(self, dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(dim, n_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)


def make_dataset(cfg: dict, split: str, seed: int) -> tuple[GroceryDataset, list[str]]:
    """One grocery split. Class order matches the rows of the anchor matrix."""
    image_dir = project_path(cfg, cfg["paths"]["image_dir"])
    splits_path = file_in(cfg, "results_dir", "splits_json", seed)
    captions_path = file_in(cfg, "results_dir", "captions_json", seed)
    names, _captions = load_caption_table(captions_path)
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    dataset = GroceryDataset(image_dir, splits[split], names, int(cfg["clip"]["image_size"]))
    return dataset, names


def encode_clean(encoder, dataset: GroceryDataset, batch_size: int, device: torch.device):
    """Run the frozen encoder once. These vectors are the linear probe's training set."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    features, labels = [], []
    with torch.no_grad():
        for images, target in loader:
            features.append(encoder(images.to(device)).cpu())
            labels.append(target)
    return torch.cat(features), torch.cat(labels)


def cached_train_features(encoder, dataset, cfg, seed, device):
    """Reuse the clean training cache when this seed already built it."""
    path = file_in(cfg, "ckpt_dir", "feature_cache", seed)
    if path.is_file():
        blob = torch.load(path, map_location="cpu", weights_only=True)
        return blob["features"], blob["labels"]
    features, labels = encode_clean(encoder, dataset, int(cfg["baselines"]["batch_size"]), device)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": features, "labels": labels}, path)
    return features, labels


def batched_accuracy(features, labels, predict, batch_size: int, device: torch.device) -> float:
    """Accuracy of a function that maps a feature batch to class ids."""
    correct = 0
    for start in range(0, len(labels), batch_size):
        pred = predict(features[start:start + batch_size].to(device)).cpu()
        correct += int((pred == labels[start:start + batch_size]).sum().item())
    return correct / len(labels)


def fit_probe(features, labels, n_classes: int, cfg: dict, seed: int, device: torch.device) -> LinearProbe:
    """Train the linear probe on cached clean features. The image tower stays frozen."""
    probe = LinearProbe(features.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=float(cfg["baselines"]["lr_probe"]))
    batch_size = int(cfg["baselines"]["batch_size"])
    generator = torch.Generator().manual_seed(seed)
    for _epoch in range(int(cfg["baselines"]["epochs"])):
        order = torch.randperm(len(labels), generator=generator)
        for start in range(0, len(labels) - batch_size + 1, batch_size):
            idx = order[start:start + batch_size]
            loss = F.cross_entropy(probe(features[idx].to(device)), labels[idx].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return probe.eval()


def fit_head(encoder, dataset, anchors, cfg, seed, device) -> InvariantHead:
    """Train InvariantHead with cross-entropy to the anchors. Augmentation is random but not learned."""
    head = InvariantHead(anchors.shape[1], anchors.shape[1]).to(device)
    augment = FixedCameraAugment().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=float(cfg["baselines"]["lr_head"]))
    batch_size = int(cfg["baselines"]["batch_size"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    anchors = anchors.to(device)
    for _epoch in range(int(cfg["baselines"]["epochs"])):
        head.train()
        for images, target in loader:
            if images.shape[0] < 2:
                continue
            images = augment(images.to(device))
            target = target.to(device)
            with torch.no_grad():
                features = encoder(images)
            loss = loss_text(head(features), anchors, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return head.eval()


def save_state(path, module, classes: list[str]) -> None:
    """Write one baseline checkpoint under ckpt/."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": module.state_dict(), "classes": classes}, path)


def main() -> None:
    args = base_parser("Train M0 zero-shot, M1 linear probe, and M2 fixed-augmentation head.").parse_args()
    set_seed(args.seed)
    cfg = load_config(args.config)
    device = pick_device(cfg["device"])
    names, captions = load_caption_table(file_in(cfg, "results_dir", "captions_json", args.seed))
    image_encoder, text_encoder = build_encoders(cfg, device)
    anchors = anchor_rows(text_encoder, names, captions)
    train_set, _names = make_dataset(cfg, "train", args.seed)
    val_set, _names = make_dataset(cfg, "val", args.seed)
    batch_size = int(cfg["baselines"]["batch_size"])
    train_features, train_labels = cached_train_features(image_encoder, train_set, cfg, args.seed, device)
    val_features, val_labels = encode_clean(image_encoder, val_set, batch_size, device)
    m0 = lambda batch: (batch @ anchors.to(batch.device).t()).argmax(dim=-1)
    m0_acc = batched_accuracy(val_features, val_labels, m0, batch_size, device)
    probe = fit_probe(train_features, train_labels, len(names), cfg, args.seed, device)
    m1_acc = batched_accuracy(val_features, val_labels, lambda batch: probe(batch).argmax(dim=-1), batch_size, device)
    head = fit_head(image_encoder, train_set, anchors, cfg, args.seed, device)
    m2 = lambda batch: (head(batch) @ anchors.to(batch.device).t()).argmax(dim=-1)
    m2_acc = batched_accuracy(val_features, val_labels, m2, batch_size, device)
    save_state(file_in(cfg, "ckpt_dir", "m1_ckpt", args.seed), probe, names)
    save_state(file_in(cfg, "ckpt_dir", "m2_ckpt", args.seed), head, names)
    csv_path = file_in(cfg, "results_dir", "baseline_csv", args.seed)
    write_csv(csv_path, [
        {"model": "M0", "split": "val", "accuracy": m0_acc, "n_images": len(val_labels), "seed": args.seed},
        {"model": "M1", "split": "val", "accuracy": m1_acc, "n_images": len(val_labels), "seed": args.seed},
        {"model": "M2", "split": "val", "accuracy": m2_acc, "n_images": len(val_labels), "seed": args.seed},
    ])
    announce(
        f"Val accuracy M0 {m0_acc:.3f}, M1 {m1_acc:.3f}, M2 {m2_acc:.3f} on {len(val_labels)} clean images.",
        f"CSV: {csv_path}",
        f"Checkpoints: {file_in(cfg, 'ckpt_dir', 'm1_ckpt', args.seed).name}, {file_in(cfg, 'ckpt_dir', 'm2_ckpt', args.seed).name}",
    )


if __name__ == "__main__":
    main()
