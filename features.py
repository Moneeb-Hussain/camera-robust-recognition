"""Frozen CLIP image tower, and text anchors built from captions.json."""
import json

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F

from kit import announce, base_parser, file_in, load_config, pick_device, project_path, set_seed, write_csv


class FrozenImageEncoder(nn.Module):
    """Frozen ViT-B/32. A [0, 1] image is normalized the CLIP way inside this module."""

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


class TextAnchorEncoder(nn.Module):
    """Turn one caption per class into a row of a unit-length anchor matrix."""

    def __init__(self, model: nn.Module, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, captions: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(captions).to(next(self.model.parameters()).device)
        with torch.no_grad():
            text = self.model.encode_text(tokens).float()
        return F.normalize(text, dim=-1)


def load_caption_table(path) -> tuple[list[str], list[str]]:
    """Read captions.json in class order. That order is the anchor row order."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    names = sorted(raw)
    return names, [raw[name] for name in names]


def build_encoders(cfg: dict, device: torch.device):
    """Load openai ViT-B/32 once and share it between the image and text towers."""
    name = cfg["clip"]["model_name"]
    model, _, _ = open_clip.create_model_and_transforms(
        name, pretrained=cfg["clip"]["pretrained"], force_quick_gelu=True,
    )
    model = model.float().to(device).eval()
    image_encoder = FrozenImageEncoder(model).to(device).eval()
    text_encoder = TextAnchorEncoder(model, open_clip.get_tokenizer(name)).to(device).eval()
    return image_encoder, text_encoder


def anchor_rows(text_encoder: TextAnchorEncoder, names: list[str], captions: list[str]) -> torch.Tensor:
    """Encode captions and refuse a matrix that is not one 512-d unit vector per class."""
    anchors = text_encoder(captions).detach().cpu()
    if anchors.shape != (len(names), 512):
        raise RuntimeError(f"Expected anchor matrix ({len(names)}, 512), got {tuple(anchors.shape)}")
    return anchors


def main() -> None:
    args = base_parser("Build frozen CLIP features and the caption anchor matrix.").parse_args()
    set_seed(args.seed)
    cfg = load_config(args.config)
    device = pick_device(cfg["device"])
    names, captions = load_caption_table(file_in(cfg, "results_dir", "captions_json", args.seed))
    _image_encoder, text_encoder = build_encoders(cfg, device)
    anchors = anchor_rows(text_encoder, names, captions)
    anchor_path = file_in(cfg, "ckpt_dir", "anchors_pt", args.seed)
    anchor_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"classes": names, "anchors": anchors}, anchor_path)
    norms = anchors.norm(dim=-1)
    csv_path = file_in(cfg, "results_dir", "anchors_csv", args.seed)
    write_csv(csv_path, [
        {"class": name, "caption": caption, "l2_norm": float(norm), "seed": args.seed}
        for name, caption, norm in zip(names, captions, norms)
    ])
    announce(
        f"Frozen {cfg['clip']['model_name']} image encoder. Inputs are [0, 1]; CLIP normalization is inside.",
        f"Anchor matrix {tuple(anchors.shape)}, mean L2 norm {float(norms.mean()):.4f}.",
        f"Saved {anchor_path} and {csv_path}.",
    )


if __name__ == "__main__":
    main()
