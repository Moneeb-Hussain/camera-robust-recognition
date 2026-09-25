"""Draw a tiny labeled image folder so the pipeline can run before real photos exist."""
import math
from pathlib import Path

import cv2
import numpy as np

from kit import announce, base_parser, file_in, load_config, project_path, set_seed, write_csv


def star_points(cx: int, cy: int, radius: int) -> np.ndarray:
    """Five-point star around a center, used as one toy class."""
    pts = []
    for i in range(10):
        ang = -math.pi / 2 + i * math.pi / 5
        rad = radius if i % 2 == 0 else radius * 0.4
        pts.append([int(cx + rad * math.cos(ang)), int(cy + rad * math.sin(ang))])
    return np.array(pts, np.int32)


def paint(canvas: np.ndarray, shape: str, color_bgr: tuple, rng: np.random.Generator) -> None:
    """Stamp one colored shape on a gray canvas, nudged so copies are not identical."""
    height, width = canvas.shape[:2]
    cx = int(rng.integers(width // 3, 2 * width // 3))
    cy = int(rng.integers(height // 3, 2 * height // 3))
    radius = int(rng.integers(width // 8, width // 5))
    if shape == "circle":
        cv2.circle(canvas, (cx, cy), radius, color_bgr, -1)
    elif shape == "square":
        cv2.rectangle(canvas, (cx - radius, cy - radius), (cx + radius, cy + radius), color_bgr, -1)
    elif shape == "triangle":
        pts = np.array([[cx, cy - radius], [cx - radius, cy + radius], [cx + radius, cy + radius]])
        cv2.fillPoly(canvas, [pts], color_bgr)
    elif shape == "star":
        cv2.fillPoly(canvas, [star_points(cx, cy, radius)], color_bgr)
    else:
        raise ValueError(f"Unknown shape {shape}")


def render_class(folder: Path, item: dict, count: int, size: int, rng: np.random.Generator) -> list[dict]:
    """Rebuild one class folder and return manifest rows for the new files."""
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.iterdir():
        if old.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            old.unlink()
    red, green, blue = item["color"]
    rows = []
    for index in range(count):
        gray = int(rng.integers(15, 45))
        canvas = np.full((size, size, 3), gray, np.uint8)
        paint(canvas, item["shape"], (blue, green, red), rng)
        name = f"img_{index:03d}.png"
        cv2.imwrite(str(folder / name), canvas)
        rows.append({"class_name": item["name"], "file": f"{item['name']}/{name}"})
    return rows


def main() -> None:
    args = base_parser("Draw toy class folders named in config.yaml.").parse_args()
    set_seed(args.seed)
    cfg = load_config(args.config)
    image_dir = project_path(cfg, cfg["paths"]["toy_dir"])
    size = int(cfg["clip"]["image_size"])
    count = int(cfg["toy"]["images_per_class"])
    rng = np.random.default_rng(args.seed)
    rows = []
    for item in cfg["toy"]["items"]:
        rows.extend(render_class(image_dir / item["name"], item, count, size, rng))
    for row in rows:
        row["seed"] = args.seed
    csv_path = file_in(cfg, "results_dir", "toy_csv", args.seed)
    write_csv(csv_path, rows)
    classes = len(cfg["toy"]["items"])
    announce(
        f"Wrote {len(rows)} toy images across {classes} classes.",
        f"Manifest: {csv_path}",
        f"Image folder: {image_dir}",
    )


if __name__ == "__main__":
    main()
