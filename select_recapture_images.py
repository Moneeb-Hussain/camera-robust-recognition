"""Copy one test photo per class into a folder ready for phone recapture."""
import random
import json
from pathlib import Path

import cv2

from kit import announce, base_parser, file_in, load_config, project_path, set_seed, write_csv


def test_paths(cfg: dict) -> list[str]:
    """Read the test split written by prep_data.py."""
    path = file_in(cfg, "results_dir", "splits_json", 0)
    if not path.is_file():
        raise RuntimeError(f"Missing {path}. Run prep_data.py first.")
    return list(json.loads(path.read_text(encoding="utf-8"))["test"])


def group_by_class(paths: list[str]) -> dict[str, list[str]]:
    """Bucket test paths by the class folder name."""
    groups: dict[str, list[str]] = {}
    for relative in paths:
        name = relative.split("/", 1)[0]
        groups.setdefault(name, []).append(relative)
    return groups


def pick_one_each(groups: dict[str, list[str]], seed: int) -> list[tuple[str, str]]:
    """Shuffle inside each class, then keep a single photo. Order is alphabetical."""
    rng = random.Random(seed)
    chosen = []
    for name in sorted(groups):
        files = list(groups[name])
        if not files:
            raise RuntimeError(f"Class {name} has no test images.")
        rng.shuffle(files)
        chosen.append((name, files[0]))
    return chosen


def write_jpg(source: Path, dest: Path) -> None:
    """Save a real JPEG, even when the grocery file was a PNG."""
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"OpenCV could not read {source}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dest), image):
        raise RuntimeError(f"OpenCV could not write {dest}")


def clear_old(folder: Path) -> None:
    """Drop a previous pick so the folder holds exactly this run."""
    if not folder.is_dir():
        return
    for old in folder.glob("*.jpg"):
        old.unlink()


def main() -> None:
    args = base_parser("Pick one test image per class for recapture.").parse_args()
    set_seed(args.seed)
    cfg = load_config(args.config)
    image_dir = project_path(cfg, cfg["paths"]["image_dir"])
    dest_dir = project_path(cfg, cfg["paths"]["recapture_dir"])
    groups = group_by_class(test_paths(cfg))
    chosen = pick_one_each(groups, args.seed)
    clear_old(dest_dir)
    rows = []
    for index, (name, relative) in enumerate(chosen, start=1):
        filename = f"{index:02d}_{name.lower()}.jpg"
        write_jpg(image_dir / relative, dest_dir / filename)
        rows.append({"filename": filename, "true_label": name})
    manifest = file_in(cfg, "results_dir", "recapture_manifest", args.seed)
    write_csv(manifest, rows)
    announce(
        f"Copied {len(rows)} images, one per class.",
        f"Folder: {dest_dir}",
        f"Manifest: {manifest}",
    )


if __name__ == "__main__":
    main()
