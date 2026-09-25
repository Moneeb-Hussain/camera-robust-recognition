"""Download Freiburg Groceries and write stratified splits, counts, and captions."""
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from kit import (
    IMAGE_SUFFIXES,
    announce,
    base_parser,
    file_in,
    load_config,
    project_path,
    read_rgb,
    set_seed,
    write_csv,
)


def server_responds(url: str, timeout: int) -> bool:
    """True when the Freiburg file server answers before the timeout."""
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status < 400
    except urllib.error.HTTPError as exc:
        return exc.code not in {404, 408, 500, 502, 503, 504}
    except Exception:
        return False


def has_images(folder: Path) -> bool:
    """True when a folder holds image files directly inside it."""
    if not folder.is_dir():
        return False
    return any(path.suffix.lower() in IMAGE_SUFFIXES for path in folder.iterdir() if path.is_file())


def class_folders(root: Path) -> list[Path]:
    """List class folders sitting directly under a root."""
    if not root.is_dir():
        return []
    return sorted(path for path in root.iterdir() if has_images(path))


def find_class_root(base: Path) -> Path:
    """Find the directory whose children are one folder per grocery class."""
    if not base.exists():
        raise RuntimeError(f"Download folder not found: {base}")
    for current, dirs, _files in os.walk(base):
        dirs[:] = [name for name in dirs if not name.startswith(".")]
        depth = len(Path(current).relative_to(base).parts)
        if depth > 4:
            dirs.clear()
            continue
        here = Path(current)
        if len(class_folders(here)) >= 2:
            return here
    raise RuntimeError(f"No class folders found under {base}")


def place_classes(found: Path, image_dir: Path) -> None:
    """Leave one folder per class in the image directory from config."""
    if found.resolve() == image_dir.resolve():
        return
    image_dir.mkdir(parents=True, exist_ok=True)
    for class_dir in class_folders(found):
        dest = image_dir / class_dir.name
        if not dest.exists():
            shutil.move(str(class_dir), str(dest))


def clone_repo(url: str, repo_dir: Path) -> None:
    """Clone the Freiburg helper repo if its download script is not already here."""
    if (repo_dir / "src" / "download_dataset.py").is_file():
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", url, str(repo_dir)], check=True)


def tarball_size(repo_dir: Path) -> int:
    """Bytes received so far for the archive the official script writes."""
    names = [
        repo_dir / "freiburg_groceries_dataset.tar.gz",
        repo_dir.parent / "freiburg_groceries_dataset.tar.gz",
    ]
    return max((path.stat().st_size for path in names if path.is_file()), default=0)


def download_official(cfg: dict, repo_dir: Path) -> None:
    """Clone the dataset repo and run its downloader. Give up if the server stalls."""
    clone_repo(cfg["groceries"]["repo_url"], repo_dir)
    script = repo_dir / "src" / "download_dataset.py"
    timeout = int(cfg["groceries"]["timeout_seconds"])
    proc = subprocess.Popen([sys.executable, str(script)], cwd=script.parent)
    last_size = -1
    stalled_at = time.time()
    while proc.poll() is None:
        size = tarball_size(repo_dir)
        if size > last_size:
            last_size = size
            stalled_at = time.time()
        elif time.time() - stalled_at > timeout:
            proc.kill()
            raise TimeoutError("Freiburg server sent no data within 30 seconds")
        time.sleep(1)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, script)


def download_kaggle(dataset: str) -> Path:
    """Download the public Kaggle copy when the university server is silent."""
    import kagglehub
    return Path(kagglehub.dataset_download(dataset))


def ensure_images(cfg: dict, image_dir: Path, repo_dir: Path) -> str:
    """Get class folders, preferring Freiburg and falling back to Kaggle."""
    if len(class_folders(image_dir)) >= 2:
        return "already on disk"
    groceries = cfg["groceries"]
    timeout = int(groceries["timeout_seconds"])
    if server_responds(groceries["dataset_url"], timeout):
        try:
            download_official(cfg, repo_dir)
            place_classes(find_class_root(repo_dir.parent), image_dir)
            return "Freiburg server"
        except (subprocess.CalledProcessError, RuntimeError, OSError):
            pass
    kaggle_root = download_kaggle(groceries["kaggle_dataset"])
    place_classes(find_class_root(kaggle_root), image_dir)
    return "Kaggle"


def list_images(class_dir: Path) -> list[Path]:
    """Sorted image files that belong to one class."""
    files = [path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(files)


def portion(count: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    """70/10/20 counts for one class. Every class with 3+ images appears in all three splits."""
    n_train = int(round(count * train_ratio))
    n_val = int(round(count * val_ratio))
    n_test = count - n_train - n_val
    if n_test < 0:
        n_train += n_test
        n_test = 0
    parts = [n_train, n_val, n_test]
    if count >= 3:
        for index in range(3):
            if parts[index] == 0:
                donor = max(range(3), key=lambda item: parts[item])
                parts[donor] -= 1
                parts[index] = 1
    return parts[0], parts[1], parts[2]


def split_classes(image_dir: Path, seed: int, train_ratio: float, val_ratio: float):
    """Shuffle inside each class, then cut train, val, and test in the same proportion."""
    rng = np.random.default_rng(seed)
    splits = {"train": [], "val": [], "test": []}
    stats = []
    for class_dir in class_folders(image_dir):
        files = list_images(class_dir)
        rng.shuffle(files)
        n_train, n_val, n_test = portion(len(files), train_ratio, val_ratio)
        groups = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }
        for name, chosen in groups.items():
            splits[name].extend(f"{class_dir.name}/{path.name}" for path in chosen)
        stats.append({
            "class": class_dir.name,
            "train": len(groups["train"]),
            "val": len(groups["val"]),
            "test": len(groups["test"]),
        })
    for name in splits:
        splits[name].sort()
    return splits, stats


def captions_for(stats: list[dict]) -> dict:
    """Map each folder name to a caption with underscores turned into spaces."""
    return {row["class"]: f"a photo of {row['class'].replace('_', ' ')}" for row in stats}


class GroceryDataset(Dataset):
    """One split. Each photo is resized to the model size when it is read, not before."""

    def __init__(self, image_dir: Path, relative_paths: list[str], class_names: list[str], image_size: int):
        self.image_dir = Path(image_dir)
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


def report_small_classes(stats: list[dict], minimum: int) -> list[str]:
    """Print classes whose training pile is smaller than the cutoff."""
    small = []
    for row in stats:
        if int(row["train"]) < minimum:
            print(f"{row['class']} has {row['train']} train images")
            small.append(row["class"])
    return small


def main() -> None:
    args = base_parser("Download Freiburg Groceries and write stratified splits.").parse_args()
    set_seed(args.seed)
    cfg = load_config(args.config)
    image_dir = project_path(cfg, cfg["paths"]["image_dir"])
    repo_dir = project_path(cfg, cfg["paths"]["upstream_repo"])
    source = ensure_images(cfg, image_dir, repo_dir)
    splits, stats = split_classes(
        image_dir, args.seed, float(cfg["groceries"]["train_ratio"]), float(cfg["groceries"]["val_ratio"]),
    )
    splits_path = file_in(cfg, "results_dir", "splits_json", args.seed)
    captions_path = file_in(cfg, "results_dir", "captions_json", args.seed)
    stats_path = file_in(cfg, "results_dir", "class_stats_csv", args.seed)
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    splits_path.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    captions_path.write_text(json.dumps(captions_for(stats), indent=2), encoding="utf-8")
    write_csv(stats_path, stats)
    small = report_small_classes(stats, int(cfg["groceries"]["min_train_images"]))
    n_images = sum(int(row["train"]) + int(row["val"]) + int(row["test"]) for row in stats)
    small_text = ", ".join(small) if small else "none"
    announce(
        f"Source: {source}. {len(stats)} classes, {n_images} images in {image_dir}.",
        f"Wrote {splits_path.name}, {captions_path.name}, and {stats_path.name}.",
        f"Split sizes train/val/test {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}. Classes under {cfg['groceries']['min_train_images']} train images: {small_text}.",
    )


if __name__ == "__main__":
    main()
