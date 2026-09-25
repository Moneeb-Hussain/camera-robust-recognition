"""Plot the training CSV: attack strength, head loss, and clean accuracy."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kit import announce, base_parser, file_in, load_config, read_csv, set_seed


def draw(rows: list[dict], dest) -> None:
    """Two panels from one training log: losses, then clean accuracy."""
    epochs = [int(row["epoch"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    axes[0].plot(epochs, [float(row["a_inv"]) for row in rows], marker="o", label="aug attack")
    axes[0].plot(epochs, [float(row["b_total"]) for row in rows], marker="o", label="head loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[1].plot(epochs, [float(row["clean_acc"]) for row in rows], marker="o", color="tab:green")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("clean accuracy")
    axes[1].set_ylim(0, 1)
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=120)
    plt.close(fig)


def main() -> None:
    args = base_parser("Plot train_seed CSV written by train.py.").parse_args()
    set_seed(args.seed)
    cfg = load_config(args.config)
    csv_path = file_in(cfg, "results_dir", "train_csv", args.seed)
    rows = read_csv(csv_path)
    dest = file_in(cfg, "plots_dir", "train_plot", args.seed)
    draw(rows, dest)
    last = rows[-1]
    announce(
        f"Read {len(rows)} rows from {csv_path.name}.",
        f"Plot: {dest}",
        f"Final epoch {last['epoch']}: head loss {float(last['b_total']):.3f}, clean accuracy {float(last['clean_acc']):.3f}.",
    )


if __name__ == "__main__":
    main()
