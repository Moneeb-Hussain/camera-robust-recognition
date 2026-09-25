"""Plot robustness CSV: accuracy versus distortion severity for each model."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kit import announce, base_parser, file_in, load_config, read_csv, set_seed


def draw(rows: list[dict], dest) -> None:
    """One panel per model. Each line is a camera distortion from the CSV."""
    models = list(dict.fromkeys(row["model"] for row in rows))
    kinds = list(dict.fromkeys(row["kind"] for row in rows))
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 4), squeeze=False)
    for axis, model in zip(axes[0], models):
        for kind in kinds:
            picked = [row for row in rows if row["model"] == model and row["kind"] == kind]
            picked.sort(key=lambda row: float(row["severity"]))
            axis.plot(
                [float(row["severity"]) for row in picked],
                [float(row["accuracy"]) for row in picked],
                marker="o",
                label=kind,
            )
        axis.set_title(model)
        axis.set_xlabel("severity")
        axis.set_ylabel("accuracy")
        axis.set_ylim(0, 1)
        axis.legend(fontsize=8)
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=120)
    plt.close(fig)


def main() -> None:
    args = base_parser("Plot robustness_seed CSV written by evaluate.py.").parse_args()
    set_seed(args.seed)
    cfg = load_config(args.config)
    csv_path = file_in(cfg, "results_dir", "robustness_csv", args.seed)
    rows = read_csv(csv_path)
    dest = file_in(cfg, "plots_dir", "robustness_plot", args.seed)
    draw(rows, dest)
    models = ", ".join(dict.fromkeys(row["model"] for row in rows))
    announce(
        f"Read {len(rows)} rows from {csv_path.name}.",
        f"Plot: {dest}",
        f"Lines show accuracy versus severity for {models}.",
    )


if __name__ == "__main__":
    main()
