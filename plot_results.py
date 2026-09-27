"""Draw the distortion sweep and the leave-one-out bars from the eval CSVs."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kit import announce, base_parser, file_in, load_config, read_csv, set_seed

PANEL_MODELS = ["M0", "M1", "M2", "M3"]
TITLE_MAP = {
    "geo": "Geometric Distortion",
    "photo": "Photometric Distortion",
    "blur": "Blur",
    "noise": "Noise",
    "moire": "Moire",
    "combined": "Combined (All Distortions)",
}


def draw_sweeps(rows: list[dict], kinds: list[str], dest) -> None:
    """One panel per distortion. Each line is a model's accuracy as severity grows."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
    for axis, kind in zip(axes.ravel(), kinds):
        for model in PANEL_MODELS:
            picked = [row for row in rows if row["model"] == model and row["kind"] == kind]
            picked.sort(key=lambda row: float(row["severity"]))
            axis.plot(
                [float(row["severity"]) for row in picked],
                [float(row["accuracy"]) for row in picked],
                marker="o",
                label=model,
            )
        axis.set_title(TITLE_MAP[kind], fontsize=11)
        axis.set_xlabel("Distortion Severity")
        axis.set_ylim(0, 1)
        axis.grid(True, alpha=0.3)
    for axis in axes[:, 0]:
        axis.set_ylabel("Top-1 Accuracy")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Experiment A: Top-1 Accuracy vs Distortion Severity (Freiburg Groceries test set)",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=120, bbox_inches="tight")
    plt.close(fig)


def draw_leaveout(rows: list[dict], severity: float, dest) -> None:
    """Bars at one severity: full M3 versus the model that never trained on that distortion."""
    title_map = {
        "geo": "Geometric",
        "photo": "Photometric",
        "blur": "Blur",
        "noise": "Noise",
        "moire": "Moire",
    }
    kinds = list(dict.fromkeys(row["held_out"] for row in rows))
    full, dropped = [], []
    for kind in kinds:
        full.append(float(next(
            row["accuracy"] for row in rows
            if row["held_out"] == kind and row["model"] == "M3" and float(row["severity"]) == severity
        )))
        dropped.append(float(next(
            row["accuracy"] for row in rows
            if row["held_out"] == kind and row["model"] == "M3-drop" and float(row["severity"]) == severity
        )))
    print(f"leave-one-out shape: ({len(rows)}, {len(rows[0]) if rows else 0})")
    print(full)
    print(dropped)
    fig, axis = plt.subplots(figsize=(8, 4))
    width = 0.36
    positions = list(range(len(kinds)))
    axis.bar([p - width / 2 for p in positions], full, width=width, label="M3")
    axis.bar([p + width / 2 for p in positions], dropped, width=width, label="M3 (trained without this kind)")
    display_labels = [title_map[kind] for kind in kinds]
    axis.set_xticks(range(len(kinds)))
    axis.set_xticklabels(display_labels)
    axis.set_ylim(0, 1)
    axis.set_title("Experiment B: Leave-One-Out Ablation (accuracy at severity 1.0)", fontsize=13)
    axis.set_xlabel("Held-Out Distortion Type")
    axis.set_ylabel("Top-1 Accuracy at Severity 1.0")
    axis.legend()
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = base_parser("Plot expA sweeps and expB leave-one-out bars.").parse_args()
    set_seed(args.seed)
    cfg = load_config(args.config)
    sweeps = read_csv(file_in(cfg, "results_dir", "expA_csv", args.seed))
    leave = read_csv(file_in(cfg, "results_dir", "expB_csv", args.seed))
    sweep_path = file_in(cfg, "plots_dir", "expA_plot", args.seed)
    leave_path = file_in(cfg, "plots_dir", "expB_plot", args.seed)
    draw_sweeps(sweeps, list(cfg["eval_recognition"]["kinds"]), sweep_path)
    draw_leaveout(leave, float(cfg["eval_recognition"]["leaveout_severity"]), leave_path)
    announce(
        f"Read {len(sweeps)} sweep rows and {len(leave)} leave-one-out rows.",
        f"Accuracy grid: {sweep_path}",
        f"Leave-one-out bars: {leave_path}",
    )


if __name__ == "__main__":
    main()
