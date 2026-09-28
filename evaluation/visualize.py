"""Generate FedAvg-versus-centralized convergence plots and benchmark tables."""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRIC_ROWS = (
    ("Accuracy", "accuracy"),
    ("Sensitivity / Recall", "sensitivity_recall"),
    ("Specificity", "specificity"),
    ("Precision", "precision"),
    ("F1-Score", "f1_score"),
    ("ROC-AUC", "roc_auc"),
)


def _load_json(path):
    with open(path, encoding="utf-8") as input_file:
        return json.load(input_file)


def _validate_curve(log, x_key, loss_key, accuracy_key, label):
    required = (x_key, loss_key, accuracy_key)
    missing = [key for key in required if key not in log]
    if missing:
        raise ValueError(f"{label} log is missing required field(s): {', '.join(missing)}")
    lengths = [len(log[key]) for key in required]
    if not lengths[0] or len(set(lengths)) != 1:
        raise ValueError(f"{label} curve fields must have the same non-zero length")


def _read_metric_file(path):
    data = _load_json(path)
    if "test_metrics" in data:
        data = data["test_metrics"]
    required = [key for _, key in METRIC_ROWS]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Metrics file {path} is missing: {', '.join(missing)}")
    return data


def _format_rate(value):
    return "N/A" if value is None else f"{value * 100:.2f}%"


def _format_delta(centralized, federated):
    if centralized is None or federated is None:
        return "N/A"
    return f"{(centralized - federated) * 100:+.2f} pp"


def create_comparison_table(centralized_metrics, federated_metrics):
    """Return a Markdown comparison table and rows for optional PNG rendering."""
    rows = []
    for label, key in METRIC_ROWS:
        central = centralized_metrics[key]
        federated = federated_metrics[key]
        rows.append((label, _format_rate(central), _format_rate(federated), _format_delta(central, federated)))

    markdown = [
        "| Metric | Centralized Pooled Baseline | Federated FedAvg (3 clients) | Delta (centralized - federated) |",
        "|---|---:|---:|---:|",
    ]
    markdown.extend(f"| {label} | {central} | {federated} | {delta} |" for label, central, federated, delta in rows)
    markdown.append("| Raw data shared | 100% pooled centrally | 0% raw data shared | Federated keeps raw data local |")
    markdown.extend([
        "",
        "> Grad-CAM explainability is planned post-30% and will use the ViT image-token representations to identify contributing radiographic regions.",
    ])
    return "\n".join(markdown) + "\n", rows


def plot_comparisons(
    fl_log_path="federated_training_log.json",
    centralized_log_path="centralized_training_log.json",
    centralized_metrics_path="centralized_test_metrics.json",
    federated_metrics_path="federated_test_metrics.json",
    output_dir=".",
):
    """Write 300-DPI validation curves and a populated comparison table."""
    fl_log = _load_json(fl_log_path)
    centralized_log = _load_json(centralized_log_path)
    _validate_curve(fl_log, "rounds", "global_val_loss", "global_val_accuracy", "Federated")
    _validate_curve(centralized_log, "epochs", "val_loss", "val_accuracy", "Centralized")
    centralized_metrics = _read_metric_file(centralized_metrics_path)
    federated_metrics = _read_metric_file(federated_metrics_path)
    os.makedirs(output_dir, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].plot(
        fl_log["rounds"], fl_log["global_val_loss"], marker="o",
        linewidth=2, label="FedAvg",
    )
    axes[0].plot(
        centralized_log["epochs"], centralized_log["val_loss"], marker="s",
        linestyle="--", linewidth=2, label="Centralized",
    )
    axes[0].set_title("Validation Loss")
    axes[0].set_xlabel("Communication rounds / epochs")
    axes[0].set_ylabel("Binary cross-entropy loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        fl_log["rounds"], [value * 100 for value in fl_log["global_val_accuracy"]],
        marker="o", linewidth=2, label="FedAvg",
    )
    axes[1].plot(
        centralized_log["epochs"], [value * 100 for value in centralized_log["val_accuracy"]],
        marker="s", linestyle="--", linewidth=2, label="Centralized",
    )
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Communication rounds / epochs")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    figure.suptitle("FedMed: Centralized Baseline vs FedAvg", fontsize=15)
    figure.tight_layout()

    plot_path = os.path.join(output_dir, "fedmed_30pct_comparison.png")
    figure.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    table_markdown, table_rows = create_comparison_table(
        centralized_metrics, federated_metrics
    )
    table_path = os.path.join(output_dir, "fedmed_30pct_comparison_table.md")
    with open(table_path, "w", encoding="utf-8") as table_file:
        table_file.write(table_markdown)

    table_figure, table_axis = plt.subplots(figsize=(11, 3.5))
    table_axis.axis("off")
    display_rows = [
        [label, central, federated, delta]
        for label, central, federated, delta in table_rows
    ]
    display_rows.append([
        "Raw data shared", "100% pooled centrally", "0% raw data shared", "Federated keeps raw data local"
    ])
    table = table_axis.table(
        cellText=display_rows,
        colLabels=["Metric", "Centralized", "FedAvg (3 clients)", "Delta"],
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.55)
    table_figure.tight_layout()
    table_png_path = os.path.join(output_dir, "fedmed_30pct_comparison_table.png")
    table_figure.savefig(table_png_path, dpi=300, bbox_inches="tight")
    plt.close(table_figure)

    print(f"Comparison plots: {plot_path}")
    print(f"Presentation table: {table_path}")
    print(f"Presentation table image: {table_png_path}")
    return plot_path, table_path, table_png_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot centralized and FedAvg convergence and create the benchmark table"
    )
    parser.add_argument("--federated-log", default="federated_training_log.json")
    parser.add_argument("--centralized-log", default="centralized_training_log.json")
    parser.add_argument("--centralized-metrics", default="centralized_test_metrics.json")
    parser.add_argument("--federated-metrics", default="federated_test_metrics.json")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()
    plot_comparisons(
        fl_log_path=args.federated_log,
        centralized_log_path=args.centralized_log,
        centralized_metrics_path=args.centralized_metrics,
        federated_metrics_path=args.federated_metrics,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
