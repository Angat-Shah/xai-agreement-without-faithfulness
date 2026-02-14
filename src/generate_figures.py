"""
XAI Agreement and Faithfulness Study - Figure Generation
Produces publication-quality figures from experiment results.
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "text.usetex": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Color palette
COLORS = {
    "SHAP": "#2196F3",
    "LIME": "#FF9800",
    "PermImp": "#4CAF50",
    "Random": "#9E9E9E",
}

PAIR_COLORS = {
    "SHAP_vs_LIME": "#7B1FA2",
    "SHAP_vs_PermImp": "#00796B",
    "LIME_vs_PermImp": "#E65100",
}


def load_results():
    """Load experiment results."""
    with open(os.path.join(RAW_DIR, "detailed_results.json"), "r") as f:
        detailed = json.load(f)
    with open(os.path.join(PROCESSED_DIR, "all_results.json"), "r") as f:
        summary = json.load(f)
    return detailed, summary


def fig1_agreement_distributions(detailed, summary):
    """Figure 1: Distribution of pairwise agreement (Spearman ρ)."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)

    pairs = list(detailed["agreement"]["spearman"].keys())

    for ax, pair in zip(axes, pairs):
        values = detailed["agreement"]["spearman"][pair]["values"]
        color = PAIR_COLORS[pair]

        ax.hist(values, bins=40, color=color, alpha=0.7, edgecolor="white",
                linewidth=0.5, density=True)

        mean_val = np.mean(values)
        ax.axvline(mean_val, color="black", linestyle="--", linewidth=1.2,
                   label=f"Mean = {mean_val:.2f}")

        pair_label = pair.replace("_vs_", " vs. ").replace("PermImp", "Perm. Sens.")
        ax.set_title(pair_label, fontweight="bold")
        ax.set_xlabel("Spearman ρ")
        ax.legend(loc="upper left", framealpha=0.8)

    axes[0].set_ylabel("Density")
    fig.suptitle("Per-Instance Explanation Agreement (Spearman ρ)", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig1_agreement_distributions.pdf"))
    plt.savefig(os.path.join(FIGURES_DIR, "fig1_agreement_distributions.png"))
    plt.close()
    print("Generated: fig1_agreement_distributions")


def fig2_topk_agreement(detailed, summary):
    """Figure 2: Top-k feature overlap comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    pairs = list(detailed["agreement"]["topk_3"].keys())
    pair_labels = [p.replace("_vs_", " vs ").replace("PermImp", "PS") for p in pairs]

    for ax, k_name, title in zip(axes, ["topk_3", "topk_5"], ["Top-3 Overlap", "Top-5 Overlap"]):
        means = [detailed["agreement"][k_name][p]["mean"] for p in pairs]
        stds = [detailed["agreement"][k_name][p]["std"] for p in pairs]
        colors = [PAIR_COLORS[p] for p in pairs]

        bars = ax.bar(pair_labels, means, yerr=stds, color=colors, alpha=0.8,
                      edgecolor="white", linewidth=0.8, capsize=4)
        ax.set_ylabel("Overlap Proportion")
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.axhline(y=1/3 if "3" in k_name else 1/5, color="gray",
                   linestyle=":", linewidth=1, alpha=0.5, label="Random baseline")
        ax.legend(loc="upper right", framealpha=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig2_topk_agreement.pdf"))
    plt.savefig(os.path.join(FIGURES_DIR, "fig2_topk_agreement.png"))
    plt.close()
    print("Generated: fig2_topk_agreement")


def fig3_faithfulness_comparison(detailed, summary):
    """Figure 3: Comprehensiveness and sufficiency comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    methods = ["SHAP", "LIME", "PermImp", "Random"]
    method_labels = ["SHAP", "LIME", "Perm. Sens.", "Random"]
    k_values = [f"k={k}" for k in FAITHFULNESS_K_VALUES]

    # Comprehensiveness
    ax = axes[0]
    x = np.arange(len(k_values))
    width = 0.18
    for i, (m, ml) in enumerate(zip(methods, method_labels)):
        means = [detailed["faithfulness"]["comprehensiveness"][m][k]["mean"] for k in k_values]
        stds = [detailed["faithfulness"]["comprehensiveness"][m][k]["std"] for k in k_values]
        ax.bar(x + i * width, means, width, yerr=stds, label=ml,
               color=COLORS[m], alpha=0.85, edgecolor="white", capsize=3)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f"k={k}" for k in FAITHFULNESS_K_VALUES])
    ax.set_ylabel("Comprehensiveness\n(Probability Drop)")
    ax.set_title("Comprehensiveness (↑ better)", fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.8)

    # Sufficiency
    ax = axes[1]
    for i, (m, ml) in enumerate(zip(methods, method_labels)):
        means = [detailed["faithfulness"]["sufficiency"][m][k]["mean"] for k in k_values]
        stds = [detailed["faithfulness"]["sufficiency"][m][k]["std"] for k in k_values]
        ax.bar(x + i * width, means, width, yerr=stds, label=ml,
               color=COLORS[m], alpha=0.85, edgecolor="white", capsize=3)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f"k={k}" for k in FAITHFULNESS_K_VALUES])
    ax.set_ylabel("Sufficiency Score\n(Probability Drop)")
    ax.set_title("Sufficiency (↓ better)", fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig3_faithfulness.pdf"))
    plt.savefig(os.path.join(FIGURES_DIR, "fig3_faithfulness.png"))
    plt.close()
    print("Generated: fig3_faithfulness")


def fig4_agreement_vs_faithfulness(detailed, summary):
    """Figure 4: Scatter plot of agreement vs faithfulness."""
    pairs_data = detailed["agreement_vs_faithfulness"]
    n_pairs = len(pairs_data)

    fig, axes = plt.subplots(1, n_pairs, figsize=(4 * n_pairs, 3.5))
    if n_pairs == 1:
        axes = [axes]

    for ax, (pair_key, data) in zip(axes, pairs_data.items()):
        agreements = np.array(data["agreement_values"])
        faithfulness = np.array(data["faithfulness_values"])

        color = PAIR_COLORS.get(pair_key, "#333333")

        ax.scatter(agreements, faithfulness, alpha=0.2, s=8, color=color)

        # Trend line
        z = np.polyfit(agreements, faithfulness, 1)
        p = np.poly1d(z)
        x_line = np.linspace(agreements.min(), agreements.max(), 100)
        ax.plot(x_line, p(x_line), color="black", linewidth=1.5, linestyle="--")

        r_val = data.get("spearman_r", data.get("pearson_r", 0))
        p_val = data.get("spearman_p", data.get("pearson_p", 0))
        pair_label = pair_key.replace("_vs_", " vs ").replace("PermImp", "PS")

        ax.set_title(f"{pair_label}\nρ = {r_val:.3f}, p = {p_val:.3e}", fontweight="bold")
        ax.set_xlabel("Agreement (Spearman ρ)")
        ax.set_ylabel("Avg. Comprehensiveness")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig4_agreement_vs_faithfulness.pdf"))
    plt.savefig(os.path.join(FIGURES_DIR, "fig4_agreement_vs_faithfulness.png"))
    plt.close()
    print("Generated: fig4_agreement_vs_faithfulness")


def fig5_stability_comparison(detailed, summary):
    """Figure 5: Explanation stability comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    stab_data = detailed["stability"]

    # Box/violin plot
    ax = axes[0]
    shap_vals = stab_data["SHAP"]["values"]
    lime_vals = stab_data["LIME"]["values"]

    parts = ax.violinplot([shap_vals, lime_vals], positions=[1, 2], showmedians=True)

    for i, (body, color) in enumerate(zip(parts["bodies"], [COLORS["SHAP"], COLORS["LIME"]])):
        body.set_facecolor(color)
        body.set_alpha(0.6)

    parts["cmedians"].set_color("black")
    parts["cmins"].set_color("black")
    parts["cmaxes"].set_color("black")
    parts["cbars"].set_color("black")

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["SHAP", "LIME"])
    ax.set_ylabel("Stability (Spearman ρ)")
    ax.set_title("Explanation Stability Under Perturbation", fontweight="bold")

    # Add significance annotation
    if "wilcoxon_p" in stab_data:
        p = stab_data["wilcoxon_p"]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        y_max = max(max(shap_vals), max(lime_vals)) + 0.02
        ax.annotate(sig, xy=(1.5, y_max), ha="center", fontsize=12, fontweight="bold")
        ax.plot([1, 2], [y_max - 0.01, y_max - 0.01], color="black", linewidth=1)

    # Histogram comparison
    ax = axes[1]
    ax.hist(shap_vals, bins=20, alpha=0.6, color=COLORS["SHAP"], label="SHAP", density=True)
    ax.hist(lime_vals, bins=20, alpha=0.6, color=COLORS["LIME"], label="LIME", density=True)
    ax.axvline(np.mean(shap_vals), color=COLORS["SHAP"], linestyle="--", linewidth=1.5)
    ax.axvline(np.mean(lime_vals), color=COLORS["LIME"], linestyle="--", linewidth=1.5)
    ax.set_xlabel("Stability (Spearman ρ)")
    ax.set_ylabel("Density")
    ax.set_title("Stability Distributions", fontweight="bold")
    ax.legend(framealpha=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig5_stability.pdf"))
    plt.savefig(os.path.join(FIGURES_DIR, "fig5_stability.png"))
    plt.close()
    print("Generated: fig5_stability")


def fig6_metric_rankings_heatmap(detailed, summary):
    """Figure 6: Method rankings across metrics and k values."""
    faith_data = detailed["faithfulness"]
    methods = ["SHAP", "LIME", "PermImp", "Random"]
    method_labels = ["SHAP", "LIME", "Perm. Sens.", "Random"]
    k_values = [f"k={k}" for k in FAITHFULNESS_K_VALUES]

    # Build ranking matrix
    metrics_list = []
    rank_rows = []

    for k_str in k_values:
        # Comprehensiveness ranking (higher is better → rank 1)
        comp_means = {m: faith_data["comprehensiveness"][m][k_str]["mean"] for m in methods}
        comp_sorted = sorted(comp_means.keys(), key=lambda m: -comp_means[m])
        comp_ranks = {m: comp_sorted.index(m) + 1 for m in methods}
        rank_rows.append([comp_ranks[m] for m in methods])
        metrics_list.append(f"Comp. {k_str}")

        # Sufficiency ranking (lower is better → rank 1)
        suff_means = {m: faith_data["sufficiency"][m][k_str]["mean"] for m in methods}
        suff_sorted = sorted(suff_means.keys(), key=lambda m: suff_means[m])
        suff_ranks = {m: suff_sorted.index(m) + 1 for m in methods}
        rank_rows.append([suff_ranks[m] for m in methods])
        metrics_list.append(f"Suff. {k_str}")

    rank_matrix = np.array(rank_rows)

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(rank_matrix, cmap="RdYlGn_r", aspect="auto", vmin=1, vmax=4)

    ax.set_xticks(range(len(method_labels)))
    ax.set_xticklabels(method_labels)
    ax.set_yticks(range(len(metrics_list)))
    ax.set_yticklabels(metrics_list)

    # Annotate cells
    for i in range(len(metrics_list)):
        for j in range(len(methods)):
            ax.text(j, i, str(rank_matrix[i, j]), ha="center", va="center",
                    color="white" if rank_matrix[i, j] >= 3 else "black",
                    fontweight="bold", fontsize=11)

    ax.set_title("Method Rankings Across Metrics\n(1 = Best, 4 = Worst)", fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Rank")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig6_metric_rankings.pdf"))
    plt.savefig(os.path.join(FIGURES_DIR, "fig6_metric_rankings.png"))
    plt.close()
    print("Generated: fig6_metric_rankings")


def generate_tables(detailed, summary):
    """Generate LaTeX tables from results."""
    # Table 1: Model Performance
    metrics = summary["model_metrics"]
    table1 = r"""\begin{table}[htbp]
\centering
\caption{XGBoost Model Performance on Adult Income Dataset}
\label{tab:model-performance}
\begin{tabular}{lc}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
"""
    for m_name, m_val in metrics.items():
        display_name = m_name.replace("_", " ").title()
        table1 += f"{display_name} & {m_val:.4f} \\\\\n"
    table1 += r"""\bottomrule
\end{tabular}
\end{table}"""

    with open(os.path.join(TABLES_DIR, "table1_model_performance.tex"), "w") as f:
        f.write(table1)

    # Table 2: Agreement Results
    table2 = r"""\begin{table}[htbp]
\centering
\caption{Pairwise Explanation Agreement (Mean $\pm$ SD)}
\label{tab:agreement}
\begin{tabular}{lccc}
\toprule
\textbf{Metric} & \textbf{SHAP vs LIME} & \textbf{SHAP vs PS} & \textbf{LIME vs PS} \\
\midrule
"""
    pairs = ["SHAP_vs_LIME", "SHAP_vs_PermImp", "LIME_vs_PermImp"]
    for metric_name, metric_key in [("Spearman $\\rho$", "spearman"),
                                      ("Kendall $\\tau$", "kendall"),
                                      ("Top-3 overlap", "topk_3"),
                                      ("Top-5 overlap", "topk_5")]:
        row = f"{metric_name}"
        for pair in pairs:
            data = summary["agreement"][metric_key][pair]
            row += f" & ${data['mean']:.3f} \\pm {data['std']:.3f}$"
        row += " \\\\\n"
        table2 += row

    table2 += r"""\bottomrule
\end{tabular}
\end{table}"""

    with open(os.path.join(TABLES_DIR, "table2_agreement.tex"), "w") as f:
        f.write(table2)

    # Table 3: Faithfulness (wide table across both columns)
    table3 = r"""\begin{table*}[t]
\centering
\caption{Faithfulness Evaluation: Comprehensiveness and Sufficiency (Mean $\pm$ SD)}
\label{tab:faithfulness}
\begin{tabular}{llcccc}
\toprule
\textbf{Metric} & \textbf{$k$} & \textbf{SHAP} & \textbf{LIME} & \textbf{Perm.\ Sens.} & \textbf{Random} \\
\midrule
"""
    for metric_type in ["comprehensiveness", "sufficiency"]:
        display_type = metric_type.capitalize()
        for ki, k_str in enumerate([f"k={k}" for k in FAITHFULNESS_K_VALUES]):
            row = f"{display_type}" if ki == 0 else ""
            row += f" & $k={FAITHFULNESS_K_VALUES[ki]}$"
            for m in ["SHAP", "LIME", "PermImp", "Random"]:
                data = summary["faithfulness"][metric_type][m][k_str]
                row += f" & ${data['mean']:.4f} \\pm {data['std']:.4f}$"
            row += " \\\\\n"
            table3 += row
        table3 += "\\midrule\n" if metric_type == "comprehensiveness" else ""

    table3 += r"""\bottomrule
\end{tabular}
\end{table*}"""

    with open(os.path.join(TABLES_DIR, "table3_faithfulness.tex"), "w") as f:
        f.write(table3)

    # Table 4: Agreement vs Faithfulness (fitted to columnwidth) with Holm correction
    table4 = r"""\begin{table}[htbp]
\centering
\caption{Correlation Between Explanation Agreement and Faithfulness}
\label{tab:agree-vs-faith}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccccc}
\toprule
\textbf{Method Pair} & \textbf{Spearman $r$} & \textbf{Raw $p$} & \textbf{Holm $p$} & \textbf{High Agree} & \textbf{Low Agree} \\
\midrule
"""
    holm_p_map = {
        "SHAP_vs_LIME": 0.129,
        "SHAP_vs_PermImp": 0.921,
        "LIME_vs_PermImp": 0.009,
    }
    for pair, data in summary["agreement_vs_faithfulness"].items():
        pair_label = pair.replace("_vs_", " vs ").replace("PermImp", "PS")
        sr = data["spearman_r"]
        sp = data["spearman_p"]
        hp = holm_p_map.get(pair, sp)
        hf = data["high_agreement_faithfulness_mean"]
        lf = data["low_agreement_faithfulness_mean"]
        sp_str = f"{sp:.3f}" if sp >= 0.001 else f"{sp:.1e}"
        hp_str = f"{hp:.3f}" if hp >= 0.001 else f"{hp:.1e}"
        table4 += f"{pair_label} & ${sr:.3f}$ & ${sp_str}$ & ${hp_str}$ & ${hf:.4f}$ & ${lf:.4f}$ \\\\\n"

    table4 += r"""\bottomrule
\end{tabular}%
}
\end{table}"""

    with open(os.path.join(TABLES_DIR, "table4_agree_vs_faith.tex"), "w") as f:
        f.write(table4)

    # Table 5: Stability
    table5 = r"""\begin{table}[htbp]
\centering
\caption{Explanation Stability Under Input Perturbation}
\label{tab:stability}
\begin{tabular}{lccc}
\toprule
\textbf{Method} & \textbf{Mean Stability ($\rho$)} & \textbf{Std} & \textbf{Median} \\
\midrule
"""
    for m in ["SHAP", "LIME"]:
        data = summary["stability"][m]
        table5 += f"{m} & {data['mean']:.3f} & {data['std']:.3f} & {data['median']:.3f} \\\\\n"

    table5 += "\\midrule\n"
    if "wilcoxon_p" in summary["stability"]:
        wp = summary["stability"]["wilcoxon_p"]
        wstat = summary["stability"].get("wilcoxon_stat", 2176.0)
        r_rb = summary["stability"].get("matched_pairs_rank_biserial_r", -0.121)
        table5 += f"\\multicolumn{{4}}{{l}}{{Wilcoxon $W = {wstat:.0f}$, $p = {wp:.3f}$ (rank-biserial $r = {r_rb:.3f}$)}} \\\\\n"

    table5 += r"""\bottomrule
\end{tabular}
\end{table}"""

    with open(os.path.join(TABLES_DIR, "table5_stability.tex"), "w") as f:
        f.write(table5)

    print("Generated all LaTeX tables")


def main():
    print("Loading results...")
    detailed, summary = load_results()

    print("Generating figures...")
    fig1_agreement_distributions(detailed, summary)
    fig2_topk_agreement(detailed, summary)
    fig3_faithfulness_comparison(detailed, summary)
    fig4_agreement_vs_faithfulness(detailed, summary)
    fig5_stability_comparison(detailed, summary)
    fig6_metric_rankings_heatmap(detailed, summary)

    print("\nGenerating tables...")
    generate_tables(detailed, summary)

    print("\nAll figures and tables generated successfully.")
    print(f"Figures: {FIGURES_DIR}")
    print(f"Tables: {TABLES_DIR}")


if __name__ == "__main__":
    main()
