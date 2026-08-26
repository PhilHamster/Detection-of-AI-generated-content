"""
visualize_results.py
=====================
Generate comparison charts for Phase 1 experimental results.

Usage (from project root):
    python "CHEAT_TEST/visualize_results.py"
"""

import os
import matplotlib
matplotlib.use("Agg")  # 无显示器环境下也能保存图片
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "CHEAT_TEST/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 实验结果数据 ──────────────────────────────────────────────
METHODS = ["TF-IDF + LR", "TF-IDF + SVM", "RoBERTa-base", "Fast-DetectGPT\n(gpt2-xl)"]
COLORS  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

# 按 source 拆分的 AUROC
RESULTS = {
    "human vs generation": [0.9978, 0.9938, 0.9992, 0.9890],
    "human vs polish":     [0.9540, 0.9579, 0.9874, 0.7408],
    "human vs fusion":     [0.7964, 0.8051, 0.8666, 0.5782],
}
OVERALL_AUROC = [0.9167, 0.9196, 0.9515, 0.7704]


# ── 图1：四方法 × 三场景 分组柱状图 ─────────────────────────
def plot_grouped_bar():
    fig, ax = plt.subplots(figsize=(11, 6))

    scenarios = list(RESULTS.keys())
    n_methods  = len(METHODS)
    n_scenarios = len(scenarios)
    x = np.arange(n_scenarios)
    bar_width = 0.18
    offsets = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * bar_width

    for i, (method, color) in enumerate(zip(METHODS, COLORS)):
        values = [RESULTS[s][i] for s in scenarios]
        bars = ax.bar(x + offsets[i], values, bar_width,
                      label=method, color=color, edgecolor="white", linewidth=0.8)
        # 在柱子顶部标注数值
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=7.5, rotation=90)

    ax.set_xlabel("Evasion Scenario", fontsize=12)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title("AUROC by Detection Method and Evasion Scenario\n(CHEAT Dataset, n_per_class=4514)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Human vs Generation\n(No Evasion)",
                         "Human vs Polish\n(AI Rewriting)",
                         "Human vs Fusion\n(Human-AI Hybrid)"],
                        fontsize=11)
    ax.set_ylim(0.50, 1.06)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, label="Random Baseline (0.5)")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig1_grouped_bar_auroc.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {path}")


# ── 图2：检测难度梯度折线图（每个方法一条线）─────────────────
def plot_gradient_lines():
    fig, ax = plt.subplots(figsize=(9, 5.5))

    scenario_labels = ["Generation\n(No Evasion)", "Polish\n(AI Rewriting)", "Fusion\n(Human-AI Hybrid)"]
    x = np.arange(len(scenario_labels))
    markers = ["o", "s", "D", "^"]

    for i, (method, color, marker) in enumerate(zip(METHODS, COLORS, markers)):
        values = [
            RESULTS["human vs generation"][i],
            RESULTS["human vs polish"][i],
            RESULTS["human vs fusion"][i],
        ]
        ax.plot(x, values, marker=marker, color=color, linewidth=2.2,
                markersize=8, label=method.replace("\n", " "))
        # 标注每个点的数值
        for xi, val in zip(x, values):
            ax.annotate(f"{val:.3f}",
                        xy=(xi, val),
                        xytext=(0, 10),
                        textcoords="offset points",
                        ha="center", fontsize=8.5, color=color)

    ax.set_xlabel("Evasion Scenario (Increasing Difficulty →)", fontsize=12)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title("Detection Difficulty Gradient Across Evasion Scenarios\n(CHEAT Dataset, n_per_class=4514)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, fontsize=11)
    ax.set_ylim(0.50, 1.05)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, label="Random Baseline (0.5)")
    ax.legend(loc="lower left", fontsize=9.5, framealpha=0.9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig2_gradient_lines.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {path}")


# ── 图3：整体AUROC横向对比条形图 ─────────────────────────────
def plot_overall_bar():
    fig, ax = plt.subplots(figsize=(8, 4.5))

    y = np.arange(len(METHODS))
    bars = ax.barh(y, OVERALL_AUROC, color=COLORS, edgecolor="white", height=0.5)

    for bar, val in zip(bars, OVERALL_AUROC):
        ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=10.5, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([m.replace("\n", " ") for m in METHODS], fontsize=11)
    ax.set_xlabel("Overall AUROC", fontsize=12)
    ax.set_title("Overall AUROC Comparison Across Detection Methods\n(CHEAT Dataset, n_per_class=4514)",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0.70, 1.02)
    ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.invert_yaxis()

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig3_overall_auroc_bar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {path}")


if __name__ == "__main__":
    plot_grouped_bar()
    plot_gradient_lines()
    plot_overall_bar()
    print(f"\nAll figures saved to {OUTPUT_DIR}/")