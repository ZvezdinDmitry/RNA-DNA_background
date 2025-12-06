from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns


def plot_correlation(
    preds: np.ndarray,
    y_val: np.ndarray,
    bins: int = 46,
    path: None | Path | str = None,
    xlabel: str = "Target",
    ylabel: str = "Predictions",
    margin_label: str = "Probability",
):
    scc = ss.spearmanr(preds, y_val)
    pcc = ss.pearsonr(preds, y_val)

    # Defining universal bins borders
    xmin, xmax = -1, y_val.max()
    ymin, ymax = -1, y_val.max()

    img = sns.jointplot(
        x=y_val,
        y=preds,
        kind="hist",
        bins=[
            np.linspace(xmin, xmax, bins + 1),
            np.linspace(ymin, ymax, bins + 1),
        ],
        marginal_kws={
            "bins": np.linspace(ymin, ymax, bins + 1),
            "stat": "probability",
        },
        vmax=2000,
        color="salmon",
        marginal_ticks=True,
    )

    plt.xlabel(xlabel, size=18)
    plt.title(f"SCC: {scc[0]:.3f}, PCC: {pcc[0]:.3f}", size=16)
    plt.ylabel(ylabel, size=18)
    plt.xticks(size=14)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.yticks(size=14)
    plt.tight_layout()
    img.ax_marg_x.set_ylabel(margin_label, fontsize=10)
    img.ax_marg_y.set_xlabel(margin_label, fontsize=10)

    if path:
        plt.savefig(
            path,
            dpi=300,
        )


def draw_interval(
    selected_preds: pd.Series | np.ndarray,
    selected_contacts: pd.Series | np.ndarray,
    start: int,
    chrom: str,
    bin_size: int = 1000,
    window_size: int = 256000,
    path: None | str | Path = None,
    ylabel_top: str = "Target",
    ylabel_bottom: str = "Predictions",
    xlabel: str = "Chromosome {}, positions in {} Kb",
):
    scc = ss.spearmanr(selected_preds, selected_contacts)
    pcc = ss.pearsonr(selected_preds, selected_contacts)
    fig, axs = plt.subplots(2, 1, figsize=(15, 5), sharex=True)
    axs[0].bar(
        np.arange(len(selected_contacts)) + start // bin_size,
        selected_contacts,
        color="salmon",
        zorder=2,
    )
    axs[1].bar(
        np.arange(len(selected_contacts)) + start // bin_size,
        selected_preds,
        color="salmon",
        zorder=2,
    )
    vmax = max(np.max(selected_contacts), np.max(selected_preds))
    plt.xticks(size=16)

    for label in axs[0].get_yticklabels():
        label.set_fontsize(16)
    for label in axs[1].get_yticklabels():
        label.set_fontsize(16)
    axs[0].set_ylim(0, vmax)
    axs[1].set_ylim(0, vmax)
    axs[0].grid(alpha=0.6, zorder=1)
    axs[1].grid(alpha=0.6, zorder=1)
    plt.suptitle(f"SCC: {scc[0]:.3f}, PCC: {pcc[0]:.3f}", size=24)
    axs[0].set_ylabel(ylabel_top, size=20)
    axs[1].set_ylabel(ylabel_bottom, size=20)
    axs[0].set_xlim(start // bin_size, (window_size + start) // bin_size)
    axs[1].set_xlim(start // bin_size, (window_size + start) // bin_size)
    plt.xlabel(xlabel.format(chrom, bin_size // 1000), size=24)
    plt.tight_layout()
    if path:
        plt.savefig(
            path,
            dpi=600,
        )
