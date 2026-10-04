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
    """Plots a 2D histogram of predictions vs targets with marginal distributions and reports SCC/PCC.

    Args:
        preds (np.ndarray): 1D array of model predictions.
        y_val (np.ndarray): 1D array of target values (same length as `preds`).
        bins (int): Number of bins along each axis for the 2D histogram. Defaults to 46.
        path (None | Path | str): Path to save the figure. If `None`, the plot is only displayed.
        xlabel (str): Label for the x-axis (target values). Defaults to 'Target'.
        ylabel (str): Label for the y-axis (predictions). Defaults to 'Predictions'.
        margin_label (str): Label for the marginal distribution axes. Defaults to 'Probability'.
    """
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
    """Visualizes target and predicted contact profiles as paired bar plots over a genomic interval.

    Args:
        selected_preds (pd.Series | np.ndarray): Predicted contact counts for the selected region.
        selected_contacts (pd.Series | np.ndarray): Observed (target) contact counts for the same region.
        start (int): Genomic start coordinate of the interval (in bp).
        chrom (str): Chromosome name (used in the x-axis label).
        bin_size (int): Size of each genomic bin in bp. Defaults to 1000.
        window_size (int): Total length of the plotted interval in bp. Defaults to 256000.
        path (None | str | Path): Path to save the figure. If `None`, the plot is only displayed.
        ylabel_top (str): Y-axis label for the top panel (target). Defaults to 'Target'.
        ylabel_bottom (str): Y-axis label for the bottom panel (predictions). Defaults to 'Predictions'.
        xlabel (str): X-axis label template with two placeholders: chromosome name and bin size in Kb.
    """
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
