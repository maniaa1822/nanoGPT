"""
Plotting functions for JS divergence analysis results.

This module provides visualization functions for different types of analysis:
- Random pairs analysis plots
- Instate analysis plots
- Cross-state heatmaps and histograms
- Threshold stability plots
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, Any


def plot_analysis(results: Dict[int, Dict[str, Any]], save_dir: Optional[Path] = None, prefix: Optional[str] = None) -> None:
    """Visualize JS distributions and thresholds.

    If running on a non-interactive backend or save_dir is provided, figures are saved
    to disk instead of shown. Filenames are based on prefix.
    """

    L_values = sorted(results.keys())
    n_plots = len(L_values)

    # Compute a global x-range for consistent comparison
    global_max = 0.0
    for L in L_values:
        if len(results[L]['js_values']) > 0:
            global_max = max(global_max, float(np.max(results[L]['js_values'])))
    if global_max <= 0:
        global_max = 1e-6

    # Plot 1: JS distributions for each L
    ncols = 2 if n_plots > 1 else 1
    nrows = int(np.ceil(n_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    for idx, L in enumerate(L_values):
        ax = axes[idx]
        js_vals = results[L]['js_values']

        # Histogram
        ax.hist(js_vals, bins=50, alpha=0.7, density=True)

        # GMM components
        x = np.linspace(0, global_max, 200)
        gmm = results[L]['gmm']

        for i in range(2):
            mean = results[L]['means'][i]
            std = results[L]['stds'][i]
            weight = results[L]['weights'][i]
            std_safe = max(1e-9, std)
            y = weight * np.exp(-(x - mean)**2 / (2 * std_safe**2)) / (std_safe * np.sqrt(2 * np.pi))
            ax.plot(x, y, label=f'Component {i+1}')

        # Threshold
        ax.axvline(results[L]['threshold'], color='red', linestyle='--', label='Threshold')

        ax.set_title(f'L={L}')
        ax.set_xlabel('JS Divergence')
        ax.set_ylabel('Density')
        ax.legend()
    # Hide any extra axes
    for k in range(n_plots, len(axes)):
        axes[k].axis('off')

    plt.tight_layout()
    noninteractive = 'agg' in plt.get_backend().lower()
    if save_dir is not None or noninteractive:
        save_dir = Path(save_dir) if save_dir is not None else Path('.')
        save_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{prefix+'_' if prefix else ''}js_histograms.png".replace(' ', '')
        out_path = save_dir / fname
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f"Saved {out_path}")
        plt.close(fig)
    else:
        plt.show()

    # Plot 2: Threshold vs L
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    thresholds = [results[L]['threshold'] for L in L_values]
    ax2.plot(L_values, thresholds, 'o-')
    ax2.set_xlabel('History Length L')
    ax2.set_ylabel('JS Threshold')
    ax2.set_title('Threshold vs History Length')
    ax2.grid(True, alpha=0.3)
    if save_dir is not None or noninteractive:
        fname2 = f"{prefix+'_' if prefix else ''}js_thresholds_vs_L.png".replace(' ', '')
        out_path2 = save_dir / fname2
        fig2.savefig(out_path2, dpi=200, bbox_inches='tight')
        print(f"Saved {out_path2}")
        plt.close(fig2)
    else:
        plt.show()


def plot_instate_analysis(results: Dict[int, Dict[str, Any]], save_dir: Optional[Path] = None, prefix: Optional[str] = None) -> None:
    """Visualize JS distributions for instate analysis (histories within same GT state)."""

    L_values = sorted(results.keys())
    if not L_values:
        print("No data to plot for instate analysis")
        return

    # Get all states that appear in the results
    all_states = set()
    for L in L_values:
        all_states.update(results[L].keys())
    all_states = sorted(all_states)

    if not all_states:
        print("No states found in results")
        return

    # Plot 1: Instate JS distributions
    n_states = len(all_states)
    n_L = len(L_values)

    # Create subplots: states x L_values
    fig, axes = plt.subplots(n_states, n_L, figsize=(4 * n_L, 3 * n_states))
    if n_states == 1:
        axes = axes.reshape(1, -1)
    if n_L == 1:
        axes = axes.reshape(-1, 1)

    for i, state in enumerate(all_states):
        for j, L in enumerate(L_values):
            ax = axes[i, j] if n_states > 1 else axes[j]

            if state in results[L] and len(results[L][state]['js_values']) > 0:
                js_vals = results[L][state]['js_values']
                threshold = results[L][state]['threshold']
                n_hist = results[L][state]['n_histories']

                # Histogram
                ax.hist(js_vals, bins=20, alpha=0.7, density=True)
                ax.axvline(threshold, color='red', linestyle='--', label='Threshold')

                ax.set_title(f'State {state}, L={L}\n({n_hist} histories)')
                if i == n_states - 1:  # Bottom row
                    ax.set_xlabel('JS Divergence')
                if j == 0:  # Leftmost column
                    ax.set_ylabel('Density')
            else:
                # No data for this state/L combination
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'State {state}, L={L}')

    plt.tight_layout()
    noninteractive = 'agg' in plt.get_backend().lower()
    if save_dir is not None or noninteractive:
        save_dir = Path(save_dir) if save_dir is not None else Path('.')
        save_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{prefix+'_' if prefix else ''}instate_js_histograms.png"
        out_path = save_dir / fname
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f"Saved {out_path}")
        plt.close(fig)
    else:
        plt.show()

    # Plot 2: Thresholds vs L for each state
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    for state in all_states:
        L_vals = []
        thresholds = []
        for L in L_values:
            if state in results[L] and len(results[L][state]['js_values']) > 0:
                L_vals.append(L)
                thresholds.append(results[L][state]['threshold'])

        if L_vals:
            ax2.plot(L_vals, thresholds, 'o-', label=f'State {state}')

    ax2.set_xlabel('History Length L')
    ax2.set_ylabel('JS Threshold')
    ax2.set_title('Instate JS Thresholds vs History Length')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    if save_dir is not None or noninteractive:
        fname2 = f"{prefix+'_' if prefix else ''}instate_js_thresholds_vs_L.png"
        out_path2 = save_dir / fname2
        fig2.savefig(out_path2, dpi=200, bbox_inches='tight')
        print(f"Saved {out_path2}")
        plt.close(fig2)
    else:
        plt.show()


def plot_cross_state_heatmaps(results: Dict[int, Dict[str, Any]], metric: str = 'mean', save_dir: Optional[Path] = None,
                             prefix: Optional[str] = None) -> None:
    """Plot heatmaps of cross-state JS divergences."""

    L_values = sorted(results.keys())
    if not L_values:
        print("No data to plot for cross-state analysis")
        return

    # Get all states across all L values
    all_states = set()
    for L in L_values:
        all_states.update(results[L]['states'])
    all_states = sorted(all_states)

    if len(all_states) < 2:
        print("Need at least 2 states for cross-state analysis")
        return

    n_L = len(L_values)
    ncols = min(3, n_L)
    nrows = int(np.ceil(n_L / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if n_L == 1:
        axes = [axes]
    elif nrows == 1:
        axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    else:
        axes = axes.flatten()

    for idx, L in enumerate(L_values):
        ax = axes[idx]

        states = results[L]['states']
        pair_stats = results[L]['pair_stats']

        # Create matrix
        n_states = len(states)
        matrix = np.zeros((n_states, n_states))

        for i, state1 in enumerate(states):
            for j, state2 in enumerate(states):
                if i == j:
                    matrix[i, j] = 0.0  # Self-comparison
                elif (state1, state2) in pair_stats:
                    matrix[i, j] = pair_stats[(state1, state2)][metric]
                    matrix[j, i] = pair_stats[(state1, state2)][metric]  # Symmetric
                elif (state2, state1) in pair_stats:
                    matrix[i, j] = pair_stats[(state2, state1)][metric]
                    matrix[j, i] = pair_stats[(state2, state1)][metric]  # Symmetric

        # Plot heatmap
        im = ax.imshow(matrix, cmap='viridis', aspect='auto')
        ax.set_xticks(range(n_states))
        ax.set_yticks(range(n_states))
        ax.set_xticklabels(states)
        ax.set_yticklabels(states)
        ax.set_title(f'L={L} ({metric})')

        # Add colorbar
        plt.colorbar(im, ax=ax)

        # Add text annotations
        for i in range(n_states):
            for j in range(n_states):
                text = ax.text(j, i, f'{matrix[i, j]:.3f}',
                             ha="center", va="center", color="white" if matrix[i, j] > matrix.max()/2 else "black")

    # Hide any extra axes
    for k in range(n_L, len(axes)):
        axes[k].axis('off')

    plt.tight_layout()
    noninteractive = 'agg' in plt.get_backend().lower()
    if save_dir is not None or noninteractive:
        save_dir = Path(save_dir) if save_dir is not None else Path('.')
        save_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{prefix+'_' if prefix else ''}{metric}_cross_state_hist.png"
        out_path = save_dir / fname
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f"Saved {out_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_cross_state_histograms(results: Dict[int, Dict[str, Any]], save_dir: Optional[Path] = None, prefix: Optional[str] = None) -> None:
    """Plot histograms of cross-state JS divergences."""

    L_values = sorted(results.keys())
    if not L_values:
        print("No data to plot")
        return

    # Create one plot per L value
    for L in L_values:
        if 'pair_stats' not in results[L]:
            continue

        pair_stats = results[L]['pair_stats']
        if not pair_stats:
            continue

        # Get all state pairs with data
        pairs_with_data = [(pair, stats) for pair, stats in pair_stats.items() if stats['n_pairs'] > 0]
        if not pairs_with_data:
            continue

        n_pairs = len(pairs_with_data)
        ncols = min(3, n_pairs)
        nrows = int(np.ceil(n_pairs / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows))
        if n_pairs == 1:
            axes = [axes]
        elif nrows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        else:
            axes = axes.flatten()

        for idx, ((state1, state2), stats) in enumerate(pairs_with_data):
            ax = axes[idx]

            js_values = stats['js_values']
            mean_val = stats['mean']
            median_val = stats['median']
            n_pairs_val = stats['n_pairs']

            # Histogram
            ax.hist(js_values, bins=20, alpha=0.7, density=True)
            ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
            ax.axvline(median_val, color='orange', linestyle='--', label=f'Median: {median_val:.3f}')

            ax.set_title(f'{state1} vs {state2}\n({n_pairs_val} pairs)')
            ax.set_xlabel('JS Divergence')
            ax.set_ylabel('Density')
            ax.legend()

        # Hide any extra axes
        for k in range(n_pairs, len(axes)):
            axes[k].axis('off')

        plt.tight_layout()
        noninteractive = 'agg' in plt.get_backend().lower()
        if save_dir is not None or noninteractive:
            save_dir = Path(save_dir) if save_dir is not None else Path('.')
            save_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{prefix+'_' if prefix else ''}L{L}_cross_state_mean.png"
            out_path = save_dir / fname
            fig.savefig(out_path, dpi=200, bbox_inches='tight')
            print(f"Saved {out_path}")
            plt.close(fig)
        else:
            plt.show()


def plot_threshold_stability(thresholds: Dict[int, float], separations: Dict[int, float],
                           save_dir: Optional[Path] = None, prefix: Optional[str] = None) -> None:
    """Plot threshold stability analysis for optimal L detection."""

    L_values = sorted(thresholds.keys())
    if not L_values:
        print("No threshold data to plot")
        return

    threshold_vals = [thresholds[L] for L in L_values]
    separation_vals = [separations.get(L, 0.0) for L in L_values]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot 1: Thresholds vs L
    ax1.plot(L_values, threshold_vals, 'o-', color='blue')
    ax1.set_xlabel('History Length L')
    ax1.set_ylabel('JS Threshold')
    ax1.set_title('JS Threshold vs History Length')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Separation vs L
    ax2.plot(L_values, separation_vals, 'o-', color='green')
    ax2.set_xlabel('History Length L')
    ax2.set_ylabel('Component Separation')
    ax2.set_title('GMM Component Separation vs History Length')
    ax2.grid(True, alpha=0.3)

    # Find and mark optimal L
    if separation_vals:
        max_sep_idx = np.argmax(separation_vals)
        optimal_L = L_values[max_sep_idx]
        ax1.axvline(optimal_L, color='red', linestyle='--', alpha=0.7, label=f'Optimal L={optimal_L}')
        ax2.axvline(optimal_L, color='red', linestyle='--', alpha=0.7, label=f'Optimal L={optimal_L}')
        ax1.legend()
        ax2.legend()

    plt.tight_layout()
    noninteractive = 'agg' in plt.get_backend().lower()
    if save_dir is not None or noninteractive:
        save_dir = Path(save_dir) if save_dir is not None else Path('.')
        save_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{prefix+'_' if prefix else ''}optimal_L_analysis.png"
        out_path = save_dir / fname
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f"Saved {out_path}")
        plt.close(fig)
    else:
        plt.show()