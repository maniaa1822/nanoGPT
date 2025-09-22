#!/usr/bin/env python3
"""
Enhanced JS divergence analysis vs history length for neural CSSR models.

This is the refactored main script that orchestrates various types of JS analysis:
- Random pairs analysis: Compute JS divergence between random history pairs
- Instate analysis: Compute JS divergence between histories within the same ground truth state
- Cross-state analysis: Compute JS divergence between histories from different ground truth states
- K-step analysis: All of the above using k-step rollout distributions
- Conditional analysis: Remove mixture effects by conditioning on first symbol

Usage examples:
  # Random pairs analysis only (works with any preset)
  python main.py --preset golden_mean --analysis_type random

  # Instate analysis on seven-state human machine (small model)
  python main.py --preset seven_state_human --analysis_type instate

  # Instate analysis on seven-state human machine (large model)
  python main.py --preset seven_state_human_large --analysis_type instate

  # Instate analysis on golden mean machine
  python main.py --preset golden_mean --analysis_type instate

  # Instate analysis on even process machine
  python main.py --preset even_process --analysis_type instate

  # Conditional cross-state analysis (seven-state human with k=4)
  python main.py --preset seven_state_human --L_max 4 --n_samples 50 --k 4 --analysis_type cross

  # Both analyses (default)
  python main.py --preset golden_mean --analysis_type both

  # Custom model and data
  python main.py --model_ckpt /path/to/model.pt --data /path/to/data.dat --analysis_type both
"""

import numpy as np
import torch
import argparse
import sys
from pathlib import Path
from typing import Optional

# Import all the modular components
from state_mapping import debug_pattern_matching
from calibration import fit_platt_params, compute_js_thresholds, find_optimal_L_thresholds, find_optimal_L_separation
from analysis_engines import (
    analyze_js_distribution,
    analyze_js_kstep,
    analyze_js_conditional_kstep,
    analyze_instate_js_distribution,
    analyze_cross_state_js_distribution,
    analyze_cross_state_js_kstep,
    analyze_cross_state_js_conditional_kstep
)
from plotting import (
    plot_analysis,
    plot_instate_analysis,
    plot_cross_state_heatmaps,
    plot_cross_state_histograms,
    plot_threshold_stability
)


def main():
    """Main orchestration function for JS divergence analysis."""
    parser = argparse.ArgumentParser(description='Analyze JS divergence distribution vs history length using nanoGPT model')
    parser.add_argument('--preset', type=str, choices=['golden_mean', 'seven_state_human', 'seven_state_human_large', 'even_process'], default='golden_mean')
    parser.add_argument('--model_ckpt', type=str, help='Override model checkpoint path')
    parser.add_argument('--data', type=str, help='Override dataset path')
    parser.add_argument('--L_max', type=int, default=4)
    parser.add_argument('--n_samples', type=int, default=4000)
    parser.add_argument('--platt_min_count', type=int, default=5)
    parser.add_argument('--analysis_type', type=str, choices=['random', 'instate', 'cross', 'both', 'all'], default='both',
                       help='Type of analysis: random pairs, instate pairs, cross-state pairs, or combinations')
    parser.add_argument('--k', type=int, default=0,
                       help='k-step rollout length for JS; 0 disables k-step analysis')
    args = parser.parse_args()

    # Resolve paths
    repo_root = Path(__file__).resolve().parents[2]
    default_model = repo_root / 'nanoGPT' / 'out-golden-mean-char' / 'ckpt.pt'
    default_data = repo_root / 'experiments' / 'datasets' / 'golden_mean' / 'golden_mean.dat'
    if args.preset == 'seven_state_human':
        # Seven-state human preset
        default_model = repo_root / 'nanoGPT' / 'out-seven-state-char' / 'ckpt.pt'
        default_data = repo_root / 'experiments' / 'datasets' / 'seven_state_human' / 'seven_state_human.dat'
    elif args.preset == 'seven_state_human_large':
        # Seven-state human large model preset
        default_model = repo_root / 'nanoGPT' / 'out-seven-state-char_large' / 'ckpt.pt'
        default_data = repo_root / 'experiments' / 'datasets' / 'seven_state_human' / 'seven_state_human.dat'
    elif args.preset == 'even_process':
        default_model = repo_root / 'nanoGPT' / 'out-even-process-char' / 'ckpt.pt'
        default_data = repo_root / 'experiments' / 'datasets' / 'even_process' / 'even_process.dat'
    model_ckpt = Path(args.model_ckpt) if args.model_ckpt is not None else default_model
    data_path = Path(args.data) if args.data is not None else default_data

    # Validate preset for instate analysis
    if args.analysis_type in ['instate', 'cross', 'both', 'all'] and args.preset not in ['seven_state_human', 'seven_state_human_large', 'golden_mean', 'even_process']:
        print("Warning: Instate and cross-state analyses are only available for seven_state_human, seven_state_human_large, golden_mean, and even_process presets")
        print(f"Your preset: {args.preset}")
        print("Switching to random analysis only")
        args.analysis_type = 'random'

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from transcssr_neural_runner import _load_nano_gpt_model, load_binary_string  # type: ignore

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, block_size = _load_nano_gpt_model(model_ckpt, device)

    # Load data and convert to numpy array of ints (0/1)
    s = load_binary_string(data_path)
    data = np.array([int(c) for c in s], dtype=np.int64)

    # Debug: print some info about the data
    print(f"Loaded data from: {data_path}")
    print(f"Data length: {len(data)}")
    print(f"Data type: {data.dtype}")
    print(f"Unique values: {np.unique(data)}")
    print(f"Sample data (first 50 chars): {s[:50]}")
    print(f"Sample data (last 50 chars): {s[-50:]}")
    print()

    # Debug: check for patterns in the data
    debug_pattern_matching(data, L=8)

    # Fit Platt calibration on-the-fly using histories up to L=args.L_max
    platt = fit_platt_params(data, model, L_max=args.L_max, min_count=args.platt_min_count)
    if platt is not None:
        print(f"Fitted Platt params: {platt}")

    # Save figures near the model checkpoint directory
    save_dir = model_ckpt.parent

    # Perform analyses based on type
    if args.analysis_type in ['random', 'both', 'all']:
        print("\n" + "="*60)
        print("RANDOM PAIRS ANALYSIS")
        print("="*60)

        # 1) Non-calibrated (raw) analysis
        results_raw = analyze_js_distribution(data, model, L_max=args.L_max, n_samples=args.n_samples, alphabet_size=2, platt_params=None)
        plot_analysis(results_raw, save_dir=save_dir, prefix=f"{args.preset}_L{args.L_max}_raw")
        print("Raw (non-calibrated) summary:")
        for L in sorted(results_raw.keys()):
            means = results_raw[L]['means']; stds = results_raw[L]['stds']
            denom = (stds[0] + stds[1])
            sep = (abs(means[1] - means[0]) / denom) if denom > 1e-12 else 0.0
            print(f"  L={L}: threshold={results_raw[L]['threshold']:.6f}, separation={sep:.3f}")

        # 2) Calibrated analysis (if Platt available)
        if platt is not None:
            results_cal = analyze_js_distribution(data, model, L_max=args.L_max, n_samples=args.n_samples, alphabet_size=2, platt_params=platt)
            plot_analysis(results_cal, save_dir=save_dir, prefix=f"{args.preset}_L{args.L_max}_calibrated")
            print("Calibrated summary:")
            for L in sorted(results_cal.keys()):
                means = results_cal[L]['means']; stds = results_cal[L]['stds']
                denom = (stds[0] + stds[1])
                sep = (abs(means[1] - means[0]) / denom) if denom > 1e-12 else 0.0
                print(f"  L={L}: threshold={results_cal[L]['threshold']:.6f}, separation={sep:.3f}")
        else:
            print("Calibration unavailable (insufficient data) — skipping calibrated plots.")

    if args.analysis_type in ['instate', 'both', 'all']:
        print("\n" + "="*60)
        print("INSTATE ANALYSIS (within same GT states)")
        print("="*60)

        # Instate analysis (raw)
        n_states = 7 if args.preset in ['seven_state_human', 'seven_state_human_large'] else 2
        results_instate_raw = analyze_instate_js_distribution(data, model, L_max=args.L_max, n_samples_per_state=args.n_samples//n_states, alphabet_size=2, platt_params=None, preset=args.preset)
        plot_instate_analysis(results_instate_raw, save_dir=save_dir, prefix=f"{args.preset}_L{args.L_max}_instate_raw")
        print("Instate raw summary:")
        for L in sorted(results_instate_raw.keys()):
            print(f"  L={L}:")
            for state in sorted(results_instate_raw[L].keys()):
                means = results_instate_raw[L][state]['means']
                stds = results_instate_raw[L][state]['stds']
                denom = (stds[0] + stds[1])
                sep = (abs(means[1] - means[0]) / denom) if denom > 1e-12 else 0.0
                n_hist = results_instate_raw[L][state]['n_histories']
                threshold = results_instate_raw[L][state]['threshold']
                print(f"    {state}: threshold={threshold:.6f}, separation={sep:.3f}, n_histories={n_hist}")

        # Instate analysis (calibrated) if Platt available
        if platt is not None:
            n_states = 7 if args.preset in ['seven_state_human', 'seven_state_human_large'] else 2
            results_instate_cal = analyze_instate_js_distribution(data, model, L_max=args.L_max, n_samples_per_state=args.n_samples//n_states, alphabet_size=2, platt_params=platt, preset=args.preset)
            plot_instate_analysis(results_instate_cal, save_dir=save_dir, prefix=f"{args.preset}_L{args.L_max}_instate_calibrated")
            print("Instate calibrated summary:")
            for L in sorted(results_instate_cal.keys()):
                print(f"  L={L}:")
                for state in sorted(results_instate_cal[L].keys()):
                    means = results_instate_cal[L][state]['means']
                    stds = results_instate_cal[L][state]['stds']
                    denom = (stds[0] + stds[1])
                    sep = (abs(means[1] - means[0]) / denom) if denom > 1e-12 else 0.0
                    n_hist = results_instate_cal[L][state]['n_histories']
                    threshold = results_instate_cal[L][state]['threshold']
                    print(f"    {state}: threshold={threshold:.6f}, separation={sep:.3f}, n_histories={n_hist}")
        else:
            print("Calibration unavailable — skipping calibrated instate plots.")

    if args.analysis_type in ['cross', 'all']:
        print("\n" + "="*60)
        print("CROSS-STATE ANALYSIS")
        print("="*60)

        results_cross_raw = analyze_cross_state_js_distribution(
            data, model, L_max=args.L_max, n_samples_per_pair=args.n_samples,
            platt_params=None, preset=args.preset)
        plot_cross_state_heatmaps(results_cross_raw, metric='mean', save_dir=save_dir,
                                  prefix=f"{args.preset}_cross_raw")
        plot_cross_state_histograms(results_cross_raw, save_dir=save_dir,
                                    prefix=f"{args.preset}_cross_raw")
        print("Cross-state raw summary:")
        for L in sorted(results_cross_raw.keys()):
            print(f"  L={L}:")
            states = results_cross_raw[L]['states']
            for i, si in enumerate(states):
                for j in range(i + 1, len(states)):
                    sj = states[j]
                    stats = results_cross_raw[L]['pair_stats'].get((si, sj))
                    if not stats or stats['n_pairs'] == 0:
                        continue
                    print(f"    {si} vs {sj}: mean={stats['mean']:.6f}, median={stats['median']:.6f}, n_pairs={stats['n_pairs']}")

        if platt is not None:
            results_cross_cal = analyze_cross_state_js_distribution(
                data, model, L_max=args.L_max, n_samples_per_pair=args.n_samples,
                platt_params=platt, preset=args.preset)
            plot_cross_state_heatmaps(results_cross_cal, metric='mean', save_dir=save_dir,
                                      prefix=f"{args.preset}_cross_calibrated")
            plot_cross_state_histograms(results_cross_cal, save_dir=save_dir,
                                        prefix=f"{args.preset}_cross_calibrated")
            print("Cross-state calibrated summary:")
            for L in sorted(results_cross_cal.keys()):
                print(f"  L={L}:")
                states = results_cross_cal[L]['states']
                for i, si in enumerate(states):
                    for j in range(i + 1, len(states)):
                        sj = states[j]
                        stats = results_cross_cal[L]['pair_stats'].get((si, sj))
                        if not stats or stats['n_pairs'] == 0:
                            continue
                        print(f"    {si} vs {sj}: mean={stats['mean']:.6f}, median={stats['median']:.6f}, n_pairs={stats['n_pairs']}")
        else:
            print("Calibration unavailable — skipping calibrated cross-state plots.")

    # K-step cross-state analysis (if k > 0 and cross-state analysis is enabled)
    if int(args.k) > 0 and args.analysis_type in ['cross', 'all']:
        print("\n" + "="*60)
        print(f"K-STEP CROSS-STATE ANALYSIS (k={args.k})")
        print("="*60)

        results_cross_k_raw = analyze_cross_state_js_kstep(
            data, model, L_max=args.L_max, k=int(args.k), n_samples_per_pair=args.n_samples,
            platt_params=None, preset=args.preset)
        plot_cross_state_heatmaps(results_cross_k_raw, metric='mean', save_dir=save_dir,
                                  prefix=f"{args.preset}_cross_k{int(args.k)}_raw")
        plot_cross_state_histograms(results_cross_k_raw, save_dir=save_dir,
                                    prefix=f"{args.preset}_cross_k{int(args.k)}_raw")
        print("K-step cross-state raw summary:")
        for L in sorted(results_cross_k_raw.keys()):
            print(f"  L={L}:")
            states = results_cross_k_raw[L]['states']
            for i, si in enumerate(states):
                for j in range(i + 1, len(states)):
                    sj = states[j]
                    stats = results_cross_k_raw[L]['pair_stats'].get((si, sj))
                    if not stats or stats['n_pairs'] == 0:
                        continue
                    print(f"    {si} vs {sj}: mean={stats['mean']:.6f}, median={stats['median']:.6f}, n_pairs={stats['n_pairs']}")

        if platt is not None:
            results_cross_k_cal = analyze_cross_state_js_kstep(
                data, model, L_max=args.L_max, k=int(args.k), n_samples_per_pair=args.n_samples,
                platt_params=platt, preset=args.preset)
            plot_cross_state_heatmaps(results_cross_k_cal, metric='mean', save_dir=save_dir,
                                      prefix=f"{args.preset}_cross_k{int(args.k)}_calibrated")
            plot_cross_state_histograms(results_cross_k_cal, save_dir=save_dir,
                                        prefix=f"{args.preset}_cross_k{int(args.k)}_calibrated")
            print("K-step cross-state calibrated summary:")
            for L in sorted(results_cross_k_cal.keys()):
                print(f"  L={L}:")
                states = results_cross_k_cal[L]['states']
                for i, si in enumerate(states):
                    for j in range(i + 1, len(states)):
                        sj = states[j]
                        stats = results_cross_k_cal[L]['pair_stats'].get((si, sj))
                        if not stats or stats['n_pairs'] == 0:
                            continue
                        print(f"    {si} vs {sj}: mean={stats['mean']:.6f}, median={stats['median']:.6f}, n_pairs={stats['n_pairs']}")
        else:
            print("Calibration unavailable — skipping calibrated k-step cross-state plots.")

    # Conditional JS analyses (removes mixture effects, focuses on successor kernels)
    if int(args.k) >= 2:
        print("\n" + "="*60)
        print(f"CONDITIONAL JS ANALYSIS (k={args.k})")
        print("="*60)
        print("Conditional JS removes mixture effects by comparing futures after conditioning")
        print("on the first symbol, better capturing successor kernel differences.")

        # Conditional random pairs analysis
        results_cond_raw = analyze_js_conditional_kstep(data, model, L_max=args.L_max, k=int(args.k), n_pairs=args.n_samples, platt_params=None)
        plot_analysis(results_cond_raw, save_dir=save_dir, prefix=f"{args.preset}_L{args.L_max}_cond_k{int(args.k)}_raw")
        print("Conditional k-step Raw (non-calibrated) summary:")
        for L in sorted(results_cond_raw.keys()):
            means = results_cond_raw[L]['means']; stds = results_cond_raw[L]['stds']
            denom = (stds[0] + stds[1])
            sep = (abs(means[1] - means[0]) / denom) if denom > 1e-12 else 0.0
            print(f"  L={L}: threshold={results_cond_raw[L]['threshold']:.6f}, separation={sep:.3f}")

        if platt is not None:
            results_cond_cal = analyze_js_conditional_kstep(data, model, L_max=args.L_max, k=int(args.k), n_pairs=args.n_samples, platt_params=platt)
            plot_analysis(results_cond_cal, save_dir=save_dir, prefix=f"{args.preset}_L{args.L_max}_cond_k{int(args.k)}_calibrated")
            print("Conditional k-step Calibrated summary:")
            for L in sorted(results_cond_cal.keys()):
                means = results_cond_cal[L]['means']; stds = results_cond_cal[L]['stds']
                denom = (stds[0] + stds[1])
                sep = (abs(means[1] - means[0]) / denom) if denom > 1e-12 else 0.0
                print(f"  L={L}: threshold={results_cond_cal[L]['threshold']:.6f}, separation={sep:.3f}")

        # Conditional cross-state analysis (if cross-state analysis is enabled)
        if args.analysis_type in ['cross', 'all']:
            print("\n" + "-"*40)
            print(f"CONDITIONAL CROSS-STATE ANALYSIS (k={args.k})")
            print("-"*40)

            results_cond_cross_raw = analyze_cross_state_js_conditional_kstep(
                data, model, L_max=args.L_max, k=int(args.k), n_samples_per_pair=args.n_samples,
                platt_params=None, preset=args.preset)
            plot_cross_state_heatmaps(results_cond_cross_raw, metric='mean', save_dir=save_dir,
                                      prefix=f"{args.preset}_cond_cross_k{int(args.k)}_raw")
            plot_cross_state_histograms(results_cond_cross_raw, save_dir=save_dir,
                                        prefix=f"{args.preset}_cond_cross_k{int(args.k)}_raw")
            print("Conditional cross-state raw summary:")
            for L in sorted(results_cond_cross_raw.keys()):
                print(f"  L={L}:")
                states = results_cond_cross_raw[L]['states']
                for i, si in enumerate(states):
                    for j in range(i + 1, len(states)):
                        sj = states[j]
                        stats = results_cond_cross_raw[L]['pair_stats'].get((si, sj))
                        if not stats or stats['n_pairs'] == 0:
                            continue
                        print(f"    {si} vs {sj}: mean={stats['mean']:.6f}, median={stats['median']:.6f}, n_pairs={stats['n_pairs']}")

            if platt is not None:
                results_cond_cross_cal = analyze_cross_state_js_conditional_kstep(
                    data, model, L_max=args.L_max, k=int(args.k), n_samples_per_pair=args.n_samples,
                    platt_params=platt, preset=args.preset)
                plot_cross_state_heatmaps(results_cond_cross_cal, metric='mean', save_dir=save_dir,
                                          prefix=f"{args.preset}_cond_cross_k{int(args.k)}_calibrated")
                plot_cross_state_histograms(results_cond_cross_cal, save_dir=save_dir,
                                            prefix=f"{args.preset}_cond_cross_k{int(args.k)}_calibrated")
                print("Conditional cross-state calibrated summary:")
                for L in sorted(results_cond_cross_cal.keys()):
                    print(f"  L={L}:")
                    states = results_cond_cross_cal[L]['states']
                    for i, si in enumerate(states):
                        for j in range(i + 1, len(states)):
                            sj = states[j]
                            stats = results_cond_cross_cal[L]['pair_stats'].get((si, sj))
                            if not stats or stats['n_pairs'] == 0:
                                continue
                            print(f"    {si} vs {sj}: mean={stats['mean']:.6f}, median={stats['median']:.6f}, n_pairs={stats['n_pairs']}")
            else:
                print("Calibration unavailable — skipping calibrated conditional cross-state plots.")

    # Optional k-step random-pairs analysis (model chaining exact 2^k paths)
    if int(args.k) > 0:
        print("\n" + "="*60)
        print(f"K-STEP ANALYSIS (k={args.k})")
        print("="*60)

        # k-step raw (non-calibrated)
        results_k_raw = analyze_js_kstep(data, model, L_max=args.L_max, k=int(args.k), n_pairs=args.n_samples, platt_params=None)
        plot_analysis(results_k_raw, save_dir=save_dir, prefix=f"{args.preset}_L{args.L_max}_k{int(args.k)}_raw")
        print("k-step Raw (non-calibrated) summary:")
        for L in sorted(results_k_raw.keys()):
            means = results_k_raw[L]['means']; stds = results_k_raw[L]['stds']
            denom = (stds[0] + stds[1])
            sep = (abs(means[1] - means[0]) / denom) if denom > 1e-12 else 0.0
            print(f"  L={L}: threshold={results_k_raw[L]['threshold']:.6f}, separation={sep:.3f}")

        # k-step calibrated (if Platt available)
        if platt is not None:
            results_k_cal = analyze_js_kstep(data, model, L_max=args.L_max, k=int(args.k), n_pairs=args.n_samples, platt_params=platt)
            plot_analysis(results_k_cal, save_dir=save_dir, prefix=f"{args.preset}_L{args.L_max}_k{int(args.k)}_calibrated")
            print("k-step Calibrated summary:")
            for L in sorted(results_k_cal.keys()):
                means = results_k_cal[L]['means']; stds = results_k_cal[L]['stds']
                denom = (stds[0] + stds[1])
                sep = (abs(means[1] - means[0]) / denom) if denom > 1e-12 else 0.0
                print(f"  L={L}: threshold={results_k_cal[L]['threshold']:.6f}, separation={sep:.3f}")
        else:
            print("Calibration unavailable — skipping calibrated k-step plots.")

    # Optimal L estimation from random pairs analysis
    print("\n" + "="*60)
    print("OPTIMAL L ESTIMATION")
    print("="*60)

    # Compute thresholds for optimal L detection
    thresholds_raw, separations_raw = compute_js_thresholds(data, model, L_max=args.L_max, n_samples=args.n_samples, platt_params=None)
    thresholds_cal, separations_cal = compute_js_thresholds(data, model, L_max=args.L_max, n_samples=args.n_samples, platt_params=platt) if platt is not None else ({}, {})

    # Find optimal L
    optimal_L_raw = find_optimal_L_thresholds(thresholds_raw)
    optimal_L_cal = find_optimal_L_thresholds(thresholds_cal) if thresholds_cal else None

    print("Optimal L estimation:")
    print(f"  Raw analysis: L={optimal_L_raw}")
    if optimal_L_cal:
        print(f"  Calibrated analysis: L={optimal_L_cal}")
    print()

    # Plot stability analysis
    plot_threshold_stability(thresholds_raw, separations_raw, save_dir=save_dir, prefix=f"{args.preset}_L{args.L_max}_optimal_L_raw")
    if thresholds_cal:
        plot_threshold_stability(thresholds_cal, separations_cal, save_dir=save_dir, prefix=f"{args.preset}_L{args.L_max}_optimal_L_cal")


if __name__ == "__main__":
    main()