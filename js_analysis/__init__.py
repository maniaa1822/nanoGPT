"""
JS divergence analysis package for neural CSSR models.

This package provides modular tools for analyzing Jensen-Shannon divergence
distributions vs history length using neural models as probability providers.
"""

__version__ = "1.0.0"

from .state_mapping import (
    get_seven_state_gt_state,
    get_golden_mean_gt_state,
    get_even_process_gt_state,
    get_gt_state,
    get_all_state_suffixes,
    collect_histories_by_state,
    find_histories_for_state,
)

from .js_metrics import (
    extract_subsequences,
    js_divergence,
    get_next_token_distribution,
    get_kstep_distribution,
    js_divergence_k,
    js_divergence_conditional_k,
)

from .analysis_engines import (
    analyze_js_distribution,
    analyze_js_kstep,
    analyze_js_conditional_kstep,
    analyze_instate_js_distribution,
    analyze_cross_state_js_distribution,
    analyze_cross_state_js_kstep,
    analyze_cross_state_js_conditional_kstep,
)

from .plotting import (
    plot_analysis,
    plot_instate_analysis,
    plot_cross_state_heatmaps,
    plot_cross_state_histograms,
    plot_threshold_stability,
)

from .calibration import (
    fit_platt_params,
    compute_js_thresholds,
    find_optimal_L_thresholds,
    find_optimal_L_separation,
)

__all__ = [
    # State mapping
    "get_seven_state_gt_state",
    "get_golden_mean_gt_state",
    "get_even_process_gt_state",
    "get_gt_state",
    "get_all_state_suffixes",
    "collect_histories_by_state",
    "find_histories_for_state",
    # JS metrics
    "extract_subsequences",
    "js_divergence",
    "get_next_token_distribution",
    "get_kstep_distribution",
    "js_divergence_k",
    "js_divergence_conditional_k",
    # Analysis engines
    "analyze_js_distribution",
    "analyze_js_kstep",
    "analyze_js_conditional_kstep",
    "analyze_instate_js_distribution",
    "analyze_cross_state_js_distribution",
    "analyze_cross_state_js_kstep",
    "analyze_cross_state_js_conditional_kstep",
    # Plotting
    "plot_analysis",
    "plot_instate_analysis",
    "plot_cross_state_heatmaps",
    "plot_cross_state_histograms",
    "plot_threshold_stability",
    # Calibration
    "fit_platt_params",
    "compute_js_thresholds",
    "find_optimal_L_thresholds",
    "find_optimal_L_separation",
]