"""
Analysis engines for different types of JS divergence analysis.

This module provides the main analysis functions for:
- Random pairs analysis
- K-step analysis
- Conditional k-step analysis
- Instate analysis (within ground truth states)
- Cross-state analysis (between ground truth states)
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from typing import Dict, Optional, Any

from js_metrics import (
    extract_subsequences,
    js_divergence,
    get_next_token_distribution,
    js_divergence_k,
    js_divergence_conditional_k
)
from state_mapping import collect_histories_by_state, get_all_state_suffixes


def analyze_js_distribution(data: np.ndarray, model, L_max: int = 10,
                           n_samples: int = 5000, alphabet_size: int = 2,
                           platt_params: Optional[dict] = None) -> Dict[int, Dict[str, Any]]:
    """Analyze JS divergence distributions for different history lengths."""

    results = {}

    for L in range(1, L_max + 1):
        print(f"\nAnalyzing L={L}")

        # Extract subsequences
        subsequences = extract_subsequences(data, L)
        if len(subsequences) < n_samples:
            n_samples_L = len(subsequences) // 2
        else:
            n_samples_L = n_samples

        # Sample random pairs
        idx = np.random.choice(len(subsequences), size=(n_samples_L, 2))

        # Compute JS divergences
        js_values = []
        for i, j in idx:
            # Get next-token distributions
            p = get_next_token_distribution(model, subsequences[i], platt_params=platt_params)
            q = get_next_token_distribution(model, subsequences[j], platt_params=platt_params)
            js_values.append(js_divergence(p, q))

        js_values = np.array(js_values)

        # Try GMM fitting with error handling
        try:
            gmm = GaussianMixture(n_components=2, random_state=42)
            gmm.fit(js_values.reshape(-1, 1))

            # Find threshold (intersection of two Gaussians)
            means = gmm.means_.flatten()
            covariances = gmm.covariances_.flatten()
            stds = np.sqrt(np.maximum(covariances, 1e-12))  # Protect against zero variance
            weights = gmm.weights_

            # Conservative threshold: mean + 2*std of lower component
            lower_idx = np.argmin(means)
            threshold = means[lower_idx] + 2 * stds[lower_idx]
        except:
            # GMM fitting failed, fall back to simple statistics
            threshold = np.percentile(js_values, 75)
            means = [np.mean(js_values)] * 2
            stds = [np.std(js_values)] * 2
            weights = [0.5, 0.5]
            gmm = None

        results[L] = {
            'js_values': js_values,
            'threshold': threshold,
            'gmm': gmm,
            'means': means,
            'stds': stds,
            'weights': weights
        }

    return results


def analyze_js_kstep(data: np.ndarray, model, L_max: int = 10, k: int = 3,
                     n_pairs: int = 5000, platt_params: Optional[dict] = None) -> Dict[int, Dict[str, Any]]:
    """
    Analyze JS divergence distributions for k-step rollout distributions across history lengths L.

    Mirrors analyze_js_distribution but uses exact k-step chained probabilities.
    """
    results: Dict[int, Dict[str, Any]] = {}
    rng = np.random.default_rng(42)

    for L in range(1, L_max + 1):
        print(f"\nAnalyzing k-step L={L}, k={k}")

        # Extract subsequences
        subsequences = extract_subsequences(data, L)
        if len(subsequences) < n_pairs:
            n_samples_L = len(subsequences) // 2
        else:
            n_samples_L = n_pairs

        # Sample random pairs
        idx = rng.choice(len(subsequences), size=(n_samples_L, 2), replace=True)

        # Compute k-step JS divergences
        js_values = []
        for i, j in idx:
            js_val = js_divergence_k(model, subsequences[i], subsequences[j], k, platt_params)
            js_values.append(js_val)

        js_values = np.array(js_values)

        # Try GMM fitting with error handling
        try:
            gmm = GaussianMixture(n_components=2, random_state=42)
            gmm.fit(js_values.reshape(-1, 1))

            # Find threshold (intersection of two Gaussians)
            means = gmm.means_.flatten()
            covariances = gmm.covariances_.flatten()
            stds = np.sqrt(np.maximum(covariances, 1e-12))
            weights = gmm.weights_

            # Conservative threshold: mean + 2*std of lower component
            lower_idx = np.argmin(means)
            threshold = means[lower_idx] + 2 * stds[lower_idx]
        except:
            # GMM fitting failed, fall back to simple statistics
            threshold = np.percentile(js_values, 75)
            means = [np.mean(js_values)] * 2
            stds = [np.std(js_values)] * 2
            weights = [0.5, 0.5]
            gmm = None

        results[L] = {
            'js_values': js_values,
            'threshold': threshold,
            'gmm': gmm,
            'means': means,
            'stds': stds,
            'weights': weights
        }

    return results


def analyze_js_conditional_kstep(data: np.ndarray, model, L_max: int = 10, k: int = 5,
                                n_pairs: int = 5000, platt_params: Optional[dict] = None) -> Dict[int, Dict[str, Any]]:
    """
    Analyze conditional JS divergence distributions for k-step rollouts.

    Uses conditional JS divergence which removes mixture effects by conditioning
    on the first symbol, providing cleaner signals for state differentiation.
    """
    results: Dict[int, Dict[str, Any]] = {}
    rng = np.random.default_rng(42)

    for L in range(1, L_max + 1):
        print(f"\nAnalyzing conditional k-step L={L}, k={k}")

        # Extract subsequences
        subsequences = extract_subsequences(data, L)
        if len(subsequences) < n_pairs:
            n_samples_L = len(subsequences) // 2
        else:
            n_samples_L = n_pairs

        # Sample random pairs
        idx = rng.choice(len(subsequences), size=(n_samples_L, 2), replace=True)

        # Compute conditional k-step JS divergences
        js_values = []
        for i, j in idx:
            js_val = js_divergence_conditional_k(model, subsequences[i], subsequences[j], k, platt_params)
            js_values.append(js_val)

        js_values = np.array(js_values)

        # Try GMM fitting with error handling
        try:
            gmm = GaussianMixture(n_components=2, random_state=42)
            gmm.fit(js_values.reshape(-1, 1))

            # Find threshold (intersection of two Gaussians)
            means = gmm.means_.flatten()
            covariances = gmm.covariances_.flatten()
            stds = np.sqrt(np.maximum(covariances, 1e-12))
            weights = gmm.weights_

            # Conservative threshold: mean + 2*std of lower component
            lower_idx = np.argmin(means)
            threshold = means[lower_idx] + 2 * stds[lower_idx]
        except:
            # GMM fitting failed, fall back to simple statistics
            threshold = np.percentile(js_values, 75)
            means = [np.mean(js_values)] * 2
            stds = [np.std(js_values)] * 2
            weights = [0.5, 0.5]
            gmm = None

        results[L] = {
            'js_values': js_values,
            'threshold': threshold,
            'gmm': gmm,
            'means': means,
            'stds': stds,
            'weights': weights
        }

    return results


def analyze_instate_js_distribution(data: np.ndarray, model, L_max: int = 10,
                                   n_samples_per_state: int = 1000, alphabet_size: int = 2,
                                   platt_params: Optional[dict] = None, preset: str = 'seven_state_human') -> Dict[int, Dict[str, Any]]:
    """Analyze JS divergence distributions for histories within the same GT state."""

    results = {}
    state_suffixes = get_all_state_suffixes(preset)

    for L in range(1, L_max + 1):
        print(f"\nAnalyzing instate L={L}")

        # For each state, find all histories that belong to that state
        state_results = {}

        histories_by_state = collect_histories_by_state(data, L, preset)

        for state, suffixes in state_suffixes.items():
            histories = histories_by_state.get(state, [])

            if len(histories) < 2:
                print(f"  {state}: {len(histories)} histories (skipping)")
                continue

            print(f"  {state}: {len(histories)} histories")

            # Sample random pairs within this state
            n_samples = min(n_samples_per_state, len(histories) * (len(histories) - 1) // 2)
            js_values = []

            for _ in range(n_samples):
                # Randomly select two different histories from this state
                i, j = np.random.choice(len(histories), size=2, replace=False)
                p = get_next_token_distribution(model, histories[i], platt_params=platt_params)
                q = get_next_token_distribution(model, histories[j], platt_params=platt_params)
                js_values.append(js_divergence(p, q))

            js_values = np.array(js_values)

            # Compute statistics safely
            if len(js_values) == 0:
                threshold = 0.0
                means = [0.0, 0.0]
                stds = [0.0, 0.0]
                weights = [0.5, 0.5]
                gmm = None
            else:
                # Check if all JS values are essentially the same (zero variance case)
                js_array = np.array(js_values)
                js_std = np.std(js_array)

                if js_std < 1e-10 or len(js_values) <= 5:
                    # All values are the same or very few samples
                    threshold = np.percentile(js_array, 75) if len(js_array) > 0 else 0.0
                    means = [np.mean(js_array)] * 2
                    stds = [0.0, 0.0]  # Zero variance
                    weights = [1.0, 0.0]
                    gmm = None
                else:
                    # Try GMM fitting with appropriate number of components
                    n_components = min(2, max(1, len(js_values)//5))
                    try:
                        gmm = GaussianMixture(n_components=n_components, random_state=42)
                        gmm.fit(js_array.reshape(-1, 1))

                        means = gmm.means_.flatten()
                        covariances = gmm.covariances_.flatten()
                        stds = np.sqrt(np.maximum(covariances, 1e-12))
                        weights = gmm.weights_

                        if n_components == 2:
                            lower_idx = np.argmin(means)
                            threshold = means[lower_idx] + 2 * stds[lower_idx]
                        else:
                            threshold = np.percentile(js_array, 75)
                    except:
                        # GMM fitting failed
                        threshold = np.percentile(js_array, 75)
                        means = [np.mean(js_array)] * 2
                        stds = [np.std(js_array)] * 2
                        weights = [0.5, 0.5]
                        gmm = None

            state_results[state] = {
                'js_values': js_values,
                'threshold': threshold,
                'gmm': gmm,
                'means': means,
                'stds': stds,
                'weights': weights,
                'n_histories': len(histories)
            }

        results[L] = state_results

    return results


def analyze_cross_state_js_distribution(data: np.ndarray, model, L_max: int = 10,
                                       n_samples_per_pair: int = 1000,
                                       platt_params: Optional[dict] = None, preset: str = 'seven_state_human') -> Dict[int, Dict[str, Any]]:
    """Analyze JS divergence distributions for histories from different GT states."""

    results = {}
    state_suffixes = get_all_state_suffixes(preset)

    for L in range(1, L_max + 1):
        print(f"\nAnalyzing cross-state L={L}")

        # Collect all histories by state
        histories_by_state = collect_histories_by_state(data, L, preset)

        # Get states with sufficient histories
        valid_states = [state for state, histories in histories_by_state.items() if len(histories) >= 2]

        if len(valid_states) < 2:
            print(f"  Not enough states with sufficient histories (need ≥2 histories each)")
            continue

        print(f"  States with sufficient data: {valid_states}")

        # For each pair of states, compute JS divergences
        pair_stats = {}

        for i, state1 in enumerate(valid_states):
            for j in range(i + 1, len(valid_states)):
                state2 = valid_states[j]

                histories1 = histories_by_state[state1]
                histories2 = histories_by_state[state2]

                print(f"    Analyzing {state1} vs {state2}: {len(histories1)} vs {len(histories2)} histories")

                # Sample pairs across states
                n_samples = min(n_samples_per_pair, len(histories1) * len(histories2))
                js_values = []

                for _ in range(n_samples):
                    # Sample one history from each state
                    h1 = histories1[np.random.choice(len(histories1))]
                    h2 = histories2[np.random.choice(len(histories2))]

                    p = get_next_token_distribution(model, h1, platt_params=platt_params)
                    q = get_next_token_distribution(model, h2, platt_params=platt_params)
                    js_values.append(js_divergence(p, q))

                js_values = np.array(js_values)

                if len(js_values) > 0:
                    pair_stats[(state1, state2)] = {
                        'js_values': js_values,
                        'mean': float(np.mean(js_values)),
                        'median': float(np.median(js_values)),
                        'std': float(np.std(js_values)),
                        'n_pairs': len(js_values)
                    }
                else:
                    pair_stats[(state1, state2)] = {
                        'js_values': np.array([]),
                        'mean': 0.0,
                        'median': 0.0,
                        'std': 0.0,
                        'n_pairs': 0
                    }

        results[L] = {
            'states': valid_states,
            'pair_stats': pair_stats,
            'histories_by_state': {state: len(histories) for state, histories in histories_by_state.items()}
        }

    return results


def analyze_cross_state_js_kstep(data: np.ndarray, model, L_max: int = 10, k: int = 3,
                                n_samples_per_pair: int = 1000,
                                platt_params: Optional[dict] = None, preset: str = 'seven_state_human') -> Dict[int, Dict[str, Any]]:
    """Analyze k-step JS divergence distributions for histories from different GT states."""

    results = {}
    state_suffixes = get_all_state_suffixes(preset)

    for L in range(1, L_max + 1):
        print(f"\nAnalyzing cross-state k-step L={L}, k={k}")

        # Collect all histories by state
        histories_by_state = collect_histories_by_state(data, L, preset)

        # Get states with sufficient histories
        valid_states = [state for state, histories in histories_by_state.items() if len(histories) >= 2]

        if len(valid_states) < 2:
            print(f"  Not enough states with sufficient histories (need ≥2 histories each)")
            continue

        print(f"  States with sufficient data: {valid_states}")

        # For each pair of states, compute k-step JS divergences
        pair_stats = {}

        for i, state1 in enumerate(valid_states):
            for j in range(i + 1, len(valid_states)):
                state2 = valid_states[j]

                histories1 = histories_by_state[state1]
                histories2 = histories_by_state[state2]

                print(f"    Analyzing {state1} vs {state2}: {len(histories1)} vs {len(histories2)} histories")

                # Sample pairs across states
                n_samples = min(n_samples_per_pair, len(histories1) * len(histories2))
                js_values = []

                for _ in range(n_samples):
                    # Sample one history from each state
                    h1 = histories1[np.random.choice(len(histories1))]
                    h2 = histories2[np.random.choice(len(histories2))]

                    js_val = js_divergence_k(model, h1, h2, k, platt_params)
                    js_values.append(js_val)

                js_values = np.array(js_values)

                if len(js_values) > 0:
                    pair_stats[(state1, state2)] = {
                        'js_values': js_values,
                        'mean': float(np.mean(js_values)),
                        'median': float(np.median(js_values)),
                        'std': float(np.std(js_values)),
                        'n_pairs': len(js_values)
                    }
                else:
                    pair_stats[(state1, state2)] = {
                        'js_values': np.array([]),
                        'mean': 0.0,
                        'median': 0.0,
                        'std': 0.0,
                        'n_pairs': 0
                    }

        results[L] = {
            'states': valid_states,
            'pair_stats': pair_stats,
            'histories_by_state': {state: len(histories) for state, histories in histories_by_state.items()}
        }

    return results


def analyze_cross_state_js_conditional_kstep(data: np.ndarray, model, L_max: int = 10, k: int = 5,
                                            n_samples_per_pair: int = 1000,
                                            platt_params: Optional[dict] = None, preset: str = 'seven_state_human') -> Dict[int, Dict[str, Any]]:
    """Analyze conditional k-step JS divergence distributions for histories from different GT states."""

    results = {}
    state_suffixes = get_all_state_suffixes(preset)

    for L in range(1, L_max + 1):
        print(f"\nAnalyzing cross-state conditional k-step L={L}, k={k}")

        # Collect all histories by state
        histories_by_state = collect_histories_by_state(data, L, preset)

        # Get states with sufficient histories
        valid_states = [state for state, histories in histories_by_state.items() if len(histories) >= 2]

        if len(valid_states) < 2:
            print(f"  Not enough states with sufficient histories (need ≥2 histories each)")
            continue

        print(f"  States with sufficient data: {valid_states}")

        # For each pair of states, compute conditional k-step JS divergences
        pair_stats = {}

        for i, state1 in enumerate(valid_states):
            for j in range(i + 1, len(valid_states)):
                state2 = valid_states[j]

                histories1 = histories_by_state[state1]
                histories2 = histories_by_state[state2]

                print(f"    Analyzing {state1} vs {state2}: {len(histories1)} vs {len(histories2)} histories")

                # Sample pairs across states
                n_samples = min(n_samples_per_pair, len(histories1) * len(histories2))
                js_values = []

                for _ in range(n_samples):
                    # Sample one history from each state
                    h1 = histories1[np.random.choice(len(histories1))]
                    h2 = histories2[np.random.choice(len(histories2))]

                    js_val = js_divergence_conditional_k(model, h1, h2, k, platt_params)
                    js_values.append(js_val)

                js_values = np.array(js_values)

                if len(js_values) > 0:
                    pair_stats[(state1, state2)] = {
                        'js_values': js_values,
                        'mean': float(np.mean(js_values)),
                        'median': float(np.median(js_values)),
                        'std': float(np.std(js_values)),
                        'n_pairs': len(js_values)
                    }
                else:
                    pair_stats[(state1, state2)] = {
                        'js_values': np.array([]),
                        'mean': 0.0,
                        'median': 0.0,
                        'std': 0.0,
                        'n_pairs': 0
                    }

        results[L] = {
            'states': valid_states,
            'pair_stats': pair_stats,
            'histories_by_state': {state: len(histories) for state, histories in histories_by_state.items()}
        }

    return results