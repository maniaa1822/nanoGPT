#!/usr/bin/env python3
"""
Enhanced JS divergence analysis vs history length for neural CSSR models.

NEW FEATURES:
- Instate analysis: Compute JS divergence between histories within the same ground truth state
- Ground truth state mapping for seven_state_human, golden_mean, and even_process presets
- Multiple analysis types: random pairs, instate pairs, or both

Usage examples:
  # Random pairs analysis only (works with any preset)
  python plot_js_vs_L.py --preset golden_mean --analysis_type random

  # Instate analysis on seven-state human machine (small model)
  python plot_js_vs_L.py --preset seven_state_human --analysis_type instate

  # Instate analysis on seven-state human machine (large model)
  python plot_js_vs_L.py --preset seven_state_human_large --analysis_type instate

  # Instate analysis on golden mean machine
  python plot_js_vs_L.py --preset golden_mean --analysis_type instate

  # Instate analysis on even process machine
  python plot_js_vs_L.py --preset even_process --analysis_type instate

  # Conditional cross-state analysis (seven-state human with k=4)
  python plot_js_vs_L.py --preset seven_state_human --L_max 4 --n_samples 50 --k 4 --analysis_type cross

  # Both analyses (default)
  python plot_js_vs_L.py --preset golden_mean --analysis_type both

  # Custom model and data
  python plot_js_vs_L.py --model_ckpt /path/to/model.pt --data /path/to/data.dat --analysis_type both
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import entropy
from typing import List, Tuple, Optional
import argparse
import torch
from pathlib import Path
import sys

def extract_subsequences(data: np.ndarray, L: int) -> List[np.ndarray]:
    """Extract all length-L subsequences from dataset."""
    subsequences = []
    for i in range(len(data) - L + 1):
        subsequences.append(data[i:i+L])
    return subsequences

def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Jensen-Shannon divergence between two distributions."""
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))

def get_seven_state_gt_state(history: np.ndarray) -> str:
    """Determine ground truth state for a history in the seven-state human machine.

    The seven states are defined by suffix patterns. We need to find the LONGEST
    suffix that matches a known state pattern, checking longer patterns first.

    State patterns (longest first):
    - AAAB: ends with 'AAAB' (4 chars) -> state 'AAAB'
    - BAAB: ends with 'BAAB' (4 chars) -> state 'BAAB'
    - BAB: ends with 'BAB' (3 chars) -> state 'BAB'
    - BAA: ends with 'BAA' (3 chars) -> state 'BAA'
    - AAA: ends with 'AAA' (3 chars) -> state 'AAA' (but NOT 'AAAB')
    - BB: ends with 'BB' (2 chars) -> state 'BB'
    - BA: ends with 'BA' (2 chars) -> state 'BA' (but NOT longer patterns)

    For histories shorter than the required pattern length, we cannot determine
    the exact state, so we return None.
    """
    history_str = ''.join(map(str, history))

    # Check for longest patterns first (4 characters, binary)
    if len(history) >= 4:
        if history_str.endswith('1110'):
            return 'aaab'
        elif history_str.endswith('0110'):
            return 'baab'

    # Check for 3-character patterns (binary)
    if len(history) >= 3:
        if history_str.endswith('010'):
            return 'bab'
        elif history_str.endswith('011'):
            return 'baa'
        elif history_str.endswith('111') and not history_str.endswith('1110'):
            return 'aaa'

    # Check for 2-character patterns (binary)
    if len(history) >= 2:
        if history_str.endswith('00'):
            return 'bb'
        elif history_str.endswith('01') and not (len(history) >= 3 and (history_str.endswith('010') or history_str.endswith('011'))):
            return 'ba'

    # Cannot determine state for shorter histories
    return None

def get_golden_mean_gt_state(history: np.ndarray) -> str:
    """Determine ground truth causal state for a history in the golden mean process.

    Golden mean causal states are based on conditional futures:
    - State A: histories ending with 1 (P(next=0)=0.5, P(next=1)=0.5)
    - State B: histories ending with 0 (P(next=0)=0, P(next=1)=1)

    The empty history (length 0) is in State A.
    """
    if len(history) == 0:
        return 'A'

    last_symbol = int(history[-1])
    return 'A' if last_symbol == 1 else 'B'


def get_even_process_gt_state(history: np.ndarray) -> str:
    """Determine ground truth causal state for a history in the even process."""
    ones_run = 0
    for bit in history[::-1]:
        if int(bit) == 1:
            ones_run += 1
        else:
            break
    return 'E' if ones_run % 2 == 0 else 'O'

def get_gt_state(history: np.ndarray, preset: str) -> str:
    """Generic function to get ground truth state for a given preset."""
    if preset in ['seven_state_human', 'seven_state_human_large']:
        return get_seven_state_gt_state(history)
    elif preset == 'golden_mean':
        return get_golden_mean_gt_state(history)
    elif preset == 'even_process':
        return get_even_process_gt_state(history)
    else:
        raise ValueError(f"Ground truth state mapping not available for preset: {preset}")

def test_seven_state_mapping():
    """Test function to verify seven-state human machine state mapping."""
    test_histories = [
        np.array([0]),        # Should be BB
        np.array([1]),        # Should be BA
        np.array([1, 0]),     # Should be BB (ends with 0)
        np.array([1, 1]),     # Should be BA (ends with 1)
        np.array([0, 1]),     # Should be BA (ends with 1)
        np.array([0, 0]),     # Should be BB (ends with 0)
        np.array([1, 0, 1]),  # Should be BB (ends with 01)
        np.array([1, 1, 1]),  # Should be BA (ends with 1)
        np.array([1, 0, 1, 1]),  # Should be BB (ends with 011)
    ]

    print("Testing seven-state human machine state mapping:")
    for hist in test_histories:
        state = get_seven_state_gt_state(hist)
        hist_str = ''.join(map(str, hist))
        print(f"  History '{hist_str}' -> State {state}")
    print()

def test_pattern_extraction():
    """Test function to verify pattern extraction logic."""
    # Create test data with known patterns
    test_data = np.array([1,1,1,0,1,0,1,0,0,1,1,0,1,1,1])  # Contains 111, 010, 00, 110, 111
    test_data_str = ''.join(map(str, test_data))
    print(f"Test data: {test_data_str}")

    # Test finding histories for bab state (pattern '010')
    L = 8
    bab_histories = find_histories_for_state(test_data, 'bab', ['010'], L)
    print(f"\nBAB state histories (L={L}):")
    for hist in bab_histories:
        hist_str = ''.join(map(str, hist))
        print(f"  '{hist_str}' (ends with '{hist_str[-3:]}')")

    # Test finding histories for bb state (pattern '00')
    bb_histories = find_histories_for_state(test_data, 'bb', ['00'], L)
    print(f"\nBB state histories (L={L}):")
    for hist in bb_histories:
        hist_str = ''.join(map(str, hist))
        print(f"  '{hist_str}' (ends with '{hist_str[-2:]}')")

    print()

def test_state_mapping_simple():
    """Test the get_seven_state_gt_state function with some examples."""
    test_histories = [
        np.array([0,1]),      # '01' - should be BA
        np.array([0,1,0]),    # '010' - should be BAB
        np.array([0,0]),      # '00' - should be BB
        np.array([1,1]),      # '11' - should be None (too short)
        np.array([1,1,1]),    # '111' - should be AAA
        np.array([1,1,1,0]),  # '1110' - should be AAAB
    ]

    print("Testing state mapping:")
    for hist in test_histories:
        state = get_seven_state_gt_state(hist)
        hist_str = ''.join(map(str, hist))
        print(f"  '{hist_str}' -> {state}")
    print()

def get_next_token_distribution(model, history: np.ndarray, platt_params: Optional[dict] = None) -> np.ndarray:
    """Get P(x|history) from your trained transformer."""
    # Convert history to model input format
    with torch.no_grad():
        # Truncate to model context window if available
        ctx = getattr(getattr(model, 'config', None), 'block_size', None)
        if ctx is not None and len(history) > int(ctx):
            ids = history[-int(ctx):]
        else:
            ids = history
        x = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        # Move to model device if possible
        try:
            device = next(model.parameters()).device
            x = x.to(device)
        except Exception:
            pass
        out = model(x)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        if logits.dim() == 3:
            logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
        if probs.shape[-1] < 2:
            return np.array([0.5, 0.5], dtype=np.float64)
        p0, p1 = float(probs[0]), float(probs[1])
        # Optional Platt calibration on p1
        if platt_params is not None and 'a' in platt_params and 'b' in platt_params:
            a = float(platt_params['a']); b = float(platt_params['b'])
            eps = 1e-12
            p1c = max(eps, min(1.0 - eps, p1))
            logit = np.log(p1c / (1.0 - p1c))
            s = a * logit + b
            p1 = float(1.0 / (1.0 + np.exp(-s)))
            p1 = max(1e-6, min(1.0 - 1e-6, p1))
            p0 = 1.0 - p1
        s = p0 + p1
        if s <= 0:
            return np.array([0.5, 0.5], dtype=np.float64)
        return np.array([p0 / s, p1 / s], dtype=np.float64)

def analyze_js_distribution(data: np.ndarray, model, L_max: int = 10, 
                           n_samples: int = 5000, alphabet_size: int = 2,
                           platt_params: Optional[dict] = None):
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

def get_kstep_distribution(model, history: np.ndarray, k: int, platt_params: Optional[dict] = None) -> np.ndarray:
    """
    Compute exact k-step distribution over binary sequences by chaining next-token probabilities.

    Returns a vector of length 2^k in lexicographic order (0..0, 0..1, ..., 1..1).
    """
    if k <= 0:
        return np.array([1.0], dtype=np.float64)

    num_paths = 1 << k
    probs = np.zeros(num_paths, dtype=np.float64)

    # stack entries: (history_tuple, depth, log_prob, index_prefix)
    initial_hist_tuple = tuple(int(x) for x in history.tolist())
    stack = [(initial_hist_tuple, 0, 0.0, 0)]

    while stack:
        hist_tuple, depth, logp, idx_prefix = stack.pop()
        if depth == k:
            probs[idx_prefix] = np.exp(logp)
            continue

        p = get_next_token_distribution(model, np.array(hist_tuple, dtype=np.int64), platt_params=platt_params)
        p0, p1 = float(p[0]), float(p[1])

        # Append 0 (left child)
        stack.append((hist_tuple + (0,), depth + 1, logp + np.log(max(1e-12, p0)), (idx_prefix << 1) | 0))
        # Append 1 (right child)
        stack.append((hist_tuple + (1,), depth + 1, logp + np.log(max(1e-12, p1)), (idx_prefix << 1) | 1))

    s = probs.sum()
    if not np.isfinite(s) or s <= 0:
        return np.full(num_paths, 1.0 / num_paths, dtype=np.float64)
    return probs / s

def js_divergence_k(model, h1: np.ndarray, h2: np.ndarray, k: int, platt_params: Optional[dict] = None) -> float:
    """JS divergence between exact k-step rollout distributions for two histories."""
    p = get_kstep_distribution(model, h1, k, platt_params)
    q = get_kstep_distribution(model, h2, k, platt_params)
    return js_divergence(p, q)

def js_divergence_conditional_k(model, h1: np.ndarray, h2: np.ndarray, k: int, platt_params: Optional[dict] = None) -> float:
    """
    Conditional JS divergence between two histories using k-step rollouts.

    This computes a weighted average of JS divergences after conditioning on the first symbol,
    which removes mixture effects and focuses on successor kernel differences.

    Args:
        model: The neural model
        h1, h2: History arrays to compare
        k: Steps to roll out (must be >= 2)
        platt_params: Optional Platt calibration parameters

    Returns:
        Conditional JS divergence: w̄₀ * JS(p|₀, q|₀) + w̄₁ * JS(p|₁, q|₁)
    """
    if k < 2:
        raise ValueError("Conditional JS requires k >= 2")

    # Get k-step distributions for both histories
    p = get_kstep_distribution(model, h1, k, platt_params)  # length 2^k
    q = get_kstep_distribution(model, h2, k, platt_params)  # length 2^k

    # Split by first bit: first half (0...) and second half (1...)
    m = 1 << (k - 1)  # 2^(k-1)

    p0, p1 = p[:m], p[m:]  # p(k)(0·|h1), p(k)(1·|h1)
    q0, q1 = q[:m], q[m:]  # p(k)(0·|h2), p(k)(1·|h2)

    # Compute marginal probabilities for first symbol
    w0p, w1p = float(p0.sum()), float(p1.sum())  # Pr(0|h1), Pr(1|h1)
    w0q, w1q = float(q0.sum()), float(q1.sum())  # Pr(0|h2), Pr(1|h2)

    # Normalize conditional distributions: p|₀^(k-1)(·|h) = p^(k)(0·|h) / w₀(h)
    p0 = p0 / max(w0p, 1e-12)
    p1 = p1 / max(w1p, 1e-12)
    q0 = q0 / max(w0q, 1e-12)
    q1 = q1 / max(w1q, 1e-12)

    # Compute average weights: w̄ₐ = ½(wₐ(h1) + wₐ(h2))
    w0_bar = 0.5 * (w0p + w0q)
    w1_bar = 1.0 - w0_bar

    # Conditional JS: weighted average of JS divergences after branching
    js_cond = (w0_bar * js_divergence(p0, q0) +
               w1_bar * js_divergence(p1, q1))

    return float(js_cond)

def analyze_js_kstep(data: np.ndarray, model, L_max: int = 10, k: int = 3,
                     n_pairs: int = 5000, platt_params: Optional[dict] = None) -> dict:
    """
    Analyze JS divergence distributions for k-step rollout distributions across history lengths L.

    Mirrors analyze_js_distribution but uses exact k-step chained probabilities.
    """
    results: dict = {}
    rng = np.random.default_rng(42)

    for L in range(1, L_max + 1):
        print(f"\nAnalyzing k-step (k={k}) L={L}")
        subsequences = extract_subsequences(data, L)
        if len(subsequences) == 0:
            results[L] = {"js_values": np.array([], dtype=np.float64), "threshold": 0.0, "gmm": None,
                          "means": [0.0, 0.0], "stds": [0.0, 0.0], "weights": [0.5, 0.5]}
            continue

        m = min(n_pairs, max(1, len(subsequences) // 2))
        pairs = rng.integers(0, len(subsequences), size=(m, 2))

        # Local cache to avoid recomputation for the same (L,k,history)
        cache: dict = {}
        def cache_key(idx: int):
            return (L, k, tuple(int(x) for x in subsequences[idx].tolist()))

        js_vals: list = []
        for i, j in pairs:
            ki, kj = cache_key(i), cache_key(j)
            if ki not in cache:
                cache[ki] = get_kstep_distribution(model, subsequences[i], k, platt_params)
            if kj not in cache:
                cache[kj] = get_kstep_distribution(model, subsequences[j], k, platt_params)
            js_vals.append(js_divergence(cache[ki], cache[kj]))

        js_values = np.array(js_vals, dtype=np.float64)

        # Fit GMM for rough thresholding (consistent with one-step pipeline)
        try:
            gmm = GaussianMixture(n_components=2, random_state=42)
            gmm.fit(js_values.reshape(-1, 1))
            means = gmm.means_.flatten()
            covariances = gmm.covariances_.flatten()
            stds = np.sqrt(np.maximum(covariances, 1e-12))
            weights = gmm.weights_
            lower_idx = int(np.argmin(means))
            threshold = float(means[lower_idx] + 2.0 * stds[lower_idx])
        except Exception:
            gmm = None
            threshold = float(np.percentile(js_values, 75)) if js_values.size > 0 else 0.0
            means = [float(js_values.mean()) if js_values.size > 0 else 0.0] * 2
            stds = [float(js_values.std()) if js_values.size > 0 else 0.0] * 2
            weights = [0.5, 0.5]

        results[L] = {
            "js_values": js_values,
            "threshold": threshold,
            "gmm": gmm,
            "means": means,
            "stds": stds,
            "weights": weights,
        }

    return results

def analyze_js_conditional_kstep(data: np.ndarray, model, L_max: int = 10, k: int = 5,
                                n_pairs: int = 5000, platt_params: Optional[dict] = None) -> dict:
    """
    Analyze conditional JS divergence distributions for k-step rollout distributions across history lengths L.

    Uses conditional JS which removes mixture effects by comparing futures after conditioning
    on the first symbol, better capturing successor kernel differences.
    """
    if k < 2:
        raise ValueError("Conditional JS analysis requires k >= 2")

    results: dict = {}
    rng = np.random.default_rng(42)

    for L in range(1, L_max + 1):
        print(f"\\nAnalyzing conditional k-step (k={k}) L={L}")
        subsequences = extract_subsequences(data, L)
        if len(subsequences) == 0:
            results[L] = {"js_values": np.array([], dtype=np.float64), "threshold": 0.0, "gmm": None,
                          "means": [0.0, 0.0], "stds": [0.0, 0.0], "weights": [0.5, 0.5]}
            continue

        m = min(n_pairs, max(1, len(subsequences) // 2))
        pairs = rng.integers(0, len(subsequences), size=(m, 2))

        # Local cache to avoid recomputation for the same (L,k,history)
        cache: dict = {}
        def cache_key(idx: int):
            return (L, k, tuple(int(x) for x in subsequences[idx].tolist()))

        js_vals: list = []
        for i, j in pairs:
            # Use conditional JS divergence instead of regular JS
            js_cond = js_divergence_conditional_k(model, subsequences[i], subsequences[j], k, platt_params)
            js_vals.append(js_cond)

        js_values = np.array(js_vals, dtype=np.float64)

        # Fit GMM for rough thresholding (consistent with other analyses)
        try:
            gmm = GaussianMixture(n_components=2, random_state=42)
            gmm.fit(js_values.reshape(-1, 1))
            means = gmm.means_.flatten()
            covariances = gmm.covariances_.flatten()
            stds = np.sqrt(np.maximum(covariances, 1e-12))
            weights = gmm.weights_
            lower_idx = int(np.argmin(means))
            threshold = float(means[lower_idx] + 2.0 * stds[lower_idx])
        except Exception:
            gmm = None
            threshold = float(np.percentile(js_values, 75)) if js_values.size > 0 else 0.0
            means = [float(js_values.mean()) if js_values.size > 0 else 0.0] * 2
            stds = [float(js_values.std()) if js_values.size > 0 else 0.0] * 2
            weights = [0.5, 0.5]

        results[L] = {
            "js_values": js_values,
            "threshold": threshold,
            "gmm": gmm,
            "means": means,
            "stds": stds,
            "weights": weights,
        }

    return results

def get_all_state_suffixes(preset: str = 'seven_state_human') -> dict:
    """Get all possible suffix patterns that define each state."""
    if preset in ['seven_state_human', 'seven_state_human_large']:
        return {
            'bb': ['00'],        # BB state: ends with 00
            'aaa': ['111'],      # AAA state: ends with 111
            'aaab': ['1110'],    # AAAB state: ends with 1110
            'ba': ['01'],        # BA state: ends with 01
            'bab': ['010'],      # BAB state: ends with 010
            'baab': ['0110'],    # BAAB state: ends with 0110
            'baa': ['011']       # BAA state: ends with 011
        }
    elif preset == 'golden_mean':
        return {
            'A': [],  # All histories ending with 1
            'B': []   # All histories ending with 0
        }
    elif preset == 'even_process':
        return {
            'E': [],  # Even parity of trailing 1s
            'O': []   # Odd parity of trailing 1s
        }
    else:
        raise ValueError(f"State suffixes not defined for preset: {preset}")


def collect_histories_by_state(data: np.ndarray, L: int, preset: str) -> dict:
    """Group all length-L histories by their ground-truth state."""
    histories_by_state: dict = {}
    for i in range(len(data) - L + 1):
        hist = data[i:i+L]
        state = get_gt_state(hist, preset)
        if state is None:
            continue
        histories_by_state.setdefault(state, []).append(hist)
    return histories_by_state


def find_histories_for_state(data: np.ndarray, state: str, suffix_patterns: list, L: int) -> list:
    """Find all L-length histories that end with patterns defining the given state."""
    if not suffix_patterns:
        return []  # No patterns defined

    histories = []

    # Find all positions where patterns for this state occur
    for i in range(len(data)):
        # Check if position i ends a pattern for this state
        for pattern in suffix_patterns:
            pattern_len = len(pattern)
            if i >= pattern_len - 1:
                # Check if data[i-pattern_len+1:i+1] matches the pattern
                candidate = data[i-pattern_len+1:i+1]
                candidate_str = ''.join(map(str, candidate))
                if candidate_str == pattern:
                    # Found pattern ending at position i
                    # Extract L-length history ending at i
                    start_pos = max(0, i - L + 1)
                    hist = data[start_pos:i+1]
                    if len(hist) == L:
                        histories.append(hist)
                    break  # Found a match for this position

    return histories

def debug_pattern_matching(data: np.ndarray, L: int = 8):
    """Debug function to check what patterns are actually found in the data."""
    patterns_to_check = {
        'bb': ['00'],
        'aaa': ['111'],
        'aaab': ['1110'],
        'ba': ['01'],
        'bab': ['010'],
        'baab': ['0110'],
        'baa': ['011']
    }

    print(f"Debug: Checking patterns in L={L} histories")
    for i in range(len(data) - L + 1):
        hist = data[i:i+L]
        hist_str = ''.join(map(str, hist))
        print(f"  History {i}: '{hist_str}'")

        for state, patterns in patterns_to_check.items():
            for pattern in patterns:
                if hist_str.endswith(pattern):
                    print(f"    -> Matches {state} pattern '{pattern}'")

    print()

def analyze_instate_js_distribution(data: np.ndarray, model, L_max: int = 10,
                                   n_samples_per_state: int = 1000, alphabet_size: int = 2,
                                   platt_params: Optional[dict] = None, preset: str = 'seven_state_human'):
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
                    gmm = GaussianMixture(n_components=n_components, random_state=42)
                    try:
                        gmm.fit(js_array.reshape(-1, 1))
                        means = gmm.means_.flatten()
                        covariances = gmm.covariances_.flatten()
                        stds = np.sqrt(np.maximum(covariances, 1e-12))  # Protect against zero variance
                        weights = gmm.weights_

                        # Conservative threshold: mean + 2*std of lower component
                        lower_idx = np.argmin(means)
                        threshold = means[lower_idx] + 2 * stds[lower_idx]
                    except:
                        # GMM fitting failed, fall back to simple statistics
                        threshold = np.percentile(js_array, 75)
                        means = [np.mean(js_array)] * 2
                        stds = [np.std(js_array)] * 2
                        weights = [0.5, 0.5]
                        gmm = None

            state_results[state] = {
                'js_values': js_values,
                'threshold': threshold,
                'gmm': gmm if len(js_values) > 10 else None,
                'means': means,
                'stds': stds,
                'weights': weights,
                'n_histories': len(histories),
                'suffix_patterns': suffixes
            }

        results[L] = state_results

    return results


def analyze_cross_state_js_distribution(data: np.ndarray, model, L_max: int = 10,
                                        n_samples_per_pair: int = 1000, platt_params: Optional[dict] = None,
                                        preset: str = 'seven_state_human') -> dict:
    """Analyze JS divergence distributions between histories from different GT states."""

    results: dict = {}
    rng = np.random.default_rng(42)
    state_names = list(get_all_state_suffixes(preset).keys())

    for L in range(1, L_max + 1):
        print(f"\nAnalyzing cross-state L={L}")
        histories_by_state = collect_histories_by_state(data, L, preset)
        pair_stats: dict = {}
        state_counts = {state: len(histories_by_state.get(state, [])) for state in state_names}

        for si_idx, state_i in enumerate(state_names):
            histories_i = histories_by_state.get(state_i, [])
            if not histories_i:
                continue

            for sj_idx, state_j in enumerate(state_names):
                if sj_idx < si_idx:
                    continue  # symmetric

                histories_j = histories_by_state.get(state_j, [])
                if not histories_j:
                    continue

                if state_i == state_j and len(histories_i) < 2:
                    continue

                samples = min(n_samples_per_pair, len(histories_i) * len(histories_j))
                if state_i == state_j:
                    # Avoid pairing identical histories when sampling with replacement
                    samples = min(n_samples_per_pair, len(histories_i) * max(len(histories_i) - 1, 0))

                if samples <= 0:
                    continue

                js_vals: List[float] = []
                for _ in range(samples):
                    idx_i = int(rng.integers(len(histories_i)))
                    if state_i == state_j:
                        if len(histories_i) < 2:
                            break
                        idx_j = int(rng.integers(len(histories_j)))
                        # Ensure distinct histories when within the same state
                        while idx_j == idx_i and len(histories_j) > 1:
                            idx_j = int(rng.integers(len(histories_j)))
                    else:
                        idx_j = int(rng.integers(len(histories_j)))

                    p = get_next_token_distribution(model, histories_i[idx_i], platt_params=platt_params)
                    q = get_next_token_distribution(model, histories_j[idx_j], platt_params=platt_params)
                    js_vals.append(js_divergence(p, q))

                js_array = np.array(js_vals, dtype=np.float64)

                stats = {
                    'js_values': js_array,
                    'mean': float(np.nanmean(js_array)) if js_array.size > 0 else float('nan'),
                    'median': float(np.nanmedian(js_array)) if js_array.size > 0 else float('nan'),
                    'std': float(np.nanstd(js_array)) if js_array.size > 0 else float('nan'),
                    'n_pairs': int(js_array.size),
                }

                pair_stats[(state_i, state_j)] = stats
                pair_stats[(state_j, state_i)] = stats

        results[L] = {
            'states': state_names,
            'pair_stats': pair_stats,
            'state_counts': state_counts,
        }

    return results

def analyze_cross_state_js_kstep(data: np.ndarray, model, L_max: int = 10, k: int = 3,
                                n_samples_per_pair: int = 1000, platt_params: Optional[dict] = None,
                                preset: str = 'seven_state_human') -> dict:
    """Analyze k-step JS divergence distributions between histories from different GT states."""

    results: dict = {}
    rng = np.random.default_rng(42)
    state_names = list(get_all_state_suffixes(preset).keys())

    for L in range(1, L_max + 1):
        print(f"\nAnalyzing k-step cross-state L={L}")
        histories_by_state = collect_histories_by_state(data, L, preset)
        pair_stats: dict = {}
        state_counts = {state: len(histories_by_state.get(state, [])) for state in state_names}

        for si_idx, state_i in enumerate(state_names):
            histories_i = histories_by_state.get(state_i, [])
            if not histories_i:
                continue

            for sj_idx, state_j in enumerate(state_names):
                if sj_idx < si_idx:
                    continue  # symmetric

                histories_j = histories_by_state.get(state_j, [])
                if not histories_j:
                    continue

                if state_i == state_j and len(histories_i) < 2:
                    continue

                samples = min(n_samples_per_pair, len(histories_i) * len(histories_j))
                if state_i == state_j:
                    # Avoid pairing identical histories when sampling with replacement
                    samples = min(n_samples_per_pair, len(histories_i) * max(len(histories_i) - 1, 0))

                if samples <= 0:
                    continue

                # Local cache to avoid recomputation for the same (L,k,history)
                cache: dict = {}
                def cache_key(idx: int, histories: list):
                    return (L, k, tuple(int(x) for x in histories[idx].tolist()))

                js_vals: List[float] = []
                for _ in range(samples):
                    idx_i = int(rng.integers(len(histories_i)))
                    if state_i == state_j:
                        if len(histories_i) < 2:
                            break
                        idx_j = int(rng.integers(len(histories_j)))
                        # Ensure distinct histories when within the same state
                        while idx_j == idx_i and len(histories_j) > 1:
                            idx_j = int(rng.integers(len(histories_j)))
                    else:
                        idx_j = int(rng.integers(len(histories_j)))

                    # Get k-step distributions with caching
                    ki, kj = cache_key(idx_i, histories_i), cache_key(idx_j, histories_j)
                    if ki not in cache:
                        cache[ki] = get_kstep_distribution(model, histories_i[idx_i], k, platt_params)
                    if kj not in cache:
                        cache[kj] = get_kstep_distribution(model, histories_j[idx_j], k, platt_params)

                    js_vals.append(js_divergence(cache[ki], cache[kj]))

                js_array = np.array(js_vals, dtype=np.float64)

                stats = {
                    'js_values': js_array,
                    'mean': float(np.nanmean(js_array)) if js_array.size > 0 else float('nan'),
                    'median': float(np.nanmedian(js_array)) if js_array.size > 0 else float('nan'),
                    'std': float(np.nanstd(js_array)) if js_array.size > 0 else float('nan'),
                    'n_pairs': int(js_array.size),
                }

                pair_stats[(state_i, state_j)] = stats
                pair_stats[(state_j, state_i)] = stats

        results[L] = {
            'states': state_names,
            'pair_stats': pair_stats,
            'state_counts': state_counts,
        }

    return results

def analyze_cross_state_js_conditional_kstep(data: np.ndarray, model, L_max: int = 10, k: int = 5,
                                             n_samples_per_pair: int = 1000, platt_params: Optional[dict] = None,
                                             preset: str = 'seven_state_human') -> dict:
    """Analyze conditional k-step JS divergence distributions between histories from different GT states."""
    if k < 2:
        raise ValueError("Conditional JS analysis requires k >= 2")

    results: dict = {}
    rng = np.random.default_rng(42)
    state_names = list(get_all_state_suffixes(preset).keys())

    for L in range(1, L_max + 1):
        print(f"\\nAnalyzing conditional k-step cross-state L={L}")
        histories_by_state = collect_histories_by_state(data, L, preset)
        pair_stats: dict = {}
        state_counts = {state: len(histories_by_state.get(state, [])) for state in state_names}

        for si_idx, state_i in enumerate(state_names):
            histories_i = histories_by_state.get(state_i, [])
            if not histories_i:
                continue

            for sj_idx, state_j in enumerate(state_names):
                if sj_idx < si_idx:
                    continue  # symmetric

                histories_j = histories_by_state.get(state_j, [])
                if not histories_j:
                    continue

                if state_i == state_j and len(histories_i) < 2:
                    continue

                samples = min(n_samples_per_pair, len(histories_i) * len(histories_j))
                if state_i == state_j:
                    # Avoid pairing identical histories when sampling with replacement
                    samples = min(n_samples_per_pair, len(histories_i) * max(len(histories_i) - 1, 0))

                if samples <= 0:
                    continue

                js_vals: List[float] = []
                for _ in range(samples):
                    idx_i = int(rng.integers(len(histories_i)))
                    if state_i == state_j:
                        if len(histories_i) < 2:
                            break
                        idx_j = int(rng.integers(len(histories_j)))
                        # Ensure distinct histories when within the same state
                        while idx_j == idx_i and len(histories_j) > 1:
                            idx_j = int(rng.integers(len(histories_j)))
                    else:
                        idx_j = int(rng.integers(len(histories_j)))

                    # Use conditional JS divergence
                    js_cond = js_divergence_conditional_k(model, histories_i[idx_i], histories_j[idx_j], k, platt_params)
                    js_vals.append(js_cond)

                js_array = np.array(js_vals, dtype=np.float64)

                stats = {
                    'js_values': js_array,
                    'mean': float(np.nanmean(js_array)) if js_array.size > 0 else float('nan'),
                    'median': float(np.nanmedian(js_array)) if js_array.size > 0 else float('nan'),
                    'std': float(np.nanstd(js_array)) if js_array.size > 0 else float('nan'),
                    'n_pairs': int(js_array.size),
                }

                pair_stats[(state_i, state_j)] = stats
                pair_stats[(state_j, state_i)] = stats

        results[L] = {
            'states': state_names,
            'pair_stats': pair_stats,
            'state_counts': state_counts,
        }

    return results

def plot_analysis(results: dict, save_dir: Optional[Path] = None, prefix: Optional[str] = None):
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
        fname = f"{prefix+'_ ' if prefix else ''}js_histograms.png".replace(' ', '')
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
        fname2 = f"{prefix+'_ ' if prefix else ''}js_thresholds_vs_L.png".replace(' ', '')
        out_path2 = save_dir / fname2
        fig2.savefig(out_path2, dpi=200, bbox_inches='tight')
        print(f"Saved {out_path2}")
        plt.close(fig2)
    else:
        plt.show()

def plot_instate_analysis(results: dict, save_dir: Optional[Path] = None, prefix: Optional[str] = None):
    """Visualize JS distributions for instate analysis (histories within same GT state).

    If running on a non-interactive backend or save_dir is provided, figures are saved
    to disk instead of shown. Filenames are based on prefix.
    """

    L_values = sorted(results.keys())
    if not L_values:
        print("No data to plot for instate analysis")
        return

    # Get all states that appear in the results
    all_states = set()
    for L in L_values:
        all_states.update(results[L].keys())
    all_states = sorted(all_states)

    # Plot 1: JS distributions for each state across L
    n_states = len(all_states)
    n_plots = len(L_values)
    ncols = min(3, n_states)
    nrows = int(np.ceil(n_states / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))

    if n_states == 1:
        axes = [axes] if nrows == 1 and ncols == 1 else axes.flatten()
    elif isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    state_idx = 0
    for state in all_states:
        ax = axes[state_idx] if state_idx < len(axes) else None
        if ax is None:
            continue

        # Plot JS distributions for this state across all L
        colors = plt.cm.viridis(np.linspace(0, 1, len(L_values)))

        for i, L in enumerate(L_values):
            if state in results[L] and len(results[L][state]['js_values']) > 0:
                js_vals = results[L][state]['js_values']
                ax.hist(js_vals, bins=30, alpha=0.6, color=colors[i],
                       label=f'L={L}', density=True)

                # Add threshold line
                threshold = results[L][state]['threshold']
                ax.axvline(threshold, color=colors[i], linestyle='--', linewidth=1)

        ax.set_title(f'State {state}')
        ax.set_xlabel('JS Divergence')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        state_idx += 1

    # Hide any extra axes
    for k in range(state_idx, len(axes)):
        axes[k].axis('off')

    plt.tight_layout()
    noninteractive = 'agg' in plt.get_backend().lower()
    if save_dir is not None or noninteractive:
        save_dir = Path(save_dir) if save_dir is not None else Path('.')
        save_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{prefix+'_ ' if prefix else ''}instate_js_histograms.png".replace(' ', '')
        out_path = save_dir / fname
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f"Saved {out_path}")
        plt.close(fig)
    else:
        plt.show()

    # Plot 2: Threshold vs L for each state
    if len(all_states) > 1:
        fig2, ax2 = plt.subplots(figsize=(12, 8))

        for state in all_states:
            thresholds = []
            L_vals = []
            for L in L_values:
                if state in results[L]:
                    thresholds.append(results[L][state]['threshold'])
                    L_vals.append(L)

            if thresholds:
                ax2.plot(L_vals, thresholds, 'o-', label=f'State {state}', linewidth=2)

        ax2.set_xlabel('History Length L')
        ax2.set_ylabel('JS Threshold')
        ax2.set_title('Instate Threshold vs History Length')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        if save_dir is not None or noninteractive:
            fname2 = f"{prefix+'_ ' if prefix else ''}instate_js_thresholds_vs_L.png".replace(' ', '')
            out_path2 = save_dir / fname2
            fig2.savefig(out_path2, dpi=200, bbox_inches='tight')
            print(f"Saved {out_path2}")
            plt.close(fig2)
        else:
            plt.show()

    # Plot 3: Number of histories per state vs L
    fig3, ax3 = plt.subplots(figsize=(10, 6))

    for state in all_states:
        n_histories = []
        L_vals = []
        for L in L_values:
            if state in results[L]:
                n_histories.append(results[L][state]['n_histories'])
                L_vals.append(L)

        if n_histories:
            ax3.plot(L_vals, n_histories, 'o-', label=f'State {state}', linewidth=2)

    ax3.set_xlabel('History Length L')
    ax3.set_ylabel('Number of Histories')
    ax3.set_title('Available Histories per State vs History Length')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    if save_dir is not None or noninteractive:
        fname3 = f"{prefix+'_ ' if prefix else ''}instate_histories_vs_L.png".replace(' ', '')
        out_path3 = save_dir / fname3
        fig3.savefig(out_path3, dpi=200, bbox_inches='tight')
        print(f"Saved {out_path3}")
        plt.close(fig3)
    else:
        plt.show()


def plot_cross_state_heatmaps(results: dict, metric: str = 'mean', save_dir: Optional[Path] = None,
                              prefix: Optional[str] = None):
    """Plot heatmaps of cross-state JS statistics for each history length."""

    metric = metric.lower()
    valid_metrics = {'mean', 'median'}
    if metric not in valid_metrics:
        raise ValueError(f"Unsupported metric '{metric}'. Choose from {valid_metrics}.")

    for L in sorted(results.keys()):
        info = results[L]
        states = info['states']
        n_states = len(states)
        if n_states == 0:
            continue

        matrix = np.full((n_states, n_states), np.nan, dtype=np.float64)
        for i, si in enumerate(states):
            for j, sj in enumerate(states):
                stats = info['pair_stats'].get((si, sj))
                if stats and stats['n_pairs'] > 0:
                    matrix[i, j] = stats[metric]

        finite_vals = matrix[np.isfinite(matrix)]
        if finite_vals.size == 0:
            continue
        vmin = float(np.nanmin(finite_vals))
        vmax = float(np.nanmax(finite_vals))
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-6

        fig, ax = plt.subplots(figsize=(4 + n_states, 3 + n_states))
        im = ax.imshow(matrix, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_xticks(range(n_states))
        ax.set_yticks(range(n_states))
        ax.set_xticklabels(states)
        ax.set_yticklabels(states)
        ax.set_title(f"Cross-state {metric} JS (L={L})")
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        midpoint = (vmin + vmax) / 2.0
        for i in range(n_states):
            for j in range(n_states):
                value = matrix[i, j]
                if not np.isfinite(value):
                    continue
                ax.text(j, i, f"{value:.3f}", ha='center', va='center',
                        color='white' if value > midpoint else 'black')

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=f'{metric} JS divergence')
        plt.tight_layout()

        noninteractive = 'agg' in plt.get_backend().lower()
        if save_dir is not None or noninteractive:
            save_dir = Path(save_dir) if save_dir is not None else Path('.')
            save_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{prefix+'_ ' if prefix else ''}L{L}_cross_state_{metric}.png".replace(' ', '')
            out_path = save_dir / fname
            fig.savefig(out_path, dpi=200, bbox_inches='tight')
            print(f"Saved {out_path}")
            plt.close(fig)
        else:
            plt.show()


def plot_cross_state_histograms(results: dict, save_dir: Optional[Path] = None, prefix: Optional[str] = None):
    """Plot histograms of cross-state JS distributions."""

    for L in sorted(results.keys()):
        info = results[L]
        states = info['states']
        n_states = len(states)
        if n_states == 0:
            continue

        fig, axes = plt.subplots(n_states, n_states, figsize=(3 * n_states, 2.5 * n_states))
        axes = np.array(axes)

        for i, si in enumerate(states):
            for j, sj in enumerate(states):
                ax = axes[i, j]
                if j < i:
                    ax.axis('off')
                    continue

                stats = info['pair_stats'].get((si, sj))
                if not stats or stats['n_pairs'] == 0:
                    ax.axis('off')
                    continue

                ax.hist(stats['js_values'], bins=30, density=True, alpha=0.7, color='tab:blue')
                ax.set_title(f"{si} vs {sj}")
                ax.set_xlabel('JS Divergence')
                ax.set_ylabel('Density')

        plt.tight_layout()
        noninteractive = 'agg' in plt.get_backend().lower()
        if save_dir is not None or noninteractive:
            save_dir = Path(save_dir) if save_dir is not None else Path('.')
            save_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{prefix+'_ ' if prefix else ''}L{L}_cross_state_hist.png".replace(' ', '')
            out_path = save_dir / fname
            fig.savefig(out_path, dpi=200, bbox_inches='tight')
            print(f"Saved {out_path}")
            plt.close(fig)
        else:
            plt.show()


def fit_platt_params(data: np.ndarray, model, L_max: int = 4, min_count: int = 5) -> Optional[dict]:
    """Fit Platt calibration parameters a,b using empirical next-token targets.

    We collect histories up to length L_max, compute model p1 and empirical p1,
    and solve weighted logistic regression with LBFGS.
    """
    try:
        device = next(model.parameters()).device
    except Exception:
        device = torch.device('cpu')

    counts = {}
    ones = {}
    N = len(data)
    for L in range(1, L_max + 1):
        for t in range(L, N):
            h = tuple(int(x) for x in data[t-L:t])
            counts[h] = counts.get(h, 0) + 1
            if int(data[t]) == 1:
                ones[h] = ones.get(h, 0) + 1

    histories = [h for h, c in counts.items() if c >= min_count]
    if not histories:
        return None

    margins = []
    targets = []
    weights = []
    ctx = getattr(getattr(model, 'config', None), 'block_size', None)
    for h in histories:
        ids = list(h)
        if ctx is not None and len(ids) > int(ctx):
            ids = ids[-int(ctx):]
        p = get_next_token_distribution(model, np.array(ids, dtype=np.int64), platt_params=None)
        p1 = float(p[1])
        p1c = max(1e-12, min(1.0 - 1e-12, p1))
        margins.append(np.log(p1c / (1.0 - p1c)))
        targets.append(ones.get(h, 0) / float(counts[h]))
        weights.append(float(counts[h]))

    if len(margins) == 0:
        return None

    margins_t = torch.tensor(margins, dtype=torch.float64, device=device)
    targets_t = torch.tensor(targets, dtype=torch.float64, device=device)
    weights_t = torch.tensor(weights, dtype=torch.float64, device=device)
    a = torch.tensor(1.0, dtype=torch.float64, device=device, requires_grad=True)
    b = torch.tensor(0.0, dtype=torch.float64, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([a, b], lr=0.25, max_iter=200)

    def closure():
        opt.zero_grad()
        s = a * margins_t + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(s, targets_t, weight=weights_t)
        loss.backward()
        return loss

    opt.step(closure)
    return {'a': float(a.detach().cpu().item()), 'b': float(b.detach().cpu().item())}

def compute_js_thresholds(data: np.ndarray, model, L_max: int = 10, n_samples: int = 5000,
                         platt_params: Optional[dict] = None) -> Tuple[dict, dict]:
    """Compute JS thresholds for different L values using random pairs."""
    thresholds = {}
    separations = {}

    for L in range(1, L_max + 1):
        print(f"Computing threshold for L={L}")

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
            p = get_next_token_distribution(model, subsequences[i], platt_params=platt_params)
            q = get_next_token_distribution(model, subsequences[j], platt_params=platt_params)
            js_values.append(js_divergence(p, q))

        js_values = np.array(js_values)

        # Fit GMM and get threshold
        try:
            gmm = GaussianMixture(n_components=2, random_state=42)
            gmm.fit(js_values.reshape(-1, 1))

            means = gmm.means_.flatten()
            covariances = gmm.covariances_.flatten()
            stds = np.sqrt(np.maximum(covariances, 1e-12))
            weights = gmm.weights_

            lower_idx = np.argmin(means)
            threshold = means[lower_idx] + 2 * stds[lower_idx]
        except:
            threshold = np.percentile(js_values, 75)
            means = [np.mean(js_values)] * 2
            stds = [np.std(js_values)] * 2
            weights = [0.5, 0.5]

        thresholds[L] = threshold
        separations[L] = abs(means[1] - means[0]) / (stds[0] + stds[1]) if (stds[0] + stds[1]) > 1e-12 else 0.0

    return thresholds, separations

def find_optimal_L_thresholds(thresholds: dict, stability_threshold: float = 0.1) -> int:
    """Find minimal L where threshold values stabilize (plateau)."""
    L_values = sorted(thresholds.keys())

    if len(L_values) < 3:
        return max(L_values)  # Need at least 3 points for stability check

    # Look for consecutive L values with relative change < threshold
    for i in range(2, len(L_values)):
        L_curr, L_prev1, L_prev2 = L_values[i], L_values[i-1], L_values[i-2]

        change1 = abs(thresholds[L_curr] - thresholds[L_prev1]) / max(thresholds[L_prev1], 1e-12)
        change2 = abs(thresholds[L_prev1] - thresholds[L_prev2]) / max(thresholds[L_prev2], 1e-12)

        if change1 < stability_threshold and change2 < stability_threshold:
            return L_prev1  # Return the L where stability starts

    return max(L_values)  # Return max L if no stability found

def find_optimal_L_separation(separations: dict, stability_threshold: float = 0.1) -> int:
    """Find minimal L where separation metric stabilizes."""
    L_values = sorted(separations.keys())

    if len(L_values) < 3:
        return max(L_values)

    # Look for consecutive L values with separation change < threshold
    for i in range(2, len(L_values)):
        L_curr, L_prev1, L_prev2 = L_values[i], L_values[i-1], L_values[i-2]

        change1 = abs(separations[L_curr] - separations[L_prev1]) / max(separations[L_prev1], 1e-12)
        change2 = abs(separations[L_prev1] - separations[L_prev2]) / max(separations[L_prev2], 1e-12)

        if change1 < stability_threshold and change2 < stability_threshold:
            return L_prev1

    return max(L_values)

def plot_threshold_stability(thresholds: dict, separations: dict, save_dir: Optional[Path] = None, prefix: Optional[str] = None):
    """Plot threshold and separation stability vs L for optimal L detection."""
    L_values = sorted(thresholds.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot thresholds
    ax1.plot(L_values, [thresholds[L] for L in L_values], 'o-', linewidth=2)
    ax1.set_xlabel('History Length L')
    ax1.set_ylabel('JS Threshold')
    ax1.set_title('Threshold vs History Length')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=find_optimal_L_thresholds(thresholds), color='red', linestyle='--', label='Optimal L')
    ax1.legend()

    # Plot separations
    ax2.plot(L_values, [separations[L] for L in L_values], 'o-', color='orange', linewidth=2)
    ax2.set_xlabel('History Length L')
    ax2.set_ylabel('Component Separation')
    ax2.set_title('Separation vs History Length')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=find_optimal_L_separation(separations), color='red', linestyle='--', label='Optimal L')
    ax2.legend()

    plt.tight_layout()

    if save_dir is not None:
        save_dir = Path(save_dir) if save_dir is not None else Path('.')
        save_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{prefix+'_ ' if prefix else ''}optimal_L_analysis.png".replace(' ', '')
        out_path = save_dir / fname
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f"Saved {out_path}")

    plt.show()

# Main execution
if __name__ == "__main__":
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
    repo_root = Path(__file__).resolve().parents[1]
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
