#!/usr/bin/env python3
"""
Recover an epsilon-machine via Predictive Partition Refinement (PPR)
using k-step predictive fingerprints from a trained binary nanoGPT model.

This implements the procedure outlined in the request:
- Build fingerprints p(·|h) for each distinct length-L history h by exact k-step chaining
- Initial radius clustering under a distance (JS or Hellinger) with tolerance tau
- Enforce unifilarity by DFA-style partition refinement on next-block consistency
- Prune unreachable and optionally merge near-identical states with identical outgoing edges

Usage example:
  python recover_ppr_machine.py \
      --preset seven_state_human --L 4 --k 6 --tau 0.12 --distance hellinger

Outputs a JSON (and optional DOT) describing the recovered unifilar machine
saved to the same directory as the model checkpoint by default.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from scipy.stats import entropy


# -------------------------
# Core probability utilities
# -------------------------

def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence between two distributions p, q.

    Returns value in [0, ln 2]. We will not rescale; thresholds should be set accordingly.
    """
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))


def hellinger_squared(p: np.ndarray, q: np.ndarray) -> float:
    """Squared Hellinger distance: H^2(p, q) = 0.5 * sum (sqrt(p_i) - sqrt(q_i))^2.

    Bounded in [0, 1]. Preferable for metric properties.
    """
    sp = np.sqrt(np.maximum(p, 0.0))
    sq = np.sqrt(np.maximum(q, 0.0))
    d = sp - sq
    return 0.5 * float(np.dot(d, d))


def get_next_token_distribution(model, history: np.ndarray) -> np.ndarray:
    """Get P(x | history) from the trained transformer (binary vocab assumed).

    Falls back to uniform if logits invalid. History is a 1-D array of ints {0,1}.
    """
    with torch.no_grad():
        ctx = getattr(getattr(model, "config", None), "block_size", None)
        if ctx is not None and len(history) > int(ctx):
            ids = history[-int(ctx):]
        else:
            ids = history

        x = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
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

        p0 = float(probs[0])
        p1 = float(probs[1])
        s = p0 + p1
        if s <= 0:
            return np.array([0.5, 0.5], dtype=np.float64)
        return np.array([p0 / s, p1 / s], dtype=np.float64)


def get_kstep_distribution(model, history: np.ndarray, k: int) -> np.ndarray:
    """Exact k-step rollout distribution over 2^k binary sequences by chaining next-token probs.

    Returns vector of length 2^k in lexicographic order (0..0, 0..1, ..., 1..1).
    """
    if k <= 0:
        return np.array([1.0], dtype=np.float64)

    num_paths = 1 << k
    probs = np.zeros(num_paths, dtype=np.float64)

    initial_hist_tuple = tuple(int(x) for x in history.tolist())
    # stack entries: (history_tuple, depth, log_prob, index_prefix)
    stack: List[Tuple[Tuple[int, ...], int, float, int]] = [(initial_hist_tuple, 0, 0.0, 0)]

    while stack:
        hist_tuple, depth, logp, idx_prefix = stack.pop()
        if depth == k:
            probs[idx_prefix] = np.exp(logp)
            continue

        p = get_next_token_distribution(model, np.array(hist_tuple, dtype=np.int64))
        p0, p1 = float(p[0]), float(p[1])

        # Append 0 then 1
        stack.append((hist_tuple + (0,), depth + 1, logp + np.log(max(1e-12, p0)), (idx_prefix << 1) | 0))
        stack.append((hist_tuple + (1,), depth + 1, logp + np.log(max(1e-12, p1)), (idx_prefix << 1) | 1))

    s = probs.sum()
    if not np.isfinite(s) or s <= 0:
        return np.full(num_paths, 1.0 / num_paths, dtype=np.float64)
    return probs / s


# -------------------------
# Data extraction utilities
# -------------------------

def extract_unique_histories(data: np.ndarray, L: int) -> List[Tuple[int, ...]]:
    """Return distinct length-L histories observed in the sequence as tuples of ints."""
    if L <= 0:
        return [tuple()] if len(data) > 0 else [tuple()]
    H: Set[Tuple[int, ...]] = set()
    for t in range(L, len(data) + 1):
        h = tuple(int(x) for x in data[t - L : t])
        H.add(h)
    return sorted(H)


def shift_append(history: Tuple[int, ...], bit: int) -> Tuple[int, ...]:
    """Drop first bit of history and append bit (0 or 1)."""
    if not history:
        return (bit,)
    return history[1:] + (int(bit),)


# -------------------------
# PPR core structures
# -------------------------

DistanceFn = str  # one of {"js", "hellinger"}


def distribution_distance(p: np.ndarray, q: np.ndarray, metric: DistanceFn) -> float:
    if metric == "js":
        return float(js_divergence(p, q))
    elif metric == "hellinger":
        return float(hellinger_squared(p, q))
    else:
        raise ValueError(f"Unknown distance metric: {metric}")


@dataclass
class StateBlock:
    """A block of histories representing a predictive state."""

    histories: List[Tuple[int, ...]]
    centroid: np.ndarray  # average fingerprint (probability vector over 2^k paths)

    def to_dict(self) -> dict:
        return {
            "histories": ["".join(str(b) for b in h) for h in self.histories],
            "centroid": self.centroid.tolist(),
        }


def compute_fingerprints(
    model,
    histories: Sequence[Tuple[int, ...]],
    k: int,
) -> Dict[Tuple[int, ...], np.ndarray]:
    """Compute model-based k-step fingerprints for each history."""
    F: Dict[Tuple[int, ...], np.ndarray] = {}
    for h in histories:
        p = get_kstep_distribution(model, np.array(h, dtype=np.int64), k)
        F[h] = p
    return F


def average_distribution(vectors: List[np.ndarray]) -> np.ndarray:
    if not vectors:
        raise ValueError("Cannot average empty list of vectors")
    m = np.mean(np.stack(vectors, axis=0), axis=0)
    s = m.sum()
    if s <= 0:
        return np.full_like(m, 1.0 / len(m))
    return m / s


def radius_cluster(
    histories: Sequence[Tuple[int, ...]],
    F: Dict[Tuple[int, ...], np.ndarray],
    tau: float,
    metric: DistanceFn,
) -> List[StateBlock]:
    """Simple radius clustering: assign to first block with centroid within tau; else new block."""
    blocks: List[StateBlock] = []
    for h in histories:
        fh = F[h]
        placed = False
        for blk in blocks:
            d = distribution_distance(fh, blk.centroid, metric)
            if d <= tau:
                blk.histories.append(h)
                blk.centroid = average_distribution([blk.centroid, fh])
                placed = True
                break
        if not placed:
            blocks.append(StateBlock(histories=[h], centroid=fh.copy()))
    return blocks


def recluster_block(
    block_histories: Sequence[Tuple[int, ...]],
    F: Dict[Tuple[int, ...], np.ndarray],
    tau: float,
    metric: DistanceFn,
) -> List[StateBlock]:
    return radius_cluster(block_histories, F, tau, metric)


def build_history_to_block_map(blocks: List[StateBlock]) -> Dict[Tuple[int, ...], int]:
    h2b: Dict[Tuple[int, ...], int] = {}
    for i, blk in enumerate(blocks):
        for h in blk.histories:
            h2b[h] = i
    return h2b


def refine_unifilar(
    blocks: List[StateBlock],
    F: Dict[Tuple[int, ...], np.ndarray],
    tau: float,
    metric: DistanceFn,
    all_histories: Set[Tuple[int, ...]],
) -> List[StateBlock]:
    """Refine partition to enforce unifilarity by splitting blocks that send symbol a to multiple next-blocks.

    We ignore successors that are not observed in all_histories (rare near sequence edges).
    """
    changed = True
    while changed:
        changed = False
        new_blocks: List[StateBlock] = []
        h2b = build_history_to_block_map(blocks)

        for blk in blocks:
            # Attempt to split by next-block consistency for a in {0,1}
            split_done = False
            for a in (0, 1):
                buckets: Dict[Optional[int], List[Tuple[int, ...]]] = {}
                for h in blk.histories:
                    hn = shift_append(h, a)
                    if hn not in all_histories:
                        # ignore unseen successor; place into a None bucket but do not cause split
                        continue
                    nb = h2b.get(hn, None)
                    buckets.setdefault(nb, []).append(h)

                # Consider only buckets with observed successors
                buckets = {k: v for k, v in buckets.items() if k is not None}
                if len(buckets) > 1:
                    # Split block into groups keyed by next-block id under symbol a
                    for _, hgroup in buckets.items():
                        new_blocks.extend(recluster_block(hgroup, F, tau, metric))
                    changed = True
                    split_done = True
                    break  # move to next original block

            if not split_done:
                new_blocks.append(blk)

        blocks = new_blocks
    return blocks


def compute_transitions(
    blocks: List[StateBlock], all_histories: Set[Tuple[int, ...]]
) -> Dict[Tuple[int, int], Optional[int]]:
    """Compute deterministic transitions delta(state_id, a) -> state_id or None if unknown.

    After refinement, transitions should be consistent for all h in a block. We compute via majority vote
    over observed successors to be robust to edge cases.
    """
    h2b = build_history_to_block_map(blocks)
    delta: Dict[Tuple[int, int], Optional[int]] = {}

    for i, blk in enumerate(blocks):
        for a in (0, 1):
            counts: Dict[int, int] = {}
            for h in blk.histories:
                hn = shift_append(h, a)
                if hn in all_histories and hn in h2b:
                    nb = h2b[hn]
                    counts[nb] = counts.get(nb, 0) + 1
            if counts:
                nb_final = max(counts.items(), key=lambda kv: kv[1])[0]
            else:
                nb_final = None
            delta[(i, a)] = nb_final

    return delta


def prune_unreachable(
    blocks: List[StateBlock],
    delta: Dict[Tuple[int, int], Optional[int]],
    data: np.ndarray,
    L: int,
) -> Tuple[List[StateBlock], Dict[Tuple[int, int], Optional[int]]]:
    """Prune states that are never visited by observed L-histories in the data.

    We mark any block containing an observed history as visited; others are dropped.
    """
    h2b = build_history_to_block_map(blocks)
    visited: Set[int] = set()
    for t in range(L, len(data) + 1):
        h = tuple(int(x) for x in data[t - L : t])
        b = h2b.get(h)
        if b is not None:
            visited.add(b)

    # remap indices
    keep_indices = sorted(visited)
    old_to_new = {old: new for new, old in enumerate(keep_indices)}

    new_blocks = [blocks[i] for i in keep_indices]
    new_delta: Dict[Tuple[int, int], Optional[int]] = {}
    for (old_i, a), nb in delta.items():
        if old_i not in old_to_new:
            continue
        new_i = old_to_new[old_i]
        new_nb = old_to_new[nb] if (nb is not None and nb in old_to_new) else None
        new_delta[(new_i, a)] = new_nb

    return new_blocks, new_delta


def merge_equivalent_states(
    blocks: List[StateBlock],
    delta: Dict[Tuple[int, int], Optional[int]],
    tau: float,
    metric: DistanceFn,
) -> Tuple[List[StateBlock], Dict[Tuple[int, int], Optional[int]]]:
    """Merge blocks whose centroids are within tau and whose outgoing transitions are identical.

    Iterates until no merges occur.
    """
    changed = True
    while changed:
        changed = False
        n = len(blocks)
        # Build transition signatures
        sigs = [(delta.get((i, 0)), delta.get((i, 1))) for i in range(n)]

        merged_parent = list(range(n))

        def find(x: int) -> int:
            while merged_parent[x] != x:
                merged_parent[x] = merged_parent[merged_parent[x]]
                x = merged_parent[x]
            return x

        def union(a: int, b: int):
            ra, rb = find(a), find(b)
            if ra != rb:
                merged_parent[rb] = ra

        for i in range(n):
            for j in range(i + 1, n):
                if sigs[i] != sigs[j]:
                    continue
                d = distribution_distance(blocks[i].centroid, blocks[j].centroid, metric)
                if d <= tau:
                    union(i, j)

        # Build new groups
        groups: Dict[int, List[int]] = {}
        for i in range(n):
            r = find(i)
            groups.setdefault(r, []).append(i)

        if len(groups) < n:
            changed = True
            # Merge
            new_blocks: List[StateBlock] = []
            old_to_new: Dict[int, int] = {}
            for new_idx, members in enumerate(groups.values()):
                hist: List[Tuple[int, ...]] = []
                vecs: List[np.ndarray] = []
                for m in members:
                    hist.extend(blocks[m].histories)
                    vecs.append(blocks[m].centroid)
                    old_to_new[m] = new_idx
                centroid = average_distribution(vecs)
                new_blocks.append(StateBlock(histories=hist, centroid=centroid))

            # Remap delta
            new_delta: Dict[Tuple[int, int], Optional[int]] = {}
            for (i, a), nb in delta.items():
                if i not in old_to_new:
                    continue
                ni = old_to_new[i]
                nnb = old_to_new[nb] if (nb is not None and nb in old_to_new) else None
                new_delta[(ni, a)] = nnb

            blocks = new_blocks
            delta = new_delta

    return blocks, delta


# -------------------------
# Serialization
# -------------------------

def machine_to_json(
    blocks: List[StateBlock], delta: Dict[Tuple[int, int], Optional[int]], L: int, k: int, metric: DistanceFn, tau: float
) -> dict:
    states = []
    for i, blk in enumerate(blocks):
        states.append(
            {
                "id": i,
                "histories": ["".join(str(b) for b in h) for h in blk.histories],
                "centroid": blk.centroid.tolist(),
            }
        )
    transitions = []
    for (i, a), nb in sorted(delta.items()):
        transitions.append({"from": i, "symbol": a, "to": nb})

    return {
        "meta": {
            "L": L,
            "k": k,
            "distance": metric,
            "tau": tau,
        },
        "num_states": len(blocks),
        "states": states,
        "transitions": transitions,
    }


def machine_to_dot(blocks: List[StateBlock], delta: Dict[Tuple[int, int], Optional[int]]) -> str:
    lines = ["digraph EpsilonMachine {", "  rankdir=LR;", "  node [shape=circle];"]
    for i, _ in enumerate(blocks):
        lines.append(f"  S{i} [label=\"S{i}\"];")
    for (i, a), nb in delta.items():
        if nb is None:
            continue
        lines.append(f"  S{i} -> S{nb} [label=\"{a}\"];")
    lines.append("}")
    return "\n".join(lines)


# -------------------------
# CLI
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Recover epsilon-machine via PPR using nanoGPT binary model")
    parser.add_argument("--preset", type=str, choices=["golden_mean", "seven_state_human"], default="golden_mean")
    parser.add_argument("--model_ckpt", type=str, help="Override model checkpoint path")
    parser.add_argument("--data", type=str, help="Override dataset path (.dat with 0/1 chars)")
    parser.add_argument("--L", type=int, default=4, help="History length")
    parser.add_argument("--k", type=int, default=6, help="Future rollout length")
    parser.add_argument("--tau", type=float, default=0.12, help="Tolerance for clustering/refinement")
    parser.add_argument("--distance", type=str, choices=["js", "hellinger"], default="hellinger")
    parser.add_argument("--write_dot", action="store_true", help="Also write a DOT graph file")
    args = parser.parse_args()

    # Resolve default paths like in plot_js_vs_L.py
    repo_root = Path(__file__).resolve().parents[1]
    default_model = repo_root / "nanoGPT" / "out-golden-mean-char" / "ckpt.pt"
    default_data = repo_root / "experiments" / "datasets" / "golden_mean" / "golden_mean.dat"
    if args.preset == "seven_state_human":
        default_model = repo_root / "nanoGPT" / "out-seven-state-char" / "ckpt.pt"
        default_data = (
            repo_root / "experiments" / "datasets" / "seven_state_human" / "seven_state_human.dat"
        )

    model_ckpt = Path(args.model_ckpt) if args.model_ckpt is not None else default_model
    data_path = Path(args.data) if args.data is not None else default_data

    # Import model/data loader from transcssr_neural_runner (kept consistent with plot_js_vs_L)
    import sys as _sys

    if str(repo_root) not in _sys.path:
        _sys.path.insert(0, str(repo_root))
    from transcssr_neural_runner import _load_nano_gpt_model, load_binary_string  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = _load_nano_gpt_model(model_ckpt, device)

    s = load_binary_string(data_path)
    data = np.array([int(c) for c in s], dtype=np.int64)
    print(f"Loaded data len={len(data)} from {data_path}")

    L = int(args.L)
    k = int(args.k)
    tau = float(args.tau)
    metric: DistanceFn = str(args.distance)

    # Step 1: histories and fingerprints
    histories = extract_unique_histories(data, L)
    Hset = set(histories)
    print(f"Distinct histories (L={L}): {len(histories)}")
    F = compute_fingerprints(model, histories, k)

    # Step 2: initial radius clustering
    blocks = radius_cluster(histories, F, tau, metric)
    print(f"Initial blocks: {len(blocks)}")

    # Step 3: refinement for unifilarity
    blocks = refine_unifilar(blocks, F, tau, metric, Hset)
    print(f"Blocks after refinement: {len(blocks)}")

    # Transitions
    delta = compute_transitions(blocks, Hset)

    # Step 4: prune by observed visits and then merge equivalents
    blocks, delta = prune_unreachable(blocks, delta, data, L)
    blocks, delta = merge_equivalent_states(blocks, delta, tau, metric)
    print(f"Final states: {len(blocks)}")

    # Save outputs
    out_dir = model_ckpt.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{args.preset}_PPR_L{L}_k{k}_{metric}_tau{tau:.2f}".replace(".", "p")
    json_path = out_dir / f"{base}.json"
    dot_path = out_dir / f"{base}.dot"

    payload = machine_to_json(blocks, delta, L=L, k=k, metric=metric, tau=tau)
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved machine JSON: {json_path}")

    if args.write_dot:
        dot = machine_to_dot(blocks, delta)
        with open(dot_path, "w") as f:
            f.write(dot)
        print(f"Saved DOT: {dot_path}")


if __name__ == "__main__":
    main()


