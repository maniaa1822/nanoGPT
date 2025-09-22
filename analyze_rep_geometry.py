"""
Representation geometry analysis for causal states (FER check).

Metrics (using prefix-only features to target pre-emission state s_t):
- linear probe accuracy (torch linear head)
- silhouette score (higher is better separation)
- inter/intra distance ratio (centroid distance / avg within-class std)
- participation ratio (feature covariance spread)

Usage:
  uv run python analyze_rep_geometry.py \
    --ckpt out-golden-mean-char/ckpt.pt --data_dir data/golden_mean \
    --states_dat ../experiments/datasets/golden_mean/golden_mean.states.dat \
    --layer final --device cpu
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from model import GPT
from probe_linear_state import (
    load_checkpoint,
    extract_hidden_states_prefix_only_with_hook,
    extract_hidden_states_with_hook,
)


def read_bin_uint16(path: Path) -> np.ndarray:
    return np.fromfile(str(path), dtype=np.uint16)


def compute_metrics(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    X = X.astype(np.float32)
    y = y.astype(np.int64)
    n, d = X.shape
    out: Dict[str, Any] = {'n': int(n), 'dim': int(d)}
    if n == 0:
        return out

    # silhouette (needs >1 label present)
    if len(np.unique(y)) > 1 and n >= 10:
        try:
            sil = float(silhouette_score(X, y, metric='euclidean'))
        except Exception:
            sil = None
    else:
        sil = None
    out['silhouette'] = sil

    # inter/intra ratio
    classes = np.unique(y)
    means = []
    intrav = []
    for c in classes:
        Xc = X[y == c]
        if len(Xc) == 0:
            continue
        mu = Xc.mean(axis=0)
        means.append(mu)
        intrav.append(Xc.std(axis=0).mean())
    inter = None
    if len(means) >= 2:
        # average pairwise centroid distance
        m = np.stack(means, axis=0)
        dists = []
        for i in range(len(m)):
            for j in range(i+1, len(m)):
                dists.append(np.linalg.norm(m[i] - m[j]))
        inter = float(np.mean(dists)) if dists else None
    intra = float(np.mean(intrav)) if intrav else None
    out['inter_centroid_dist'] = inter
    out['intra_avg_std'] = intra
    out['inter_over_intra'] = (inter / intra) if (inter is not None and intra and intra > 0) else None

    # participation ratio from covariance eigenvalues
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = (Xc.T @ Xc) / max(1, (n - 1))
    try:
        evals = np.linalg.eigvalsh(cov)
        evals = np.clip(evals, 0.0, None)
        s1 = float(np.sum(evals))
        s2 = float(np.sum(evals ** 2))
        pr = (s1 * s1) / s2 if s2 > 0 else None
    except Exception:
        pr = None
    out['participation_ratio'] = pr
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--data_dir', type=str, required=True)
    ap.add_argument('--states_dat', type=str, required=True)
    ap.add_argument('--layer', type=str, default='final', help="'final', integer index, or 'all'")
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--out', type=str, default=None)
    ap.add_argument('--pca_png', type=str, default=None, help='If set, save PCA scatter (2D) colored by state')
    args = ap.parse_args()

    model, margs = load_checkpoint(Path(args.ckpt))
    device = args.device
    model.to(device)

    def get_hook_module(layer_spec: str):
        if layer_spec == 'final':
            return model.transformer.ln_f
        else:
            li = int(layer_spec)
            return model.transformer.h[li]

    # data
    train_tokens = read_bin_uint16(Path(args.data_dir) / 'train.bin')
    states_chars = Path(args.states_dat).read_text().strip()
    # labels for s_t
    char_to_idx = {'A': 0, 'B': 1}
    y_all = np.fromiter((char_to_idx[c] for c in states_chars), dtype=np.int64)

    layers: List[str]
    if args.layer == 'all':
        layers = [str(i) for i in range(len(model.transformer.h))] + ['final']
    else:
        layers = [args.layer]

    results = []
    for ls in layers:
        hook_module = get_hook_module(ls)
        X = extract_hidden_states_prefix_only_with_hook(model, hook_module, train_tokens, device)
        y = y_all[1:len(X)+1]
        metrics = compute_metrics(X, y)
        result = {
            'layer': ls,
            'n_samples': metrics.get('n'),
            'emb_dim': metrics.get('dim'),
            'silhouette': metrics.get('silhouette'),
            'inter_centroid_dist': metrics.get('inter_centroid_dist'),
            'intra_avg_std': metrics.get('intra_avg_std'),
            'inter_over_intra': metrics.get('inter_over_intra'),
            'participation_ratio': metrics.get('participation_ratio'),
        }
        results.append(result)
        print(json.dumps(result, indent=2))

        if args.pca_png and ls == layers[-1]:  # draw only for the last layer analyzed
            pca = PCA(n_components=2)
            X2 = pca.fit_transform(X)
            plt.figure(figsize=(5, 4))
            mask0 = y == 0
            mask1 = y == 1
            plt.scatter(X2[mask0, 0], X2[mask0, 1], s=2, alpha=0.4, label='state 0')
            plt.scatter(X2[mask1, 0], X2[mask1, 1], s=2, alpha=0.4, label='state 1')
            plt.legend()
            plt.title(f'PCA layer={ls}')
            plt.tight_layout()
            plt.savefig(args.pca_png)

    if args.out:
        Path(args.out).write_text(json.dumps({'results': results}, indent=2))


if __name__ == '__main__':
    main()


