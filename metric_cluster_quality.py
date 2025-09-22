import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, v_measure_score

from probe_linear_state import load_checkpoint, read_bin_uint16, extract_hidden_states_prefix_only_with_hook


def load_states(path: Path) -> np.ndarray:
    chars = path.read_text().strip()
    mapping = {ch: i for i, ch in enumerate(sorted(set(chars)))}
    return np.fromiter((mapping[c] for c in chars), dtype=np.int64)


def purity_score(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    # contingency matrix
    k = int(max(pred_labels.max() if len(pred_labels) else 0, 0) + 1)
    s = int(max(true_labels.max() if len(true_labels) else 0, 0) + 1)
    M = np.zeros((k, s), dtype=np.int64)
    for p, t in zip(pred_labels, true_labels):
        M[int(p), int(t)] += 1
    return float(np.sum(M.max(axis=1)) / max(len(true_labels), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--data_dir', type=str, required=True)
    ap.add_argument('--states_dat', type=str, required=True)
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--k', type=int, default=None, help='number of clusters; default: #states')
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()

    model, _ = load_checkpoint(Path(args.ckpt))
    model.to(args.device).eval()

    tokens = read_bin_uint16(Path(args.data_dir) / 'train.bin').astype(np.int64)
    states = load_states(Path(args.states_dat))

    bs_ctx = int(model.config.block_size)
    L = (len(tokens) // bs_ctx) * bs_ctx
    tokens = tokens[:L]
    states = states[:L]

    X = extract_hidden_states_prefix_only_with_hook(model, model.transformer.ln_f, tokens, args.device)
    y = states[1:len(X)+1]

    n_states = int(max(y.max() if len(y) else 0, 0) + 1)
    k = args.k or n_states

    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    c = km.fit_predict(X)

    purity = purity_score(y, c)
    nmi = float(normalized_mutual_info_score(y, c)) if len(np.unique(y)) > 1 else None
    vmeas = float(v_measure_score(y, c)) if len(np.unique(y)) > 1 else None

    out = {
        'n_samples': int(len(y)),
        'n_states': int(n_states),
        'k': int(k),
        'purity': float(purity),
        'nmi': nmi,
        'v_measure': vmeas,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(out)


if __name__ == '__main__':
    main()


