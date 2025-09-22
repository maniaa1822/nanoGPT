import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

from probe_linear_state import load_checkpoint, read_bin_uint16, extract_hidden_states_prefix_only_with_hook


def load_states(path: Path) -> np.ndarray:
    chars = path.read_text().strip()
    mapping = {ch: i for i, ch in enumerate(sorted(set(chars)))}
    return np.fromiter((mapping[c] for c in chars), dtype=np.int64), mapping


def transitions_from_labels(labels: np.ndarray, k: int) -> np.ndarray:
    T = np.zeros((k, k), dtype=np.float64)
    for a, b in zip(labels[:-1], labels[1:]):
        T[int(a), int(b)] += 1.0
    T = T + 1e-8
    T = T / T.sum(axis=1, keepdims=True)
    return T


def kl_rows(P: np.ndarray, Q: np.ndarray) -> float:
    eps = 1e-8
    P = np.clip(P, eps, 1.0)
    Q = np.clip(Q, eps, 1.0)
    return float(np.sum(P * (np.log(P) - np.log(Q))))


def jaccard_adj(A: np.ndarray, B: np.ndarray, thr: float = 1e-3) -> float:
    EA = (A > thr).astype(np.int32).ravel()
    EB = (B > thr).astype(np.int32).ravel()
    inter = int(np.sum(EA & EB))
    union = int(np.sum(EA | EB))
    return float(inter / max(union, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--data_dir', type=str, required=True)
    ap.add_argument('--states_dat', type=str, required=True)
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--k_clusters', type=int, default=None, help='defaults to #states')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--bs', type=int, default=16384)
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()

    model, _ = load_checkpoint(Path(args.ckpt))
    model.to(args.device).eval()

    tokens = read_bin_uint16(Path(args.data_dir) / 'train.bin').astype(np.int64)
    states_all, mapping = load_states(Path(args.states_dat))
    n_states = len(mapping)
    k = args.k_clusters or n_states

    # Align lengths and extract prefix-only features for s_t and next-state labels s_{t+1}
    bs_ctx = int(model.config.block_size)
    L = (len(tokens) // bs_ctx) * bs_ctx
    tokens = tokens[:L]
    states_all = states_all[:L]
    X = extract_hidden_states_prefix_only_with_hook(model, model.transformer.ln_f, tokens, args.device)
    s_t = states_all[1:len(X)+1]
    s_next = states_all[2:len(X)+2] if len(states_all) >= len(X) + 2 else s_t[1:]

    # Next-state linear probe
    X_t = torch.from_numpy(X[:len(s_next)]).to(args.device)
    y_next = torch.from_numpy(s_next.astype(np.int64)).to(args.device)
    head = nn.Linear(model.config.n_embd, n_states).to(args.device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2)
    crit = nn.CrossEntropyLoss()

    def batches(X, y, B):
        n = X.size(0)
        for i in range(0, n, B):
            yield X[i:i+B], y[i:i+B]

    head.train()
    for _ in range(args.epochs):
        for xb, yb in batches(X_t, y_next, args.bs):
            opt.zero_grad(set_to_none=True)
            logits = head(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

    with torch.no_grad():
        logits = []
        for xb, _ in batches(X_t, y_next, args.bs):
            logits.append(head(xb))
        preds = torch.cat(logits, dim=0).argmax(dim=1).cpu().numpy()
        next_state_acc = float((preds == y_next[:preds.shape[0]].cpu().numpy()).mean())

    # Cluster transitions and compare to true
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    c = km.fit_predict(X)[:len(s_t)]
    T_hidden = transitions_from_labels(c, k)
    T_true = transitions_from_labels(s_t, n_states)

    # Map clusters to states via Hungarian on confusion
    conf = np.zeros((k, n_states), dtype=np.int64)
    for ci, si in zip(c, s_t):
        conf[int(ci), int(si)] += 1
    row_ind, col_ind = linear_sum_assignment(-conf)

    # Build mapped transition matrix P (rows mapped to states), then map columns similarly
    P = np.zeros((n_states, k), dtype=np.float64)
    for i_c, j_s in zip(row_ind, col_ind):
        P[j_s, :] = T_hidden[i_c, :]
    Q = np.zeros((n_states, n_states), dtype=np.float64)
    for i_c, j_s in zip(row_ind, col_ind):
        Q[:, j_s] = P[:, i_c]

    mapped = Q if Q.shape == T_true.shape else Q[:, :T_true.shape[1]]
    kl = kl_rows(T_true, np.clip(mapped, 1e-8, 1.0))
    jacc = jaccard_adj(T_true, np.clip(mapped, 1e-8, 1.0))

    out = {
        'n_states': int(n_states),
        'k_clusters': int(k),
        'next_state_probe_acc': next_state_acc,
        'transition_row_kl': kl,
        'transition_graph_jaccard': jacc,
        'assignment_rows_cluster_to_state': col_ind.tolist(),
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(out)


if __name__ == '__main__':
    main()


