import os
import argparse
import pickle
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

from model import GPT
from probe_linear_state import (
    load_checkpoint,
    read_bin_uint16,
    extract_hidden_states_prefix_only_with_hook,
)
import rep_viz as rv


def load_meta(data_dir: str):
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        return pickle.load(f)


def load_states(states_dat: str) -> Tuple[np.ndarray, dict]:
    chars = Path(states_dat).read_text().strip()
    mapping = {ch: i for i, ch in enumerate(sorted(set(chars)))}
    states = np.fromiter((mapping[c] for c in chars), dtype=np.int64)
    return states, mapping


@torch.no_grad()
def extract_prefix_features_and_nextp(model: GPT,
                                      tokens: np.ndarray,
                                      device: str,
                                      idx_one: Optional[int] = None,
                                      max_positions: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      H: (N, d) prefix-only hidden features aligned to s_t (positions 1..)
      p1: (N,) probability of token '1' (or of idx_one) for next token at each position
    Alignment mirrors extract_hidden_states_prefix_only_with_hook.
    """
    model.eval()
    block_size = int(model.config.block_size)

    feat_list: list[torch.Tensor] = []
    prob_list: list[torch.Tensor] = []

    captured = {}

    def hook_fn(module, inp, out):
        captured['h'] = out.detach().cpu()  # (1, T, C)

    handle = model.transformer.ln_f.register_forward_hook(hook_fn)
    prev_last_h: Optional[torch.Tensor] = None
    prev_last_p: Optional[torch.Tensor] = None
    try:
        # iterate contiguous chunks
        limit = len(tokens) - (len(tokens) % block_size)
        for i in range(0, limit, block_size):
            chunk = torch.from_numpy(tokens[i:i + block_size].astype(np.int64)).unsqueeze(0).to(device)
            # full-position logits by passing dummy targets
            logits, _ = model(chunk, targets=chunk)
            logits = logits.squeeze(0).cpu()  # (T, V)
            if idx_one is not None:
                p = F.softmax(logits, dim=-1)[:, idx_one]  # (T,)
            else:
                # fallback: assume binary vocab and index 1 is token '1'
                p = F.softmax(logits, dim=-1)[:, 1]
            _ = model(chunk)  # to populate captured['h']
            h = captured['h'].squeeze(0)  # (T, C)

            if prev_last_h is not None and prev_last_p is not None:
                feat_list.append(prev_last_h.unsqueeze(0))
                prob_list.append(prev_last_p.view(1, 1))
            if h.size(0) > 1:
                feat_list.append(h[:-1, :])
                prob_list.append(p[:-1].unsqueeze(1))
            prev_last_h = h[-1, :]
            prev_last_p = p[-1]

            if max_positions is not None:
                cur_n = sum(x.size(0) for x in feat_list)
                if cur_n >= max_positions:
                    break

        if prev_last_h is not None and prev_last_p is not None:
            feat_list.append(prev_last_h.unsqueeze(0))
            prob_list.append(prev_last_p.view(1, 1))
    finally:
        handle.remove()

    if not feat_list:
        return np.empty((0, int(model.config.n_embd)), dtype=np.float32), np.empty((0,), dtype=np.float32)

    H = torch.cat(feat_list, dim=0).numpy().astype(np.float32)
    p1 = torch.cat(prob_list, dim=0).squeeze(1).numpy().astype(np.float32)
    return H, p1


def build_true_transition(states: np.ndarray, S: int) -> np.ndarray:
    T = np.zeros((S, S), dtype=np.float64)
    for a, b in zip(states[:-1], states[1:]):
        T[int(a), int(b)] += 1.0
    T = T + 1e-8
    T = T / T.sum(axis=1, keepdims=True)
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--data_dir', type=str, required=True)
    ap.add_argument('--states_dat', type=str, required=True)
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--out_dir', type=str, required=True)
    ap.add_argument('--embed', type=str, default='pca', choices=['pca', 'umap', 'tsne'])
    ap.add_argument('--sample', type=int, default=5000)
    ap.add_argument('--traj_len', type=int, default=200)
    ap.add_argument('--layers', type=str, default='final', help="Comma-separated list of layers: 'final' or integer indices")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    model, _ = load_checkpoint(Path(args.ckpt))
    model.to(args.device).eval()

    # data
    tokens = read_bin_uint16(Path(args.data_dir) / 'train.bin').astype(np.int64)
    meta = load_meta(args.data_dir)
    stoi = meta.get('stoi', None)
    idx_one = int(stoi['1']) if (stoi and '1' in stoi) else 1

    # states
    states_all, mapping = load_states(args.states_dat)

    # trim to block multiple
    bs = int(model.config.block_size)
    L = (len(tokens) // bs) * bs
    tokens = tokens[:L]
    states_all = states_all[:L]

    # extract features and aligned next-prob (final layer)
    H, p1 = extract_prefix_features_and_nextp(model, tokens, args.device, idx_one=idx_one, max_positions=None)
    # Align lengths: prefix-only features should align to s_t (positions 1..L-1). If we produced L, drop first.
    if H.shape[0] == len(tokens):
        H = H[1:, :]
        p1 = p1[1:]
    s = states_all[1:1 + len(H)]

    # scatter with next-prob overlay
    rv.plot_state_scatter(H, s, next_probs=p1, title='State scatter with P(1) overlay', embed_method=args.embed, sample=args.sample)
    Path(os.path.join(args.out_dir, 'viz_scatter.png')).parent.mkdir(parents=True, exist_ok=True)
    plt_path = os.path.join(args.out_dir, 'viz_scatter.png')
    import matplotlib.pyplot as plt
    plt.savefig(plt_path, dpi=150)
    plt.close()

    # centroid graph overlay using true transitions
    S = int(s.max()) + 1
    T_true = build_true_transition(s, S)
    rv.plot_centroid_graph(H, s, T_true, embed_method='pca')
    plt_path = os.path.join(args.out_dir, 'viz_centroids.png')
    plt.savefig(plt_path, dpi=150)
    plt.close()

    # trajectory plot from first traj_len positions with token annotations (next token at each t)
    H_seq = H[: args.traj_len]
    s_seq = s[: args.traj_len]
    tok_seq = tokens[1:1 + len(H_seq)]
    rv.plot_hidden_trajectory(H_seq, s_seq, tokens_seq=tok_seq, embed_method='pca')
    plt_path = os.path.join(args.out_dir, 'viz_trajectory.png')
    plt.savefig(plt_path, dpi=150)
    plt.close()

    # save arrays for later interactive use (final layer)
    np.savez_compressed(os.path.join(args.out_dir, 'viz_cache.npz'), H=H, states=s, p1=p1)

    # Optionally extract features for additional layers and save per-layer caches
    layers = [ls.strip() for ls in args.layers.split(',') if ls.strip()]
    def get_hook_module(layer_spec: str):
        if layer_spec == 'final':
            return model.transformer.ln_f
        else:
            li = int(layer_spec)
            return model.transformer.h[li]
    for ls in layers:
        try:
            hook_module = get_hook_module(ls)
        except Exception:
            continue
        X = extract_hidden_states_prefix_only_with_hook(model, hook_module, tokens, args.device)
        if X.shape[0] == len(tokens):
            X = X[1:, :]
        s_l = states_all[1:1 + len(X)]
        outp = os.path.join(args.out_dir, f'viz_cache_layer_{ls}.npz')
        np.savez_compressed(outp, H=X.astype(np.float32), states=s_l)

    print({
        'H_shape': H.shape,
        'states_shape': s.shape,
        'p1_shape': p1.shape,
        'scatter_png': os.path.abspath(os.path.join(args.out_dir, 'viz_scatter.png')),
        'centroids_png': os.path.abspath(os.path.join(args.out_dir, 'viz_centroids.png')),
        'trajectory_png': os.path.abspath(os.path.join(args.out_dir, 'viz_trajectory.png')),
    })


if __name__ == '__main__':
    main()


