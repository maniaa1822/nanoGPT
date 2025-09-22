import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from probe_linear_state import load_checkpoint, read_bin_uint16, extract_hidden_states_prefix_only_with_hook


def load_states(path: Path) -> np.ndarray:
    chars = path.read_text().strip()
    mapping = {ch: i for i, ch in enumerate(sorted(set(chars)))}
    return np.fromiter((mapping[c] for c in chars), dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--src_data_dir', type=str, required=True)
    ap.add_argument('--src_states_dat', type=str, required=True)
    ap.add_argument('--tgt_data_dir', type=str, required=True)
    ap.add_argument('--tgt_states_dat', type=str, required=True)
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--bs', type=int, default=16384)
    ap.add_argument('--lr', type=float, default=1e-2)
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()

    model, _ = load_checkpoint(Path(args.ckpt))
    device = args.device
    model.to(device).eval()

    # Source
    src_tokens = read_bin_uint16(Path(args.src_data_dir) / 'train.bin').astype(np.int64)
    src_states = load_states(Path(args.src_states_dat))
    bs_ctx = int(model.config.block_size)
    Ls = (len(src_tokens) // bs_ctx) * bs_ctx
    src_tokens, src_states = src_tokens[:Ls], src_states[:Ls]
    X_src = extract_hidden_states_prefix_only_with_hook(model, model.transformer.ln_f, src_tokens, device)
    y_src = src_states[1:len(X_src)+1]

    # Target
    tgt_tokens = read_bin_uint16(Path(args.tgt_data_dir) / 'val.bin').astype(np.int64)
    tgt_states = load_states(Path(args.tgt_states_dat))
    Lt = (len(tgt_tokens) // bs_ctx) * bs_ctx
    tgt_tokens, tgt_states = tgt_tokens[:Lt], tgt_states[:Lt]
    X_tgt = extract_hidden_states_prefix_only_with_hook(model, model.transformer.ln_f, tgt_tokens, device)
    y_tgt = tgt_states[1:len(X_tgt)+1]

    # Train probe on source, evaluate on target
    Xs = torch.from_numpy(X_src).to(device)
    Ys = torch.from_numpy(y_src.astype(np.int64)).to(device)
    Xt = torch.from_numpy(X_tgt).to(device)
    Yt = torch.from_numpy(y_tgt.astype(np.int64)).to(device)

    n_states = int(max(Ys.max().item() if Ys.numel() else 1, Yt.max().item() if Yt.numel() else 1) + 1)
    head = nn.Linear(model.config.n_embd, n_states).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    def batches(X, y, B):
        n = X.size(0)
        for i in range(0, n, B):
            yield X[i:i+B], y[i:i+B]

    head.train()
    for _ in range(args.epochs):
        for xb, yb in batches(Xs, Ys, args.bs):
            opt.zero_grad(set_to_none=True)
            logits = head(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

    head.eval()
    with torch.no_grad():
        logits_t = []
        for xb, _ in batches(Xt, Yt, args.bs):
            logits_t.append(head(xb))
        preds_t = torch.cat(logits_t, dim=0).argmax(dim=1)
        acc_t = float((preds_t == Yt[:preds_t.size(0)]).float().mean().item())

    out = {
        'n_src_positions': int(len(y_src)),
        'n_tgt_positions': int(len(y_tgt)),
        'transfer_acc': acc_t,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(out)


if __name__ == '__main__':
    main()


