import os
import math
import argparse
import pickle
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn

from model import GPT
from analyze_logits import get_next_log_probs
from probe_linear_state import load_checkpoint, read_bin_uint16, extract_hidden_states_prefix_only_with_hook


def load_meta(data_dir: str):
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    return meta


@torch.no_grad()
def eval_model_full_context_nll(model: GPT, device: str, tokens: np.ndarray, stride: int = 10) -> float:
    nll_sum, count = 0.0, 0
    for i in range(1, len(tokens), stride):
        context = tokens[max(0, i - model.config.block_size):i]
        target = int(tokens[i])
        if len(context) < 1:
            continue
        logprobs = get_next_log_probs(model, device, context.tolist())
        nll_sum += (-float(logprobs[target]))
        count += 1
    return (nll_sum / max(count, 1)), count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--train_data_dir', type=str, required=True)
    ap.add_argument('--val_data_dir', type=str, required=True)
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--bs', type=int, default=16384)
    ap.add_argument('--lr', type=float, default=1e-2)
    ap.add_argument('--stride', type=int, default=10)
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()

    model, _ = load_checkpoint(Path(args.ckpt))
    device = args.device
    model.to(device).eval()

    # Load datasets
    _ = load_meta(args.train_data_dir)
    meta_va = load_meta(args.val_data_dir)
    vocab_size = int(meta_va['vocab_size'])
    train_tokens = read_bin_uint16(Path(args.train_data_dir) / 'train.bin').astype(np.int64)
    val_tokens = read_bin_uint16(Path(args.val_data_dir) / 'val.bin').astype(np.int64)

    # Trim to multiples of block_size for efficient chunking
    bs_ctx = int(model.config.block_size)
    def trim(x: np.ndarray) -> np.ndarray:
        return x[: (len(x) // bs_ctx) * bs_ctx]
    train_tokens = trim(train_tokens)
    val_tokens = trim(val_tokens)

    # Extract prefix-only features for h_t
    X_tr = extract_hidden_states_prefix_only_with_hook(model, model.transformer.ln_f, train_tokens, device)
    X_va = extract_hidden_states_prefix_only_with_hook(model, model.transformer.ln_f, val_tokens, device)

    # Next-token labels aligned to positions 1..len(X)
    y_tr = train_tokens[1:len(X_tr)+1]
    y_va = val_tokens[1:len(X_va)+1]

    # Train linear head: p(x_{t+1} | h_t)
    Xtr = torch.from_numpy(X_tr).to(device)
    Xva = torch.from_numpy(X_va).to(device)
    ytr = torch.from_numpy(y_tr.astype(np.int64)).to(device)
    yva = torch.from_numpy(y_va.astype(np.int64)).to(device)

    head = nn.Linear(model.config.n_embd, vocab_size).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    def batches(X: torch.Tensor, y: torch.Tensor, B: int):
        n = X.size(0)
        for i in range(0, n, B):
            yield X[i:i+B], y[i:i+B]

    head.train()
    for _ in range(args.epochs):
        for xb, yb in batches(Xtr, ytr, args.bs):
            opt.zero_grad(set_to_none=True)
            logits = head(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

    @torch.no_grad()
    def avg_nll_head(X: torch.Tensor, y: torch.Tensor) -> float:
        head.eval()
        nll_sum, count = 0.0, 0
        for xb, yb in batches(X, y, args.bs):
            logits = head(xb)
            logp = logits.log_softmax(dim=-1)
            nll_sum += float((-logp.gather(1, yb.view(-1, 1)).squeeze(1)).sum().item())
            count += int(yb.numel())
        return nll_sum / max(count, 1)

    nll_head = avg_nll_head(Xva, yva)
    nll_full, ctx_count = eval_model_full_context_nll(model, device, val_tokens, stride=args.stride)

    out = {
        'vocab_size': vocab_size,
        'n_val_positions_head': int(len(y_va)),
        'n_val_positions_full_ctx': int(ctx_count),
        'avg_nll_head_nats': float(nll_head),
        'avg_nll_fullctx_nats': float(nll_full),
        'gap_bits': float((nll_head - nll_full) / math.log(2)),
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(out)


if __name__ == '__main__':
    main()


