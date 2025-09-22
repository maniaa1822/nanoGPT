"""
Linear probe for causal states on a NanoGPT checkpoint (Golden Mean).

Loads:
- Model checkpoint from out-golden-mean-char/ckpt.pt (override with --ckpt)
- Tokenized train/val from nanoGPT/data/golden_mean/{train,val}.bin
- Ground-truth states from notebook_experiments/.../golden_mean.states.dat

Extracts final hidden states (after ln_f) per token using non-overlapping
windows of size block_size and trains a Logistic Regression to predict
the causal state (A/B) at each position. Reports train/val accuracy.
"""
import argparse
import os
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import GPT, GPTConfig


def find_states_dat() -> Path:
    here = Path(__file__).parent
    repo_root = (here / '..').resolve()
    candidates = [
        repo_root / 'notebook_experiments' / 'golden_mean' / 'data' / 'golden_mean' / 'golden_mean.states.dat',
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("golden_mean.states.dat not found under experiments/datasets/golden_mean")


def load_checkpoint(ckpt_path: Path) -> Tuple[GPT, dict]:
    ckpt = torch.load(str(ckpt_path), map_location='cpu')
    args = ckpt['model_args']
    config = GPTConfig(
        block_size=args['block_size'],
        vocab_size=args['vocab_size'],
        n_layer=args['n_layer'],
        n_head=args['n_head'],
        n_embd=args['n_embd'],
        dropout=args.get('dropout', 0.0),
        bias=args.get('bias', True),
        state_head_classes=args.get('state_head_classes', 0),
        state_loss_weight=args.get('state_loss_weight', 0.0),
    )
    model = GPT(config)
    state_dict = ckpt['model']
    # handle checkpoints saved under torch.compile / DataParallel that prefix keys with "_orig_mod."
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model, args


def read_bin_uint16(path: Path) -> np.ndarray:
    return np.fromfile(str(path), dtype=np.uint16)


def extract_hidden_states_with_hook(model: GPT, hook_module: torch.nn.Module, token_ids: np.ndarray, device: str) -> np.ndarray:
    block_size = model.config.block_size
    embd = model.config.n_embd
    tokens = torch.from_numpy(token_ids.astype(np.int64))
    features: List[torch.Tensor] = []

    # Hook to capture final hidden states (after ln_f) per forward
    captured = {}
    def hook_fn(module, inp, out):
        captured['h'] = out.detach().cpu()

    handle = hook_module.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            for i in range(0, len(tokens) - block_size + 1, block_size):
                chunk = tokens[i:i + block_size].unsqueeze(0)  # (1, T)
                chunk = chunk.to(device)
                _ = model(chunk)  # logits unused
                h = captured['h']  # (1, T, C)
                features.append(h.squeeze(0))
    finally:
        handle.remove()

    if not features:
        return np.empty((0, embd), dtype=np.float32)
    H = torch.cat(features, dim=0)  # (N, C)
    return H.numpy().astype(np.float32)


def extract_hidden_states_prefix_only_with_hook(model: GPT, hook_module: torch.nn.Module, token_ids: np.ndarray, device: str) -> np.ndarray:
    """
    Extract features for pre-emission state s_t by only feeding prefix up to t-1.
    Efficiently implemented by chunking and shifting hidden states by one within each chunk
    and carrying over the last hidden of previous chunk to cover boundaries.
    Returns features for positions t = 1..L (aligned to tokens positions), length L-1.
    """
    block_size = model.config.block_size
    embd = model.config.n_embd
    tokens = torch.from_numpy(token_ids.astype(np.int64))

    captured = {}
    def hook_fn(module, inp, out):
        captured['h'] = out.detach().cpu()  # (1, T, C)

    handle = hook_module.register_forward_hook(hook_fn)
    features: List[torch.Tensor] = []
    prev_last = None
    try:
        with torch.no_grad():
            for i in range(0, len(tokens) - block_size + 1, block_size):
                chunk = tokens[i:i + block_size].unsqueeze(0).to(device)
                _ = model(chunk)
                h = captured['h'].squeeze(0)  # (T, C)
                if prev_last is not None:
                    features.append(prev_last.unsqueeze(0))  # boundary t = i
                # add shifted within-chunk: positions t = i+1 .. i+block_size-1
                if h.size(0) > 1:
                    features.append(h[:-1, :])
                prev_last = h[-1, :]
        if prev_last is not None:
            # final boundary t = last_index
            features.append(prev_last.unsqueeze(0))
    finally:
        handle.remove()

    if not features:
        return np.empty((0, embd), dtype=np.float32)
    H = torch.cat(features, dim=0)  # expected length L-1 (plus last boundary makes L?)
    # We produced (num_chunks*block_size - 1) + 1 = num_chunks*block_size = L features; but s_0 has no prefix, drop first
    if H.size(0) == len(token_ids):
        H = H[1:, :]
    return H.numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=str(Path(__file__).parent / 'out-golden-mean-char' / 'ckpt.pt'))
    parser.add_argument('--data_dir', type=str, default=str(Path(__file__).parent / 'data' / 'golden_mean'))
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--batch_size', type=int, default=8192)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--target_state', type=str, default='pre', choices=['pre', 'next_from_states', 'next_from_symbol'],
                        help='Which labels to use: pre-emission state, next state via .states, or next state derived from symbol')
    parser.add_argument('--class_weight', action='store_true', help='Use inverse-frequency class weights')
    parser.add_argument('--states_dat', type=str, default=None, help='Optional explicit path to GT states .dat file')
    parser.add_argument('--layer', type=str, default='final', help="Feature layer: 'final' or integer index (0-based) of transformer block")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    data_dir = Path(args.data_dir)
    states_dat_path = Path(args.states_dat) if args.states_dat else find_states_dat()

    model, margs = load_checkpoint(ckpt_path)
    device = args.device
    model.to(device)

    # load tokens
    train_tokens = read_bin_uint16(data_dir / 'train.bin')
    val_tokens = read_bin_uint16(data_dir / 'val.bin')

    # labels per config
    states_chars = states_dat_path.read_text().strip()
    n_all = len(states_chars)
    n_train = int(n_all * 0.9)
    if args.target_state == 'pre':
        # direct pre-emission state labels
        char_to_idx = {'A': 0, 'B': 1}
        train_labels = np.fromiter((char_to_idx[c] for c in states_chars[:n_train]), dtype=np.int64)
        val_labels = np.fromiter((char_to_idx[c] for c in states_chars[n_train:]), dtype=np.int64)
    elif args.target_state == 'next_from_states':
        # shift states by +1 (next state's index aligned to current token position)
        char_to_idx = {'A': 0, 'B': 1}
        next_states_all = states_chars[1:]  # drop first, next-state for pos i is states[i+1]
        # split
        train_labels = np.fromiter((char_to_idx[c] for c in next_states_all[:n_train]), dtype=np.int64)
        val_labels = np.fromiter((char_to_idx[c] for c in next_states_all[n_train:]), dtype=np.int64)
    else:
        # derive next state from current symbol: 0 -> B(1), 1 -> A(0)
        sym_to_next_state = {0: 1, 1: 0}
        train_labels = np.fromiter((sym_to_next_state[int(t)] for t in train_tokens), dtype=np.int64)
        val_labels = np.fromiter((sym_to_next_state[int(t)] for t in val_tokens), dtype=np.int64)

    # ensure lengths consistent with block packing
    bs = model.config.block_size
    def trim_to_blocks(x: np.ndarray) -> np.ndarray:
        L = (len(x) // bs) * bs
        return x[:L]

    train_tokens = trim_to_blocks(train_tokens)
    val_tokens = trim_to_blocks(val_tokens)
    train_labels = trim_to_blocks(train_labels)
    val_labels = trim_to_blocks(val_labels)

    # choose hook module for features
    if args.layer == 'final':
        hook_module = model.transformer.ln_f
    else:
        try:
            layer_idx = int(args.layer)
        except Exception:
            raise ValueError("--layer must be 'final' or an integer index")
        if layer_idx < 0 or layer_idx >= len(model.transformer.h):
            raise ValueError(f"--layer index out of range: {layer_idx}")
        hook_module = model.transformer.h[layer_idx]

    # extract representations
    if args.target_state == 'pre':
        # prefix-only features for s_t
        X_train = extract_hidden_states_prefix_only_with_hook(model, hook_module, train_tokens, device)
        X_val = extract_hidden_states_prefix_only_with_hook(model, hook_module, val_tokens, device)
        # drop the first label (s_0 undefined)
        train_labels = train_labels[1:len(X_train)+1]
        val_labels = val_labels[1:len(X_val)+1]
    else:
        X_train = extract_hidden_states_with_hook(model, hook_module, train_tokens, device)
        X_val = extract_hidden_states_with_hook(model, hook_module, val_tokens, device)

    # labels per position
    y_train = train_labels[: len(X_train)]
    y_val = val_labels[: len(X_val)]

    # convert to torch tensors
    X_train_t = torch.from_numpy(X_train).to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    y_train_t = torch.from_numpy(y_train.astype(np.int64)).to(device)
    y_val_t = torch.from_numpy(y_val.astype(np.int64)).to(device)

    num_classes = int(max(y_train.max() if len(y_train) else 1, y_val.max() if len(y_val) else 1) + 1)
    emb_dim = int(model.config.n_embd)
    clf = nn.Linear(emb_dim, num_classes).to(device)
    if args.class_weight:
        # inverse frequency weights from train labels
        classes, counts = np.unique(y_train, return_counts=True)
        freq = counts / counts.sum()
        inv = 1.0 / np.maximum(freq, 1e-12)
        weights = inv / inv.sum() * len(classes)
        class_weights = torch.tensor([weights[c] if c in classes else 1.0 for c in range(num_classes)], dtype=torch.float32, device=device)
    else:
        class_weights = None
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(clf.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def iterate_batches(X: torch.Tensor, y: torch.Tensor, batch_size: int):
        n = X.size(0)
        for i in range(0, n, batch_size):
            yield X[i:i+batch_size], y[i:i+batch_size]

    # train loop
    clf.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        total = 0
        correct = 0
        for xb, yb in iterate_batches(X_train_t, y_train_t, args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            logits = clf(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += int((preds == yb).sum().item())
            total += int(xb.size(0))
        train_epoch_loss = total_loss / max(total, 1)
        train_epoch_acc = correct / max(total, 1)
        # quick val eval each epoch
        clf.eval()
        with torch.no_grad():
            logits_val = []
            for xb, _ in iterate_batches(X_val_t, y_val_t, args.batch_size):
                logits_val.append(clf(xb))
            logits_val = torch.cat(logits_val, dim=0) if logits_val else torch.empty(0, num_classes, device=device)
            preds_val = logits_val.argmax(dim=1) if logits_val.numel() else torch.empty(0, dtype=torch.long, device=device)
            val_acc = (preds_val == y_val_t[: preds_val.size(0)]).float().mean().item() if preds_val.numel() else 0.0
        clf.train()
        print({'epoch': epoch+1, 'train_loss': train_epoch_loss, 'train_acc': train_epoch_acc, 'val_acc': val_acc})

    # final metrics
    clf.eval()
    with torch.no_grad():
        train_logits = []
        for xb, _ in iterate_batches(X_train_t, y_train_t, args.batch_size):
            train_logits.append(clf(xb))
        train_preds = torch.cat(train_logits, dim=0).argmax(dim=1) if train_logits else torch.empty(0, dtype=torch.long, device=device)
        final_train_acc = (train_preds == y_train_t[: train_preds.size(0)]).float().mean().item() if train_preds.numel() else 0.0

        val_logits = []
        for xb, _ in iterate_batches(X_val_t, y_val_t, args.batch_size):
            val_logits.append(clf(xb))
        val_preds = torch.cat(val_logits, dim=0).argmax(dim=1) if val_logits else torch.empty(0, dtype=torch.long, device=device)
        final_val_acc = (val_preds == y_val_t[: val_preds.size(0)]).float().mean().item() if val_preds.numel() else 0.0

    # label balance diagnostics
    uniq_tr, cnt_tr = np.unique(y_train, return_counts=True)
    uniq_va, cnt_va = np.unique(y_val, return_counts=True)

    print({
        'n_train_positions': int(len(X_train)),
        'n_val_positions': int(len(X_val)),
        'emb_dim': int(model.config.n_embd),
        'block_size': int(bs),
        'final_train_acc': float(final_train_acc),
        'final_val_acc': float(final_val_acc),
        'epochs': int(args.epochs),
        'lr': float(args.lr),
        'batch_size': int(args.batch_size),
        'weight_decay': float(args.weight_decay),
        'target_state': args.target_state,
        'class_weight': bool(args.class_weight),
        'label_dist_train': {int(k): int(v) for k, v in zip(uniq_tr, cnt_tr)},
        'label_dist_val': {int(k): int(v) for k, v in zip(uniq_va, cnt_va)},
    })


if __name__ == '__main__':
    main()


