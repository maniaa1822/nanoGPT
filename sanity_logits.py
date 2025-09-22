"""
Sanity check next-token behavior for binary datasets.

Reports:
- p(0|last=0), p(0|last=1) using true token mapping from meta.pkl
- p(same|last), p(opposite|last)
- top-1 next token accuracy (vs ground truth)

Usage:
  uv run python sanity_logits.py --out_dir out-golden-mean-char --dataset golden_mean --split val --device cpu
"""
import os
import json
import argparse
import pickle
import numpy as np
import torch

from model import GPTConfig, GPT


def load_checkpoint_and_model(out_dir: str, device: str):
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return checkpoint, model


@torch.no_grad()
def get_next_log_probs(model: GPT, device: str, context_tokens: np.ndarray):
    x = torch.tensor(context_tokens, dtype=torch.long, device=device)[None, ...]
    logits, _ = model(x)
    return torch.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', type=str, required=True)
    ap.add_argument('--dataset', type=str, required=True)
    ap.add_argument('--split', type=str, default='val', choices=['train', 'val'])
    ap.add_argument('--num_positions', type=int, default=2000)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--device', type=str, default='cpu')
    args = ap.parse_args()

    ckpt, model = load_checkpoint_and_model(args.out_dir, args.device)
    data_dir = os.path.join('data', args.dataset)
    split_file = 'val.bin' if args.split == 'val' else 'train.bin'
    data = np.memmap(os.path.join(data_dir, split_file), dtype=np.uint16, mode='r')
    tokens = data.astype(np.int64)

    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    stoi = meta.get('stoi', {'0': 0, '1': 1})
    itos = meta.get('itos', {0: '0', 1: '1'})
    idx_zero = int(stoi.get('0', 0))
    idx_one = int(stoi.get('1', 1))

    start = 1  # need at least 1 token of context
    idxs = list(range(start, len(tokens) - 1, args.stride))
    if len(idxs) > args.num_positions:
        idxs = idxs[:args.num_positions]

    p0_given0_sum = 0.0
    p0_given1_sum = 0.0
    cnt0 = 0
    cnt1 = 0

    psame_sum = 0.0
    popp_sum = 0.0
    acc_top1 = 0
    total = 0

    for i in idxs:
        last_tok = int(tokens[i - 1])
        target = int(tokens[i])
        context = tokens[max(0, i - model.config.block_size): i]
        logprobs = get_next_log_probs(model, args.device, context)
        probs = torch.exp(logprobs)

        p0 = float(probs[idx_zero].item())
        p1 = float(probs[idx_one].item())

        if last_tok == idx_zero:
            p0_given0_sum += p0
            cnt0 += 1
        elif last_tok == idx_one:
            p0_given1_sum += p0
            cnt1 += 1

        psame_sum += p0 if last_tok == idx_zero else p1
        popp_sum += p1 if last_tok == idx_zero else p0

        pred = int(torch.argmax(probs).item())
        if pred == target:
            acc_top1 += 1
        total += 1

    result = {
        'out_dir': args.out_dir,
        'split': args.split,
        'vocab_size': int(meta['vocab_size']),
        'idx_zero': idx_zero,
        'idx_one': idx_one,
        'num_positions': total,
        'p0_given_last0': (p0_given0_sum / cnt0) if cnt0 else None,
        'p0_given_last1': (p0_given1_sum / cnt1) if cnt1 else None,
        'p_same_given_last': psame_sum / total if total else None,
        'p_opp_given_last': popp_sum / total if total else None,
        'top1_accuracy': acc_top1 / total if total else None,
    }
    print(json.dumps(result, indent=2))
    out_path = os.path.join(args.out_dir, 'sanity_logits.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == '__main__':
    main()



