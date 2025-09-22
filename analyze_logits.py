"""
Analyze how next-token logits depend on context length for a trained nanoGPT model.

Outputs, per k in [1..max_k]:
- avg NLL (nats and bits) on the chosen split when only the last k tokens are used
- estimated p(token=0 | last=0) and p(token=0 | last=1) for binary datasets

Usage examples:
  uv run --with torch --with numpy python analyze_logits.py \
      --out_dir out-golden-mean-char --split val --max_k 64 --num_positions 5000 --device cpu

Results are printed and also saved to {out_dir}/logit_analysis.json
"""
import os
import json
import math
import argparse
import pickle
import csv
from typing import List, Dict, Any

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


def find_dataset_dir(checkpoint: Dict[str, Any], fallback_dataset: str) -> str:
    if 'config' in checkpoint and 'dataset' in checkpoint['config']:
        dataset = checkpoint['config']['dataset']
    else:
        dataset = fallback_dataset
    data_dir = os.path.join('data', dataset)
    if not os.path.exists(os.path.join(data_dir, 'meta.pkl')):
        raise FileNotFoundError(f"meta.pkl not found under {data_dir}. Ensure dataset is prepared.")
    return data_dir


@torch.no_grad()
def get_next_log_probs(model: GPT, device: str, context_tokens: List[int]):
    if len(context_tokens) < 1:
        raise ValueError("Context must contain at least 1 token")
    x = torch.tensor(context_tokens, dtype=torch.long, device=device)[None, ...]
    logits, _ = model(x)
    logprobs = torch.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)  # [vocab]
    return logprobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='out-golden-mean-char')
    parser.add_argument('--dataset', type=str, default='golden_mean')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'])
    parser.add_argument('--max_k', type=int, default=64)
    parser.add_argument('--num_positions', type=int, default=5000,
                        help='number of positions to sample for evaluation (<= available tokens)')
    parser.add_argument('--stride', type=int, default=10,
                        help='evaluate every `stride` positions (applied before num_positions)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dump_csv', action='store_true', help='dump per-position per-k probabilities to CSV')
    parser.add_argument('--csv_out', type=str, default=None, help='path to CSV (default: {out_dir}/suffix_logits.csv)')
    args = parser.parse_args()

    checkpoint, model = load_checkpoint_and_model(args.out_dir, args.device)
    data_dir = find_dataset_dir(checkpoint, args.dataset)

    # load dataset split
    split_file = 'val.bin' if args.split == 'val' else 'train.bin'
    data = np.memmap(os.path.join(data_dir, split_file), dtype=np.uint16, mode='r')
    tokens = data.astype(np.int64)
    vocab_size = None
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
        vocab_size = meta['vocab_size']
        # Respect the dataset's token mapping for '0'/'1'
        stoi = meta.get('stoi', None)
        itos = meta.get('itos', None)
        if stoi is not None and '0' in stoi:
            idx_zero = int(stoi['0'])
        else:
            # fallback: assume id 0 is '0'
            idx_zero = 0

    # choose positions to evaluate
    # ensure we have at least max_k context, so start at index >= max_k
    start_index = args.max_k
    idxs = list(range(start_index, len(tokens) - 1, args.stride))
    if len(idxs) > args.num_positions:
        idxs = idxs[:args.num_positions]

    results = []
    print(f"Evaluating {len(idxs)} positions, vocab_size={vocab_size}")

    # For binary datasets, track p(0|last=0/1)
    track_binary = vocab_size == 2

    # Optional CSV dump setup
    csv_writer = None
    csv_fh = None
    if args.dump_csv:
        csv_path = args.csv_out or os.path.join(args.out_dir, 'suffix_logits.csv')
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        csv_fh = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_fh)
        header = ['pos_idx', 'k', 'target', 'context'] + [f'p_{i}' for i in range(vocab_size)]
        csv_writer.writerow(header)

    for k in range(1, args.max_k + 1):
        nll_sum = 0.0
        count = 0
        # binary diagnostics
        p0_given_last0_sum = 0.0
        p0_given_last1_sum = 0.0
        last0_count = 0
        last1_count = 0

        for i in idxs:
            context = tokens[i - k:i]
            target = int(tokens[i])
            # crop to model block_size if needed
            if len(context) > model.config.block_size:
                context = context[-model.config.block_size:]
            logprobs = get_next_log_probs(model, args.device, context.tolist())
            nll_sum += (-float(logprobs[target]))
            count += 1

            if csv_writer is not None:
                probs = torch.exp(logprobs).cpu().numpy().tolist()
                # represent context as space-separated token ids to be general
                context_str = ' '.join(str(int(t)) for t in context.tolist())
                row = [i, k, target, context_str] + probs
                csv_writer.writerow(row)

            if track_binary and len(context) >= 1:
                last_tok = int(context[-1])
                # probability assigned to token that corresponds to symbol '0'
                p0 = float(torch.exp(logprobs[idx_zero]).item())
                if last_tok == idx_zero:
                    p0_given_last0_sum += p0
                    last0_count += 1
                else:
                    p0_given_last1_sum += p0
                    last1_count += 1

        avg_nll = nll_sum / max(1, count)
        bits = avg_nll / math.log(2)
        summary = {
            'k': k,
            'avg_nll_nats': avg_nll,
            'avg_nll_bits': bits,
            'num_positions': count,
        }
        if track_binary:
            summary.update({
                'p0_given_last0': (p0_given_last0_sum / last0_count) if last0_count else None,
                'p0_given_last1': (p0_given_last1_sum / last1_count) if last1_count else None,
                'count_last0': last0_count,
                'count_last1': last1_count,
            })
        results.append(summary)
        print(f"k={k:3d}  nll(bits)={bits:.4f}  positions={count}" + (
            f"  p0|0={summary['p0_given_last0']:.3f}  p0|1={summary['p0_given_last1']:.3f}" if track_binary else ''
        ))

    out_path = os.path.join(args.out_dir, 'logit_analysis.json')
    with open(out_path, 'w') as f:
        json.dump({'results': results, 'max_k': args.max_k, 'split': args.split}, f, indent=2)
    print(f"Saved analysis to {out_path}")

    if csv_fh is not None:
        csv_fh.close()
        print(f"Saved per-position probabilities CSV to {csv_path}")


if __name__ == '__main__':
    main()


