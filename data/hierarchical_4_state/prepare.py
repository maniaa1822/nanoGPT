"""
Prepare the Hierarchical 4-State dataset for character-level language modeling.
Mirrors other binary dataset preps: map characters to ints and write train/val bin files.

Input discovery order:
1) data/hierarchical_4_state/input.txt
2) data/hierarchical_4_state/hierarchical_4_state.dat
3) experiments/datasets/hierarchical_4_state/hierarchical_4_state.dat
4) experiments/datasets/hierarchical_4_state_10k/hierarchical_4_state/hierarchical_4_state.dat
"""
import os
import pickle
import numpy as np


def find_input_path() -> str:
    here = os.path.dirname(__file__)
    repo_root = os.path.abspath(os.path.join(here, "../../.."))
    candidates = [
        os.path.join(here, "input.txt"),
        os.path.join(here, "hierarchical_4_state.dat"),
        os.path.join(repo_root, "experiments", "datasets", "hierarchical_4_state", "hierarchical_4_state.dat"),
        os.path.join(repo_root, "experiments", "datasets", "hierarchical_4_state_10k", "hierarchical_4_state", "hierarchical_4_state.dat"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "No input file found. Place sequence into data/hierarchical_4_state/input.txt or hierarchical_4_state.dat, "
        "or ensure experiments/datasets/hierarchical_4_state/hierarchical_4_state.dat (or the 10k variant) exists."
    )


input_file_path = find_input_path()
with open(input_file_path, "r") as f:
    raw = f.read()

data = "".join(ch for ch in raw if ch in ("0", "1"))
if len(data) == 0:
    raise ValueError("Input appears empty after filtering to '0'/'1'.")

print(f"length of dataset in characters: {len(data):,}")

chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", "".join(chars))
print(f"vocab size: {vocab_size:,}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s: str):
    return [stoi[c] for c in s]

def decode(l):
    return "".join([itos[i] for i in l])

n = len(data)
train_data = data[: int(n * 0.9)]
val_data = data[int(n * 0.9) :]

train_ids = np.array(encode(train_data), dtype=np.uint16)
val_ids = np.array(encode(val_data), dtype=np.uint16)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

out_dir = os.path.dirname(__file__)
train_ids.tofile(os.path.join(out_dir, "train.bin"))
val_ids.tofile(os.path.join(out_dir, "val.bin"))

meta = {
    "vocab_size": vocab_size,
    "itos": itos,
    "stoi": stoi,
}
with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

print("Wrote train.bin, val.bin and meta.pkl to:", out_dir)
