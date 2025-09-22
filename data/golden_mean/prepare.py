"""
Prepare the Golden Mean dataset for character-level language modeling.
This mirrors the char-level Shakespeare prep: map characters to ints and
write train.bin, val.bin, and meta.pkl. The dataset is expected to be a
string over the alphabet {"0","1"} (e.g., golden_mean.dat).

Input discovery order (first existing is used):
1) data/golden_mean/input.txt
2) data/golden_mean/golden_mean.dat
3) experiments/datasets/golden_mean/golden_mean.dat (repo-relative)
"""
import os
import pickle
import numpy as np


def find_input_path() -> str:
    here = os.path.dirname(__file__)
    repo_root = os.path.abspath(os.path.join(here, "../../.."))
    candidates = [
        os.path.join(here, "input.txt"),
        os.path.join(here, "golden_mean.dat"),
        os.path.join(
            repo_root, "notebook_experiments", "test_golden_mean", "golden_mean", "golden_mean.dat"
        ),
        os.path.join(
            repo_root, "notebook_experiments", "golden_mean", "data", "golden_mean", "golden_mean.dat"
        ),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "No input file found. Place your sequence into data/golden_mean/input.txt or golden_mean.dat, "
        "or ensure experiments/datasets/golden_mean/golden_mean.dat exists."
    )


input_file_path = find_input_path()
with open(input_file_path, "r") as f:
    raw = f.read()

# Keep only '0' and '1' characters; drop any whitespace/newlines/others
data = "".join(ch for ch in raw if ch in ("0", "1"))
if len(data) == 0:
    raise ValueError("Input appears empty after filtering to '0'/'1'.")

print(f"length of dataset in characters: {len(data):,}")

# Build vocabulary and mappings
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

# Train/val split
n = len(data)
train_data = data[: int(n * 0.9)]
val_data = data[int(n * 0.9) :]

# Encode and export
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



