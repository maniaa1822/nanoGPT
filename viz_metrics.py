import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rep_viz as rv


def load_cache(path: str):
    z = np.load(path)
    H = z['H']
    s = z['states']
    p1 = z['p1'] if 'p1' in z.files else None
    return H, s, p1


def plot_redundancy(cache_path: str, out_png: str, subsample: int = 3000, threads_cap: int = 2):
    H, s, _ = load_cache(cache_path)
    n = H.shape[0]
    m = min(subsample, n)
    rng = np.random.default_rng(0)
    idx = rng.choice(n, m, replace=False)
    Hs, ss = H[idx], s[idx]
    fr = np.linspace(0.1, 1.0, 10)
    fr, acc, auc = rv.redundancy_auc(Hs, ss, fractions=fr, n_trials=3, C=1.0, seed=0)
    rv.plot_redundancy_curve(fr, acc, auc, title='Redundancy (subsampled)')
    plt.savefig(out_png, dpi=140)
    plt.close()


def plot_kernel_pred(cache_path: str, out_txt: str):
    H, s, p1 = load_cache(cache_path)
    if p1 is None:
        with open(out_txt, 'w') as f:
            f.write('p1 missing in cache')
        return
    S = int(s.max()) + 1
    pred_avg = []
    for k in range(S):
        mask = s == k
        pred_avg.append(float(p1[mask].mean()) if mask.any() else float('nan'))
    with open(out_txt, 'w') as f:
        f.write(str({'pred_mean_p1_per_state': pred_avg}))


def cka_from_layer_caches(cache_paths: list[str], labels: list[str], out_png: str, subsample: int = 4000):
    rng = np.random.default_rng(0)
    H_list = []
    for path in cache_paths:
        z = np.load(path)
        H = z['H']
        n = H.shape[0]
        m = min(subsample, n)
        idx = rng.choice(n, m, replace=False)
        H_list.append(H[idx])
    rv.plot_cka_matrix(H_list, labels)
    plt.savefig(out_png, dpi=140)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', type=str, required=True)
    ap.add_argument('--layers', type=str, default='0,1,final')
    args = ap.parse_args()

    cache = os.path.join(args.model_dir, 'viz_cache.npz')
    os.makedirs(args.model_dir, exist_ok=True)

    plot_redundancy(cache, os.path.join(args.model_dir, 'viz_redundancy.png'))
    plot_kernel_pred(cache, os.path.join(args.model_dir, 'viz_kernel_pred.txt'))

    # CKA across selected layers (requires per-layer caches)
    layers = [ls.strip() for ls in args.layers.split(',') if ls.strip()]
    layer_paths = []
    labels = []
    for ls in layers:
        p = os.path.join(args.model_dir, f'viz_cache_layer_{ls}.npz')
        if os.path.exists(p):
            layer_paths.append(p)
            labels.append(ls)
    if len(layer_paths) >= 2:
        cka_from_layer_caches(layer_paths, labels, os.path.join(args.model_dir, 'viz_cka.png'))
    else:
        with open(os.path.join(args.model_dir, 'viz_cka.txt'), 'w') as f:
            f.write('Not enough layer caches for CKA')


if __name__ == '__main__':
    main()


