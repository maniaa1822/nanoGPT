import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap  # type: ignore
    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False


# -------------------------
# Utilities
# -------------------------
def pca_reduce(H: np.ndarray, n_components: int = 2):
    p = PCA(n_components=n_components)
    Z = p.fit_transform(H)
    return Z, p


def umap_reduce(H: np.ndarray, n_components: int = 2):
    if _HAS_UMAP:
        reducer = umap.UMAP(n_components=n_components)
        Z = reducer.fit_transform(H)
        return Z, reducer
    else:
        reducer = TSNE(n_components=n_components, init='pca')
        Z = reducer.fit_transform(H)
        return Z, reducer


def ensure_2d_embedding(H: np.ndarray, method: str = 'umap', n_components: int = 2):
    if H.shape[1] <= n_components:
        return H[:, :n_components], None
    if method == 'umap':
        return umap_reduce(H, n_components) if _HAS_UMAP else pca_reduce(H, n_components)
    if method == 'tsne':
        return TSNE(n_components=n_components, init='pca').fit_transform(H), None
    return pca_reduce(H, n_components)


# -------------------------
# 1) Layer scatter: embedding colored by state + next-token intensity
# -------------------------
def plot_state_scatter(H: np.ndarray,
                       states: np.ndarray,
                       next_probs: np.ndarray | None = None,
                       title: str | None = None,
                       embed_method: str = 'umap',
                       sample: int | None = 5000,
                       figsize: tuple[int, int] = (8, 6)) -> None:
    """
    H: (N,d) hidden vectors
    states: (N,) integer labels 0..S-1
    next_probs: (N,) float in [0,1] representing predicted P(token=1|h) or other scalar per point
    """
    N = H.shape[0]
    if sample is not None and N > sample:
        idx = np.random.choice(N, sample, replace=False)
        Hs = H[idx]
        states_s = states[idx]
        next_p_s = None if next_probs is None else next_probs[idx]
    else:
        Hs, states_s, next_p_s = H, states, next_probs

    Z, _ = ensure_2d_embedding(Hs, method=embed_method)

    plt.figure(figsize=figsize)
    sc = plt.scatter(Z[:, 0], Z[:, 1], s=6, c=states_s, alpha=0.8)
    plt.title(title or "Hidden states (colored by causal state)")
    plt.xlabel("dim1")
    plt.ylabel("dim2")
    plt.colorbar(sc, label='causal state')

    if next_p_s is not None:
        plt.scatter(Z[:, 0], Z[:, 1], s=6, c=next_p_s, cmap='viridis', alpha=0.6)
        cbar = plt.colorbar()
        cbar.set_label('predicted P(token=1)')
    plt.tight_layout()


# -------------------------
# 2) State-centroid graph overlay
# -------------------------
def plot_centroid_graph(H: np.ndarray,
                        states: np.ndarray,
                        true_transition_matrix: np.ndarray,
                        state_names: list[str] | None = None,
                        embed_method: str = 'pca',
                        figsize: tuple[int, int] = (8, 6)) -> None:
    """
    Compute centroids per state, embed centroids in 2D, and draw arrows according to transition matrix.
    """
    S = int(states.max()) + 1
    centroids = np.zeros((S, H.shape[1]))
    counts = np.zeros(S, dtype=int)
    for s in range(S):
        mask = states == s
        if mask.sum() == 0:
            continue
        centroids[s] = H[mask].mean(axis=0)
        counts[s] = mask.sum()

    Zc, _ = ensure_2d_embedding(centroids, method=embed_method, n_components=2)
    plt.figure(figsize=figsize)
    plt.scatter(Zc[:, 0], Zc[:, 1], s=150)
    for s in range(S):
        label = f"{state_names[s] if state_names else s}\n(n={counts[s]})"
        plt.text(Zc[s, 0], Zc[s, 1], label, fontsize=9, ha='center', va='center')

    max_w = float(np.max(true_transition_matrix)) if true_transition_matrix.size else 1.0
    for i in range(S):
        for j in range(S):
            w = float(true_transition_matrix[i, j])
            if w > 0:
                start = Zc[i]
                end = Zc[j]
                dx, dy = end - start
                plt.arrow(start[0], start[1], dx * 0.85, dy * 0.85,
                          head_width=0.02, length_includes_head=True,
                          alpha=min(0.9, 0.2 + 0.8 * (w / max_w)))
    plt.title("State centroids & transition graph overlay")
    plt.xlabel("dim1")
    plt.ylabel("dim2")
    plt.tight_layout()


# -------------------------
# 3) Trajectory plot: path of hidden states over time (projected)
# -------------------------
def plot_hidden_trajectory(H_seq: np.ndarray,
                           states_seq: np.ndarray,
                           tokens_seq: np.ndarray | None = None,
                           embed_method: str = 'pca',
                           window: tuple[int, int] | None = None,
                           figsize: tuple[int, int] = (10, 4)) -> None:
    """
    Plot the projection trajectory of hidden states in order.
    H_seq: (T,d)
    states_seq: (T,)
    tokens_seq: (T,) optional
    window: (start, end) slice to visualize a segment
    """
    if window is not None:
        s, e = window
        H_seq = H_seq[s:e]
        states_seq = states_seq[s:e]
        if tokens_seq is not None:
            tokens_seq = tokens_seq[s:e]

    Z, _ = ensure_2d_embedding(H_seq, method=embed_method, n_components=2)
    plt.figure(figsize=figsize)
    plt.plot(Z[:, 0], Z[:, 1], '-o', markersize=3, alpha=0.6)
    for i, (x, y) in enumerate(Z):
        plt.scatter(x, y, s=18, c=[states_seq[i]])
    plt.title("Hidden-state trajectory (time-ordered)")
    plt.xlabel("dim1")
    plt.ylabel("dim2")
    if tokens_seq is not None:
        ones_idx = np.where(tokens_seq == 1)[0]
        plt.scatter(Z[ones_idx, 0], Z[ones_idx, 1], marker='x', s=40)
    plt.tight_layout()


# -------------------------
# 4) Redundancy AUC curve (subspace probing)
# -------------------------
from sklearn.linear_model import LogisticRegression


def redundancy_auc(H: np.ndarray,
                   states: np.ndarray,
                   fractions: np.ndarray = np.linspace(0.05, 1.0, 20),
                   n_trials: int = 8,
                   C: float = 1.0,
                   seed: int = 0):
    """
    returns fractions and mean accuracies per fraction
    """
    rng = np.random.RandomState(seed)
    N, d = H.shape
    accuracies = np.zeros((len(fractions), n_trials), dtype=np.float64)
    for i, f in enumerate(fractions):
        k = max(1, int(math.floor(float(f) * d)))
        for t in range(n_trials):
            idx = rng.choice(d, k, replace=False)
            Hs = H[:, idx]
            perm = rng.permutation(N)
            split = int(0.8 * N)
            tr, te = perm[:split], perm[split:]
            clf = LogisticRegression(max_iter=200, C=C).fit(Hs[tr], states[tr])
            acc = clf.score(Hs[te], states[te])
            accuracies[i, t] = acc
    mean_acc = accuracies.mean(axis=1)
    auc = float(np.trapz(mean_acc, fractions) / (fractions[-1] - fractions[0]))
    return fractions, mean_acc, auc


def plot_redundancy_curve(fractions: np.ndarray,
                          mean_acc: np.ndarray,
                          auc: float,
                          title: str | None = None,
                          figsize: tuple[int, int] = (6, 4)) -> None:
    plt.figure(figsize=figsize)
    plt.plot(fractions, mean_acc, marker='o')
    plt.title((title or "Subspace redundancy curve") + f"    AUC={auc:.3f}")
    plt.xlabel("fraction of features used")
    plt.ylabel("probe accuracy")
    plt.grid(True)
    plt.tight_layout()


# -------------------------
# 5) Fracture Index heatmap by layer & step
# -------------------------
def plot_fi_heatmap(fi_matrix: np.ndarray,
                    layers: list[int] | list[str],
                    steps: list[int] | list[str],
                    title: str = "Fracture Index heatmap",
                    figsize: tuple[int, int] = (8, 6)) -> None:
    """
    fi_matrix: 2D array shape (len(layers), len(steps))
    """
    plt.figure(figsize=figsize)
    im = plt.imshow(fi_matrix, aspect='auto', origin='lower')
    plt.colorbar(im, label='Fracture Index (lower better)')
    plt.yticks(np.arange(len(layers)), layers)
    plt.xticks(np.arange(len(steps)), steps, rotation=45)
    plt.xlabel("training step")
    plt.ylabel("layer")
    plt.title(title)
    plt.tight_layout()


# -------------------------
# 6) Kernel recovery bar charts + KL
# -------------------------
def per_state_kernel_metrics(pred_next_probs: np.ndarray,
                             true_next_probs: np.ndarray | None,
                             states: np.ndarray,
                             state_names: list[str] | None = None,
                             figsize: tuple[int, int] = (10, 4)):
    """
    pred_next_probs: (N, |V|) predicted distribution per token
    true_next_probs: optional (|S|, |V|) true distribution per state (if available)
    states: (N,)
    """
    S = int(states.max()) + 1
    pred_avg = []
    for s in range(S):
        mask = states == s
        if mask.sum() == 0:
            pred_avg.append(np.zeros(pred_next_probs.shape[1], dtype=np.float64))
            continue
        pred_mean = pred_next_probs[mask].mean(axis=0)
        pred_avg.append(pred_mean)
    pred_avg = np.stack(pred_avg, axis=0)
    return pred_avg


def plot_state_kernels(pred_avg: np.ndarray,
                       true_avg: np.ndarray,
                       state_names: list[str] | None = None,
                       figsize: tuple[int, int] = (10, 4)) -> None:
    S = pred_avg.shape[0]
    V = pred_avg.shape[1]
    fig, axs = plt.subplots(1, S, figsize=(min(16, 3 * S), 4))
    if S == 1:
        axs = [axs]
    for s in range(S):
        axs[s].bar(np.arange(V) - 0.2, true_avg[s], width=0.4)
        axs[s].bar(np.arange(V) + 0.2, pred_avg[s], width=0.4)
        axs[s].set_title(state_names[s] if state_names else f"State {s}")
        axs[s].set_ylim(0, 1.0)
    plt.suptitle("True vs Predicted next-token kernels per state")
    plt.tight_layout()


# -------------------------
# 7) CKA matrix (linear CKA)
# -------------------------
def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    linear CKA between two matrices X(N,d1) and Y(N,d2)
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    HSIC = np.sum((Xc.T @ Yc) ** 2)
    denom = np.sqrt(np.sum((Xc.T @ Xc) ** 2) * np.sum((Yc.T @ Yc) ** 2))
    return float(HSIC / (denom + 1e-12))


def plot_cka_matrix(H_list: list[np.ndarray], labels: list[str], figsize: tuple[int, int] = (8, 6)) -> None:
    L = len(H_list)
    C = np.zeros((L, L), dtype=np.float64)
    for i in range(L):
        for j in range(L):
            C[i, j] = linear_cka(H_list[i], H_list[j])
    plt.figure(figsize=figsize)
    im = plt.imshow(C, vmin=0, vmax=1, cmap='viridis')
    plt.colorbar(im, label='linear CKA')
    plt.xticks(np.arange(L), labels, rotation=45)
    plt.yticks(np.arange(L), labels)
    plt.title("CKA similarity matrix")
    plt.tight_layout()


