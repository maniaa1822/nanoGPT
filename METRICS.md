# nanoGPT Representation Metrics & Visualizations

This guide documents the metrics and plots we use to evaluate how training techniques (e.g., auxiliary heads, curricula) shape hidden representations in nanoGPT models. All commands use uv.

## Quickstart (Golden Mean)

- Generate visuals + caches (scatter with next-prob overlay, centroids, trajectory):
```bash
cd /home/matteo/NeuralCSSR/nanoGPT
uv run python viz_generate.py \
  --ckpt out-golden-mean-char/ckpt.pt \
  --data_dir data/golden_mean \
  --states_dat ../notebook_experiments/golden_mean/data/golden_mean/golden_mean.states.dat \
  --device cuda --out_dir out-golden-mean-char --embed pca --sample 5000 --traj_len 200 --layers 0,1,final
```
- Redundancy curve, kernel summary, CKA (requires per-layer caches):
```bash
uv run python viz_metrics.py --model_dir out-golden-mean-char --layers 0,1,final
```
- Core metrics (probe, clustering, dynamics, perplexity gap):
```bash
# linear probe (prints final acc)
uv run python probe_linear_state.py --ckpt out-golden-mean-char/ckpt.pt \
  --data_dir data/golden_mean --device cpu --epochs 8 --lr 5e-3 --batch_size 8192 \
  --class_weight --target_state pre --layer final

# clustering quality (writes JSON)
uv run --with torch --with numpy --with scikit-learn python metric_cluster_quality.py \
  --ckpt out-golden-mean-char/ckpt.pt --data_dir data/golden_mean \
  --states_dat ../notebook_experiments/golden_mean/data/golden_mean/golden_mean.states.dat \
  --device cpu --out out-golden-mean-char/cluster_metrics.json

# dynamics consistency (writes JSON)
uv run --with torch --with numpy --with scikit-learn python metric_dynamics.py \
  --ckpt out-golden-mean-char/ckpt.pt --data_dir data/golden_mean \
  --states_dat ../notebook_experiments/golden_mean/data/golden_mean/golden_mean.states.dat \
  --device cpu --out out-golden-mean-char/dynamics_metrics.json

# perplexity gap (writes JSON)
uv run --with torch --with numpy python metric_perplexity_gap.py \
  --ckpt out-golden-mean-char/ckpt.pt --train_data_dir data/golden_mean \
  --val_data_dir data/golden_mean --device cpu --out out-golden-mean-char/perplexity_gap.json
```

Tip: For heavy plots (UMAP/TSNE, redundancy, CKA) consider subsampling and limiting BLAS threads:
```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 <your uv run ...>
```

---

## Metrics (what, run, output, interpret)

### Linear probe state accuracy (sufficiency)
- What: Train a linear classifier from prefix-only hidden `h_t` to causal state `s_t`.
- Run:
```bash
uv run python probe_linear_state.py --ckpt <OUT_DIR>/ckpt.pt \
  --data_dir data/<dataset> --device cpu --epochs 8 --lr 5e-3 --batch_size 8192 \
  --class_weight --target_state pre --layer final
```
- Output: Console (`final_val_acc`).
- Interpret: Higher = states are more linearly decodable from `h_t`. Very sensitive to representation shaping.

### Clustering quality (purity, NMI, V-measure)
- What: k-means (k = #states) on `h_t` vs true states.
- Run:
```bash
uv run --with torch --with numpy --with scikit-learn python metric_cluster_quality.py \
  --ckpt <OUT_DIR>/ckpt.pt --data_dir data/<dataset> \
  --states_dat <path/to/*.states.dat> --device cpu --out <OUT_DIR>/cluster_metrics.json
```
- Output: `<OUT_DIR>/cluster_metrics.json`.
- Interpret: Conservative; improves when reps naturally form state clusters. May stay flat even if probe improves.

### Dynamics consistency (transitions)
- What: (a) `h_t → s_{t+1}` probe accuracy; (b) cluster-induced transition matrix vs true (row KL, graph Jaccard).
- Run:
```bash
uv run --with torch --with numpy --with scikit-learn python metric_dynamics.py \
  --ckpt <OUT_DIR>/ckpt.pt --data_dir data/<dataset> \
  --states_dat <path/to/*.states.dat> --device cpu --out <OUT_DIR>/dynamics_metrics.json
```
- Output: `<OUT_DIR>/dynamics_metrics.json`.
- Interpret: Next-state probe often saturates on simple FSMs; structure scores improve as cluster→state mapping clarifies. Prefer supervised centroids for cleaner mapping.

### Perplexity gap (sufficiency alt)
- What: Compare NLL of linear head on `h_t` vs LM head with full context.
- Run:
```bash
uv run --with torch --with numpy python metric_perplexity_gap.py \
  --ckpt <OUT_DIR>/ckpt.pt --train_data_dir data/<dataset> --val_data_dir data/<dataset> \
  --device cpu --out <OUT_DIR>/perplexity_gap.json
```
- Output: `<OUT_DIR>/perplexity_gap.json`.
- Interpret: Negative `gap_bits` ⇒ `h_t` alone is highly predictive. For rigor, compare on matched positions.

### Transfer probe (generalization)
- What: Train probe on source machine reps; evaluate on target.
- Run:
```bash
uv run --with torch --with numpy python metric_transfer_probe.py \
  --ckpt <OUT_DIR>/ckpt.pt \
  --src_data_dir data/<src_dataset> --src_states_dat <path/to/src.states.dat> \
  --tgt_data_dir data/<tgt_dataset> --tgt_states_dat <path/to/tgt.states.dat> \
  --device cpu --out <OUT_DIR>/transfer_metrics.json
```
- Output: `<OUT_DIR>/transfer_metrics.json`.
- Interpret: Higher = better cross-machine reuse of representations.

---

## Visualizations (what, run, output, interpret)

### Scatter + next-prob overlay, centroid graph, trajectory
- What: 2D embedding colored by state (overlay predicted next-token probability), centroid graph, time trajectory with token marks.
- Run:
```bash
uv run python viz_generate.py \
  --ckpt <OUT_DIR>/ckpt.pt --data_dir data/<dataset> \
  --states_dat <path/to/*.states.dat> --device cuda \
  --out_dir <OUT_DIR> --embed pca --sample 5000 --traj_len 200 --layers 0,1,final
```
- Outputs:
  - `<OUT_DIR>/viz_scatter.png`
  - `<OUT_DIR>/viz_centroids.png`
  - `<OUT_DIR>/viz_trajectory.png`
  - `<OUT_DIR>/viz_cache.npz` (arrays: `H`, `states`, `p1`)
  - `<OUT_DIR>/viz_cache_layer_<L>.npz` (per requested layers)
- Interpret:
  - Clean lobes per state and consistent intensity = sufficiency + invariance.
  - Centroid arrows mirror FSM topology when geometry aligns.
  - Trajectories that quickly reach centroids and follow edges indicate attractor-like dynamics.

### Redundancy curve, kernel summary, CKA
- What: Subspace probing (redundancy), per-state mean predicted kernel summary, and layer CKA.
- Run:
```bash
uv run python viz_metrics.py --model_dir <OUT_DIR> --layers 0,1,final
```
- Outputs:
  - `<OUT_DIR>/viz_redundancy.png`
  - `<OUT_DIR>/viz_kernel_pred.txt` (mean P(1|h) per state)
  - `<OUT_DIR>/viz_cka.png` (if ≥2 layer caches), else `viz_cka.txt`
- Interpret:
  - High accuracy at small fractions ⇒ replicated/robust signal; low then rising ⇒ compact coding.
  - Kernel summary should reflect true next-symbol distributions per state (extend with true-KL overlay).
  - CKA: higher similarity across layers/checkpoints/seeds ⇒ more consistent representations.

---

## Interpretation cheat sheet
- Probe accuracy (prefix-only) ↑: stronger state sufficiency.
- Silhouette, inter/intra ↑; scatter separation: cleaner clustering by state.
- Trajectory adheres to centroids/edges: stable, Markovian dynamics in rep space.
- Redundancy curve higher at small fractions: replicated state features; combine with participation ratio to disambiguate compact vs replicated coding.
- Transition structure KL ↓ / Jaccard ↑: hidden clusters respect FSM topology (use supervised centroids for mapping).
- Perplexity gap near/below 0: `h_t` is highly predictive.
- Transfer probe ↑: better cross-machine generalization.

## Performance tips
- Cap threads: `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2`.
- Subsample for heavy plots; use `--sample` in `viz_generate.py` and defaults in `viz_metrics.py`.

## Extend
- Matched-position NLL for perplexity gap; true per-state kernel + KL; supervised-centroid transitions; CKA/FI across steps/seeds; optional MINE for I(h; future).
