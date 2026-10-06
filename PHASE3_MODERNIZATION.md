# Phase 3 — RLDiff Modernization Record

Completed 2026-10-05, on the local RTX 3060 Laptop GPU (6GB VRAM, driver 550.163.01, CUDA 12.4).
Moves RLDiff onto the exact same stack as the parent's Phase 2 bump
(`../PHASE2_MODERNIZATION.md`), since RLDiff imports parent internals directly and loads
parent-built checkpoints.

## Target stack

Identical to the parent's Phase 2 choice — Python 3.11, PyTorch 2.6.0+cu124, torch_geometric
2.8.0.post1, e3nn 0.6.0, matching `torch-scatter`/`sparse`/`cluster`/`spline-conv` wheel builds.

## Changes

**Environment files** (`inference_env.yml`, `training_env.yml`): bumped the full torch/CUDA/PyG/e3nn
stack to match the parent. Two fixes needed beyond a straight version bump:
- `rdkit` moved from a conda dependency to pip (matching the parent's own approach): the
  conda-channel `rdkit=2026.3.6` requires a much newer `libboost` than `smina`'s pinned version
  supports — an unresolvable conda-level conflict once rdkit was bumped this far.
- Added `scikit-learn`, the same undeclared dependency found in the parent's Phase 2 (RLDiff
  imports the parent's `datasets/process_mols.py`, which imports `sklearn.neighbors` directly).

**Code cleanup** (per the roadmap's Phase 3 scope):
- Removed the unused `from torch.autograd import Variable` import in
  `utils/compute_probability_utils.py`.
- Deleted the orphaned `utils/interaction.py`, `utils/reward_utils.py`, `utils/system_prep.py` —
  confirmed unreachable from any entry point (`train.py`, `inference.py`, `src/*.py`), importing
  only each other. The original developers never finished wiring these in.
- GNINA (external prebuilt binary, used for `--minimize_and_rerank`) was not re-validated — not
  installed on this machine. Flagged in the original roadmap as needing a build compatible with
  the target CUDA/driver; still an open item whenever `--minimize_and_rerank` is actually needed.
- The namespace-package `sys.path` integration with the parent repo needed no changes — directory
  layout is untouched.

## Validation

### Inference — 3dpf smoke test (same complex as the parent's Phase 0/1/2 baselines)
Zero source changes needed for the inference path itself — only the environment fix. Both
checkpoints (RL-fine-tuned score model from `oxpig/RLDiff`, confidence model from
`plainerman/DiffDock-Pocket`) auto-downloaded and loaded correctly.

| | Phase 1 (torch 1.13.1) | Phase 3 (torch 2.6.0) |
|---|---|---|
| Top-1 ligand RMSD to crystal pose | 0.35 Å | 0.40 Å |

Both comfortably under the 2 Å threshold; the small difference is consistent with sampling being
stochastic (no fixed seed), not a regression.

### Training — RL fine-tuning smoke test
This took real debugging, entirely in test-harness setup rather than modernization code:

1. `pdbbind_dir` in `train_config.yaml` is resolved via `f"{base_dir}/{args.pdbbind_dir}"` in
   `train.py` (string concatenation against the parent repo root, not `os.path.join`) — it must be
   a path relative to the parent repo root, not absolute. Required placing a scratch dataset under
   the parent repo rather than under `/tmp`.
2. `num_complexes_to_sample` must be evenly divisible by `num_updates_per_sampling` (an assertion
   in the training loop) — a smoke-test parameter choice, not a bug.
3. **Found a real, independent bug while validating**: `utils/val_utils.py`'s
   `trajectory_generation_val` reads `args.pdbbind_full_path` (used directly in `os.path.join`,
   unrelated to `pdbbind_dir`) to locate ground-truth structures for reward computation. The
   checked-in `train_config.yaml` leaves this `null`, so every validation complex throws
   `TypeError: expected str, bytes or os.PathLike object, not NoneType` inside a caught
   `except Exception`. Worse, **the surrounding loop has no overall give-up condition** — it just
   resets the dataloader and retries indefinitely when every complex fails, which is how two
   earlier attempts at this smoke test ran for over an hour and then a full 2 hours with zero
   progress, looking like a hang rather than a fast, loud failure. Not something this phase
   touched in source (it's orthogonal to the version bump, and setting `pdbbind_full_path`
   correctly in the training config is arguably just "required configuration"), but worth a recommendation:
   **add a hard iteration cap to this retry loop** so a bad config fails fast instead of spinning
   forever. Worked around for this test by setting `pdbbind_full_path` correctly.

Once configured correctly (1 epoch, 1 complex, no branching — deliberately minimal to isolate the
training step itself), the full loop completed cleanly:
```
EXIT CODE: 0
Epoch 1 - Loss: 0.0000, Norm Reward: 0.0000
Epoch 1 - Validation Trajectory Generation metrics: {'successful_complexes': 2, 'avg_raw_reward': 0.952, 'pb_valid_fraction': 0.5, ...}
Saved final model: .../final_model.pt
Training and Evaluation completed.
```
`Loss: 0.0000` is expected, not a bug: with zero branching and one sample per complex, there's no
variance for the policy-gradient advantage term to act on — the degenerate case of this specific
minimal config, not evidence of a broken training step. The reward values computed throughout
(0.92–1.0, both in training-time sampling and validation) are sane and consistent, confirming the
PoseBusters-based reward pipeline itself works correctly on the modernized stack. Data loading,
checkpoint loading (75.8M-param pretrained score model), forward trajectory sampling, the PPO
gradient-accumulation/update step, and checkpoint saving all ran without error.

This smoke test used a maximally minimal configuration specifically to isolate "does the training
step execute at all" from "is training meaningful" — it does not validate convergence or that the
RL fine-tuning actually improves pose validity, which would require a real training run (branching
enabled, many epochs, full PDBBind) best done on the HPC fleet (Phase 4), not this 6GB laptop GPU.

## Files changed
`RLDiff/inference_env.yml`, `RLDiff/training_env.yml`, `RLDiff/utils/compute_probability_utils.py`;
deleted `RLDiff/utils/interaction.py`, `RLDiff/utils/reward_utils.py`, `RLDiff/utils/system_prep.py`.

## Not yet done
- GNINA binary compatibility with the target CUDA/driver — untested, no local install.
- The `trajectory_generation_val` infinite-retry-on-total-failure gap noted above is unfixed;
  recommend a hard iteration cap if RLDiff's own maintainers pick this up.
- Real-scale RL fine-tuning validation (branching enabled, full dataset, meaningful convergence
  check) is deferred to Phase 4 on the HPC fleet — this laptop's 6GB GPU isn't the target training
  hardware, and even this minimal smoke test took close to an hour once the config bugs were fixed.
