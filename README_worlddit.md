# WorldDiT reconstruction on LIBERO-Object

A from-scratch reconstruction of [WorldDiT](https://arxiv.org/abs/2607.23909)
(Bagel Labs, "A Unified Diffusion Architecture for World and Action
Modeling") — a small (~400M param) diffusion-transformer robot policy that
jointly learns action generation and an auxiliary future-frame-prediction
objective, then drops the frame-prediction head at inference. Trained here
on the LIBERO-Object suite as a learning exercise and an efficiency
comparison against the sibling `~/dev/vla0` project (VLA-0 on Qwen3-VL-4B,
~4.47B params).

The official release ([`bageldotcom/worlddit`](https://huggingface.co/bageldotcom/worlddit))
ships **inference-only** code — the RGB world-modeling pathway is stripped
out entirely since it's unused at deployment. Everything needed to actually
*train* the model (the RGB pathway, the flow-matching training loop, the
data windowing) was reconstructed here from the paper's text and figures. See
[`ASSUMPTIONS.md`](ASSUMPTIONS.md) for exactly which parts are paper-verified
vs. judgment calls.

## Major components

**Reused verbatim from the release** (`scripts/reference_inference.py`,
unmodified copy kept for diffing):
- `VisionEncoder` — frozen MAE ViT-B/16 image encoder
- `PerceiverResampler` — compresses each frame's 196 patch tokens to 16 latents
- `DFDiTBlock` — AdaLN-modulated DiT block (the shared backbone's building block)
- CLIP text/state encoding path

**New, reconstructed from the paper** (`scripts/model.py`):
- `WorldModelSampler` — extends the release's action-only `ActionSampler`
  with an RGB tokenizer/decoder, RGB positional/type embeddings, and an
  extended block-causal attention mask that adds an RGB token block per
  the paper's "action-safe" attention requirement (action queries can't
  see noised RGB targets)
- `WorldDiTTrainingPolicy.training_step()` — flow-matching loss over both
  action and RGB targets, `Eq. 5`'s weighted sum (`w_action=0.1, w_rgb=0.001`)
- `WorldDiTTrainingPolicy.generate()` — the actual inference/sampling path
  (20-step Euler integration from Gaussian noise), ported from the release's
  `ActionSampler.forward()` and extended to skip RGB entirely at inference
  (verified empirically: peak memory during `generate()` is lower than during
  a training step, confirming RGB truly isn't computed)

**Parameter count**: 356.9M total / 119.9M trainable (paper reports
399.084M / 135.107M — same ballpark, not exact; see `ASSUMPTIONS.md`).

## Window layout

Verified line-by-line against the paper's Eq. 1–4 (Sec. 2.1) and
Fig. 3 — see the chat history for the full derivation. A training window has
`N=10` steps: `C=3` context steps (images + state, ending at the "current"
step) + `H=7` action-chunk steps (starting **at** the last context step, not
after it) + 1 RGB target frame at the very last step of the window (`N-1`).

```
window-local index:   0   1   2   3   4   5   6   7   8   9
                      [--- context (C=3) ---]
                                [------- action target (H=7) -------]
                                                                     [RGB target]
```

Implemented in `scripts/data.py`'s `WorldDiTWindowDataset`.

## Setup

```bash
conda activate wdit   # separate env from the sibling vla0 project
# torch 2.10.0+cu128, clip 1.0, timm 1.0.29, einops(-exts), lerobot 0.4.4, wandb, matplotlib
```

Data: reuses the sibling `~/dev/vla0` project's already-downloaded
`lerobot/libero` dataset and `data/libero_object_index.json` episode split
directly (no separate download). MAE + CLIP dependency weights and the
released per-suite checkpoints are pulled from `bageldotcom/worlddit` into
`worlddit_ref/dependencies/` (`mae_pretrain_vit_base.pth`, `ViT-B-32.pt`,
~344MB + ~354MB).

Model config lives at `worlddit_ref/config.json`; the reference release also
ships `eval.py`, which needs a real LIBERO/robosuite/MuJoCo checkout for
closed-loop rollout eval — **not used here**, same constraint the sibling
vla0 project has held all along on this headless machine.

## Run guide

**Sanity checks first, before trusting a real run** (`scripts/overfit_check.py`):
```bash
# Can the model memorize a single window? (~1-2 min, cheap)
python overfit_check.py --scope sample --epochs 300 --lr 1e-4
# Can it generalize across all windows of one episode? (~5-10 min)
python overfit_check.py --scope episode --epochs 30 --lr 1e-4
```
Both write comparison figures + report MAE. Confirmed working: single-window
overfit reaches MAE ~0.12, episode-scope (133 windows) reaches mean MAE
~0.14 — real learning, not memorization-only, and no RGB-pathway bug
(action-only vs. combined-loss training converge to the same MAE at equal
step count, ruling out RGB as disruptive).

**Training** (`scripts/train.py`) — epoch-based, shuffles each epoch, cosine
LR schedule down to a floor fraction of peak, checkpoints once per epoch
(always overwriting the same file, so disk use stays bounded):
```bash
python train.py \
  --epochs 28 --batch-size 48 --stride 4 \
  --lr 1e-4 --lr-min-rate 0.1 \
  --eval-every 200 --eval-stride 16 \
  --output ../outputs/run1 --wandb-run-name wdit-object-run1
```
`--stride` subsamples window start-frames per episode (must be `<= H=7` to
guarantee every ground-truth action is covered by some window). Measured
throughput at `batch=48`: ~4.8-4.9s/step on this machine's Tesla T4
(15.6GB) — `--stride 4` (15,084 windows) at 28 epochs is ~11.8h, chosen to
roughly match the paper's own 30-epoch fine-tune recipe within a ~12h budget.
Batch-size ceiling (`scripts/find_max_batch_size.py`, real forward+backward
sweep): fits up to `batch=64` (14.9GB), recommend `batch=48` (11.75GB, real
margin) over the edge.

Eval uses a **fixed seed** (`--eval-seed`, default 12345) for the
flow-matching noise/timestep draw, so `eval/loss_action` is a repeatable
point estimate rather than fresh random noise each call — without this,
`eval/loss_action` (49-value target) is dominated by per-call sampling
variance in a way `eval/loss_rgb` (~98,300-value target) isn't, purely from
the dimensionality gap (verified: coefficient of variation 0.88 vs 0.15
pre-fix). RNG state is saved/restored around eval so it doesn't perturb
training's own random stream.

**Current run status**: `run1` ran 13 of 28 planned epochs (~8.7h, step
4258/8820) before being manually interrupted; `outputs/run1/checkpoint.pt`
holds that state. Mid-training eval quality was already strong: **mean
overall MAE ~0.06-0.09** on held-out validation windows (see below), well
past pure-noise territory.

**Visualization** (`scripts/visualize.py`) — predicted vs. ground-truth
7-channel (dx, dy, dz, d_roll, d_pitch, d_yaw, gripper) comparison plots for
any checkpoint and any train/val sample:
```bash
# random samples across the whole split
python visualize.py --checkpoint ../outputs/run1/checkpoint.pt --split val --n-samples 4

# every episode in the split, 10 evenly-spaced timepoints each,
# organized into one subfolder per episode
python visualize.py --checkpoint ../outputs/run1/checkpoint.pt --split val \
  --per-episode --points-per-episode 10 --output ../outputs/visualize_val
```
Omit `--checkpoint` to sanity-check untrained weights. If the GPU is busy
with a live training run, force CPU-only inference with
`CUDA_VISIBLE_DEVICES="" python visualize.py ...` to avoid any memory
contention risk with the training process (verified zero GPU footprint via
`nvidia-smi` when done this way).

## Caveats

- **Timing resolution likely doesn't match the paper.** LIBERO's official
  demonstration-collection script sets `control_freq=20` (verified directly
  from LIBERO's own source) — i.e. native 20Hz. The `lerobot/libero` HF
  dataset this project (and the sibling vla0 project) trains on is exported
  at **10Hz** (`meta/info.json`, independently corroborated). Neither the
  WorldDiT nor VLA-0 paper states an explicit fps/Hz number, but both train
  via LIBERO's standard benchmark tooling on raw demonstrations, which
  strongly implies native-rate (20Hz) data. Our `H=7`-step windows almost
  certainly span **~2x the real-world time duration** (0.7s vs. a likely
  0.35s) that the paper's identically-numbered windows do. Step *counts*
  match the paper exactly; real-world *time horizon* is not verified to
  match and probably doesn't. Fixing this would mean sourcing native-rate
  LIBERO data (raw HDF5 + robosuite) instead of the `lerobot/libero`
  re-export — a real dependency addition this project has deliberately
  avoided so far.
- **RGB pathway is a from-paper reconstruction, not verified against real
  code** — the release never shipped it. Empirically: architecture spec
  (depth, hidden size, heads, register tokens, C/H/N, RGB token counts) is a
  verified exact match to the paper's Sec. 3.1.1; the actual masking/attention
  details beyond the one explicit "action-safe" requirement are judgment
  calls (see `ASSUMPTIONS.md`).
- **Param count is off from the paper's** (356.9M/119.9M vs. 399.084M/135.107M
  trainable) — not investigated further; likely a sizing detail in the RGB
  pathway (Linear vs. MLP projections).
- **No closed-loop rollout eval.** Everything here is open-loop: one
  `generate()` call from one window, compared against the recorded ground
  truth. Real success-rate numbers would need the LIBERO/robosuite
  simulator, which this project has not integrated (same tradeoff the
  sibling vla0 project made).
- **Known weak spot: gripper near grasp/release transitions.** Consistently
  the largest error source across every check run so far (e.g. MAE spikes to
  0.8-1.4 on the gripper channel specifically at transition frames, while
  tracking near-perfectly elsewhere) — the same failure mode independently
  observed in the sibling vla0/Qwen3-VL project, suggesting it's a genuine
  property of the task/data rather than a bug in either implementation.
- **Training run was manually interrupted**, not run to completion (13/28
  epochs). The LR schedule was tuned for a full 28-epoch run, so the
  checkpoint's LR at interruption doesn't reflect a properly-annealed final
  state — treat it as a strong mid-training checkpoint, not a finished model.
