# Wan-DiT tower on LIBERO-Object

A from-scratch, small (~313M param) diffusion-transformer robot policy built
on **Wan2.1**'s architecture (VAE + DiT block design), not its pretrained
weights — the DiT is trained fully from random init here, only the video VAE
is frozen and reused. Separate sub-project from
[`README_worlddit.md`](README_worlddit.md)'s MAE-based WorldDiT
reconstruction; shares the `scripts/` directory but no code.

## Model architecture

```
5 context frames (heatmap-overlaid)         instruction (text)
        │                                          │
  Wan-VAE encoder (frozen, 126.9M)          UMT5-XXL (offline, cached)
   1 anchor + 1x4 causal-conv chunk           per-token (≤512, 4096) embed
        │                                          │
  Wan patchify conv (frozen)                dit.text_embedding
        │                                    (Linear 4096→1024→1024,
   392 video tokens (dim=1024)                Wan's own native module)
        │                                          │
        └──────────────┬───────────────────────────┘
                        │
          10x WanAttentionBlock (from scratch)
          self-attn+RoPE (video/action/heatmap tokens)
          cross-attn → text context (every block)
          AdaLN-Zero FFN, one shared timestep
                        │
        ┌───────────────┴───────────────┐
   action tokens (x7)              heatmap-query tokens (x7)
        │                                │
  action_detokenizer               heatmap_head
  MLP → (7,7) velocity             MLP → (7, 2, 20) row/col bin logits
  (flow-matching, Euler            (soft cross-entropy vs.
   integration at inference)        linearly-interpolated pixel target)
```

- **No robot-state input** — conditions only on video + instruction, deliberately, to stay embodiment-agnostic.
- **Text conditioning is real Wan-style cross-attention**, not a pooled token folded into self-attention. An earlier version used a single pooled CLIP embedding fed into self-attention while the DiT's actual cross-attention sat disconnected (constant zeros) — verified via a causal test that this made the model provably unable to use the instruction at all. Now: UMT5-XXL's real per-token embeddings, projected by Wan's own native `dit.text_embedding` (previously unused), fed as `context` to every block's `cross_attn`.
- **Heatmap head is coordinate-binned, not a dense spatial map**: two independent 20-way categorical distributions (row, col) per future step, decoded via soft-argmax — sub-pixel precision, not snapped to a grid cell.
- Params: **313.2M total / 186.4M trainable** (VAE frozen). Current default config: `dim=1024`, 10 blocks, 16 heads, `ffn_dim=4096`.

## Project structure

```
scripts/
  model_wan_tower.py            WanTowerPolicy: forward/generate, flow-matching + heatmap losses
  data_wan_tower.py              LiberoWanTowerDataset, collate, EE-pixel heatmap overlay
  train_wan_tower.py              training entry point
  rollout_eval_tower.py           closed-loop LIBERO rollout eval + trajectory-video rendering
  precompute_ee_pixels.py         offline: 3D EE pos -> 2D pixel cache (per demo)
  precompute_text_embeddings.py   offline: UMT5-XXL -> per-instruction embedding cache
  wan_vae.py / wan_dit.py / wan_attention.py   vendored Wan2.1 VAE + DiT + attention dispatcher
  wan_t5.py / wan_tokenizers.py   vendored Wan2.1 UMT5-XXL encoder (precompute step only)
  batch_size_sweep.py            [stale] targets the frozen-1.3B/LoRA path, not current scope

worlddit_ref/dependencies/        pretrained weights (see Assets below)
data/libero_object_raw/
  libero_object/*.hdf5            raw LIBERO-Object demos (10 tasks)
  ee_pixels/*.npz                 cached EE->pixel projections (from precompute_ee_pixels.py)
  text_embeddings_umt5.pt         cached UMT5 embeddings (from precompute_text_embeddings.py)

outputs/rollout_tower_run1/       durable rollout eval results (results.json + videos)
scripts/outputs/<run-name>/checkpoint.pt   training checkpoints (per --output)
```

## Assets: what to download, and what auto-downloads

| Asset | Size | Needed for | Auto or manual |
|---|---|---|---|
| `Wan2.1_VAE.pth` | 508MB | training + eval (frozen video encoder) | **manual** — see below |
| `ViT-B-32.pt` (CLIP) | 354MB | training + eval (image preprocessing only) | **manual** — see below |
| `models_t5_umt5-xxl-enc-bf16.pth` | 11.4GB | one-time text cache precompute only | **manual** — see below |
| `google/umt5-xxl` tokenizer files | ~1MB | text cache precompute only | **auto** — `AutoTokenizer.from_pretrained` fetches on first run, cached under `~/.cache/huggingface` |
| `Wan2.1_DiT_1.3B.safetensors` | 5.7GB | **not used** by the current from-scratch model | manual, only if resuming the separate pretrained-1.3B/LoRA path (`batch_size_sweep.py`) |
| raw LIBERO-Object HDF5 demos | — | training/eval data | pre-existing in this environment, not part of this sub-project's setup |
| LIBERO/robosuite checkout (`~/LIBERO`) | — | `rollout_eval_tower.py` only | pre-existing in this environment, not documented in either README |

Manual downloads (into `worlddit_ref/dependencies/`):
```bash
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Wan-AI/Wan2.1-T2V-1.3B', 'Wan2.1_VAE.pth', local_dir='worlddit_ref/dependencies')
hf_hub_download('Wan-AI/Wan2.1-T2V-1.3B', 'models_t5_umt5-xxl-enc-bf16.pth', local_dir='worlddit_ref/dependencies')
"
# ViT-B-32.pt: standard OpenAI CLIP checkpoint (see clip.load's own download helper,
# or reuse the copy already in worlddit_ref/dependencies/ from the sibling WorldDiT setup)
```

The UMT5 checkpoint is only needed transiently, to build `text_embeddings_umt5.pt` once — it's not
loaded again by training or eval afterward.

## Conda environments

Dependency conflicts forced a 3-env split this session:

| env | has | used by |
|---|---|---|
| `wdit` | torch, clip, no robosuite | `train_wan_tower.py` |
| `wdit_eval` | robosuite/LIBERO/MuJoCo, diffusers | `precompute_ee_pixels.py`, `rollout_eval_tower.py` |
| `wdit_tower` | newer `huggingface_hub`/`transformers` (needed by `wan_tokenizers.py`'s `AutoTokenizer`, incompatible with `wdit`'s older pins) | `precompute_text_embeddings.py` only |

## Setup (one-time, before first training run)

```bash
conda activate wdit_eval
cd scripts
python precompute_ee_pixels.py                     # ~1 min, all 10 tasks

conda activate wdit_tower
python precompute_text_embeddings.py                # ~30s once the UMT5 checkpoint is downloaded
```

## Recommended train command

```bash
conda activate wdit
cd scripts
python train_wan_tower.py \
  --stride 8 --epochs 8 --batch-size 16 \
  --eval-every 50 --ckpt-every 250 \
  --output outputs/tower-run1 --wandb-run-name tower-run1
```
- `--stride 8`: default `--stride 16` only yields 4,336 train windows — halving it to 8 (8,428 windows) roughly doubles genuine data diversity without over-sampling near-duplicate frames (`--stride 4`'s windows are only 0.2s apart at 20Hz — too correlated to add much).
- `--batch-size 16`: measured at 9.67GB peak (real forward+backward+`optimizer.step()`), ~5GB margin below the true ceiling (`bs=22` succeeds at 14.90GB, `bs=24` OOMs) — safe for a long unattended run. Bump to `--batch-size 18` (12.80GB) for better throughput with still-comfortable margin.
- `--ckpt-every 250`: saves (overwriting the same file) roughly every ~35 minutes, independent of epoch length — matters once an epoch runs past an hour.
- At `dim=1024`/10 blocks/stride=8, this is ~527 steps/epoch x 8 epochs ≈ 4,216 steps ≈ **~10 hours** at the measured ~8.5s/step.

## 2-episode eval command

```bash
conda activate wdit_eval
cd scripts
python rollout_eval_tower.py \
  --checkpoint outputs/tower-run1/checkpoint.pt \
  --suite libero_object --tasks 1 --episodes 2 \
  --output-dir ../outputs/rollout_tower_eval \
  --save-video --save-trajectory-video
```
Runs task 0 ("pick up the alphabet soup and place it in the basket"), episodes 0 and 1, closed-loop
with temporal action ensembling. `--save-trajectory-video` additionally renders a side-by-side video
(plain | predicted future-EE-position trajectory, decoded from the heatmap head) per episode —
useful for diagnosing *why* a rollout failed, not just whether it did. Success is LIBERO's own BDDL
goal-predicate check (`In(alphabet_soup_1, basket_1_contain_region)` — real geometric
contact+containment, not a proxy), read from `outputs/rollout_tower_eval/results.json`.

## Known caveats

- **The 10-hour checkpoint trained before the cross-attention rewrite is now incompatible** with the
  current model (different parameter set — confirmed by an exact 1,574,912-param delta from removing
  the old CLIP-token text module). A fresh training run is needed to get a checkpoint that actually
  exercises the new text-conditioning pathway.
- **Object-identity confusion, not random failure.** Diagnosed directly against the simulator: the
  pre-rewrite model consistently picked up `salad_dressing_1` (a distractor) instead of the actual
  target `alphabet_soup_1`, across multiple episodes — a real, repeatable grounding failure, not noise.
  The cross-attention rewrite targets exactly this; whether it actually fixes it is an open question a
  real training run (not yet done post-rewrite) will answer.
- `batch_size_sweep.py` has a broken call signature post-rewrite (targets the separate, currently
  out-of-scope frozen-1.3B/LoRA path) — needs updating before reuse.
