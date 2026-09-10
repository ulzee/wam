#!/usr/bin/env python3
"""Training loop for the Wan-DiT tower (state-free conditioning, soft-binned
heatmap head) on LIBERO-Object. Mirrors train.py's conventions (epoch-based,
shuffles window order each epoch, cosine LR decay to a floor fraction of
peak, checkpoints once per epoch overwriting the same file, seed-fixed eval
for a repeatable point estimate) adapted for this tower's own model/data.

Default config is the validated "current config" from the architecture
diagram: dim=1024, 10 blocks, 16 heads, ffn_dim=4096, from-scratch (no
pretrained DiT weights) -- confirmed to fit at batch size 16 with real
margin and to actually learn (declining loss on both heads) in a 40-step
fixed-batch overfit check before this script was written.
"""

import argparse
import time

import clip
import numpy as np
import torch

from data_wan_tower import LiberoWanTowerDataset, collate_tower, split_train_val_demos
from model_wan_tower import (
    WanTowerPolicy, soft_bin_target, soft_ce_loss, save_trainable_state, load_partial_state, HEATMAP_BINS,
)
from clip_text_sequence import clip_encode_text_sequence


def compute_losses(model, batch, clip_model, clip_mean, clip_std, device, w_action, w_heatmap, eval_seed=None):
    primary = (batch["image_primary"].to(device) * clip_std + clip_mean) * 2 - 1
    action_target = batch["action_target"].to(device)
    future_px = batch["future_px_primary"].to(device)

    with torch.no_grad():
        tokens = clip.tokenize(batch["instruction"], truncate=True).to(device)
        text_embed_seq, text_context_lens = clip_encode_text_sequence(clip_model, tokens)

    cpu_rng_state = cuda_rng_state = None
    if eval_seed is not None:
        # Fix the flow-matching noise/timestep draw for eval so the metric is
        # a repeatable point estimate rather than fresh random noise every
        # call, then restore the exact prior RNG state so training's own
        # random stream is undisturbed. Same trick as train.py's eval loop.
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state(device) if device == "cuda" else None
        torch.manual_seed(eval_seed)
        if device == "cuda":
            torch.cuda.manual_seed(eval_seed)

    pred_v, pred_hm, target_v = model(primary, text_embed_seq, text_context_lens, action_target=action_target)

    if eval_seed is not None:
        torch.set_rng_state(cpu_rng_state)
        if device == "cuda":
            torch.cuda.set_rng_state(cuda_rng_state, device)

    loss_action = torch.nn.functional.mse_loss(pred_v, target_v)
    target_row = soft_bin_target(future_px[..., 0], image_size=128, bins=HEATMAP_BINS)
    target_col = soft_bin_target(future_px[..., 1], image_size=128, bins=HEATMAP_BINS)
    loss_heatmap = soft_ce_loss(pred_hm[:, :, 0, :], target_row) + soft_ce_loss(pred_hm[:, :, 1, :], target_col)
    total = w_action * loss_action + w_heatmap * loss_heatmap
    return loss_action, loss_heatmap, total


def collate_with_instruction(samples):
    batch = collate_tower(samples)
    batch["instruction"] = [s["instruction"] for s in samples]
    return batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default="/home/ubuntu/dev/wdit/data/libero_object_raw")
    p.add_argument("--vae", default="../worlddit_ref/dependencies/Wan2.1_VAE.pth")
    p.add_argument("--clip", default="../worlddit_ref/dependencies/ViT-B-32.pt",
                    help="used for both image preprocessing AND text conditioning now -- real per-token "
                         "cross-attention over CLIP's own ln_final sequence (see clip_text_sequence.py), "
                         "not a single pooled vector")
    p.add_argument("--dit-dim", type=int, default=1024)
    p.add_argument("--dit-blocks", type=int, default=10)
    p.add_argument("--dit-heads", type=int, default=16)
    p.add_argument("--dit-ffn", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=0,
                    help="stop after N total steps regardless of --epochs; 0 = disabled (default). "
                         "For smoke-testing the script itself, not for real runs.")
    p.add_argument("--batch-size", type=int, default=16,
                    help="validated operating point for the default config (12.42GB, real margin; "
                         "ceiling ~22 before OOM)")
    p.add_argument("--lr", type=float, default=3e-4,
                    help="higher than train.py's 1e-4 default -- this DiT is trained fully from "
                         "random init, not fine-tuned from pretrained weights, and 3e-4 is what the "
                         "40-step overfit check already validated as stable")
    p.add_argument("--lr-min-rate", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0,
                    help="max grad norm; 0 disables. Not present in train.py (mostly-frozen fine-tune) "
                         "but this model trains fully from scratch on real diverse data for the first "
                         "time here, past only a 40-step fixed-batch check -- clip as a safety margin")
    p.add_argument("--w-action", type=float, default=1.0)
    p.add_argument("--w-heatmap", type=float, default=1.0)
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--val-demos-per-task", type=int, default=2)
    p.add_argument("--eval-stride", type=int, default=32,
                    help="coarser than train stride -- eval is a monitoring signal, not the real metric")
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--eval-every", type=int, default=0,
                    help="run eval every N steps; 0 disables eval entirely (default, for smoke tests)")
    p.add_argument("--eval-seed", type=int, default=12345)
    p.add_argument("--wandb-project", default="wdit-tower-libero")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--output", default=None,
                    help="dir to save the checkpoint in (e.g. outputs/tower-run1); omit to skip saving")
    p.add_argument("--init-from", default=None,
                    help="partially initialize from a prior checkpoint.pt (e.g. from a different text-encoder "
                         "variant) -- loads only tensors whose key AND shape match the current model, skips "
                         "the rest (they train from random init). See model_wan_tower.py's load_partial_state "
                         "for exactly what transfers: DiT self_attn/ffn/cross_attn (dim->dim, independent of "
                         "text_dim) and the action/heatmap heads carry over; dit.text_embedding's first layer "
                         "(sized by text_dim) does not if the source used a different text encoder.")
    p.add_argument("--ckpt-every", type=int, default=0,
                    help="also save (overwriting the same checkpoint.pt) every N steps, not just once per "
                         "epoch; 0 = epoch-end only (default). Matters once an epoch takes long enough that "
                         "losing it to a crash/preemption would be expensive -- irrelevant for short runs.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    dit_config = dict(
        model_type="t2v", dim=args.dit_dim, ffn_dim=args.dit_ffn, freq_dim=256, text_dim=512,
        out_dim=16, num_heads=args.dit_heads, num_layers=args.dit_blocks, text_len=77, in_dim=16,
    )
    model = WanTowerPolicy(args.vae, None, dit_config).to(device)
    if args.init_from:
        load_partial_state(model, args.init_from)
    model.train()
    n_total = sum(p_.numel() for p_ in model.parameters())
    n_trainable = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    print(f"DiT config: dim={args.dit_dim} blocks={args.dit_blocks} heads={args.dit_heads} ffn={args.dit_ffn}")
    print(f"total params: {n_total:,}  trainable: {n_trainable:,}")

    clip_model, preprocess = clip.load(args.clip, device="cpu")
    clip_model = clip_model.to(device).eval()
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 1, 3, 1, 1)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 1, 3, 1, 1)

    splits = split_train_val_demos(args.raw_dir, val_demos_per_task=args.val_demos_per_task, seed=0)
    train_map = {task: ids[0] for task, ids in splits.items()}
    val_map = {task: ids[1] for task, ids in splits.items()}
    ds = LiberoWanTowerDataset(args.raw_dir, train_map, preprocess, stride=args.stride)
    val_ds = None
    if args.eval_every > 0:
        val_ds = LiberoWanTowerDataset(args.raw_dir, val_map, preprocess, stride=args.eval_stride)
    print(f"train windows: {len(ds)}")
    if val_ds is not None:
        print(f"val windows: {len(val_ds)}")

    if not args.no_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    optimizer = torch.optim.AdamW([p_ for p_ in model.parameters() if p_.requires_grad], lr=args.lr)
    steps_per_epoch = -(-len(ds) // args.batch_size)  # ceil
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr * args.lr_min_rate
    )
    print(f"steps/epoch: {steps_per_epoch}  total steps: {total_steps}  "
          f"LR: {args.lr:.2e} -> {args.lr * args.lr_min_rate:.2e} (cosine)  grad_clip={args.grad_clip}")

    rng = np.random.default_rng(args.seed)
    global_step = 0
    t_run0 = time.perf_counter()
    for epoch in range(args.epochs):
        perm = rng.permutation(len(ds))
        for start in range(0, len(perm), args.batch_size):
            idxs = perm[start:start + args.batch_size]
            batch = collate_with_instruction([ds[int(i)] for i in idxs])

            t0 = time.perf_counter()
            loss_action, loss_heatmap, total_loss = compute_losses(
                model, batch, clip_model, clip_mean, clip_std, device, args.w_action, args.w_heatmap
            )
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            grad_norm = None
            if args.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p_ for p_ in model.parameters() if p_.requires_grad], args.grad_clip
                )
            optimizer.step()
            current_lr = scheduler.get_last_lr()[0]
            scheduler.step()
            if device == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            gn_str = f" grad_norm={grad_norm.item():.2f}" if grad_norm is not None else ""
            print(f"epoch {epoch} step {global_step}: loss_action={loss_action.item():.4f} "
                  f"loss_heatmap={loss_heatmap.item():.4f} total={total_loss.item():.4f} "
                  f"lr={current_lr:.2e}{gn_str}  ({dt:.2f}s)")

            log = {
                "train/loss_action": loss_action.item(),
                "train/loss_heatmap": loss_heatmap.item(),
                "train/loss_total": total_loss.item(),
                "train/step_time_s": dt,
                "train/lr": current_lr,
                "train/epoch": epoch,
            }
            if grad_norm is not None:
                log["train/grad_norm"] = grad_norm.item()
            if device == "cuda":
                log["train/peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9

            if val_ds is not None and (global_step + 1) % args.eval_every == 0:
                model.eval()
                eval_samples = [val_ds[i % len(val_ds)] for i in range(args.eval_batch_size)]
                eval_batch = collate_with_instruction(eval_samples)
                with torch.no_grad():
                    ev_action, ev_heatmap, ev_total = compute_losses(
                        model, eval_batch, clip_model, clip_mean, clip_std, device,
                        args.w_action, args.w_heatmap, eval_seed=args.eval_seed,
                    )
                model.train()
                print(f"  eval @ step {global_step}: loss_action={ev_action.item():.4f} "
                      f"loss_heatmap={ev_heatmap.item():.4f} total={ev_total.item():.4f}")
                log.update({
                    "eval/loss_action": ev_action.item(),
                    "eval/loss_heatmap": ev_heatmap.item(),
                    "eval/loss_total": ev_total.item(),
                })

            if not args.no_wandb:
                import wandb
                wandb.log(log, step=global_step)
            global_step += 1

            if args.output and args.ckpt_every > 0 and global_step % args.ckpt_every == 0:
                ckpt_path = f"{args.output}/checkpoint.pt"
                save_trainable_state(model, ckpt_path)
                print(f"  step {global_step}: saved checkpoint: {ckpt_path}")

            if args.max_steps > 0 and global_step >= args.max_steps:
                if args.output:
                    ckpt_path = f"{args.output}/checkpoint.pt"
                    save_trainable_state(model, ckpt_path)
                    print(f"max-steps reached -- saved checkpoint: {ckpt_path}")
                print(f"\nstopped early at --max-steps={args.max_steps}")
                if device == "cuda":
                    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
                print(f"total training time: {time.perf_counter() - t_run0:.0f}s")
                if not args.no_wandb:
                    import wandb
                    wandb.finish()
                print("\nTRAINING DONE")
                return

        if args.output:
            ckpt_path = f"{args.output}/checkpoint.pt"
            save_trainable_state(model, ckpt_path)
            print(f"epoch {epoch} done -- saved checkpoint: {ckpt_path} "
                  f"({time.perf_counter() - t_run0:.0f}s elapsed)")

    if device == "cuda":
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"\npeak GPU memory: {peak_gb:.2f} GB")
    print(f"total training time: {time.perf_counter() - t_run0:.0f}s")
    if not args.no_wandb:
        import wandb
        wandb.finish()
    print("\nTRAINING DONE")


if __name__ == "__main__":
    main()
