"""Sweep batch size for the full WanTowerPolicy (VAE + LoRA-wrapped DiT +
action/heatmap heads) with a real optimizer step at each size, to find the
largest batch that actually trains on this machine -- not just fits a
forward pass. Run in the wdit_tower env."""
import sys
import time
from pathlib import Path

import torch
import clip
from safetensors.torch import load_file
from peft import LoraConfig, get_peft_model

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_wan_tower import LiberoWanTowerDataset, collate_tower, split_train_val_demos
from model_wan_tower import WanTowerPolicy, soft_bin_target, soft_ce_loss, HEATMAP_BINS

device = "cuda"
raw_dir = "/home/ubuntu/dev/wdit/data/libero_object_raw"

splits = split_train_val_demos(raw_dir, val_demos_per_task=2, seed=0)
train_map = {task: ids[0] for task, ids in splits.items()}
clip_model, preprocess = clip.load("../worlddit_ref/dependencies/ViT-B-32.pt", device="cpu")
ds = LiberoWanTowerDataset(raw_dir, train_map, preprocess, stride=16)
print(f"dataset windows available: {len(ds)}")

dit_config = dict(model_type="t2v", dim=1536, ffn_dim=8960, freq_dim=256, text_dim=4096,
                   out_dim=16, num_heads=12, num_layers=30, text_len=512, in_dim=16)
policy = WanTowerPolicy("../worlddit_ref/dependencies/Wan2.1_VAE.pth",
                         "../worlddit_ref/dependencies/Wan2.1_DiT_1.3B.safetensors", dit_config)
sd = load_file("../worlddit_ref/dependencies/Wan2.1_DiT_1.3B.safetensors")
policy.dit.load_state_dict(sd, strict=False)
del sd

lora_cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["q", "k", "v", "o"], lora_dropout=0.0)
policy.dit = get_peft_model(policy.dit, lora_cfg)
policy = policy.to(device)
policy.train()

n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in policy.parameters())
print(f"trainable: {n_trainable:,} / total: {n_total:,}")

opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=1e-4)
clip_model = clip_model.to(device)

clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 1, 3, 1, 1)
clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 1, 3, 1, 1)


def build_batch(bs):
    samples = [ds[i % len(ds)] for i in range(bs)]
    batch = collate_tower(samples)
    primary = (batch["image_primary"].to(device) * clip_std + clip_mean) * 2 - 1
    action_target = batch["action_target"].to(device)
    future_px = batch["future_px_primary"].to(device)
    with torch.no_grad():
        text_embed = clip_model.encode_text(clip.tokenize(batch2instr(samples)).to(device)).float()
    return primary, action_target, future_px, text_embed


def batch2instr(samples):
    return [s["instruction"] for s in samples]


for bs in [5, 6, 7]:
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        primary, action_target, future_px, text_embed = build_batch(bs)

        t0 = time.perf_counter()
        pred_v, pred_hm, target_v = policy(primary, text_embed, action_target=action_target)
        loss_action = torch.nn.functional.mse_loss(pred_v, target_v)

        # future_px: (B,7,2) = (row, col) at 128x128 -> soft (linearly interpolated) bin targets
        target_row = soft_bin_target(future_px[..., 0], image_size=128, bins=HEATMAP_BINS)
        target_col = soft_bin_target(future_px[..., 1], image_size=128, bins=HEATMAP_BINS)
        loss_hm = soft_ce_loss(pred_hm[:, :, 0, :], target_row) + soft_ce_loss(pred_hm[:, :, 1, :], target_col)
        loss = loss_action + loss_hm

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"bs={bs:3d}  OK   peak_mem={peak:.2f}GB  step_time={dt:.2f}s  loss_action={loss_action.item():.3f} loss_hm={loss_hm.item():.3f}")
    except torch.cuda.OutOfMemoryError:
        print(f"bs={bs:3d}  OOM")
        torch.cuda.empty_cache()
        break
