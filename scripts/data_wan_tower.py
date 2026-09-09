"""Dataset for the Wan-DiT tower experiment: 5-frame (1 anchor + 1x4 chunk)
context window to genuinely exercise Wan-VAE's native causal-conv temporal
chunking, plus a Gaussian keypoint-heatmap overlay drawn directly onto the
context frames at the (known, exact) end-effector pixel location -- role 2
of the keypoint plan, using the precomputed projections from
precompute_ee_pixels.py (privileged simulator geometry, verified against
real rendered frames).

Reuses ACTION_HORIZON, list_raw_libero_tasks, split_train_val_demos from the
existing data.py -- only the context window and the added EE/heatmap
features are new."""

from __future__ import annotations

import random
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

# Deliberately self-contained -- data.py imports lerobot unconditionally at
# module level (needed for the WorldDiTWindowDataset/lerobot-libero path),
# which conflicts with peft's transformers/huggingface-hub requirements in
# the wdit env. This tower pipeline never touches lerobot, so these three
# pieces (copied from data.py, not re-exported) let it import cleanly in an
# env where lerobot isn't installed at all, e.g. wdit_tower.

ACTION_HORIZON = 7


def _task_name_from_hdf5_stem(stem: str) -> str:
    return stem.removesuffix("_demo").replace("_", " ")


def list_raw_libero_tasks(raw_dir: str) -> dict[str, Path]:
    """{task_instruction: hdf5_path} for every *_demo.hdf5 file directly under raw_dir/libero_object/."""
    root = Path(raw_dir)
    search_dir = root / "libero_object" if (root / "libero_object").is_dir() else root
    return {_task_name_from_hdf5_stem(p.stem): p for p in sorted(search_dir.glob("*_demo.hdf5"))}


def split_train_val_demos(raw_dir: str, val_demos_per_task: int = 2, seed: int = 0):
    """Episode(demo)-level train/val split. Returns {task: (train_demo_ids, val_demo_ids)}."""
    tasks = list_raw_libero_tasks(raw_dir)
    rng = random.Random(seed)
    splits = {}
    for task, path in tasks.items():
        with h5py.File(path, "r") as f:
            demo_ids = sorted(int(k.split("_")[1]) for k in f["data"].keys())
        rng.shuffle(demo_ids)
        val_ids = sorted(demo_ids[:val_demos_per_task])
        train_ids = sorted(demo_ids[val_demos_per_task:])
        splits[task] = (train_ids, val_ids)
    return splits


CONTEXT_STEPS_TOWER = 5  # 1 anchor + 1x4 causal-conv chunk -- Wan-VAE's native temporal unit
WINDOW_LEN_TOWER = CONTEXT_STEPS_TOWER + ACTION_HORIZON  # 12
HEATMAP_SIGMA = 4.0  # pixels, at the native 128x128 resolution


def draw_gaussian(img: np.ndarray, row: int, col: int, sigma: float = HEATMAP_SIGMA) -> np.ndarray:
    """img: (H, W, 3) uint8, modified in place with a red-channel-boosted
    Gaussian blob at (row, col) -- a visible, differentiable-in-spirit
    marker baked directly into the pixels the VAE will encode."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    g = np.exp(-((yy - row) ** 2 + (xx - col) ** 2) / (2 * sigma ** 2))
    out = img.astype(np.float32)
    out[..., 0] = np.clip(out[..., 0] + 255.0 * g, 0, 255)  # boost red channel at the keypoint
    out[..., 1] = np.clip(out[..., 1] * (1 - 0.6 * g), 0, 255)  # dim other channels for contrast
    out[..., 2] = np.clip(out[..., 2] * (1 - 0.6 * g), 0, 255)
    return out.astype(np.uint8)


class LiberoWanTowerDataset(torch.utils.data.Dataset):
    """Windowed dataset for the Wan-DiT tower: CONTEXT_STEPS_TOWER=5 context
    frames (heatmap-overlaid) + ACTION_HORIZON=7 action chunk, plus the raw
    3D end-effector trajectory and future 2D pixel targets for the new
    heatmap-prediction head."""

    def __init__(self, raw_dir: str, task_demo_map: dict[str, list[int]], image_processor, stride: int = 8):
        self.image_processor = image_processor
        tasks = list_raw_libero_tasks(raw_dir)
        self.files = {task: h5py.File(tasks[task], "r") for task in task_demo_map}

        ee_pixel_dir = Path(raw_dir) / "ee_pixels"
        self.ee_pixels = {}  # task -> npz
        for task in task_demo_map:
            stem = Path(tasks[task]).stem
            npz_path = ee_pixel_dir / f"{stem}_ee_pixels.npz"
            if not npz_path.exists():
                raise FileNotFoundError(
                    f"missing precomputed EE pixels for task '{task}': {npz_path} "
                    "-- run precompute_ee_pixels.py first"
                )
            self.ee_pixels[task] = np.load(npz_path)

        self.index = []  # (task, demo_id, start_frame)
        for task, demo_ids in task_demo_map.items():
            data = self.files[task]["data"]
            for demo_id in demo_ids:
                length = data[f"demo_{demo_id}"]["actions"].shape[0]
                self.index.extend(
                    (task, demo_id, start) for start in range(0, max(length - WINDOW_LEN_TOWER, 1), stride)
                )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        task, demo_id, start = self.index[idx]
        demo = self.files[task]["data"][f"demo_{demo_id}"]
        obs = demo["obs"]
        ee_px = self.ee_pixels[task]
        px_primary = ee_px[f"demo_{demo_id}_agentview"]  # (T, 2) int32, (row, col) at 128x128
        px_wrist = ee_px[f"demo_{demo_id}_eye_in_hand"]

        primary = obs["agentview_rgb"][start : start + WINDOW_LEN_TOWER]  # (WINDOW_LEN_TOWER, 128, 128, 3)
        wrist = obs["eye_in_hand_rgb"][start : start + WINDOW_LEN_TOWER]

        ee_pos = obs["ee_pos"][start : start + CONTEXT_STEPS_TOWER]
        ee_ori = obs["ee_ori"][start : start + CONTEXT_STEPS_TOWER]
        gripper = obs["gripper_states"][start : start + CONTEXT_STEPS_TOWER]
        state = torch.from_numpy(np.concatenate([ee_pos, ee_ori, gripper], axis=1)).float()  # (5, 8)

        action = torch.from_numpy(
            demo["actions"][start + CONTEXT_STEPS_TOWER - 1 : start + CONTEXT_STEPS_TOWER - 1 + ACTION_HORIZON]
        ).float()  # (7, 7)

        # heatmap overlay baked directly into the context frames, at the raw
        # 128x128 resolution (matching the precomputed pixel coords) before
        # any resize/normalization -- role 2 of the keypoint plan.
        context_primary_frames = []
        context_wrist_frames = []
        for t in range(CONTEXT_STEPS_TOWER):
            fidx = start + t
            p_frame = draw_gaussian(primary[t].copy(), int(px_primary[fidx, 0]), int(px_primary[fidx, 1]))
            w_frame = draw_gaussian(wrist[t].copy(), int(px_wrist[fidx, 0]), int(px_wrist[fidx, 1]))
            context_primary_frames.append(self.image_processor(Image.fromarray(p_frame)))
            context_wrist_frames.append(self.image_processor(Image.fromarray(w_frame)))
        context_primary = torch.stack(context_primary_frames)  # (5, 3, 224, 224)
        context_wrist = torch.stack(context_wrist_frames)

        # future 3D EE trajectory (auxiliary, role 1) and future 2D pixel
        # targets (new heatmap-prediction head, role 2 extended) -- both
        # over the same H=7 action horizon, starting at the same offset as
        # the action chunk.
        fut_start = start + CONTEXT_STEPS_TOWER - 1
        ee_pos_target = torch.from_numpy(obs["ee_pos"][fut_start : fut_start + ACTION_HORIZON]).float()  # (7, 3)
        future_px_primary = torch.from_numpy(px_primary[fut_start : fut_start + ACTION_HORIZON].copy()).long()  # (7, 2)
        future_px_wrist = torch.from_numpy(px_wrist[fut_start : fut_start + ACTION_HORIZON].copy()).long()

        return {
            "image_primary": context_primary,
            "image_wrist": context_wrist,
            "state": state,
            "instruction": task,
            "action_target": action,
            "ee_pos_target": ee_pos_target,
            "future_px_primary": future_px_primary,
            "future_px_wrist": future_px_wrist,
            "episode_index": demo_id,
            "frame_index": start,
        }


def collate_tower(samples: list[dict]) -> dict:
    import clip

    text_tokens = clip.tokenize(
        [s["instruction"] for s in samples for _ in range(CONTEXT_STEPS_TOWER)], truncate=True
    )
    text_tokens = text_tokens.view(len(samples), CONTEXT_STEPS_TOWER, -1)
    return {
        "image_primary": torch.stack([s["image_primary"] for s in samples]),
        "image_wrist": torch.stack([s["image_wrist"] for s in samples]),
        "state": torch.stack([s["state"] for s in samples]),
        "text_token": text_tokens,
        "action_target": torch.stack([s["action_target"] for s in samples]),
        "ee_pos_target": torch.stack([s["ee_pos_target"] for s in samples]),
        "future_px_primary": torch.stack([s["future_px_primary"] for s in samples]),
        "future_px_wrist": torch.stack([s["future_px_wrist"] for s in samples]),
    }
