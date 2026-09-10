#!/usr/bin/env python3
"""Closed-loop LIBERO rollout evaluation for the Wan-DiT tower checkpoint.

Adapted from rollout_eval.py (the WorldDiT rollout harness) for this model's
real differences:
  - single camera (agentview) only, no wrist
  - no robot-state input at all
  - CONTEXT_STEPS_TOWER=5 context frames, not 3
  - the model was trained on frames with a Gaussian heatmap overlay drawn at
    the live EE pixel position (data_wan_tower.py's draw_gaussian) -- so
    rollout must draw that same overlay on every live frame using the SAME
    projection precompute_ee_pixels.py used (robosuite camera_utils on the
    live sim), or the model sees out-of-distribution input. This is the one
    genuinely new piece of machinery this script adds over the WorldDiT one.
  - WanTowerPolicy doesn't hold its own CLIP model (unlike WorldDiTTrainingPolicy),
    so this script loads CLIP itself for both image preprocessing and text
    encoding, matching train_wan_tower.py's convention.

Needs a real LIBERO checkout + headless EGL rendering + robosuite -- see
README.md for the wdit_eval conda env / LIBERO clone setup this depends on.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import clip
import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_wan_tower import CONTEXT_STEPS_TOWER, draw_gaussian
from model_wan_tower import ACTION_HORIZON, WanTowerPolicy, decode_heatmap_to_px, load_trainable_state

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
IMG_SIZE = 128

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 1, 3, 1, 1)


def configure_libero(libero_root: Path, output: Path):
    package = libero_root / "libero" / "libero"
    required = (package / "bddl_files", package / "init_files", package / "assets")
    if any(not path.is_dir() for path in required):
        raise FileNotFoundError(f"invalid LIBERO checkout: {libero_root}")
    config_dir = output / "libero_config"
    output.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(exist_ok=True)
    paths = {
        "assets": str(package / "assets"),
        "bddl_files": str(package / "bddl_files"),
        "benchmark_root": str(package),
        "datasets": str(package.parent / "datasets"),
        "init_states": str(package / "init_files"),
    }
    (config_dir / "config.yaml").write_text(json.dumps(paths), encoding="utf-8")
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    sys.path.insert(0, str(libero_root))


def finish_action(action: torch.Tensor):
    action = action.detach().cpu().numpy().copy()
    action[-1] = 1.0 if action[-1] > 0.5 else -1.0
    return action


def frame_from_obs(observation) -> np.ndarray:
    """Right-side-up view for a human watching the saved video -- same flip
    rollout_eval.py applies; the model itself gets the raw (un-flipped)
    convention below, matching training data."""
    return observation["agentview_image"][::-1].copy()


def live_ee_pixel(sim, ee_pos_3d) -> tuple[int, int]:
    """Same projection + row-flip convention as precompute_ee_pixels.py,
    computed live against the running sim instead of a replayed state."""
    import robosuite.utils.camera_utils as cu
    transform = cu.get_camera_transform_matrix(sim, "agentview", camera_height=IMG_SIZE, camera_width=IMG_SIZE)
    pixel = cu.project_points_from_world_to_camera(ee_pos_3d[None, :], transform, IMG_SIZE, IMG_SIZE)[0]
    row_flipped = IMG_SIZE - 1 - pixel[0]
    return int(row_flipped), int(pixel[1])


TRAJ_START_COLOR = np.array([80, 200, 255])   # near-term (next step): cyan
TRAJ_END_COLOR = np.array([255, 40, 80])      # far-term (7 steps out): red


def to_display_frame(viewable_frame: np.ndarray, traj_px_raw: np.ndarray | None = None, upscale: int = 8) -> np.ndarray:
    """viewable_frame: (H,W,3) uint8 in the flipped, human-viewable convention
    (frame_from_obs's output). traj_px_raw: (ACTION_HORIZON,2) predicted
    (row,col) in the RAW/unflipped convention the model itself uses (same
    space as live_ee_pixel/precompute_ee_pixels.py) -- flip rows once here to
    align with the flipped frame being drawn on, same correction applied
    everywhere else in this pipeline; None (no prediction yet, e.g. during
    warmup) just returns the upscaled plain frame. Returns an upscaled RGB
    frame, with the predicted future-position trajectory drawn as a
    time-graded polyline when traj_px_raw is given.

    decode_heatmap_to_px returns continuous floating-point pixel positions
    (soft-argmax over the bin distribution, not a discretized bin index --
    see soft_bin_target/decode_heatmap_to_px), so the underlying precision
    here is already sub-pixel at the native 128x128 resolution; LANCZOS (not
    NEAREST) resizing plus markers/lines sized relative to `upscale` is what
    makes that sub-pixel precision actually visible rather than lost to a
    handful of blocky native pixels."""
    h, w = viewable_frame.shape[:2]
    img = Image.fromarray(viewable_frame).resize((w * upscale, h * upscale), Image.LANCZOS)
    if traj_px_raw is None:
        return np.array(img)
    draw = ImageDraw.Draw(img)
    n = len(traj_px_raw)
    pts = []
    for row, col in traj_px_raw:
        flipped_row = (h - 1 - row) * upscale
        x = col * upscale
        pts.append((float(x), float(flipped_row)))
    line_width = max(2, round(upscale * 0.35))
    for i in range(n):
        t = i / max(n - 1, 1)
        color = tuple(int(round(c)) for c in (1 - t) * TRAJ_START_COLOR + t * TRAJ_END_COLOR)
        if i > 0:
            draw.line([pts[i - 1], pts[i]], fill=color, width=line_width)
        r = upscale * (0.55 if i < n - 1 else 0.85)
        draw.ellipse([pts[i][0] - r, pts[i][1] - r, pts[i][0] + r, pts[i][1] + r], fill=color, outline=(0, 0, 0))
    return np.array(img)


class TowerPolicyRunner:
    """Receding-horizon controller for the Wan-DiT tower: rolling
    CONTEXT_STEPS_TOWER observation window (heatmap-overlaid, single camera),
    queries the model for a 7-step action chunk, temporally ensembles
    overlapping chunk predictions across consecutive queries, executes
    execution_horizon actions before replanning. Ensemble logic is identical
    to rollout_eval.py's -- generic to any ACTION_HORIZON=7 chunk output."""

    def __init__(self, model, preprocess, clip_model, sim, temperature: float,
                 execution_horizon: int, max_steps: int, sampling_steps: int, device: str):
        self.model = model
        self.preprocess = preprocess
        self.clip_model = clip_model
        self.sim = sim
        self.temperature = temperature
        self.execution_horizon = execution_horizon
        self.max_steps = max_steps
        self.sampling_steps = sampling_steps
        self.device = device

    def reset(self, instruction: str):
        self.primary = deque(maxlen=CONTEXT_STEPS_TOWER)
        self.pending = deque()
        self.predictions = torch.zeros(self.max_steps, self.max_steps + ACTION_HORIZON, 7, device=self.device)
        self.valid = torch.zeros(self.max_steps, self.max_steps + ACTION_HORIZON, dtype=torch.bool, device=self.device)
        self.last_traj_px = None  # (ACTION_HORIZON,2) predicted future (row,col), raw/unflipped convention
        with torch.no_grad():
            self.text_embed = self.clip_model.encode_text(
                clip.tokenize([instruction], truncate=True).to(self.device)
            ).float()

    def observe(self, observation):
        raw = observation["agentview_image"]  # raw/native MuJoCo convention, matches training storage
        row, col = live_ee_pixel(self.sim, observation["robot0_eef_pos"])
        overlaid = draw_gaussian(raw.copy(), row, col)
        clip_tensor = self.preprocess(Image.fromarray(overlaid))  # CLIP-normalized
        self.primary.append(clip_tensor.unsqueeze(0))  # (1,3,224,224)

    def prefill(self, observations):
        for observation in observations[-CONTEXT_STEPS_TOWER:-1]:
            self.observe(observation)

    def ensemble(self, chunk: torch.Tensor, timestep: int):
        self.predictions[timestep, timestep : timestep + ACTION_HORIZON] = chunk
        self.valid[timestep, timestep : timestep + ACTION_HORIZON] = True
        selected = []
        for target in range(timestep, timestep + self.execution_horizon):
            mask = self.valid[: timestep + 1, target]
            actions = self.predictions[: timestep + 1, target][mask]
            weights = np.exp(-self.temperature * np.arange(len(actions)))
            weights = torch.as_tensor(weights / weights.sum(), device=self.device).unsqueeze(1)
            selected.append(finish_action((actions * weights).sum(0)))
        self.pending.extend(selected[1:])
        return selected[0]

    @torch.inference_mode()
    def act(self, observation, timestep: int):
        self.observe(observation)
        if self.pending:
            return self.pending.popleft()
        if len(self.primary) != CONTEXT_STEPS_TOWER:
            raise RuntimeError(f"evaluation requires {CONTEXT_STEPS_TOWER} real context observations")
        clip_frames = torch.cat(tuple(self.primary), dim=0).unsqueeze(0).to(self.device)  # (1,5,3,224,224)
        primary = (clip_frames * CLIP_STD.to(self.device) + CLIP_MEAN.to(self.device)) * 2 - 1
        chunk, heatmap_logits = self.model.generate(
            primary, self.text_embed, sampling_steps=self.sampling_steps, return_heatmap=True
        )
        chunk = chunk[0]
        row_px = decode_heatmap_to_px(heatmap_logits[0, :, 0, :], image_size=IMG_SIZE)
        col_px = decode_heatmap_to_px(heatmap_logits[0, :, 1, :], image_size=IMG_SIZE)
        self.last_traj_px = torch.stack([row_px, col_px], dim=-1).cpu().numpy()  # (ACTION_HORIZON,2)
        return self.ensemble(chunk, timestep)


def evaluate(args, model, preprocess, clip_model, device):
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[args.suite]()
    total = args.tasks * args.episodes
    results = []
    progress = tqdm(range(total), desc="rollout", dynamic_ncols=True)
    for evaluation_id in progress:
        task_id, episode_index = divmod(evaluation_id, args.episodes)
        episode_id = args.episode_offset + episode_index
        task = suite.get_task(task_id)
        bddl = args.libero_path / "libero" / "libero" / "bddl_files" / task.problem_folder / task.bddl_file
        environment = OffScreenRenderEnv(
            bddl_file_name=str(bddl), camera_heights=IMG_SIZE, camera_widths=IMG_SIZE, render_gpu_device_id=0,
        )
        t0 = time.perf_counter()
        try:
            environment.reset()
            environment.seed(66)
            sim = environment.env.sim
            initial_states = torch.load(
                args.libero_path / "libero" / "libero" / "init_files" / task.problem_folder / task.init_states_file,
                weights_only=False,
            )
            if episode_id >= len(initial_states):
                raise IndexError(f"episode {episode_id} is unavailable for task {task_id}")
            observation = environment.set_init_state(initial_states[episode_id])
            need_frames = args.save_video or args.save_trajectory_video
            frames = [frame_from_obs(observation)] if need_frames else None
            traj_at_frame = [None] if args.save_trajectory_video else None  # no prediction before first act()
            warmup = []
            for _ in range(5):
                observation, _, _, _ = environment.step(np.zeros(7))
                warmup.append(copy.deepcopy(observation))
                if need_frames:
                    frames.append(frame_from_obs(observation))
                if traj_at_frame is not None:
                    traj_at_frame.append(None)
            runner = TowerPolicyRunner(
                model, preprocess, clip_model, sim, args.temperature,
                args.execution_horizon, args.max_steps, args.sampling_steps, device,
            )
            runner.reset(task.language)
            runner.prefill(warmup)
            observation = warmup[-1]
            success = 0
            steps = 0
            env_steps = 0
            for steps in range(1, args.max_steps + 1):
                action = runner.act(observation, steps - 1)
                done = False
                for _ in range(args.action_repeat):
                    observation, _, done, _ = environment.step(action)
                    env_steps += 1
                    if need_frames:
                        frames.append(frame_from_obs(observation))
                    if traj_at_frame is not None:
                        traj_at_frame.append(runner.last_traj_px)
                    if done:
                        break
                if done:
                    success = 1
                    break
            dt = time.perf_counter() - t0
            results.append({
                "eval_id": evaluation_id, "task": task_id, "task_name": task.language,
                "episode": episode_id, "success": success, "steps": steps, "env_steps": env_steps,
                "action_repeat": args.action_repeat, "wall_time_s": round(dt, 2),
            })
            progress.set_postfix(successes=sum(r["success"] for r in results), last_s=f"{dt:.1f}")
            print(f"  task={task_id} ({task.language!r}) episode={episode_id}: "
                  f"{'SUCCESS' if success else 'fail'} in {steps} policy steps / "
                  f"{env_steps} env steps (action_repeat={args.action_repeat}) ({dt:.1f}s)")

            label = "success" if success else "fail"
            if args.save_video:
                video_dir = args.output_dir / "videos"
                video_dir.mkdir(parents=True, exist_ok=True)
                video_path = video_dir / f"task{task_id}_ep{episode_id}_{label}.mp4"
                imageio.mimsave(video_path, frames, fps=args.video_fps)
                print(f"    saved video: {video_path} ({len(frames)} frames)")

            if args.save_trajectory_video:
                video_dir = args.output_dir / "videos"
                video_dir.mkdir(parents=True, exist_ok=True)
                combined = [
                    np.concatenate([
                        to_display_frame(frames[i], upscale=args.traj_upscale),
                        to_display_frame(frames[i], traj_at_frame[i], upscale=args.traj_upscale),
                    ], axis=1)
                    for i in range(len(frames))
                ]
                traj_video_path = video_dir / f"task{task_id}_ep{episode_id}_{label}_traj.mp4"
                imageio.mimsave(traj_video_path, combined, fps=args.video_fps)
                print(f"    saved trajectory video: {traj_video_path} ({len(combined)} frames, "
                      f"plain | predicted-future-xy-trajectory)")
        finally:
            environment.close()

    print()
    for task_id in range(args.tasks):
        values = [r["success"] for r in results if r["task"] == task_id]
        if values:
            print(f"Task {task_id}: {sum(values)}/{len(values)} ({np.mean(values):.1%})")
    successes = sum(r["success"] for r in results)
    print(f"Overall: {successes}/{len(results)} ({successes / len(results):.1%})")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "suite": args.suite, "checkpoint": str(args.checkpoint), "episodes": len(results),
        "successes": successes, "success_rate": successes / len(results) if results else None,
        "results": results,
    }
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output_dir / 'results.json'}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", default="libero_object", choices=SUITES)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--vae", default="../worlddit_ref/dependencies/Wan2.1_VAE.pth")
    p.add_argument("--clip", default="../worlddit_ref/dependencies/ViT-B-32.pt")
    p.add_argument("--dit-dim", type=int, default=1024)
    p.add_argument("--dit-blocks", type=int, default=10)
    p.add_argument("--dit-heads", type=int, default=16)
    p.add_argument("--dit-ffn", type=int, default=4096)
    p.add_argument("--libero-path", type=Path, default=Path("~/LIBERO").expanduser())
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--tasks", type=int, default=10)
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--episode-offset", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--execution-horizon", type=int, choices=(1, 3), default=3)
    p.add_argument("--temperature", type=float, default=0.01)
    p.add_argument("--sampling-steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=66)
    p.add_argument("--action-repeat", type=int, default=1)
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--save-trajectory-video", action="store_true",
                    help="also save a side-by-side video per episode (plain | predicted future EE xy "
                         "trajectory drawn on each frame, decoded via soft-argmax from the heatmap head's "
                         "final-sampling-step prediction, held constant between replans)")
    p.add_argument("--traj-upscale", type=int, default=8,
                    help="upscale factor for --save-trajectory-video frames (LANCZOS-resized, markers/lines "
                         "sized relative to this) -- higher makes the sub-pixel soft-argmax precision of the "
                         "predicted trajectory actually visible instead of lost to a few blocky 128px pixels")
    p.add_argument("--video-fps", type=int, default=20)
    return p


def main():
    args = build_parser().parse_args()
    args.libero_path = args.libero_path.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    os.environ.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    configure_libero(args.libero_path, args.output_dir)

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading model (vae={args.vae}, clip={args.clip}) on {device} ...")
    dit_config = dict(
        model_type="t2v", dim=args.dit_dim, ffn_dim=args.dit_ffn, freq_dim=256, text_dim=4096,
        out_dim=16, num_heads=args.dit_heads, num_layers=args.dit_blocks, text_len=512, in_dim=16,
    )
    model = WanTowerPolicy(args.vae, None, dit_config).to(device)
    load_trainable_state(model, args.checkpoint)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    clip_model, preprocess = clip.load(str(args.clip), device="cpu")
    clip_model = clip_model.to(device).eval()

    evaluate(args, model, preprocess, clip_model, device)


if __name__ == "__main__":
    main()
