#!/usr/bin/env python3
"""Offline precompute: project the ground-truth 3D end-effector position
(obs/ee_pos) into 2D pixel coordinates for the agentview and eye_in_hand
cameras, for every frame of every demo in the raw LIBERO-Object HDF5 files.

Uses robosuite's own camera calibration (privileged simulator info -- exact,
no pose estimator), verified against real rendered frames in an earlier
sanity check (see outputs/ee_keypoint_check/).

Runs in the wdit_eval conda env (has robosuite/LIBERO). Output is cached to
a sidecar .npz per task so the actual training data loader (wdit env, no
robosuite dependency) can just load precomputed pixel coordinates.

Usage: python precompute_ee_pixels.py --raw-dir ../data/libero_object_raw
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, "/home/ubuntu/LIBERO")
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import robosuite.utils.camera_utils as cu

CAMERAS = [("agentview", "agentview"), ("robot0_eye_in_hand", "eye_in_hand")]
IMG_SIZE = 128


def bddl_path_for_task(raw_dir: Path, task_stem: str) -> str:
    instruction = task_stem.removesuffix("_demo").replace("_", " ")
    bddl_name = task_stem.removesuffix("_demo") + ".bddl"
    return str(Path(get_libero_path("bddl_files")) / "libero_object" / bddl_name)


def process_task_file(hdf5_path: Path, out_path: Path):
    bddl = bddl_path_for_task(hdf5_path.parent, hdf5_path.stem)
    if not Path(bddl).exists():
        print(f"  SKIP {hdf5_path.name}: no bddl at {bddl}")
        return

    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=IMG_SIZE, camera_widths=IMG_SIZE, render_gpu_device_id=0)
    env.reset()
    sim = env.env.sim

    results = {}  # demo_id -> {cam_tag: (T,2) pixel array}
    with h5py.File(hdf5_path, "r") as f:
        demo_ids = sorted(int(k.split("_")[1]) for k in f["data"].keys())
        for demo_id in demo_ids:
            demo = f["data"][f"demo_{demo_id}"]
            states = demo["states"][:]
            ee_pos = demo["obs"]["ee_pos"][:]
            T = states.shape[0]

            per_cam = {tag: np.zeros((T, 2), dtype=np.int32) for _, tag in CAMERAS}
            for t in range(T):
                sim.set_state_from_flattened(states[t])
                sim.forward()
                p3d = ee_pos[t][None, :]
                for cam_name, tag in CAMERAS:
                    transform = cu.get_camera_transform_matrix(sim, cam_name, camera_height=IMG_SIZE, camera_width=IMG_SIZE)
                    pixel = cu.project_points_from_world_to_camera(p3d, transform, IMG_SIZE, IMG_SIZE)[0]
                    # flip row to match the raw (un-flipped) MuJoCo-native storage convention
                    # used by LIBERO's own HDF5 frames (verified empirically -- see conversation).
                    row_flipped = IMG_SIZE - 1 - pixel[0]
                    per_cam[tag][t] = [row_flipped, pixel[1]]
            results[demo_id] = per_cam

    env.close()

    save_dict = {}
    for demo_id, per_cam in results.items():
        for tag, arr in per_cam.items():
            save_dict[f"demo_{demo_id}_{tag}"] = arr
    np.savez_compressed(out_path, **save_dict)
    print(f"  saved {out_path.name}  ({len(results)} demos)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default="/home/ubuntu/dev/wdit/data/libero_object_raw")
    p.add_argument("--out-dir", default="/home/ubuntu/dev/wdit/data/libero_object_raw/ee_pixels")
    p.add_argument("--tasks", type=int, default=None, help="limit to first N task files, for a quick smoke test")
    args = p.parse_args()

    raw_dir = Path(args.raw_dir)
    search_dir = raw_dir / "libero_object" if (raw_dir / "libero_object").is_dir() else raw_dir
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_files = sorted(search_dir.glob("*_demo.hdf5"))
    if args.tasks:
        task_files = task_files[: args.tasks]
    print(f"{len(task_files)} task file(s) to process")

    t0 = time.perf_counter()
    for i, hdf5_path in enumerate(task_files):
        print(f"[{i+1}/{len(task_files)}] {hdf5_path.name}")
        out_path = out_dir / (hdf5_path.stem + "_ee_pixels.npz")
        process_task_file(hdf5_path, out_path)
    print(f"\nDone in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
