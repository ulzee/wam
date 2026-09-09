#!/usr/bin/env python3
"""Offline precompute: run Wan's own UMT5-XXL text encoder once over every
LIBERO-Object task instruction and cache the resulting per-token embeddings
to disk. Same "expensive computation once, cheap lookup at train/eval time"
pattern as precompute_ee_pixels.py.

Avoids ever materializing the model in fp32 (which needs ~22.7GB and OOMs
this 14.56GB GPU even transiently): builds on the meta device, then
to_empty()'s straight into bf16 on GPU, then loads the checkpoint via a
CPU-staged state dict so only one bf16 copy (~11.4GB) is ever GPU-resident.

Run in the wdit_tower env (needs a newer huggingface_hub/transformers than
the base wdit env has -- see wan_tokenizers.py's AutoTokenizer dependency).
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_wan_tower import list_raw_libero_tasks
from wan_t5 import umt5_xxl
from wan_tokenizers import HuggingfaceTokenizer


def load_encoder(checkpoint_path: str, device: str = "cuda", dtype=torch.bfloat16):
    with torch.device("meta"):
        model = umt5_xxl(encoder_only=True, return_tokenizer=False, dtype=dtype, device="meta")
    model = model.to_empty(device=device)
    sd = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(sd, strict=True)
    del sd
    return model.eval().requires_grad_(False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default="/home/ubuntu/dev/wdit/data/libero_object_raw")
    p.add_argument("--checkpoint", default="/home/ubuntu/dev/wdit/worlddit_ref/dependencies/models_t5_umt5-xxl-enc-bf16.pth")
    p.add_argument("--tokenizer", default="google/umt5-xxl")
    p.add_argument("--text-len", type=int, default=512, help="max tokenize length, matches Wan's own T5_CONTEXT_TOKEN_NUMBER")
    p.add_argument("--out", default="/home/ubuntu/dev/wdit/data/libero_object_raw/text_embeddings_umt5.pt")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    instructions = sorted(list_raw_libero_tasks(args.raw_dir).keys())
    print(f"{len(instructions)} instructions to encode:")
    for instr in instructions:
        print(" ", repr(instr))

    t0 = time.perf_counter()
    model = load_encoder(args.checkpoint, device=device)
    print(f"encoder loaded in {time.perf_counter()-t0:.1f}s, peak GPU mem: "
          f"{torch.cuda.max_memory_allocated()/1e9:.2f}GB" if device == "cuda" else "")

    tokenizer = HuggingfaceTokenizer(name=args.tokenizer, seq_len=args.text_len, clean="whitespace")
    ids, mask = tokenizer(instructions, return_mask=True, add_special_tokens=True)
    ids, mask = ids.to(device), mask.to(device)
    seq_lens = mask.gt(0).sum(dim=1).long()

    with torch.no_grad():
        context = model(ids, mask)  # (N, text_len, 4096), padded
    per_sample = [u[:v].float().cpu() for u, v in zip(context, seq_lens)]  # real length each, fp32 for storage

    max_len = max(t.shape[0] for t in per_sample)
    dim = per_sample[0].shape[1]
    padded = torch.zeros(len(instructions), max_len, dim, dtype=torch.float32)
    lengths = torch.zeros(len(instructions), dtype=torch.long)
    for i, t in enumerate(per_sample):
        padded[i, : t.shape[0]] = t
        lengths[i] = t.shape[0]

    print(f"\nreal per-instruction token counts: {lengths.tolist()}")
    print(f"padded cache shape: {tuple(padded.shape)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"instructions": instructions, "embeddings": padded, "lengths": lengths}, args.out)
    print(f"\nsaved cache to {args.out} ({Path(args.out).stat().st_size/1e6:.2f} MB)")
    print(f"total time: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
