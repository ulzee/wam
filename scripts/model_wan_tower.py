"""Wan-DiT tower for LIBERO-Object: fine-tune Wan2.1-1.3B's actual pretrained
DiT (not just its VAE) to predict actions, jointly with a lightweight
image-space keypoint-heatmap head.

SIMPLIFICATIONS in this first working version (see conversation for the
full target design -- these are deliberate scope cuts for a first real,
tested smoke test, not the final architecture):
  - Single camera (agentview) only. Wan's native forward() assumes one
    video per batch item; multi-camera token concatenation is real
    additional work, kept out of this round to isolate the core question
    (does Wan's pretrained DiT help at all) from a second variable.
  - One shared global timestep across context + targets, not the
    per-token-group split (context frozen at tau=0, targets noisy)
    described in the design conversation. Wan's block natively only
    supports one timestep per sequence; using it as-is is a real,
    acknowledged simplification, not the target mechanism.

What's NOT simplified: Wan's own pretrained weights (VAE + DiT blocks) are
used and fine-tuned as-is, loaded via the vendored wan_vae.py/wan_dit.py.
The 5-frame (1 anchor + 1x4 chunk) context window genuinely exercises the
VAE's native causal-conv temporal chunking. New tokens (text, action
target, heatmap-query) are appended to the video token sequence -- verified
that Wan's own rope_apply leaves any tokens beyond the video grid's
f*h*w length completely unrotated (identity), so this required zero
modification to Wan's own block/attention code to support cleanly.

No robot-state input: deliberately excluded. Proprioceptive state (EE
pose/gripper) is embodiment-specific and was cut so the model only ever
conditions on things a human demonstration could also provide (video +
instruction), keeping it agnostic for the eventual cross-embodiment goal.

Heatmap head is coordinate-binned, not a dense spatial heatmap: each future
step predicts two independent categorical distributions (row bin, col bin)
over HEATMAP_BINS=20 positions each via softmax/cross-entropy, instead of a
flattened 14x14=196-way spatial classification. Cheaper head, cheaper loss,
same 2D localization signal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wan_vae import WanVAE
from wan_dit import WanModel, sinusoidal_embedding_1d, rope_params

DIM_DEFAULT = 1536  # Wan2.1-1.3B model width; overridden per-instance by dit_config["dim"]
ACTION_HORIZON = 7
HEATMAP_BINS = 20  # per-axis coordinate bins for the row/col categorical heatmap head


def soft_bin_target(coord_px: torch.Tensor, image_size: int = 128, bins: int = HEATMAP_BINS) -> torch.Tensor:
    """coord_px: (...,) pixel coordinate along one axis. Returns (..., bins):
    a probability distribution linearly split between the two nearest bin
    centers, never a hard round to a single bin -- e.g. a pixel exactly
    between bin 1 and bin 2 gets 0.5 on each, matching soft-argmax/DSNT-style
    coordinate binning. Only clamps to keep both bin indices in range (for a
    pixel at or past the last bin center); it never collapses a legitimate
    in-range position down to one bin."""
    pos = (coord_px.float() / image_size * bins).clamp(0, bins - 1)
    low = pos.floor().long()
    high = (low + 1).clamp(max=bins - 1)
    frac = pos - low.float()
    target = torch.zeros(*coord_px.shape, bins, device=coord_px.device, dtype=pos.dtype)
    target.scatter_(-1, low.unsqueeze(-1), (1 - frac).unsqueeze(-1))
    target.scatter_add_(-1, high.unsqueeze(-1), frac.unsqueeze(-1))  # no-op add if high==low (edge bin)
    return target


def soft_ce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """logits, target: (..., bins). target is a probability distribution
    (not necessarily one-hot) -- standard soft cross-entropy."""
    return -(target * torch.log_softmax(logits, dim=-1)).sum(-1).mean()


class WanTowerPolicy(nn.Module):
    def __init__(self, vae_path: str, dit_path: str, dit_config: dict):
        super().__init__()
        self.dim = dit_config["dim"]  # sized from the DiT config -- supports both the real
        DIM = self.dim               # 1.3B checkpoint and a smaller from-scratch variant
        vae = WanVAE(z_dim=16, vae_pth=vae_path, dtype=torch.float32, device="cpu")
        self.vae_model = vae.model  # frozen VAE encoder, real submodule so .to()/dtype casts propagate
        self.register_buffer("vae_mean", vae.mean.clone())
        self.register_buffer("vae_std", vae.std.clone())
        self.vae_model.requires_grad_(False)

        self.dit = WanModel(**dit_config)  # pretrained weights loaded by caller (if any), then fine-tuned

        # new, small, trainable heads -- everything below is new, nothing pretrained
        self.text_embedding = nn.Sequential(nn.Linear(512, DIM), nn.GELU(), nn.Linear(DIM, DIM))  # replaces Wan's umT5-sized text_embedding; CLIP (512-d) in
        self.action_tokenizer = nn.Sequential(nn.Linear(7, DIM * 2), nn.GELU(), nn.Linear(DIM * 2, DIM))
        self.action_detokenizer = nn.Sequential(nn.Linear(DIM, DIM * 2), nn.GELU(), nn.Linear(DIM * 2, 7))
        self.action_query = nn.Parameter(torch.randn(1, ACTION_HORIZON, DIM) * 0.02)

        self.heatmap_query = nn.Parameter(torch.randn(1, ACTION_HORIZON, DIM) * 0.02)
        self.heatmap_head = nn.Sequential(
            nn.Linear(DIM, DIM), nn.GELU(), nn.Linear(DIM, 2 * HEATMAP_BINS)
        )  # per future step: row-bin logits (20) + col-bin logits (20), softmaxed independently

        self.action_type = nn.Parameter(torch.randn(1, 1, DIM) * 0.02)
        self.heatmap_type = nn.Parameter(torch.randn(1, 1, DIM) * 0.02)

    def encode_context_video(self, primary_5frame: torch.Tensor) -> torch.Tensor:
        """primary_5frame: (B, 5, 3, 224, 224) in [-1, 1] (already converted
        from CLIP-normalized upstream). Returns patchified video tokens
        (B, 2*14*14, DIM) via a real multi-frame VAE encode (genuine 1+4
        causal-conv chunking) + Wan's own patchify conv."""
        vae_dtype = self.vae_model.conv1.weight.dtype
        clip = primary_5frame.permute(0, 2, 1, 3, 4).to(vae_dtype)  # (B,5,3,H,W) -> (B,3,5,H,W)
        scale = [self.vae_mean, 1.0 / self.vae_std]
        # The frozen VAE needs no gradients, so its own forward-pass peak
        # memory (dominated by early, full-resolution conv layers) is
        # independent of the "real" batch size everything else trains at --
        # chunk it internally so a large logical batch never forces the VAE
        # to materialize all B samples' activations simultaneously.
        vae_chunk = 8
        with torch.no_grad():
            latent = torch.cat(
                [self.vae_model.encode(clip[i : i + vae_chunk], scale) for i in range(0, clip.shape[0], vae_chunk)],
                dim=0,
            )  # (B, 16, 2, 28, 28), fp32
        latent = latent.to(self.dit.patch_embedding.weight.dtype)
        tokens = self.dit.patch_embedding(latent)  # (B, DIM, 2, 14, 14)
        grid = tokens.shape[2:]  # (2, 14, 14)
        tokens = tokens.flatten(2).transpose(1, 2)  # (B, 2*14*14, DIM)
        return tokens, grid

    def forward(self, primary_5frame, text_embed, action_target=None, tau=None,
                noisy_action=None, video_tokens=None, grid=None):
        """primary_5frame: (B,5,3,224,224) in [-1,1].
        text_embed: (B,512) pooled CLIP text embedding.
        action_target: (B,7,7) clean actions, for building the flow-matching
        noisy input during training; ignored if noisy_action is given.
        tau: (B,) flow-matching timestep in [0,1]; if None, sampled here.
        noisy_action: (B,7,7) explicit current ODE state, for inference
        sampling (generate() supplies this each Euler step instead of having
        forward() draw fresh noise) -- takes priority over action_target.
        video_tokens/grid: pass in encode_context_video()'s own output to
        skip re-encoding the (unchanging, frozen) video context on repeated
        calls -- generate() uses this so a multi-step sampling loop pays for
        the VAE encode once, not once per step.
        Returns: pred_action_velocity (B,7,7), pred_heatmap_logits (B,7,2,20).
        No robot-state input -- conditions only on video + instruction, so
        nothing here is embodiment-specific.
        """
        B = primary_5frame.shape[0]
        device = primary_5frame.device
        if self.dit.freqs.device != device:
            self.dit.freqs = self.dit.freqs.to(device)
        if video_tokens is None:
            video_tokens, grid = self.encode_context_video(primary_5frame)
        f, h, w = grid
        video_len = f * h * w

        text_tokens = self.text_embedding(text_embed).unsqueeze(1)  # (B,1,DIM)

        if tau is None:
            tau = torch.rand(B, device=device, dtype=primary_5frame.dtype)
        if noisy_action is not None:
            target_velocity = None
        elif action_target is not None:
            eps = torch.randn_like(action_target)
            noisy_action = (1 - tau.view(B, 1, 1)) * eps + tau.view(B, 1, 1) * action_target
            target_velocity = action_target - eps
        else:
            noisy_action = torch.randn(B, ACTION_HORIZON, 7, device=device)
            target_velocity = None
        action_in = self.action_tokenizer(noisy_action) + self.action_query + self.action_type  # (B,7,DIM)
        heatmap_in = self.heatmap_query.expand(B, -1, -1) + self.heatmap_type  # (B,7,DIM)

        x = torch.cat([video_tokens, text_tokens, action_in, heatmap_in], dim=1)
        seq_len = x.shape[1]

        # SIMPLIFICATION (documented above): one shared global timestep for
        # the whole sequence, Wan's native mechanism, unmodified. No outer
        # autocast: Wan's own modules already force fp32 internally at the
        # numerically-sensitive spots (LayerNorm/RMSNorm/RoPE all cast
        # internally regardless of caller context) -- mixing that with an
        # outer autocast produced a broken mixed-dtype backward graph.
        work_dtype = self.dit.time_projection[1].weight.dtype
        sinu = sinusoidal_embedding_1d(self.dit.freq_dim, tau.float() * 1000).to(work_dtype)
        e = self.dit.time_embedding(sinu)
        e0 = self.dit.time_projection(e).unflatten(1, (6, self.dit.dim)).float()

        grid_sizes = torch.tensor([[f, h, w]] * B, dtype=torch.long, device=device)
        seq_lens = torch.tensor([seq_len] * B, dtype=torch.long, device=device)
        dummy_context = torch.zeros(B, 1, self.dit.dim, device=device, dtype=x.dtype)
        context_lens = None

        if True:
            for block in self.dit.blocks:
                x = block(
                    x, e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes,
                    freqs=self.dit.freqs, context=dummy_context, context_lens=context_lens,
                )

        action_out = x[:, video_len + 1 : video_len + 1 + ACTION_HORIZON]
        heatmap_out = x[:, video_len + 1 + ACTION_HORIZON :]
        pred_velocity = self.action_detokenizer(action_out.to(self.action_detokenizer[0].weight.dtype))
        pred_heatmap_logits = self.heatmap_head(heatmap_out.to(self.heatmap_head[0].weight.dtype))
        pred_heatmap_logits = pred_heatmap_logits.view(B, ACTION_HORIZON, 2, HEATMAP_BINS)

        return pred_velocity, pred_heatmap_logits, target_velocity

    @torch.no_grad()
    def generate(self, primary_5frame, text_embed, sampling_steps: int = 20, return_heatmap: bool = False):
        """Inference entry point: encode the (frozen, unchanging) video
        context once, then flow-sample the action chunk via Euler
        integration -- same convention as WorldDiT's WorldModelSampler.sample
        (linspace 0->1 over sampling_steps, action += velocity/sampling_steps).
        Returns (B, ACTION_HORIZON, 7), or (action, pred_heatmap_logits) if
        return_heatmap -- the heatmap head's prediction from the FINAL
        sampling step (tau->1, once the action tokens have converged), since
        there's no single "correct" tau to read it at otherwise."""
        B = primary_5frame.shape[0]
        device = primary_5frame.device
        dtype = primary_5frame.dtype
        video_tokens, grid = self.encode_context_video(primary_5frame)
        action = torch.randn(B, ACTION_HORIZON, 7, device=device, dtype=dtype)
        pred_heatmap_logits = None
        for t in torch.linspace(0.0, 1.0, sampling_steps + 1, device=device)[:-1]:
            tau = torch.full((B,), float(t.item()), device=device, dtype=dtype)
            velocity, pred_heatmap_logits, _ = self.forward(
                primary_5frame, text_embed, tau=tau, noisy_action=action,
                video_tokens=video_tokens, grid=grid,
            )
            action = action + velocity / sampling_steps
        if return_heatmap:
            return action, pred_heatmap_logits
        return action


def decode_heatmap_to_px(logits: torch.Tensor, image_size: int = 128, bins: int = HEATMAP_BINS) -> torch.Tensor:
    """logits: (..., bins) row-bin or col-bin logits. Returns (...,) expected
    pixel position via soft-argmax (probability-weighted bin index) -- the
    inverse of soft_bin_target's pixel->bin-position mapping, and consistent
    with it: a distribution split 0.5/0.5 across bins 1 and 2 decodes back to
    the pixel exactly between their centers, not snapped to either."""
    probs = torch.softmax(logits, dim=-1)
    bin_idx = torch.arange(bins, device=logits.device, dtype=probs.dtype)
    expected_bin = (probs * bin_idx).sum(-1)
    return expected_bin * (image_size / bins)


def save_trainable_state(model: "WanTowerPolicy", path) -> None:
    """Save everything except the frozen Wan-VAE encoder (~127M params that
    never change and are reloaded from vae_path at construction anyway) --
    same exclusion convention as model.py's save_trainable_state."""
    from pathlib import Path as _Path
    state = {k: v for k, v in model.state_dict().items() if not k.startswith("vae_model.")}
    _Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(path))


def load_trainable_state(model: "WanTowerPolicy", path) -> None:
    """Load a checkpoint saved by save_trainable_state. strict=False since the
    checkpoint intentionally omits the frozen vae_model keys."""
    state = torch.load(str(path), map_location=next(model.parameters()).device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad_missing = [k for k in missing if not k.startswith("vae_model.")]
    if bad_missing or unexpected:
        raise RuntimeError(f"checkpoint load mismatch -- missing (non-frozen): {bad_missing}, unexpected: {unexpected}")
