"""Per-token CLIP text sequence extraction -- bypasses CLIP's own EOT-only
pooling + text_projection to get the full per-position hidden state instead,
for use as real Wan-style cross-attention context (see model_wan_tower.py's
`context`/`context_lens` in forward()/generate()). Same role as wan_t5.py's
UMT5 encoder wrapper -- a real per-token sequence, not one pooled vector --
just architecturally much smaller (CLIP ViT-B/32 text tower: 512-wide, 77
tokens, 63.4M params total) and needing no separate download or offline
cache, since CLIP is already loaded here for image preprocessing anyway.
"""
import torch


def clip_encode_text_sequence(clip_model, tokens: torch.Tensor):
    """tokens: (B, 77) int64, from clip.tokenize(...). Returns:
      sequence: (B, 77, transformer.width) -- CLIP's own ln_final output,
        the per-token hidden state before EOT-selection/text_projection
        (transformer.width=512 for ViT-B/32).
      lengths: (B,) real (unpadded) token count per sample, EOT position + 1
        -- CLIP's tokenizer assigns EOT the highest token id, same trick
        encode_text() uses internally, here used for context_lens masking
        instead of single-position selection.
    """
    x = clip_model.token_embedding(tokens).type(clip_model.dtype)
    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x).type(clip_model.dtype)  # (B, 77, width)
    lengths = tokens.argmax(dim=-1) + 1
    return x, lengths
