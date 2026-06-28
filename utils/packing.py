"""In-memory 16-frame chunk packing for classifier inference."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import torch

from utils.feature_extraction import FeatureBundle


@dataclass
class InferenceChunk:
    composite: torch.Tensor  # (12, T, H, W)
    attn: torch.Tensor  # (320, T, 64, 64)


def chunk_rgb_tensor(tensor: torch.Tensor, clip_len: int = 16) -> List[torch.Tensor]:
    """Split an RGB video tensor (3, T, H, W) into non-overlapping chunks."""
    if tensor.ndim != 4 or tensor.shape[0] != 3:
        raise ValueError(f"Expected RGB tensor (3, T, H, W), got {tuple(tensor.shape)}")

    _, total_frames, _, _ = tensor.shape
    chunks: List[torch.Tensor] = []
    for start in range(0, total_frames - clip_len + 1, clip_len):
        chunk = tensor[:, start : start + clip_len]
        if chunk.shape[1] < clip_len:
            continue
        chunks.append(chunk.contiguous())
    return chunks


def reshape_attention_chunk(chunk: torch.Tensor) -> torch.Tensor:
    """Reshape a temporal attention chunk to (320, T, 64, 64)."""
    if chunk.ndim == 4:
        return chunk.permute(1, 0, 2, 3).contiguous().float()
    if chunk.ndim != 3:
        raise ValueError(f"Unsupported attention shape: {tuple(chunk.shape)}")

    clip_len, num_tokens, channels = chunk.shape
    side = int(math.isqrt(num_tokens))
    if side * side != num_tokens:
        raise ValueError(f"Attention token count is not square: {num_tokens}")
    chunk = chunk.permute(0, 2, 1).reshape(clip_len, channels, side, side)
    return chunk.permute(1, 0, 2, 3).contiguous().float()


def chunk_attention(attn: torch.Tensor, clip_len: int = 16) -> List[torch.Tensor]:
    """Split attention features (T, tokens, C) into reshaped chunks."""
    if attn.ndim != 3:
        raise ValueError(f"Expected attention tensor (T, tokens, C), got {tuple(attn.shape)}")

    total_frames = attn.shape[0]
    chunks: List[torch.Tensor] = []
    for start in range(0, total_frames - clip_len + 1, clip_len):
        chunk = attn[start : start + clip_len]
        if chunk.shape[0] < clip_len:
            continue
        chunks.append(reshape_attention_chunk(chunk))
    return chunks


def pack_inference_chunks(
    feature_bundle: FeatureBundle,
    clip_len: int = 16,
) -> List[InferenceChunk]:
    """Pack Hallo features into classifier-ready 16-frame chunks."""
    modalities = (
        feature_bundle.original,
        feature_bundle.inverted,
        feature_bundle.reconstructed,
        feature_bundle.residual,
    )
    rgb_chunk_lists = [chunk_rgb_tensor(modality, clip_len=clip_len) for modality in modalities]
    attn_chunks = chunk_attention(feature_bundle.attn_feat, clip_len=clip_len)

    if not attn_chunks:
        return []

    num_chunks = min(len(chunks) for chunks in rgb_chunk_lists + [attn_chunks])
    if num_chunks == 0:
        return []

    packed: List[InferenceChunk] = []
    for idx in range(num_chunks):
        composite = torch.cat(
            [rgb_chunk_lists[0][idx], rgb_chunk_lists[1][idx], rgb_chunk_lists[2][idx], rgb_chunk_lists[3][idx]],
            dim=0,
        )
        packed.append(InferenceChunk(composite=composite, attn=attn_chunks[idx]))
    return packed
