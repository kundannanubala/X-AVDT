"""In-memory 16-frame chunk packing for classifier inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


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
