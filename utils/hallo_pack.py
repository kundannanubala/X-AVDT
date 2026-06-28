"""Pack Hallo clip outputs the same way as hallo/preprocess_videos.py pack-features."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import torch

from utils.packing import InferenceChunk, chunk_rgb_tensor

HALLO_ROOT = Path(__file__).resolve().parent.parent / "hallo"
if str(HALLO_ROOT) not in sys.path:
    sys.path.insert(0, str(HALLO_ROOT))

FEATURE_VIDEOS = ("original", "inverted", "reconstructed", "residual")


def _load_hallo_preprocess():
    import preprocess_videos as hallo_preprocess

    return hallo_preprocess


def pack_chunks_from_hallo_clip(clip_dir: Path, clip_len: int = 16) -> List[InferenceChunk]:
    """Build classifier chunks from one hallo/extract_features.py output directory."""
    hallo_preprocess = _load_hallo_preprocess()
    clip_dir = Path(clip_dir)
    if not clip_dir.is_dir():
        raise FileNotFoundError(f"Hallo clip directory not found: {clip_dir}")

    rgb_tensors = []
    for modality in FEATURE_VIDEOS:
        video_path = clip_dir / f"{modality}.mp4"
        if not video_path.is_file():
            raise FileNotFoundError(f"Missing Hallo modality video: {video_path}")
        frames = hallo_preprocess.extract_frames_from_video(video_path)
        tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).contiguous()
        rgb_tensors.append(tensor)

    attn_path = clip_dir / "attn_feat.pt"
    if not attn_path.is_file():
        raise FileNotFoundError(f"Missing attention features: {attn_path}")
    attn = torch.load(attn_path, map_location="cpu")

    rgb_chunk_lists = [chunk_rgb_tensor(tensor, clip_len=clip_len) for tensor in rgb_tensors]
    attn_chunks = []
    total_frames = attn.shape[0]
    for start in range(0, total_frames - clip_len + 1, clip_len):
        chunk = attn[start : start + clip_len]
        if chunk.shape[0] < clip_len:
            continue
        attn_chunks.append(hallo_preprocess.reshape_attention_chunk(chunk).float())

    if not attn_chunks:
        return []

    num_chunks = min(len(chunks) for chunks in rgb_chunk_lists + [attn_chunks])
    packed: List[InferenceChunk] = []
    for idx in range(num_chunks):
        composite = torch.cat(
            [rgb_chunk_lists[0][idx], rgb_chunk_lists[1][idx], rgb_chunk_lists[2][idx], rgb_chunk_lists[3][idx]],
            dim=0,
        )
        packed.append(InferenceChunk(composite=composite, attn=attn_chunks[idx]))
    return packed
