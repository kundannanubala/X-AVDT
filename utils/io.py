"""I/O helpers for the inference engine."""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from utils.preprocessing import VIDEO_EXTS


@dataclass
class VideoResult:
    video: str
    label: str
    score: float
    chunk_scores: List[float]
    num_chunks: int
    max_score: float = 0.0


@dataclass
class InferenceResults:
    meta: dict[str, Any]
    videos: List[VideoResult] = field(default_factory=list)
    errors: List[dict[str, str]] = field(default_factory=list)


def discover_videos(video_dir: Path) -> List[Path]:
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")
    videos = sorted(
        path for path in video_dir.rglob("*") if path.suffix.lower() in VIDEO_EXTS
    )
    if not videos:
        raise RuntimeError(f"No videos found under {video_dir}")
    return videos


def make_work_dir(base: Optional[Path] = None) -> Path:
    if base is not None:
        work_dir = Path(base)
        work_dir.mkdir(parents=True, exist_ok=True)
        return work_dir
    return Path(tempfile.mkdtemp(prefix="xavdt_inference_"))


def cleanup_work_dir(work_dir: Path, keep: bool) -> None:
    if keep or not work_dir.exists():
        return
    shutil.rmtree(work_dir, ignore_errors=True)


def write_results(output_path: Path, results: InferenceResults) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": results.meta,
        "videos": [asdict(video) for video in results.videos],
        "errors": results.errors,
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def build_meta(
    checkpoint: str,
    threshold: float,
    num_videos: int,
    **extra: Any,
) -> dict[str, Any]:
    meta = {
        "checkpoint": checkpoint,
        "threshold": threshold,
        "num_videos": num_videos,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }
    meta.update(extra)
    return meta
