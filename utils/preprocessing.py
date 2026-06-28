"""Video frame and audio extraction for Hallo feature extraction."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for video frame/audio extraction.")


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def extract_video(
    video_path: Path,
    frame_dir: Path,
    audio_path: Path,
    fps: float = 25.0,
    size: Optional[Tuple[int, int]] = (512, 512),
    duration: Optional[float] = 5.0,
    force: bool = False,
) -> None:
    """Extract PNG frames and 16 kHz WAV audio from a single video."""
    require_ffmpeg()
    frame_dir.mkdir(parents=True, exist_ok=True)
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    if not force and any(frame_dir.glob("*.png")) and audio_path.exists():
        return

    for old_frame in frame_dir.glob("*.png"):
        old_frame.unlink()
    if audio_path.exists():
        audio_path.unlink()

    video_filter = f"fps={fps}"
    if size is not None:
        width, height = size
        video_filter = f"{video_filter},scale={width}:{height}"

    image_cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video_path)]
    if duration is not None:
        image_cmd.extend(["-t", str(duration)])
    image_cmd.extend(["-vf", video_filter, "-start_number", "0", str(frame_dir / "%04d.png")])

    audio_cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video_path)]
    if duration is not None:
        audio_cmd.extend(["-t", str(duration)])
    audio_cmd.extend(["-ac", "2", "-ar", "16000", "-vn", str(audio_path)])

    run_command(image_cmd)
    run_command(audio_cmd)


def prepare_clip_frames(
    video_path: Path,
    video_root: Path,
    work_dir: Path,
    fps: float = 25.0,
    size: Optional[Tuple[int, int]] = (512, 512),
    duration: Optional[float] = 5.0,
    force: bool = False,
) -> Tuple[Path, Path, str]:
    """Extract frames/audio for one video into the Hallo-expected layout.

    Returns:
        frame_dir: directory containing extracted PNG frames
        wav_path: path to extracted WAV audio
        clip_id: relative clip identifier (used as subdirectory name)
    """
    clip_id = str(video_path.relative_to(video_root).with_suffix(""))
    frames_root = work_dir / "frames"
    frame_dir = frames_root / "frame" / clip_id
    wav_path = (frames_root / "wav" / clip_id).with_suffix(".wav")

    extract_video(
        video_path=video_path,
        frame_dir=frame_dir,
        audio_path=wav_path,
        fps=fps,
        size=size,
        duration=duration,
        force=force,
    )
    return frame_dir, wav_path, clip_id
