#!/usr/bin/env python3
"""X-AVDT inference engine — score raw talking-head videos for deepfake detection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.detector import load_classifier, score_video
from utils.feature_extraction import HalloExtractor
from utils.io import InferenceResults, VideoResult, build_meta, cleanup_work_dir, discover_videos, make_work_dir, write_results
from utils.hallo_pack import pack_chunks_from_hallo_clip
from utils.preprocessing import prepare_clip_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run end-to-end X-AVDT inference on a folder of raw videos. "
            "Requires Hallo weights under hallo/pretrained_models/ and "
            "the detector checkpoint at models/model_best.pt."
        )
    )
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing raw videos.")
    parser.add_argument("--output", type=str, default="results/inference.json", help="Output JSON path.")
    parser.add_argument("--checkpoint", type=str, default="models/model_best.pt", help="X-AVDT detector checkpoint.")
    parser.add_argument("--work_dir", type=str, default=None, help="Intermediate working directory.")
    parser.add_argument("--keep_intermediates", action="store_true", help="Keep intermediate frame/feature files.")
    parser.add_argument("--duration", type=float, default=5.0, help="Max seconds per video.")
    parser.add_argument("--fps", type=float, default=25.0, help="Frame extraction FPS.")
    parser.add_argument("--size", type=int, nargs=2, default=(512, 512), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--threshold", type=float, default=0.5, help="Fake/real decision threshold on P(fake).")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu (auto-detected if omitted).")
    parser.add_argument("--norm", type=str, choices=["batch", "instance", "layer"], default="batch")
    parser.add_argument("--hallo_config", type=str, default=None, help="Optional Hallo inference config path.")
    parser.add_argument("--clip_len", type=int, default=16, help="Frames per classifier chunk.")
    return parser.parse_args()


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Detector checkpoint not found: {checkpoint}\n"
            "Download model_best.pt from the README link and place it under models/."
        )

    video_dir = Path(args.video_dir)
    videos = discover_videos(video_dir)
    duration = None if args.duration is not None and args.duration <= 0 else args.duration
    size = tuple(args.size)

    work_dir = make_work_dir(Path(args.work_dir) if args.work_dir else None)
    hallo_cache = work_dir / "hallo_cache"
    hallo_cache.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(videos)} video(s). Work dir: {work_dir}")
    print(f"Loading Hallo extractor on {device}...")
    hallo = HalloExtractor(config_path=args.hallo_config, device=device, cache_dir=hallo_cache)

    print(f"Loading classifier from {checkpoint}...")
    classifier = load_classifier(checkpoint, norm=args.norm, device=device)

    results = InferenceResults(
        meta=build_meta(
            checkpoint=str(checkpoint),
            threshold=args.threshold,
            num_videos=len(videos),
            device=str(device),
            duration=args.duration,
            fps=args.fps,
            size=list(size),
        )
    )

    try:
        for video_path in tqdm(videos, desc="Inferring"):
            try:
                frame_dir, wav_path, clip_id = prepare_clip_frames(
                    video_path=video_path,
                    video_root=video_dir,
                    work_dir=work_dir,
                    fps=args.fps,
                    size=size,
                    duration=duration,
                )
                hallo_clip_dir = work_dir / "hallo_features" / clip_id
                hallo.extract_to_dir(frame_dir=frame_dir, wav_path=wav_path, output_dir=hallo_clip_dir)
                chunks = pack_chunks_from_hallo_clip(hallo_clip_dir, clip_len=args.clip_len)
                if not chunks:
                    raise ValueError("No valid 16-frame chunks produced from extracted features.")

                scored = score_video(classifier, chunks, device=device, threshold=args.threshold)
                results.videos.append(
                    VideoResult(
                        video=str(video_path.relative_to(video_dir)),
                        label=scored["label"],
                        score=scored["score"],
                        chunk_scores=scored["chunk_scores"],
                        num_chunks=scored["num_chunks"],
                        max_score=scored["max_score"],
                    )
                )
            except Exception as exc:
                results.errors.append({"video": str(video_path), "error": str(exc)})
                print(f"[Error] {video_path}: {exc}")
    finally:
        cleanup_work_dir(work_dir, keep=args.keep_intermediates)

    output_path = Path(args.output)
    write_results(output_path, results)

    print("\n==== Inference Summary ====")
    for item in results.videos:
        print(f"{item.video}: {item.label} (score={item.score:.4f}, chunks={item.num_chunks})")
    if results.errors:
        print(f"\n{len(results.errors)} video(s) failed.")
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
