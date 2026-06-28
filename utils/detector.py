"""X-AVDT classifier loading and inference."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Lambda
from torchvision.transforms.functional import normalize

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_NETWORK = REPO_ROOT / "train" / "utils" / "network.py"


def _load_network_module():
    spec = importlib.util.spec_from_file_location("xavdt_network", TRAIN_NETWORK)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load classifier module from {TRAIN_NETWORK}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["xavdt_network"] = module
    spec.loader.exec_module(module)
    return module


_network_module = _load_network_module()
Classifier = _network_module.Classifier
load_checkpoint_into_model = _network_module.load_checkpoint_into_model

from utils.packing import InferenceChunk  # noqa: E402


def _to_float01(x: torch.Tensor) -> torch.Tensor:
    return x.float() / 255.0


class NormalizeVideo:
    """Per-frame channel-wise normalization for a 4D video tensor (C, T, H, W)."""

    def __init__(self, mean: List[float], std: List[float]):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(video.device).view(-1)
        std = self.std.to(video.device).view(-1)
        _, time_steps, _, _ = video.shape
        out = torch.empty_like(video)
        for t in range(time_steps):
            out[:, t, :, :] = normalize(video[:, t, :, :], mean=mean, std=std)
        return out


_COMPOSITE_TRANSFORM = Compose([
    Lambda(_to_float01),
    NormalizeVideo([0.45] * 12, [0.225] * 12),
])


def load_classifier(ckpt_path: str | Path, norm: str = "batch", device: torch.device | None = None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Classifier(norm_layer=norm).to(device)

    missing, unexpected = load_checkpoint_into_model(
        model, ckpt_path, strict=False, inference_only=True
    )
    if missing:
        print(f"[Warn] missing keys: {missing}")
    if unexpected:
        print(f"[Warn] unexpected keys: {unexpected}")

    model.eval()
    return model


def normalize_composite(tensor: torch.Tensor) -> torch.Tensor:
    return _COMPOSITE_TRANSFORM(tensor)


@torch.inference_mode()
def predict_chunk(
    model: Classifier,
    composite: torch.Tensor,
    attn: torch.Tensor,
    device: torch.device,
) -> float:
    composite = normalize_composite(composite).unsqueeze(0).to(device=device, dtype=torch.float)
    attn = attn.unsqueeze(0).to(device=device, dtype=torch.float)
    logits, _ = model(composite, attn)
    return float(F.softmax(logits, dim=1)[0, 1].item())


def score_video(
    model: Classifier,
    chunks: List[InferenceChunk],
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    if not chunks:
        raise ValueError("No inference chunks available for scoring.")

    chunk_scores = [predict_chunk(model, chunk.composite, chunk.attn, device) for chunk in chunks]
    score = float(sum(chunk_scores) / len(chunk_scores))
    max_score = float(max(chunk_scores))
    label = "fake" if score >= threshold else "real"
    return {
        "label": label,
        "score": score,
        "max_score": max_score,
        "chunk_scores": chunk_scores,
        "num_chunks": len(chunk_scores),
    }
