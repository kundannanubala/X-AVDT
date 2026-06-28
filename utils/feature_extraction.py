"""Hallo feature extraction wrapper for inference."""

from __future__ import annotations

import glob
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
HALLO_ROOT = REPO_ROOT / "hallo"
if str(HALLO_ROOT) not in sys.path:
    sys.path.insert(0, str(HALLO_ROOT))


@dataclass
class FeatureBundle:
    original: torch.Tensor  # (3, T, H, W) uint8
    inverted: torch.Tensor
    reconstructed: torch.Tensor
    residual: torch.Tensor
    attn_feat: torch.Tensor  # (T, 4096, 320)


def _images_to_tensor(images: List[Union[str, Image.Image]]) -> torch.Tensor:
    frames = []
    for image in images:
        if isinstance(image, Image.Image):
            arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
        else:
            with Image.open(image) as opened:
                arr = np.asarray(opened.convert("RGB"), dtype=np.uint8)
        frames.append(torch.from_numpy(arr).permute(2, 0, 1))
    if not frames:
        raise ValueError("No frames available to build a video tensor.")
    return torch.stack(frames, dim=1).contiguous()


def _compute_residual(original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    residual = torch.abs(original.float() - reconstructed.float())
    return residual.clamp(0, 255).to(torch.uint8)


class HalloExtractor:
    def __init__(
        self,
        config_path: str | Path | None = None,
        device: torch.device | None = None,
        cache_dir: str | Path | None = None,
    ):
        import extract_features as ef

        self._ef = ef
        ef.load_runtime_dependencies()

        from omegaconf import OmegaConf

        config_path = Path(config_path or HALLO_ROOT / "configs/inference/default.yaml")
        self.config = OmegaConf.load(str(config_path))
        self.config = ef.resolve_config_paths(self.config)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = Path(cache_dir or self.config.save_path)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        ef.save_path = str(self.cache_dir)

        weight_dtype = self._resolve_weight_dtype(self.config.weight_dtype)
        self.weight_dtype = weight_dtype

        from diffusers import AutoencoderKL
        from hallo.models.audio_proj import AudioProjModel
        from hallo.models.face_locator import FaceLocator
        from hallo.models.image_proj import ImageProjModel
        from hallo.models.unet_2d_condition import UNet2DConditionModel
        from hallo.models.unet_3d import UNet3DConditionModel

        config = self.config
        self.vae = AutoencoderKL.from_pretrained(config.vae.model_path)
        reference_unet = UNet2DConditionModel.from_pretrained(config.base_model_path, subfolder="unet")
        denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            config.base_model_path,
            config.motion_module_path,
            subfolder="unet",
            unet_additional_kwargs=OmegaConf.to_container(config.unet_additional_kwargs),
            use_landmark=False,
        )
        face_locator = FaceLocator(conditioning_embedding_channels=320)
        image_proj = ImageProjModel(
            cross_attention_dim=denoising_unet.config.cross_attention_dim,
            clip_embeddings_dim=512,
            clip_extra_context_tokens=4,
        )
        audio_proj = AudioProjModel(
            seq_len=5,
            blocks=12,
            channels=768,
            intermediate_dim=512,
            output_dim=768,
            context_tokens=32,
        ).to(device=self.device, dtype=weight_dtype)

        for module in (self.vae, image_proj, reference_unet, denoising_unet, face_locator, audio_proj):
            module.requires_grad_(False)

        reference_unet.enable_gradient_checkpointing()
        denoising_unet.enable_gradient_checkpointing()

        self.net = ef.Net(reference_unet, denoising_unet, face_locator, image_proj, audio_proj)
        ckpt_path = os.path.join(config.audio_ckpt_dir, "net.pth")
        missing, unexpected = self.net.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        if missing or unexpected:
            raise RuntimeError(f"Failed to load Hallo checkpoint from {ckpt_path}")

        self.vae = self.vae.to(self.device, dtype=weight_dtype)
        self.net = self.net.to(self.device, dtype=weight_dtype)
        self.net.eval()

    @staticmethod
    def _resolve_weight_dtype(weight_dtype: str) -> torch.dtype:
        if weight_dtype == "fp16":
            return torch.float16
        if weight_dtype == "bf16":
            return torch.bfloat16
        if weight_dtype == "fp32":
            return torch.float32
        return torch.float32

    def _prepare_audio(self, wav_path: str | Path):
        from hallo.datasets.audio_processor import AudioProcessor

        config = self.config
        sample_rate = config.data.driving_audio.sample_rate
        if sample_rate != 16000:
            raise ValueError("audio sample rate must be 16000")

        clip_length = config.data.n_sample_frames
        audio_separator_model_file = config.audio_separator.model_path
        with AudioProcessor(
            sample_rate,
            config.data.export_video.fps,
            config.wav2vec.model_path,
            config.wav2vec.features == "last",
            os.path.dirname(audio_separator_model_file),
            os.path.basename(audio_separator_model_file),
            os.path.join(self.cache_dir, "audio_preprocess"),
        ) as audio_processor:
            audio_emb, audio_length = audio_processor.preprocess(str(wav_path), clip_length)
        return self._ef.process_audio_emb(audio_emb), audio_length

    @torch.inference_mode()
    def extract(self, frame_dir: str | Path, wav_path: str | Path) -> FeatureBundle:
        frame_dir = Path(frame_dir)
        wav_path = Path(wav_path)
        source_image_path = sorted(glob.glob(str(frame_dir / "*.png")))
        if not source_image_path:
            raise ValueError(f"No PNG frames found in {frame_dir}")

        audio_emb, audio_length = self._prepare_audio(wav_path)

        inverted_latents, _, z0, inverted_images, attn_feat = self._ef.inversion_process(
            self.config,
            self.device,
            self.weight_dtype,
            source_image_path,
            audio_emb,
            audio_length,
            self.vae,
            self.net,
        )
        _, z0_recon, reconstructed_images = self._ef.reconstruction_process(
            self.config,
            inverted_latents,
            self.device,
            self.weight_dtype,
            source_image_path,
            audio_emb,
            audio_length,
            self.vae,
            self.net,
        )

        min_len = min(
            len(source_image_path),
            len(inverted_images),
            len(reconstructed_images),
            attn_feat.shape[0],
            z0.shape[2],
            z0_recon.shape[2],
        )
        if min_len < self.config.data.n_sample_frames:
            raise ValueError(
                f"Clip too short for inference ({min_len} frames); "
                f"need at least {self.config.data.n_sample_frames}."
            )

        original_tensor = _images_to_tensor(source_image_path[:min_len])
        inverted_tensor = _images_to_tensor(inverted_images[:min_len])
        reconstructed_tensor = _images_to_tensor(reconstructed_images[:min_len])
        residual_tensor = _compute_residual(original_tensor, reconstructed_tensor)

        return FeatureBundle(
            original=original_tensor,
            inverted=inverted_tensor,
            reconstructed=reconstructed_tensor,
            residual=residual_tensor,
            attn_feat=attn_feat[:min_len].detach().cpu(),
        )
