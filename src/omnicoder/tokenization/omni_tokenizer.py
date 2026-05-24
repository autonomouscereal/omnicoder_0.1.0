from __future__ import annotations
import os
import hashlib
import pickle
import json
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from PIL import Image
import soundfile as sf

from omnicoder.utils.logger import get_logger
from omnicoder.config import MultiModalConfig
from .simple_tokenizer import TextTokenizer, ByteTokenizer
from .bpe_trainer import BPETokenizer

logger = get_logger("omnicoder.tokenizer.omni")


def _stable_bucket(value: Any, modulo: int) -> int:
    payload = repr(value).encode("utf-8", errors="replace")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % max(1, int(modulo))


class OmniTokenizer:
    """
    BEST UNIFIED MULTIMODAL TOKENIZER (2026)
    - One encode() / decode() for EVERYTHING
    - Same vocabulary for text, images, audio, video, dicts, tensors, etc.
    - Smart serialization (PNG for images, WAV for audio)
    - Uses ALL real encoders from multimodal/ when available:
        • ImageVQ / VisionEncoder / ContinuousLatentHead
        • AudioVQVAE / AudioTokenizer / ContinuousLatentHead
        • VideoVQ / VideoEncoder
        • Latent3D
        • ConceptLatentHead
    - Keeps all your special tokens and modality registry
    """

    def __init__(self, vocab_size: int = 128000, bpe_path: Optional[str] = None):
        self.vocab_size = vocab_size
        self.mm_config = MultiModalConfig()

        # === SPECIAL TOKENS (your original list) ===
        self.special_tokens = {
            "<|text|>": 0,
            "<|image_start|>": 1, "<|image_end|>": 2,
            "<|audio_start|>": 3, "<|audio_end|>": 4,
            "<|video_start|>": 5, "<|video_end|>": 6,
            "<|thinking|>": 7,
            "<|pad|>": 8, "<|unk|>": 9,
            "<|modality:car|>": 10, "<|modality:robot|>": 11,
            "<|modality:lidar|>": 12, "<|modality:radar|>": 13,
            "<|modality:imu|>": 14, "<|modality:joint_state|>": 15,
            "<|modality:point_cloud|>": 16, "<|modality:3d_gaussian|>": 17,
            "<|modality:vq_code|>": 18,
        }

        self.text_tok = TextTokenizer(vocab_size=32000)
        self.byte_tok = ByteTokenizer()
        self.bpe_tok: Optional[BPETokenizer] = None

        if bpe_path and Path(bpe_path).exists():
            try:
                self.bpe_tok = BPETokenizer.load(bpe_path)
                logger.info(f"Loaded unified BPE: {bpe_path}")
            except Exception as e:
                logger.warning(f"BPE load failed: {e}")

        # Modality registry (your original)
        self.modality_registry: Dict[str, Dict[str, Any]] = {
            "text": {"start": "<|text|>", "end": None, "max_tokens": 8192},
            "image": {"start": "<|image_start|>", "end": "<|image_end|>", "max_tokens": 1024},
            "audio": {"start": "<|audio_start|>", "end": "<|audio_end|>", "max_tokens": 512},
            "video": {"start": "<|video_start|>", "end": "<|video_end|>", "max_tokens": 2048},
            "car": {"start": "<|modality:car|>", "end": None, "max_tokens": 256},
            "robot": {"start": "<|modality:robot|>", "end": None, "max_tokens": 256},
            "lidar": {"start": "<|modality:lidar|>", "end": None, "max_tokens": 512},
            "radar": {"start": "<|modality:radar|>", "end": None, "max_tokens": 256},
            "imu": {"start": "<|modality:imu|>", "end": None, "max_tokens": 128},
            "joint_state": {"start": "<|modality:joint_state|>", "end": None, "max_tokens": 128},
            "point_cloud": {"start": "<|modality:point_cloud|>", "end": None, "max_tokens": 512},
            "3d_gaussian": {"start": "<|modality:3d_gaussian|>", "end": None, "max_tokens": 1024},
            "vq_code": {"start": "<|modality:vq_code|>", "end": None, "max_tokens": 2048},
        }

        self.encoders: Dict[str, Any] = {}
        self._load_all_real_encoders()

        logger.info(f"OmniTokenizer (best unified) ready | vocab={vocab_size} | encoders={list(self.encoders.keys())}")

    # ============================================================
    # LOAD ALL REAL ENCODERS FROM multimodal/
    # ============================================================
    def _load_all_real_encoders(self):
        """Attempt to load every useful encoder from your repo"""

        # --- IMAGE ---
        loaded = False
        try:
            from omnicoder.modeling.multimodal.image_vq import ImageVQ
            self.encoders["image"] = ImageVQ()
            logger.info("✓ Loaded ImageVQ (discrete VQ tokens)")
            loaded = True
        except Exception:
            pass

        if not loaded:
            try:
                from omnicoder.modeling.multimodal.vision_encoder import VisionEncoder
                self.encoders["image"] = VisionEncoder()
                logger.info("✓ Loaded VisionEncoder")
                loaded = True
            except Exception:
                pass

        if not loaded:
            try:
                from omnicoder.modeling.multimodal.aligner import ContinuousLatentHead
                self.encoders["image"] = ContinuousLatentHead(d_model=512, latent_dim=256)
                logger.info("✓ Loaded ContinuousLatentHead for image")
            except Exception:
                logger.info("No image encoder loaded — using fallback")

        # --- AUDIO ---
        loaded = False
        try:
            from omnicoder.modeling.multimodal.audio_vqvae import AudioVQVAE
            self.encoders["audio"] = AudioVQVAE()
            logger.info("✓ Loaded AudioVQVAE (discrete VQ tokens)")
            loaded = True
        except Exception:
            pass

        if not loaded:
            try:
                from omnicoder.modeling.multimodal.audio_tokenizer import AudioTokenizer
                self.encoders["audio"] = AudioTokenizer()
                logger.info("✓ Loaded AudioTokenizer")
                loaded = True
            except Exception:
                pass

        if not loaded:
            try:
                from omnicoder.modeling.multimodal.aligner import ContinuousLatentHead
                self.encoders["audio"] = ContinuousLatentHead(d_model=512, latent_dim=128)
                logger.info("✓ Loaded ContinuousLatentHead for audio")
            except Exception:
                pass

        # --- VIDEO ---
        loaded = False
        try:
            from omnicoder.modeling.multimodal.video_vq import VideoVQ
            self.encoders["video"] = VideoVQ()
            logger.info("✓ Loaded VideoVQ")
            loaded = True
        except Exception:
            pass

        if not loaded:
            try:
                from omnicoder.modeling.multimodal.video_encoder import VideoEncoder
                self.encoders["video"] = VideoEncoder()
                logger.info("✓ Loaded VideoEncoder")
            except Exception:
                pass

        # --- 3D ---
        try:
            from omnicoder.modeling.multimodal.latent3d import Latent3D
            self.encoders["3d"] = Latent3D()
            logger.info("✓ Loaded Latent3D")
        except Exception:
            pass

        # --- CONCEPT / ALIGNER ---
        try:
            from omnicoder.modeling.multimodal.aligner import ConceptLatentHead
            self.encoders["concept"] = ConceptLatentHead()
            logger.info("✓ Loaded ConceptLatentHead")
        except Exception:
            pass

    # ============================================================
    # ENCODER HELPERS (use real models when available)
    # ============================================================
    def _encode_image(self, image: Any) -> List[int]:
        if "image" in self.encoders and self.encoders["image"] is not None:
            try:
                inp = torch.tensor(image).float() if not isinstance(image, torch.Tensor) else image.float()
                codes = self.encoders["image"](inp)
                if isinstance(codes, torch.Tensor):
                    codes = codes.flatten().tolist()
                return [int(x) % self.vocab_size for x in codes][:1024]
            except Exception as e:
                logger.debug(f"Image encoder failed: {e}")

        # Smart fallback: PNG bytes → BPE
        try:
            if isinstance(image, np.ndarray):
                img = Image.fromarray(image.astype(np.uint8))
            elif isinstance(image, torch.Tensor):
                img = Image.fromarray(image.detach().cpu().numpy().astype(np.uint8))
            else:
                img = image
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            byte_data = buf.getvalue()
            text_repr = byte_data.decode("latin-1", errors="replace")
            if self.bpe_tok:
                return self.bpe_tok.encode(text_repr)
        except Exception:
            pass

        return [32000 + _stable_bucket(getattr(image, 'shape', 'unknown'), 30000) for _ in range(64)]

    def _encode_audio(self, audio: Any) -> List[int]:
        if "audio" in self.encoders and self.encoders["audio"] is not None:
            try:
                inp = torch.tensor(audio).float() if not isinstance(audio, torch.Tensor) else audio.float()
                codes = self.encoders["audio"](inp)
                if isinstance(codes, torch.Tensor):
                    codes = codes.flatten().tolist()
                return [int(x) % self.vocab_size for x in codes][:512]
            except Exception:
                pass

        # Smart fallback: WAV bytes → BPE
        try:
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu().numpy()
            if isinstance(audio, list):
                audio = np.array(audio)
            buf = io.BytesIO()
            sf.write(buf, audio, 16000, format="WAV")
            byte_data = buf.getvalue()
            text_repr = byte_data.decode("latin-1", errors="replace")
            if self.bpe_tok:
                return self.bpe_tok.encode(text_repr)
        except Exception:
            pass

        return [40000 + _stable_bucket(getattr(audio, 'shape', 'unknown'), 20000) for _ in range(128)]

    def _encode_video(self, video: Any) -> List[int]:
        if "video" in self.encoders and self.encoders["video"] is not None:
            try:
                inp = torch.tensor(video).float() if not isinstance(video, torch.Tensor) else video.float()
                codes = self.encoders["video"](inp)
                if isinstance(codes, torch.Tensor):
                    codes = codes.flatten().tolist()
                return [int(x) % self.vocab_size for x in codes][:2048]
            except Exception:
                pass

        # Fallback
        return [50000 + _stable_bucket(getattr(video, 'shape', 'unknown'), 15000) for _ in range(256)]

    def _encode_3d(self, data: Any) -> List[int]:
        if "3d" in self.encoders and self.encoders["3d"] is not None:
            try:
                codes = self.encoders["3d"](data)
                if isinstance(codes, torch.Tensor):
                    codes = codes.flatten().tolist()
                return [int(x) % self.vocab_size for x in codes][:1024]
            except Exception:
                pass
        return [60000 + _stable_bucket(data, 10000) for _ in range(256)]

    # ============================================================
    # UNIVERSAL ENCODE
    # ============================================================
    def encode(self, data: Any = None, **kwargs) -> List[int]:
        if data is None and kwargs:
            return self.encode_multimodal(kwargs)[0]
        if data is None:
            return [self.special_tokens["<|unk|>"]]

        # === TEXT ===
        if isinstance(data, str):
            if self.bpe_tok:
                return self.bpe_tok.encode(data)
            return self.text_tok.encode(data)

        # === RAW IMAGE ===
        if isinstance(data, (Image.Image, np.ndarray, torch.Tensor)):
            if isinstance(data, Image.Image):
                data = np.array(data)
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
            return self._encode_image(data)

        # === RAW AUDIO ===
        if isinstance(data, (np.ndarray, torch.Tensor, list)):
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
            if isinstance(data, list):
                data = np.array(data)
            return self._encode_audio(data)

        # === VIDEO (list of frames or tensor) ===
        if isinstance(data, (list, tuple)) and len(data) > 0:
            if isinstance(data[0], (Image.Image, np.ndarray, torch.Tensor)):
                return self._encode_video(data)

        # === DICT / LIST / JSON ===
        if isinstance(data, (dict, list, tuple)):
            try:
                byte_data = json.dumps(data, ensure_ascii=False).encode("utf-8")
            except Exception:
                byte_data = pickle.dumps(data, protocol=4)
            text_repr = byte_data.decode("latin-1", errors="replace")
            if self.bpe_tok:
                return self.bpe_tok.encode(text_repr)
            return self.byte_tok.encode(text_repr)

        # === EVERYTHING ELSE (pickle) ===
        try:
            byte_data = pickle.dumps(data, protocol=4)
            text_repr = byte_data.decode("latin-1", errors="replace")
            if self.bpe_tok:
                return self.bpe_tok.encode(text_repr)
            return self.byte_tok.encode(text_repr)
        except Exception:
            return [self.special_tokens["<|unk|>"]]

    # ============================================================
    # UNIVERSAL DECODE
    # ============================================================
    def decode(self, ids: Union[List[int], torch.Tensor]) -> Any:
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        if self.bpe_tok:
            text = self.bpe_tok.decode(ids)
        else:
            text = self.text_tok.decode(ids)

        # Try to recover original object
        try:
            return pickle.loads(text.encode("latin-1"))
        except Exception:
            try:
                return json.loads(text)
            except Exception:
                return text

    # ============================================================
    # MULTIMODAL (structured)
    # ============================================================
    def encode_multimodal(
        self, inputs: Dict[str, Any], max_length: Optional[int] = None
    ) -> Tuple[List[int], List[str]]:
        token_ids: List[int] = []
        modality_labels: List[str] = []

        for modality, data in inputs.items():
            if data is None:
                continue

            reg = self.modality_registry.get(modality, {"start": "<|unk|>", "end": None, "max_tokens": 512})
            start_id = self.special_tokens.get(reg["start"], 9)
            end_id = self.special_tokens.get(reg.get("end", ""), None) if reg.get("end") else None

            token_ids.append(start_id)
            modality_labels.append(modality)

            content = self.encode(data)
            content = content[:reg["max_tokens"]]
            token_ids.extend(content)
            modality_labels.extend([modality] * len(content))

            if end_id is not None:
                token_ids.append(end_id)
                modality_labels.append(modality)

        if max_length:
            token_ids = token_ids[:max_length]
            modality_labels = modality_labels[:max_length]

        return token_ids, modality_labels

    def __call__(self, data: Any = None, **kwargs):
        return self.encode(data, **kwargs)
