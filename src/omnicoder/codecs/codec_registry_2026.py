from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from omnicoder.tokenization.omni_ledger_2026 import DEFAULT_LEDGER

CodecDirection = Literal["input", "output", "bidirectional"]


@dataclass(frozen=True)
class CodecSpec:
    codec_id: str
    modality: str
    direction: CodecDirection
    ledger_ranges: tuple[str, ...]
    representation: str
    status: str
    source: str
    training_targets: tuple[str, ...]
    notes: tuple[str, ...]


CODECS: tuple[CodecSpec, ...] = (
    CodecSpec(
        codec_id="text_qwen_chatml_bytes_bpe",
        modality="text_code_tool",
        direction="bidirectional",
        ledger_ranges=("text", "control", "tool_agent"),
        representation="discrete token ids",
        status="active",
        source="https://github.com/QwenLM/Qwen3.6",
        training_targets=("next_token_ce", "tool_schema_ce", "action_token_weighting"),
        notes=("GGUF bridge should use a qwen-compatible tokenizer path.",),
    ),
    CodecSpec(
        codec_id="cosmos_visual_semantic",
        modality="image_video_understanding",
        direction="bidirectional",
        ledger_ranges=("vision_semantic", "vision_residual", "time_space"),
        representation="continuous/discrete visual tokens, spatial 8x/16x and temporal 4x/8x style compression",
        status="planned_external_codec",
        source="https://github.com/NVIDIA/Cosmos-Tokenizer",
        training_targets=("image_text_contrastive", "latent_reconstruction", "caption_grounding", "edit_preservation"),
        notes=("Raw pixels stay outside the trunk; trunk sees ledger packets and flow targets.",),
    ),
    CodecSpec(
        codec_id="qwen3_omni_speech_rvq",
        modality="speech_tts",
        direction="bidirectional",
        ledger_ranges=("speech_tts", "time_space"),
        representation="speech semantic/prosody/RVQ codec tokens",
        status="planned_external_codec",
        source="https://github.com/QwenLM/Qwen3-Omni",
        training_targets=("codec_token_ce", "prosody_loss", "speaker_similarity", "turn_taking_latency"),
        notes=("Keep reasoning trunk separate from low-latency renderer/talker path.",),
    ),
    CodecSpec(
        codec_id="ltx_2_3_video_audio_latents",
        modality="text_image_video_to_video_audio",
        direction="output",
        ledger_ranges=("vision_semantic", "vision_residual", "audio_music", "time_space", "flow"),
        representation="trunk-conditioned video/audio latent flow plan",
        status="teacher_and_renderer",
        source="https://ltx.io/model/ltx-2-3",
        training_targets=("shot_plan_ce", "spatiotemporal_flow_loss", "audio_video_sync", "temporal_consistency"),
        notes=("Used first as teacher/renderer; student learns prompts/plans/critiques and latent flow targets.",),
    ),
    CodecSpec(
        codec_id="ace_step_1_5_music_plan_latents",
        modality="music_audio",
        direction="output",
        ledger_ranges=("audio_music", "music_control", "time_space", "flow"),
        representation="LM song plan plus music/audio latent diffusion targets",
        status="teacher_and_renderer",
        source="https://arxiv.org/abs/2602.00744",
        training_targets=("music_plan_ce", "beat_key_structure", "lyric_alignment", "latent_flow_loss"),
        notes=("Use for music generation/editing distillation, not as an in-trunk adapter.",),
    ),
    CodecSpec(
        codec_id="gpt_image_2_reference_teacher",
        modality="image_generation_edit",
        direction="output",
        ledger_ranges=("vision_semantic", "vision_residual", "flow"),
        representation="image generation/editing teacher traces and rubric labels",
        status="optional_api_teacher",
        source="https://developers.openai.com/api/docs/guides/image-generation",
        training_targets=("prompt_rewrite_ce", "edit_instruction_following", "ocr_text_rendering", "composition_rubric"),
        notes=("Only use where credentials/licensing/policy allow; store provenance and model version.",),
    ),
    CodecSpec(
        codec_id="gemini_omni_video_audio_reference",
        modality="any_input_to_video_audio",
        direction="output",
        ledger_ranges=("vision_semantic", "vision_residual", "audio_music", "time_space", "flow"),
        representation="video-with-audio capability reference and eval rubric",
        status="reference_only",
        source="https://deepmind.google/models/gemini-omni/",
        training_targets=("video_prompt_rubric", "audio_video_alignment", "multimodal_instruction_following"),
        notes=("Use public behavior/model-card lessons; do not assume unpublished internals.",),
    ),
)


def registry_manifest() -> dict:
    ledger = DEFAULT_LEDGER.as_metadata()
    return {
        "schema": "omnicoder2026_codec_registry_v1",
        "trunk_rule": "the dense trunk consumes typed ledger tokens and supervised flow targets; raw media codecs live at the edge",
        "adapter_rule": "no learned modality adapters inside the trunk",
        "ledger": ledger,
        "codecs": [asdict(codec) for codec in CODECS],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Emit the Omnicoder 2026 edge codec registry")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    manifest = registry_manifest()
    text = json.dumps(manifest, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
