#!/usr/bin/env python3
"""Run bounded real-config Omnicoder 20B profiling variants on the AI server."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


TRUTHY = {"1", "true", "yes", "on"}
DEFAULT_VARIANTS: list[dict[str, Any]] = [
    {
        "name": "baseline_clean",
        "steps": 1,
        "env": {
            "OMNICODER_FAKE_QUANT": "1",
            "OMNICODER_FAKE_QUANT_CHUNK_ROWS": "16",
            "OMNICODER_ACTIVATION_CHECKPOINTING": "1",
            "OMNICODER_PIPELINE_STAGE_SCHEDULE": "gpipe",
            "OMNICODER_PIPELINE_MICROBATCHES": "1",
            "OMNICODER_BATCH_SIZE": "1",
            "OMNICODER_STAGE_ORDER": "text",
            "OMNICODER2026_LOSS_DIAGNOSTICS_INTERVAL": "0",
        },
    },
    {
        "name": "fakequant_off",
        "steps": 1,
        "env": {
            "OMNICODER_FAKE_QUANT": "0",
            "OMNICODER_PROFILE_ALLOW_FAKE_QUANT_OFF": "1",
            "OMNICODER_ACTIVATION_CHECKPOINTING": "1",
            "OMNICODER_PIPELINE_STAGE_SCHEDULE": "gpipe",
            "OMNICODER_PIPELINE_MICROBATCHES": "1",
            "OMNICODER_BATCH_SIZE": "1",
            "OMNICODER_STAGE_ORDER": "text",
            "OMNICODER2026_LOSS_DIAGNOSTICS_INTERVAL": "0",
        },
    },
    {
        "name": "fakequant_chunk256",
        "steps": 1,
        "env": {
            "OMNICODER_FAKE_QUANT": "1",
            "OMNICODER_FAKE_QUANT_CHUNK_ROWS": "256",
            "OMNICODER_ACTIVATION_CHECKPOINTING": "1",
            "OMNICODER_PIPELINE_STAGE_SCHEDULE": "gpipe",
            "OMNICODER_PIPELINE_MICROBATCHES": "1",
            "OMNICODER_BATCH_SIZE": "1",
            "OMNICODER_STAGE_ORDER": "text",
            "OMNICODER2026_LOSS_DIAGNOSTICS_INTERVAL": "0",
        },
    },
    {
        "name": "fakequant_chunk256_loss64",
        "steps": 1,
        "env": {
            "OMNICODER_FAKE_QUANT": "1",
            "OMNICODER_FAKE_QUANT_CHUNK_ROWS": "256",
            "OMNICODER_MAX_LOSS_TOKENS_PER_SAMPLE": "64",
            "OMNICODER_ACTIVATION_CHECKPOINTING": "1",
            "OMNICODER_PIPELINE_STAGE_SCHEDULE": "gpipe",
            "OMNICODER_PIPELINE_MICROBATCHES": "1",
            "OMNICODER_BATCH_SIZE": "1",
            "OMNICODER_STAGE_ORDER": "text",
            "OMNICODER2026_LOSS_DIAGNOSTICS_INTERVAL": "0",
        },
    },
    {
        "name": "gdn2_compiled_fakequant_chunk256_loss64",
        "default": False,
        "steps": 4,
        "env": {
            "OMNICODER_FAKE_QUANT": "1",
            "OMNICODER_FAKE_QUANT_CHUNK_ROWS": "256",
            "OMNICODER_MAX_LOSS_TOKENS_PER_SAMPLE": "64",
            "OMNICODER_ACTIVATION_CHECKPOINTING": "1",
            "OMNICODER_PIPELINE_STAGE_SCHEDULE": "gpipe",
            "OMNICODER_PIPELINE_MICROBATCHES": "1",
            "OMNICODER_BATCH_SIZE": "1",
            "OMNICODER_STAGE_ORDER": "text",
            "OMNICODER2026_GDN2_COMPILED_CHUNKS": "1",
            "OMNICODER2026_GDN2_COMPILED_MODE": "chunked",
            "OMNICODER2026_GDN2_COMPILED_CHUNK_TOKENS": "32",
            "OMNICODER2026_LOSS_DIAGNOSTICS_INTERVAL": "4",
        },
    },
    {
        "name": "gdn2_jit_q4_loss64",
        "default": False,
        "steps": 2,
        "env": {
            "OMNICODER_FAKE_QUANT": "1",
            "OMNICODER_FAKE_QUANT_CHUNK_ROWS": "256",
            "OMNICODER_MAX_LOSS_TOKENS_PER_SAMPLE": "64",
            "OMNICODER_ACTIVATION_CHECKPOINTING": "1",
            "OMNICODER_PIPELINE_STAGE_SCHEDULE": "gpipe",
            "OMNICODER_PIPELINE_MICROBATCHES": "1",
            "OMNICODER_BATCH_SIZE": "1",
            "OMNICODER_STAGE_ORDER": "text",
            "OMNICODER2026_GDN2_JIT_SCAN": "1",
            "OMNICODER2026_LOSS_DIAGNOSTICS_INTERVAL": "0",
        },
    },
    {
        "name": "fakequant_chunk1024_loss64",
        "steps": 1,
        "env": {
            "OMNICODER_FAKE_QUANT": "1",
            "OMNICODER_FAKE_QUANT_CHUNK_ROWS": "1024",
            "OMNICODER_MAX_LOSS_TOKENS_PER_SAMPLE": "64",
            "OMNICODER_ACTIVATION_CHECKPOINTING": "1",
            "OMNICODER_PIPELINE_STAGE_SCHEDULE": "gpipe",
            "OMNICODER_PIPELINE_MICROBATCHES": "1",
            "OMNICODER_BATCH_SIZE": "1",
            "OMNICODER_STAGE_ORDER": "text",
            "OMNICODER2026_LOSS_DIAGNOSTICS_INTERVAL": "0",
        },
    },
    {
        "name": "fakequant_chunk2048_loss64",
        "steps": 1,
        "env": {
            "OMNICODER_FAKE_QUANT": "1",
            "OMNICODER_FAKE_QUANT_CHUNK_ROWS": "2048",
            "OMNICODER_MAX_LOSS_TOKENS_PER_SAMPLE": "64",
            "OMNICODER_ACTIVATION_CHECKPOINTING": "1",
            "OMNICODER_PIPELINE_STAGE_SCHEDULE": "gpipe",
            "OMNICODER_PIPELINE_MICROBATCHES": "1",
            "OMNICODER_BATCH_SIZE": "1",
            "OMNICODER_STAGE_ORDER": "text",
            "OMNICODER2026_LOSS_DIAGNOSTICS_INTERVAL": "0",
        },
    },
    {
        "name": "fakequant_chunk4096_loss64",
        "default": False,
        "steps": 1,
        "env": {
            "OMNICODER_FAKE_QUANT": "1",
            "OMNICODER_FAKE_QUANT_CHUNK_ROWS": "4096",
            "OMNICODER_MAX_LOSS_TOKENS_PER_SAMPLE": "64",
            "OMNICODER_ACTIVATION_CHECKPOINTING": "1",
            "OMNICODER_PIPELINE_STAGE_SCHEDULE": "gpipe",
            "OMNICODER_PIPELINE_MICROBATCHES": "1",
            "OMNICODER_BATCH_SIZE": "1",
            "OMNICODER_STAGE_ORDER": "text",
            "OMNICODER2026_LOSS_DIAGNOSTICS_INTERVAL": "0",
        },
    },
    {
        "name": "fakequant_chunk8192_loss64",
        "default": False,
        "steps": 1,
        "env": {
            "OMNICODER_FAKE_QUANT": "1",
            "OMNICODER_FAKE_QUANT_CHUNK_ROWS": "8192",
            "OMNICODER_MAX_LOSS_TOKENS_PER_SAMPLE": "64",
            "OMNICODER_ACTIVATION_CHECKPOINTING": "1",
            "OMNICODER_PIPELINE_STAGE_SCHEDULE": "gpipe",
            "OMNICODER_PIPELINE_MICROBATCHES": "1",
            "OMNICODER_BATCH_SIZE": "1",
            "OMNICODER_STAGE_ORDER": "text",
            "OMNICODER2026_LOSS_DIAGNOSTICS_INTERVAL": "0",
        },
    },
    {
        "name": "headroom_16_16_32_q4_chunk2048_loss64",
        "default": False,
        "steps": 1,
        "env": {
            "OMNICODER_FAKE_QUANT": "1",
            "OMNICODER_FAKE_QUANT_CHUNK_ROWS": "2048",
            "OMNICODER_MAX_LOSS_TOKENS_PER_SAMPLE": "64",
            "OMNICODER_PLACEMENT_LAYER_COUNTS": "16,16,32",
            "OMNICODER_ACTIVATION_CHECKPOINTING": "1",
            "OMNICODER_PIPELINE_STAGE_SCHEDULE": "gpipe",
            "OMNICODER_PIPELINE_MICROBATCHES": "1",
            "OMNICODER_BATCH_SIZE": "1",
            "OMNICODER_STAGE_ORDER": "text",
            "OMNICODER2026_LOSS_DIAGNOSTICS_INTERVAL": "0",
        },
    },
    {
        "name": "headroom_16_16_32_q4_chunk8192_loss64",
        "default": False,
        "steps": 1,
        "env": {
            "OMNICODER_FAKE_QUANT": "1",
            "OMNICODER_FAKE_QUANT_CHUNK_ROWS": "8192",
            "OMNICODER_MAX_LOSS_TOKENS_PER_SAMPLE": "64",
            "OMNICODER_PLACEMENT_LAYER_COUNTS": "16,16,32",
            "OMNICODER_ACTIVATION_CHECKPOINTING": "1",
            "OMNICODER_PIPELINE_STAGE_SCHEDULE": "gpipe",
            "OMNICODER_PIPELINE_MICROBATCHES": "1",
            "OMNICODER_BATCH_SIZE": "1",
            "OMNICODER_STAGE_ORDER": "text",
            "OMNICODER2026_LOSS_DIAGNOSTICS_INTERVAL": "0",
        },
    },
    {
        "name": "ffn_chunk512_headroom_q4_chunk8192_loss64",
        "default": False,
        "steps": 1,
        "env": {
            "OMNICODER_FAKE_QUANT": "1",
            "OMNICODER_FAKE_QUANT_CHUNK_ROWS": "8192",
            "OMNICODER_FFN_CHUNK_TOKENS": "512",
            "OMNICODER_MAX_LOSS_TOKENS_PER_SAMPLE": "64",
            "OMNICODER_PLACEMENT_LAYER_COUNTS": "16,16,32",
            "OMNICODER_ACTIVATION_CHECKPOINTING": "1",
            "OMNICODER_PIPELINE_STAGE_SCHEDULE": "gpipe",
            "OMNICODER_PIPELINE_MICROBATCHES": "1",
            "OMNICODER_BATCH_SIZE": "1",
            "OMNICODER_STAGE_ORDER": "text",
            "OMNICODER2026_LOSS_DIAGNOSTICS_INTERVAL": "0",
        },
    },
    {
        "name": "ffn_chunk1024_headroom_q4_chunk8192_loss64",
        "default": False,
        "steps": 1,
        "env": {
            "OMNICODER_FAKE_QUANT": "1",
            "OMNICODER_FAKE_QUANT_CHUNK_ROWS": "8192",
            "OMNICODER_FFN_CHUNK_TOKENS": "1024",
            "OMNICODER_MAX_LOSS_TOKENS_PER_SAMPLE": "64",
            "OMNICODER_PLACEMENT_LAYER_COUNTS": "16,16,32",
            "OMNICODER_ACTIVATION_CHECKPOINTING": "1",
            "OMNICODER_PIPELINE_STAGE_SCHEDULE": "gpipe",
            "OMNICODER_PIPELINE_MICROBATCHES": "1",
            "OMNICODER_BATCH_SIZE": "1",
            "OMNICODER_STAGE_ORDER": "text",
            "OMNICODER2026_LOSS_DIAGNOSTICS_INTERVAL": "0",
        },
    },
    {
        "name": "gpipe_batch2_headroom_q4_chunk8192_ffn1024_loss64",
        "default": False,
        "steps": 1,
        "env": {
            "OMNICODER_FAKE_QUANT": "1",
            "OMNICODER_FAKE_QUANT_CHUNK_ROWS": "8192",
            "OMNICODER_FFN_CHUNK_TOKENS": "1024",
            "OMNICODER_MAX_LOSS_TOKENS_PER_SAMPLE": "64",
            "OMNICODER_PLACEMENT_LAYER_COUNTS": "16,16,32",
            "OMNICODER_ACTIVATION_CHECKPOINTING": "1",
            "OMNICODER_PIPELINE_STAGE_SCHEDULE": "gpipe",
            "OMNICODER_PIPELINE_MICROBATCHES": "2",
            "OMNICODER_BATCH_SIZE": "2",
            "OMNICODER_STAGE_ORDER": "text",
            "OMNICODER2026_LOSS_DIAGNOSTICS_INTERVAL": "0",
        },
    },
    {
        "name": "onef1b_batch2_headroom_q4_chunk8192_ffn1024_loss64",
        "default": False,
        "steps": 1,
        "env": {
            "OMNICODER_FAKE_QUANT": "1",
            "OMNICODER_FAKE_QUANT_CHUNK_ROWS": "8192",
            "OMNICODER_FFN_CHUNK_TOKENS": "1024",
            "OMNICODER_MAX_LOSS_TOKENS_PER_SAMPLE": "64",
            "OMNICODER_PLACEMENT_LAYER_COUNTS": "16,16,32",
            "OMNICODER_ACTIVATION_CHECKPOINTING": "1",
            "OMNICODER_PIPELINE_STAGE_SCHEDULE": "1f1b",
            "OMNICODER_PIPELINE_MICROBATCHES": "2",
            "OMNICODER_BATCH_SIZE": "2",
            "OMNICODER_STAGE_ORDER": "text",
            "OMNICODER2026_LOSS_DIAGNOSTICS_INTERVAL": "0",
        },
    },
    {
        "name": "gpipe_batch2_headroom_q4_chunk4096_ffn1024_loss64",
        "default": False,
        "steps": 1,
        "env": {
            "OMNICODER_FAKE_QUANT": "1",
            "OMNICODER_FAKE_QUANT_CHUNK_ROWS": "4096",
            "OMNICODER_FFN_CHUNK_TOKENS": "1024",
            "OMNICODER_MAX_LOSS_TOKENS_PER_SAMPLE": "64",
            "OMNICODER_PLACEMENT_LAYER_COUNTS": "16,16,32",
            "OMNICODER_ACTIVATION_CHECKPOINTING": "1",
            "OMNICODER_PIPELINE_STAGE_SCHEDULE": "gpipe",
            "OMNICODER_PIPELINE_MICROBATCHES": "2",
            "OMNICODER_BATCH_SIZE": "2",
            "OMNICODER_STAGE_ORDER": "text",
            "OMNICODER2026_LOSS_DIAGNOSTICS_INTERVAL": "0",
        },
    },
    {
        "name": "onef1b_batch2_headroom_q4_chunk4096_ffn1024_loss64",
        "default": False,
        "steps": 1,
        "env": {
            "OMNICODER_FAKE_QUANT": "1",
            "OMNICODER_FAKE_QUANT_CHUNK_ROWS": "4096",
            "OMNICODER_FFN_CHUNK_TOKENS": "1024",
            "OMNICODER_MAX_LOSS_TOKENS_PER_SAMPLE": "64",
            "OMNICODER_PLACEMENT_LAYER_COUNTS": "16,16,32",
            "OMNICODER_ACTIVATION_CHECKPOINTING": "1",
            "OMNICODER_PIPELINE_STAGE_SCHEDULE": "1f1b",
            "OMNICODER_PIPELINE_MICROBATCHES": "2",
            "OMNICODER_BATCH_SIZE": "2",
            "OMNICODER_STAGE_ORDER": "text",
            "OMNICODER2026_LOSS_DIAGNOSTICS_INTERVAL": "0",
        },
    },
    {
        "name": "p2p_on_ffn_chunk1024_headroom_q4_chunk8192_loss64",
        "default": False,
        "steps": 1,
        "env": {
            "NCCL_P2P_DISABLE": "0",
            "OMNICODER_FAKE_QUANT": "1",
            "OMNICODER_FAKE_QUANT_CHUNK_ROWS": "8192",
            "OMNICODER_FFN_CHUNK_TOKENS": "1024",
            "OMNICODER_MAX_LOSS_TOKENS_PER_SAMPLE": "64",
            "OMNICODER_PLACEMENT_LAYER_COUNTS": "16,16,32",
            "OMNICODER_ACTIVATION_CHECKPOINTING": "1",
            "OMNICODER_PIPELINE_STAGE_SCHEDULE": "gpipe",
            "OMNICODER_PIPELINE_MICROBATCHES": "1",
            "OMNICODER_BATCH_SIZE": "1",
            "OMNICODER_STAGE_ORDER": "text",
            "OMNICODER2026_LOSS_DIAGNOSTICS_INTERVAL": "0",
        },
    },
    {
        "name": "gdn2_compiled_headroom_q4_chunk8192_ffn1024_loss64",
        "default": False,
        "steps": 1,
        "env": {
            "OMNICODER_FAKE_QUANT": "1",
            "OMNICODER_FAKE_QUANT_CHUNK_ROWS": "8192",
            "OMNICODER_FFN_CHUNK_TOKENS": "1024",
            "OMNICODER_MAX_LOSS_TOKENS_PER_SAMPLE": "64",
            "OMNICODER_PLACEMENT_LAYER_COUNTS": "16,16,32",
            "OMNICODER_ACTIVATION_CHECKPOINTING": "1",
            "OMNICODER_PIPELINE_STAGE_SCHEDULE": "gpipe",
            "OMNICODER_PIPELINE_MICROBATCHES": "1",
            "OMNICODER_BATCH_SIZE": "1",
            "OMNICODER_STAGE_ORDER": "text",
            "OMNICODER2026_GDN2_COMPILED_CHUNKS": "1",
            "OMNICODER2026_GDN2_COMPILED_MODE": "chunked",
            "OMNICODER2026_GDN2_COMPILED_CHUNK_TOKENS": "32",
            "OMNICODER2026_LOSS_DIAGNOSTICS_INTERVAL": "0",
        },
    },
    {
        "name": "gdn2_jit_headroom_q4_chunk8192_ffn1024_loss64",
        "default": False,
        "steps": 1,
        "env": {
            "OMNICODER_FAKE_QUANT": "1",
            "OMNICODER_FAKE_QUANT_CHUNK_ROWS": "8192",
            "OMNICODER_FFN_CHUNK_TOKENS": "1024",
            "OMNICODER_MAX_LOSS_TOKENS_PER_SAMPLE": "64",
            "OMNICODER_PLACEMENT_LAYER_COUNTS": "16,16,32",
            "OMNICODER_ACTIVATION_CHECKPOINTING": "1",
            "OMNICODER_PIPELINE_STAGE_SCHEDULE": "gpipe",
            "OMNICODER_PIPELINE_MICROBATCHES": "1",
            "OMNICODER_BATCH_SIZE": "1",
            "OMNICODER_STAGE_ORDER": "text",
            "OMNICODER2026_GDN2_JIT_SCAN": "1",
            "OMNICODER2026_LOSS_DIAGNOSTICS_INTERVAL": "0",
        },
    },
    {
        "name": "fakequant_chunk2048_loss64_diagnostics",
        "default": False,
        "steps": 2,
        "env": {
            "OMNICODER_FAKE_QUANT": "1",
            "OMNICODER_FAKE_QUANT_CHUNK_ROWS": "2048",
            "OMNICODER_MAX_LOSS_TOKENS_PER_SAMPLE": "64",
            "OMNICODER_ACTIVATION_CHECKPOINTING": "1",
            "OMNICODER_PIPELINE_STAGE_SCHEDULE": "gpipe",
            "OMNICODER_PIPELINE_MICROBATCHES": "1",
            "OMNICODER_BATCH_SIZE": "1",
            "OMNICODER_STAGE_ORDER": "text",
            "OMNICODER2026_LOSS_DIAGNOSTICS_INTERVAL": "1",
        },
    },
    {
        "name": "block_timing_q4_chunk2048_loss64",
        "default": False,
        "steps": 1,
        "env": {
            "OMNICODER_FAKE_QUANT": "1",
            "OMNICODER_FAKE_QUANT_CHUNK_ROWS": "2048",
            "OMNICODER_MAX_LOSS_TOKENS_PER_SAMPLE": "64",
            "OMNICODER_ACTIVATION_CHECKPOINTING": "1",
            "OMNICODER_PIPELINE_STAGE_SCHEDULE": "gpipe",
            "OMNICODER_PIPELINE_MICROBATCHES": "1",
            "OMNICODER_BATCH_SIZE": "1",
            "OMNICODER_STAGE_ORDER": "text",
            "OMNICODER2026_BLOCK_TIMING": "1",
            "OMNICODER2026_BLOCK_TIMING_CUDA_SYNC": "0",
            "OMNICODER2026_LOSS_DIAGNOSTICS_INTERVAL": "0",
        },
    },
    {
        "name": "checkpoint_segment2_q4_chunk2048_loss64",
        "default": False,
        "steps": 1,
        "env": {
            "OMNICODER_FAKE_QUANT": "1",
            "OMNICODER_FAKE_QUANT_CHUNK_ROWS": "2048",
            "OMNICODER_MAX_LOSS_TOKENS_PER_SAMPLE": "64",
            "OMNICODER_ACTIVATION_CHECKPOINTING": "1",
            "OMNICODER2026_ACTIVATION_CHECKPOINT_SEGMENT_SIZE": "2",
            "OMNICODER_PIPELINE_STAGE_SCHEDULE": "gpipe",
            "OMNICODER_PIPELINE_MICROBATCHES": "1",
            "OMNICODER_BATCH_SIZE": "1",
            "OMNICODER_STAGE_ORDER": "text",
            "OMNICODER2026_LOSS_DIAGNOSTICS_INTERVAL": "0",
        },
    },
    {
        "name": "checkpoint_segment2_q4_chunk8192_loss64",
        "default": False,
        "steps": 1,
        "env": {
            "OMNICODER_FAKE_QUANT": "1",
            "OMNICODER_FAKE_QUANT_CHUNK_ROWS": "8192",
            "OMNICODER_MAX_LOSS_TOKENS_PER_SAMPLE": "64",
            "OMNICODER_ACTIVATION_CHECKPOINTING": "1",
            "OMNICODER2026_ACTIVATION_CHECKPOINT_SEGMENT_SIZE": "2",
            "OMNICODER_PIPELINE_STAGE_SCHEDULE": "gpipe",
            "OMNICODER_PIPELINE_MICROBATCHES": "1",
            "OMNICODER_BATCH_SIZE": "1",
            "OMNICODER_STAGE_ORDER": "text",
            "OMNICODER2026_LOSS_DIAGNOSTICS_INTERVAL": "0",
        },
    },
    {
        "name": "checkpoint_segment4_q4_chunk2048_loss64",
        "default": False,
        "steps": 1,
        "env": {
            "OMNICODER_FAKE_QUANT": "1",
            "OMNICODER_FAKE_QUANT_CHUNK_ROWS": "2048",
            "OMNICODER_MAX_LOSS_TOKENS_PER_SAMPLE": "64",
            "OMNICODER_ACTIVATION_CHECKPOINTING": "1",
            "OMNICODER2026_ACTIVATION_CHECKPOINT_SEGMENT_SIZE": "4",
            "OMNICODER_PIPELINE_STAGE_SCHEDULE": "gpipe",
            "OMNICODER_PIPELINE_MICROBATCHES": "1",
            "OMNICODER_BATCH_SIZE": "1",
            "OMNICODER_STAGE_ORDER": "text",
            "OMNICODER2026_LOSS_DIAGNOSTICS_INTERVAL": "0",
        },
    },
    {
        "name": "reasoning_effort2_q4_chunk2048_loss64",
        "default": False,
        "steps": 1,
        "env": {
            "OMNICODER_FAKE_QUANT": "1",
            "OMNICODER_FAKE_QUANT_CHUNK_ROWS": "2048",
            "OMNICODER_MAX_LOSS_TOKENS_PER_SAMPLE": "64",
            "OMNICODER_ACTIVATION_CHECKPOINTING": "1",
            "OMNICODER2026_PIPELINE_REASONING_EFFORT": "2",
            "OMNICODER_PIPELINE_STAGE_SCHEDULE": "gpipe",
            "OMNICODER_PIPELINE_MICROBATCHES": "1",
            "OMNICODER_BATCH_SIZE": "1",
            "OMNICODER_STAGE_ORDER": "text",
            "OMNICODER2026_LOSS_DIAGNOSTICS_INTERVAL": "0",
        },
    },
    {
        "name": "reasoning_efforthigh_q4_chunk2048_loss64",
        "default": False,
        "steps": 1,
        "env": {
            "OMNICODER_FAKE_QUANT": "1",
            "OMNICODER_FAKE_QUANT_CHUNK_ROWS": "2048",
            "OMNICODER_MAX_LOSS_TOKENS_PER_SAMPLE": "64",
            "OMNICODER_ACTIVATION_CHECKPOINTING": "1",
            "OMNICODER2026_PIPELINE_REASONING_EFFORT": "high",
            "OMNICODER_PIPELINE_STAGE_SCHEDULE": "gpipe",
            "OMNICODER_PIPELINE_MICROBATCHES": "1",
            "OMNICODER_BATCH_SIZE": "1",
            "OMNICODER_STAGE_ORDER": "text",
            "OMNICODER2026_LOSS_DIAGNOSTICS_INTERVAL": "0",
        },
    },
    {
        "name": "gpipe_mb2_q4_chunk2048_loss64",
        "default": False,
        "steps": 1,
        "env": {
            "OMNICODER_FAKE_QUANT": "1",
            "OMNICODER_FAKE_QUANT_CHUNK_ROWS": "2048",
            "OMNICODER_MAX_LOSS_TOKENS_PER_SAMPLE": "64",
            "OMNICODER_ACTIVATION_CHECKPOINTING": "1",
            "OMNICODER_PIPELINE_STAGE_SCHEDULE": "gpipe",
            "OMNICODER_PIPELINE_MICROBATCHES": "2",
            "OMNICODER_BATCH_SIZE": "2",
            "OMNICODER_STAGE_ORDER": "text",
            "OMNICODER2026_LOSS_DIAGNOSTICS_INTERVAL": "0",
        },
    },
    {
        "name": "onef1b_mb2_q4_chunk2048_loss64",
        "default": False,
        "steps": 1,
        "env": {
            "OMNICODER_FAKE_QUANT": "1",
            "OMNICODER_FAKE_QUANT_CHUNK_ROWS": "2048",
            "OMNICODER_MAX_LOSS_TOKENS_PER_SAMPLE": "64",
            "OMNICODER_ACTIVATION_CHECKPOINTING": "1",
            "OMNICODER_PIPELINE_STAGE_SCHEDULE": "1f1b",
            "OMNICODER_PIPELINE_MICROBATCHES": "2",
            "OMNICODER_BATCH_SIZE": "2",
            "OMNICODER_STAGE_ORDER": "text",
            "OMNICODER2026_LOSS_DIAGNOSTICS_INTERVAL": "0",
        },
    },
    {
        "name": "actckpt_off_q4_chunk2048_loss64",
        "default": False,
        "steps": 1,
        "env": {
            "OMNICODER_FAKE_QUANT": "1",
            "OMNICODER_FAKE_QUANT_CHUNK_ROWS": "2048",
            "OMNICODER_MAX_LOSS_TOKENS_PER_SAMPLE": "64",
            "OMNICODER_ACTIVATION_CHECKPOINTING": "0",
            "OMNICODER_PIPELINE_STAGE_SCHEDULE": "gpipe",
            "OMNICODER_PIPELINE_MICROBATCHES": "1",
            "OMNICODER_BATCH_SIZE": "1",
            "OMNICODER_STAGE_ORDER": "text",
            "OMNICODER2026_LOSS_DIAGNOSTICS_INTERVAL": "0",
        },
    },
    {
        "name": "actckpt_off_q4_loss64",
        "default": False,
        "steps": 1,
        "env": {
            "OMNICODER_FAKE_QUANT": "1",
            "OMNICODER_FAKE_QUANT_CHUNK_ROWS": "256",
            "OMNICODER_MAX_LOSS_TOKENS_PER_SAMPLE": "64",
            "OMNICODER_ACTIVATION_CHECKPOINTING": "0",
            "OMNICODER_PIPELINE_STAGE_SCHEDULE": "gpipe",
            "OMNICODER_PIPELINE_MICROBATCHES": "1",
            "OMNICODER_BATCH_SIZE": "1",
            "OMNICODER_STAGE_ORDER": "text",
            "OMNICODER2026_LOSS_DIAGNOSTICS_INTERVAL": "0",
        },
    },
    {
        "name": "placement_18_18_28_q4_loss64",
        "steps": 1,
        "env": {
            "OMNICODER_FAKE_QUANT": "1",
            "OMNICODER_FAKE_QUANT_CHUNK_ROWS": "256",
            "OMNICODER_MAX_LOSS_TOKENS_PER_SAMPLE": "64",
            "OMNICODER_PLACEMENT_LAYER_COUNTS": "18,18,28",
            "OMNICODER_ACTIVATION_CHECKPOINTING": "1",
            "OMNICODER_PIPELINE_STAGE_SCHEDULE": "gpipe",
            "OMNICODER_PIPELINE_MICROBATCHES": "1",
            "OMNICODER_BATCH_SIZE": "1",
            "OMNICODER_STAGE_ORDER": "text",
            "OMNICODER2026_LOSS_DIAGNOSTICS_INTERVAL": "0",
        },
    },
    {
        "name": "placement_18_18_28_q4_chunk2048_loss64",
        "default": False,
        "steps": 1,
        "env": {
            "OMNICODER_FAKE_QUANT": "1",
            "OMNICODER_FAKE_QUANT_CHUNK_ROWS": "2048",
            "OMNICODER_MAX_LOSS_TOKENS_PER_SAMPLE": "64",
            "OMNICODER_PLACEMENT_LAYER_COUNTS": "18,18,28",
            "OMNICODER_ACTIVATION_CHECKPOINTING": "1",
            "OMNICODER_PIPELINE_STAGE_SCHEDULE": "gpipe",
            "OMNICODER_PIPELINE_MICROBATCHES": "1",
            "OMNICODER_BATCH_SIZE": "1",
            "OMNICODER_STAGE_ORDER": "text",
            "OMNICODER2026_LOSS_DIAGNOSTICS_INTERVAL": "0",
        },
    },
    {
        "name": "placement_20_20_24_q4_loss64",
        "steps": 1,
        "env": {
            "OMNICODER_FAKE_QUANT": "1",
            "OMNICODER_FAKE_QUANT_CHUNK_ROWS": "256",
            "OMNICODER_MAX_LOSS_TOKENS_PER_SAMPLE": "64",
            "OMNICODER_PLACEMENT_LAYER_COUNTS": "20,20,24",
            "OMNICODER_ACTIVATION_CHECKPOINTING": "1",
            "OMNICODER_PIPELINE_STAGE_SCHEDULE": "gpipe",
            "OMNICODER_PIPELINE_MICROBATCHES": "1",
            "OMNICODER_BATCH_SIZE": "1",
            "OMNICODER_STAGE_ORDER": "text",
            "OMNICODER2026_LOSS_DIAGNOSTICS_INTERVAL": "0",
        },
    },
    {
        "name": "placement_20_20_24_q4_chunk2048_loss64",
        "default": False,
        "steps": 1,
        "env": {
            "OMNICODER_FAKE_QUANT": "1",
            "OMNICODER_FAKE_QUANT_CHUNK_ROWS": "2048",
            "OMNICODER_MAX_LOSS_TOKENS_PER_SAMPLE": "64",
            "OMNICODER_PLACEMENT_LAYER_COUNTS": "20,20,24",
            "OMNICODER_ACTIVATION_CHECKPOINTING": "1",
            "OMNICODER_PIPELINE_STAGE_SCHEDULE": "gpipe",
            "OMNICODER_PIPELINE_MICROBATCHES": "1",
            "OMNICODER_BATCH_SIZE": "1",
            "OMNICODER_STAGE_ORDER": "text",
            "OMNICODER2026_LOSS_DIAGNOSTICS_INTERVAL": "0",
        },
    },
    {
        "name": "placement_21_21_22_q4_loss64",
        "steps": 1,
        "env": {
            "OMNICODER_FAKE_QUANT": "1",
            "OMNICODER_FAKE_QUANT_CHUNK_ROWS": "256",
            "OMNICODER_MAX_LOSS_TOKENS_PER_SAMPLE": "64",
            "OMNICODER_PLACEMENT_LAYER_COUNTS": "21,21,22",
            "OMNICODER_ACTIVATION_CHECKPOINTING": "1",
            "OMNICODER_PIPELINE_STAGE_SCHEDULE": "gpipe",
            "OMNICODER_PIPELINE_MICROBATCHES": "1",
            "OMNICODER_BATCH_SIZE": "1",
            "OMNICODER_STAGE_ORDER": "text",
            "OMNICODER2026_LOSS_DIAGNOSTICS_INTERVAL": "0",
        },
    },
    {
        "name": "placement_21_21_22_q4_chunk2048_loss64",
        "default": False,
        "steps": 1,
        "env": {
            "OMNICODER_FAKE_QUANT": "1",
            "OMNICODER_FAKE_QUANT_CHUNK_ROWS": "2048",
            "OMNICODER_MAX_LOSS_TOKENS_PER_SAMPLE": "64",
            "OMNICODER_PLACEMENT_LAYER_COUNTS": "21,21,22",
            "OMNICODER_ACTIVATION_CHECKPOINTING": "1",
            "OMNICODER_PIPELINE_STAGE_SCHEDULE": "gpipe",
            "OMNICODER_PIPELINE_MICROBATCHES": "1",
            "OMNICODER_BATCH_SIZE": "1",
            "OMNICODER_STAGE_ORDER": "text",
            "OMNICODER2026_LOSS_DIAGNOSTICS_INTERVAL": "0",
        },
    },
    {
        "name": "onef1b_batch2_q4_loss64",
        "steps": 1,
        "env": {
            "OMNICODER_FAKE_QUANT": "1",
            "OMNICODER_FAKE_QUANT_CHUNK_ROWS": "256",
            "OMNICODER_MAX_LOSS_TOKENS_PER_SAMPLE": "64",
            "OMNICODER_ACTIVATION_CHECKPOINTING": "1",
            "OMNICODER_PIPELINE_STAGE_SCHEDULE": "1f1b",
            "OMNICODER_PIPELINE_MICROBATCHES": "2",
            "OMNICODER_BATCH_SIZE": "2",
            "OMNICODER_STAGE_ORDER": "text",
            "OMNICODER2026_LOSS_DIAGNOSTICS_INTERVAL": "0",
        },
    },
    {
        "name": "dataloader_workers2_q4_loss64",
        "steps": 2,
        "env": {
            "OMNICODER_FAKE_QUANT": "1",
            "OMNICODER_FAKE_QUANT_CHUNK_ROWS": "256",
            "OMNICODER_MAX_LOSS_TOKENS_PER_SAMPLE": "64",
            "OMNICODER_ACTIVATION_CHECKPOINTING": "1",
            "OMNICODER_PIPELINE_STAGE_SCHEDULE": "gpipe",
            "OMNICODER_PIPELINE_MICROBATCHES": "1",
            "OMNICODER_BATCH_SIZE": "1",
            "OMNICODER_STAGE_ORDER": "text",
            "OMNICODER2026_DATALOADER_NUM_WORKERS": "2",
            "OMNICODER2026_DATALOADER_PERSISTENT_WORKERS": "1",
            "OMNICODER2026_DATALOADER_PREFETCH_FACTOR": "2",
            "OMNICODER2026_LOSS_DIAGNOSTICS_INTERVAL": "0",
        },
    },
    {
        "name": "fakequant_chunk512_loss64",
        "steps": 1,
        "env": {
            "OMNICODER_FAKE_QUANT": "1",
            "OMNICODER_FAKE_QUANT_CHUNK_ROWS": "512",
            "OMNICODER_MAX_LOSS_TOKENS_PER_SAMPLE": "64",
            "OMNICODER_ACTIVATION_CHECKPOINTING": "1",
            "OMNICODER_PIPELINE_STAGE_SCHEDULE": "gpipe",
            "OMNICODER_PIPELINE_MICROBATCHES": "1",
            "OMNICODER_BATCH_SIZE": "1",
            "OMNICODER_STAGE_ORDER": "text",
            "OMNICODER2026_LOSS_DIAGNOSTICS_INTERVAL": "0",
        },
    },
    {
        "name": "actckpt_off_fakequant_off",
        "steps": 1,
        "env": {
            "OMNICODER_FAKE_QUANT": "0",
            "OMNICODER_PROFILE_ALLOW_FAKE_QUANT_OFF": "1",
            "OMNICODER_ACTIVATION_CHECKPOINTING": "0",
            "OMNICODER_PIPELINE_STAGE_SCHEDULE": "gpipe",
            "OMNICODER_PIPELINE_MICROBATCHES": "1",
            "OMNICODER_BATCH_SIZE": "1",
            "OMNICODER_STAGE_ORDER": "text",
            "OMNICODER2026_LOSS_DIAGNOSTICS_INTERVAL": "0",
        },
    },
    {
        "name": "onef1b_batch2_fakequant_off",
        "steps": 1,
        "env": {
            "OMNICODER_FAKE_QUANT": "0",
            "OMNICODER_PROFILE_ALLOW_FAKE_QUANT_OFF": "1",
            "OMNICODER_ACTIVATION_CHECKPOINTING": "1",
            "OMNICODER_PIPELINE_STAGE_SCHEDULE": "1f1b",
            "OMNICODER_PIPELINE_MICROBATCHES": "2",
            "OMNICODER_BATCH_SIZE": "2",
            "OMNICODER_STAGE_ORDER": "text",
            "OMNICODER2026_LOSS_DIAGNOSTICS_INTERVAL": "0",
        },
    },
    {
        "name": "block_timing_fakequant_off",
        "steps": 1,
        "env": {
            "OMNICODER_FAKE_QUANT": "0",
            "OMNICODER_PROFILE_ALLOW_FAKE_QUANT_OFF": "1",
            "OMNICODER_ACTIVATION_CHECKPOINTING": "1",
            "OMNICODER_PIPELINE_STAGE_SCHEDULE": "gpipe",
            "OMNICODER_PIPELINE_MICROBATCHES": "1",
            "OMNICODER_BATCH_SIZE": "1",
            "OMNICODER_STAGE_ORDER": "text",
            "OMNICODER2026_BLOCK_TIMING": "1",
            "OMNICODER2026_BLOCK_TIMING_CUDA_SYNC": "0",
            "OMNICODER2026_LOSS_DIAGNOSTICS_INTERVAL": "0",
        },
    },
    {
        "name": "block_timing_sync_fakequant_off",
        "steps": 1,
        "env": {
            "OMNICODER_FAKE_QUANT": "0",
            "OMNICODER_PROFILE_ALLOW_FAKE_QUANT_OFF": "1",
            "OMNICODER_ACTIVATION_CHECKPOINTING": "1",
            "OMNICODER_PIPELINE_STAGE_SCHEDULE": "gpipe",
            "OMNICODER_PIPELINE_MICROBATCHES": "1",
            "OMNICODER_BATCH_SIZE": "1",
            "OMNICODER_STAGE_ORDER": "text",
            "OMNICODER2026_BLOCK_TIMING": "1",
            "OMNICODER2026_BLOCK_TIMING_CUDA_SYNC": "1",
            "OMNICODER2026_LOSS_DIAGNOSTICS_INTERVAL": "0",
        },
    },
]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def profile_corpus_paths(matrix_root: Path, matrix_tag: str) -> tuple[Path, str]:
    host_root = matrix_root / "profile_corpus"
    container_root = f"/workspace/weights/training_runs/profile_matrix_{matrix_tag}/profile_corpus"
    return host_root, container_root


def ensure_profile_corpus(matrix_root: Path, matrix_tag: str) -> str:
    host_root, container_root = profile_corpus_paths(matrix_root, matrix_tag)
    train_path = host_root / "train_text_profile.jsonl"
    eval_path = host_root / "eval_text_profile.jsonl"
    rows: list[dict[str, Any]] = []
    prompts = [
        (
            "Explain why bounded profiling should avoid scanning the full dataset lake.",
            "Bounded profiling should use a tiny clean manifest so measured time reflects model compute, optimizer work, and diagnostics instead of corpus discovery or disk I/O.",
        ),
        (
            "Summarize the training target coverage rule.",
            "The loss mask must cover assistant answer tokens and routed media artifact tokens while preserving prompt and media-input tokens as shared context.",
        ),
        (
            "What should the 20B profiling probe record?",
            "It should record per-rank step spans, optimizer diagnostics, optional block timings, target-token coverage, loss, schedule, and whether checkpoints were intentionally skipped.",
        ),
        (
            "State the fake-quant profiling boundary.",
            "Fake-quant-off runs are only no-checkpoint TPS isolation probes; production 20B training still requires the q4 fake-quant training path.",
        ),
    ]
    for index, (prompt, answer) in enumerate(prompts):
        identity = json.dumps({"prompt": prompt, "answer": answer, "index": index}, sort_keys=True)
        rows.append(
            {
                "schema": "omnicoder.profile_matrix_training_row_2026.v1",
                "record_id": f"profile_text_train_{index:02d}",
                "source_id": f"profile_matrix_{matrix_tag}_{index:02d}",
                "source_uri": "profile_matrix://synthetic/compute-isolation",
                "source_date": "2026-05-30",
                "modality": "text",
                "modalities": ["text"],
                "split": "train",
                "quality_score": 0.99,
                "contamination_status": "clean",
                "payload_sha256": hashlib.sha256(identity.encode("utf-8")).hexdigest(),
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ],
            }
        )
    eval_rows = [
        {
            **rows[0],
            "record_id": "profile_text_eval_00",
            "source_id": f"profile_matrix_{matrix_tag}_eval_00",
            "split": "eval",
        }
    ]
    write_jsonl(train_path, rows)
    write_jsonl(eval_path, eval_rows)
    manifest_path = host_root / "curation_manifest.json"
    train_container = f"{container_root}/train_text_profile.jsonl"
    eval_container = f"{container_root}/eval_text_profile.jsonl"
    manifest = {
        "schema": "omnicoder.real_training_curation_manifest_2026.v1",
        "schema_version": "2026-05-23",
        "status": "ok",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "profile_name": f"profile_matrix_{matrix_tag}",
        "train_all_jsonl": train_container,
        "curated_jsonl": train_container,
        "per_modality_jsonl": {"text": train_container},
        "per_modality_split_jsonl": {"text": {"train": train_container, "eval": eval_container}},
        "records": len(rows),
        "eval_records": len(eval_rows),
        "modalities": {"text": len(rows)},
        "split_counts": {"text": {"train": len(rows), "eval": len(eval_rows), "test": 0}},
        "profile_matrix_compute_isolation": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return f"{container_root}/curation_manifest.json"


def run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True, check=False, timeout=timeout)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _summarize_numeric(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = [float(value) for value in values]
    return {
        "count": len(ordered),
        "min_sec": min(ordered),
        "max_sec": max(ordered),
        "mean_sec": sum(ordered) / float(len(ordered)),
    }


def parse_launch_output(stdout: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key in {"container", "out_dir", "host_out_dir"}:
            out[key] = value
    return out


def inspect_container(container: str, repo: Path) -> dict[str, Any]:
    proc = run(
        [
            "docker",
            "inspect",
            "-f",
            "{{.State.Status}} {{.State.Running}} {{.State.ExitCode}} {{.State.OOMKilled}}",
            container,
        ],
        cwd=repo,
    )
    if proc.returncode != 0:
        return {"exists": False, "returncode": proc.returncode, "stderr": proc.stderr.strip()}
    parts = proc.stdout.strip().split()
    return {
        "exists": True,
        "status": parts[0] if len(parts) > 0 else "unknown",
        "running": parts[1].lower() == "true" if len(parts) > 1 else False,
        "exit_code": int(parts[2]) if len(parts) > 2 and parts[2].lstrip("-").isdigit() else None,
        "oom_killed": parts[3].lower() == "true" if len(parts) > 3 else None,
    }


def wait_container(container: str, repo: Path, timeout_seconds: int, poll_seconds: int) -> dict[str, Any]:
    deadline = time.time() + float(timeout_seconds)
    last: dict[str, Any] = {}
    while time.time() < deadline:
        last = inspect_container(container, repo)
        if last.get("exists") and not last.get("running"):
            return last
        time.sleep(max(1, int(poll_seconds)))
    last = inspect_container(container, repo)
    last["timed_out"] = True
    return last


def summarize_run(host_out_dir: Path, container: str, repo: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {"host_out_dir": str(host_out_dir), "container": container}
    log_path = host_out_dir / "train.log"
    if not log_path.exists():
        loss_logs = sorted((host_out_dir / "logs").glob("*_loss.jsonl"))
        if loss_logs:
            log_path = loss_logs[-1]
    log_rows = load_jsonl(log_path)
    step_rows = [row for row in log_rows if "step" in row and "loss" in row]
    summary["train_log"] = str(log_path)
    summary["steps_logged"] = len(step_rows)
    if step_rows:
        summary["first_loss"] = step_rows[0].get("loss")
        summary["last_loss"] = step_rows[-1].get("loss")
        summary["last_global_step"] = step_rows[-1].get("step")
        summary["last_optimized_target_tokens"] = step_rows[-1].get("optimized_target_tokens")
        summary["last_valid_target_tokens"] = step_rows[-1].get("valid_target_tokens")
        valid_tokens = _float_or_none(step_rows[-1].get("valid_target_tokens"))
        optimized_tokens = _float_or_none(step_rows[-1].get("optimized_target_tokens"))
        if valid_tokens and valid_tokens > 0.0 and optimized_tokens is not None:
            summary["last_target_token_coverage"] = optimized_tokens / valid_tokens
        else:
            summary["last_target_token_coverage"] = None
        summary["last_seq_len"] = step_rows[-1].get("seq_len")
        summary["last_batch_size"] = step_rows[-1].get("batch_size")
        summary["max_loss_tokens_per_sample"] = step_rows[-1].get("max_loss_tokens_per_sample")
        summary["loss_diagnostics_collected"] = step_rows[-1].get("loss_diagnostics_collected")
        summary["pipeline_schedule"] = step_rows[-1].get("pipeline_schedule")
        summary["pipeline_microbatches"] = step_rows[-1].get("pipeline_microbatches")
        summary["microbatch_size"] = step_rows[-1].get("microbatch_size")
    diag_dir = host_out_dir / "diagnostics"
    timing_files = sorted(diag_dir.glob("*_step_timing.rank*.jsonl"))
    timings: list[dict[str, Any]] = []
    phase_values: dict[str, list[float]] = {}
    lm_loss_phase_values: dict[str, list[float]] = {}
    optimizer_values: dict[str, list[float]] = {}
    rank_skews: list[float] = []
    for path in timing_files:
        rank_match = re.search(r"\.rank(\d+)\.", path.name)
        rank = int(rank_match.group(1)) if rank_match else None
        rows = [row for row in load_jsonl(path) if row.get("event") == "pipeline_step_timing"]
        if not rows:
            continue
        for row in rows:
            total = _float_or_none(row.get("total_sec"))
            if total is not None:
                phase_values.setdefault("total_sec", []).append(total)
            skew = _float_or_none(row.get("rank_skew_sec"))
            if skew is not None:
                rank_skews.append(skew)
            spans = row.get("spans") if isinstance(row.get("spans"), dict) else {}
            for key, value in spans.items():
                numeric = _float_or_none(value)
                if numeric is not None:
                    phase_values.setdefault(str(key), []).append(numeric)
            lm_loss_timing = row.get("lm_loss_timing") if isinstance(row.get("lm_loss_timing"), dict) else {}
            lm_total = _float_or_none(lm_loss_timing.get("total_sec"))
            if lm_total is not None:
                lm_loss_phase_values.setdefault("total_sec", []).append(lm_total)
            lm_spans = lm_loss_timing.get("spans") if isinstance(lm_loss_timing.get("spans"), dict) else {}
            for key, value in lm_spans.items():
                numeric = _float_or_none(value)
                if numeric is not None:
                    lm_loss_phase_values.setdefault(str(key), []).append(numeric)
            optimizer_diagnostics = row.get("optimizer_diagnostics") if isinstance(row.get("optimizer_diagnostics"), dict) else {}
            for key, value in optimizer_diagnostics.items():
                numeric = _float_or_none(value)
                if numeric is not None:
                    optimizer_values.setdefault(str(key), []).append(numeric)
        last = rows[-1]
        spans = last.get("spans") if isinstance(last.get("spans"), dict) else {}
        opt = last.get("optimizer_diagnostics") if isinstance(last.get("optimizer_diagnostics"), dict) else {}
        lm_loss_timing = last.get("lm_loss_timing") if isinstance(last.get("lm_loss_timing"), dict) else {}
        timings.append(
            {
                "rank": rank,
                "path": str(path),
                "rows": len(rows),
                "total_sec": last.get("total_sec"),
                "schedule_step_sec": spans.get("schedule_step_sec"),
                "optimizer_step_sec": spans.get("optimizer_step_sec"),
                "broadcast_inputs_sec": spans.get("broadcast_inputs_sec"),
                "telemetry_sec": spans.get("telemetry_sec"),
                "rank_skew_sec": last.get("rank_skew_sec"),
                "phase_spans": spans,
                "optimizer_diagnostics": opt,
                "lm_loss_timing": lm_loss_timing,
            }
        )
    summary["rank_timing"] = timings
    summary["phase_timing_files"] = [str(path) for path in timing_files]
    summary["phase_timing_summary"] = {key: _summarize_numeric(values) for key, values in sorted(phase_values.items())}
    summary["lm_loss_timing_summary"] = {key: _summarize_numeric(values) for key, values in sorted(lm_loss_phase_values.items())}
    summary["optimizer_diagnostics_summary"] = {key: _summarize_numeric(values) for key, values in sorted(optimizer_values.items())}
    if rank_skews:
        summary["rank_skew_summary"] = _summarize_numeric(rank_skews)
    if timings:
        totals = [float(item["total_sec"]) for item in timings if isinstance(item.get("total_sec"), (int, float))]
        schedules = [float(item["schedule_step_sec"]) for item in timings if isinstance(item.get("schedule_step_sec"), (int, float))]
        if totals:
            summary["max_total_sec"] = max(totals)
            summary["min_total_sec"] = min(totals)
            if min(totals) > 0.0:
                summary["total_step_skew_ratio"] = max(totals) / min(totals)
            if isinstance(summary.get("last_seq_len"), (int, float)):
                summary["sequence_tokens_per_sec"] = float(summary["last_seq_len"]) / max(totals)
                batch_size = _float_or_none(summary.get("last_batch_size")) or 1.0
                summary["training_tokens_per_sec"] = (float(summary["last_seq_len"]) * max(1.0, batch_size)) / max(totals)
            if isinstance(summary.get("last_optimized_target_tokens"), (int, float)):
                summary["optimized_target_tokens_per_sec"] = float(summary["last_optimized_target_tokens"]) / max(totals)
        if schedules:
            summary["max_schedule_step_sec"] = max(schedules)
            summary["min_schedule_step_sec"] = min(schedules)
            if min(schedules) > 0.0:
                summary["schedule_step_skew_ratio"] = max(schedules) / min(schedules)
    block_files = sorted(diag_dir.glob("*_block_timing.rank*.jsonl"))
    summary["block_timing_files"] = [str(path) for path in block_files]
    if block_files:
        block_summary: list[dict[str, Any]] = []
        for path in block_files:
            rows = load_jsonl(path)
            span_totals: dict[str, float] = {}
            span_counts: dict[str, int] = {}
            for row in rows:
                for record in row.get("records", []) if isinstance(row.get("records"), list) else []:
                    for span in record.get("spans", []) if isinstance(record.get("spans"), list) else []:
                        name = str(span.get("name") or "unknown")
                        sec = float(span.get("sec") or 0.0)
                        span_totals[name] = span_totals.get(name, 0.0) + sec
                        span_counts[name] = span_counts.get(name, 0) + 1
            block_summary.append({"path": str(path), "span_totals": span_totals, "span_counts": span_counts})
        summary["block_timing_summary"] = block_summary
    checkpoint_files = list(host_out_dir.rglob("*.complete.json")) if host_out_dir.exists() else []
    summary["checkpoint_complete_files"] = [str(path) for path in checkpoint_files]
    summary["no_checkpoint_written"] = not checkpoint_files
    logs_proc = run(["docker", "logs", "--tail", "80", container], cwd=repo, timeout=60)
    summary["container_log_tail"] = logs_proc.stdout[-12000:] if logs_proc.stdout else logs_proc.stderr[-12000:]
    return summary


def launch_variant(repo: Path, matrix_root: Path, matrix_tag: str, variant: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    name = str(variant["name"])
    run_tag = f"{matrix_tag}_{name}"
    out_dir = f"weights/training_runs/profile_matrix_{matrix_tag}/{name}"
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in variant.get("env", {}).items()})
    env.update(
        {
            "OMNICODER_REPO": str(repo),
            "OMNICODER_RUN_TAG": run_tag,
            "OMNICODER_CONTAINER_NAME": f"omnicoder_profile_{run_tag}",
            "OMNICODER_OUT_DIR": out_dir,
            "OMNICODER_MODE": args.mode,
            "OMNICODER_STEPS_PER_STAGE": str(int(variant.get("steps", args.steps))),
            "OMNICODER_SEQ_LEN": str(int(variant.get("seq_len", args.seq_len))),
            "OMNICODER_SAVE_INTERVAL": "0",
            "OMNICODER2026_SKIP_FINAL_SAVE": "1",
            "OMNICODER_DETACH": "1",
            "OMNICODER2026_STEP_TIMING_INTERVAL": "1",
            "OMNICODER2026_TELEMETRY_INTERVAL": "1",
            "OMNICODER2026_RANK_SKEW_INTERVAL": "1",
            "OMNICODER2026_CHECKPOINT_DATA_HASH_POLICY": "never",
        }
    )
    if args.profile:
        env["OMNICODER_PROFILE"] = args.profile
    if args.curation_manifest:
        env["OMNICODER_CURATION_MANIFEST"] = args.curation_manifest
    elif args.profile_corpus:
        env["OMNICODER_CURATION_MANIFEST"] = ensure_profile_corpus(matrix_root, matrix_tag)
        env["OMNICODER_EXTERNAL_CURATION_PREFLIGHT_MAX_RECORDS"] = str(min(16, int(env.get("OMNICODER_EXTERNAL_CURATION_PREFLIGHT_MAX_RECORDS", "16") or 16)))
        env["OMNICODER_DENSE_LAUNCH_PREFLIGHT_MAX_RECORDS"] = str(min(16, int(env.get("OMNICODER_DENSE_LAUNCH_PREFLIGHT_MAX_RECORDS", "16") or 16)))
    launcher = repo / "scripts" / "ai_server_fast_pipeline_20b.sh"
    launch = run(["bash", str(launcher)], cwd=repo, env=env, timeout=120)
    result: dict[str, Any] = {
        "variant": name,
        "requested_env": {key: env[key] for key in sorted(env) if key.startswith("OMNICODER") or key.startswith("NCCL_")},
        "launch_returncode": launch.returncode,
        "launch_stdout": launch.stdout,
        "launch_stderr": launch.stderr,
    }
    parsed = parse_launch_output(launch.stdout)
    result.update(parsed)
    if launch.returncode != 0:
        result["status"] = "launch_failed"
        return result
    container = parsed.get("container") or env["OMNICODER_CONTAINER_NAME"]
    state = wait_container(container, repo, int(args.timeout_seconds), int(args.poll_seconds))
    result["container_state"] = state
    if args.cleanup_containers and state.get("timed_out") and state.get("running"):
        stop_proc = run(["docker", "stop", container], cwd=repo, timeout=120)
        result["timeout_stop"] = {
            "returncode": stop_proc.returncode,
            "stdout": stop_proc.stdout.strip(),
            "stderr": stop_proc.stderr.strip(),
        }
        state = inspect_container(container, repo)
        state["timed_out"] = True
        result["container_state_after_timeout_stop"] = state
    host_out_dir = Path(parsed.get("host_out_dir") or (matrix_root / name))
    result.update(summarize_run(host_out_dir, container, repo))
    result["no_checkpoint_requested"] = (
        env.get("OMNICODER_SAVE_INTERVAL") == "0" and env.get("OMNICODER2026_SKIP_FINAL_SAVE") == "1"
    )
    checkpoint_violation = bool(result["no_checkpoint_requested"] and result.get("checkpoint_complete_files"))
    if checkpoint_violation:
        result["no_checkpoint_violation"] = True
        result["failure_reason"] = "checkpoint_written_in_no_checkpoint_profile"
    if state.get("timed_out"):
        result["status"] = "timed_out"
    elif checkpoint_violation:
        result["status"] = "failed"
    elif state.get("exit_code") == 0:
        result["status"] = "passed"
    else:
        result["status"] = "failed"
    if args.cleanup_containers and not state.get("running"):
        run(["docker", "rm", container], cwd=repo, timeout=60)
    return result


def select_variants(wanted: set[str]) -> tuple[list[dict[str, Any]], list[str]]:
    if wanted:
        variants = [variant for variant in DEFAULT_VARIANTS if str(variant["name"]) in wanted]
        names = {str(variant["name"]) for variant in variants}
        return variants, sorted(wanted - names)
    return [variant for variant in DEFAULT_VARIANTS if bool(variant.get("default", True))], []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.getenv("OMNICODER_REPO", "/home/cereal/omnicoder_2026_work"))
    parser.add_argument("--profile", default=os.getenv("OMNICODER_PROFILE", "profiles/training_orchestration_2026.json"))
    parser.add_argument("--curation-manifest", default=os.getenv("OMNICODER_CURATION_MANIFEST", ""))
    parser.add_argument("--matrix-tag", default=os.getenv("OMNICODER_PROFILE_MATRIX_TAG", ""))
    parser.add_argument("--variants", default="", help="Comma-separated subset of variant names")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--mode", default="run-full")
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--profile-corpus", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cleanup-containers", action="store_true", default=os.getenv("OMNICODER_PROFILE_MATRIX_CLEANUP", "0").lower() in TRUTHY)
    args = parser.parse_args(argv)

    repo = Path(args.repo).resolve()
    if not repo.exists():
        print(f"repo does not exist: {repo}", file=sys.stderr)
        return 2
    tag = args.matrix_tag.strip() or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    matrix_root = Path(os.getenv("OMNICODER_WEIGHTS_ROOT", "/home/cereal/omnicoder_2026_work/weights")) / "training_runs" / f"profile_matrix_{tag}"
    matrix_root.mkdir(parents=True, exist_ok=True)
    wanted = {item.strip() for item in args.variants.split(",") if item.strip()}
    variants, missing = select_variants(wanted)
    if missing:
        print(f"unknown variants: {', '.join(missing)}", file=sys.stderr)
        return 2
    summary: dict[str, Any] = {
        "schema": "omnicoder.profile_matrix_20b_2026.v1",
        "matrix_tag": tag,
        "repo": str(repo),
        "matrix_root": str(matrix_root),
        "started_at": time.time(),
        "variants": [],
    }
    summary_path = matrix_root / "profile_matrix_summary.json"
    for variant in variants:
        result = launch_variant(repo, matrix_root, tag, variant, args)
        summary["variants"].append(result)
        summary["updated_at"] = time.time()
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({"variant": variant["name"], "status": result.get("status"), "summary": str(summary_path)}), flush=True)
        if result.get("status") == "timed_out":
            break
    failed = [item for item in summary["variants"] if item.get("status") not in {"passed"}]
    summary["finished_at"] = time.time()
    summary["status"] = "passed" if not failed else "failed"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(str(summary_path))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
