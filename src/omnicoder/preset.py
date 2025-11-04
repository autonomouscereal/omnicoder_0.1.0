import os as _os
from omnicoder.utils.logger import get_logger as _get_logger

# Re-export builder utilities from inference.generate to satisfy imports and keep logs stable
try:
    from omnicoder.inference.generate import (
        build_mobile_model as _build_mobile_model,
        build_mobile_model_by_name as _build_by_name,
    )
except Exception as _e:  # pragma: no cover
    _build_mobile_model = None  # type: ignore
    _build_by_name = None  # type: ignore


def build_mobile_model(preset, rope_scale=None, rope_base=None, multi_token=2, mem_slots=0, skip_init=False):
    log = _get_logger("omnicoder.preset")
    try:
        log.info("build_mobile_model enter")
    except Exception:
        pass
    if _build_mobile_model is None:
        raise RuntimeError("omnicoder.preset: build_mobile_model unavailable")
    return _build_mobile_model(preset, rope_scale=rope_scale, rope_base=rope_base, multi_token=multi_token, mem_slots=mem_slots, skip_init=skip_init)


def build_by_name(preset_name: str, rope_scale=None, rope_base=None, multi_token=2, mem_slots=0, skip_init=False):
    log = _get_logger("omnicoder.preset")
    try:
        log.info(
            "build_by_name enter name=%s rope_scale=%s rope_base=%s multi_token=%s mem_slots=%s",
            preset_name, str(rope_scale), str(rope_base), str(multi_token), str(mem_slots)
        )
    except Exception:
        pass
    if _build_by_name is None:
        raise RuntimeError("omnicoder.preset: build_by_name unavailable")
    model = _build_by_name(preset_name, rope_scale=rope_scale, rope_base=rope_base, multi_token=multi_token, mem_slots=mem_slots, skip_init=skip_init)
    try:
        log.info("build_by_name exit name=%s", preset_name)
    except Exception:
        pass
    return model


