import pytest

from omnicoder.modeling.multimodal.vocab_map import VocabLayout, map_image_tokens, map_video_tokens, map_audio_tokens


def test_vocab_layout_non_overlap():
    layout = VocabLayout()
    layout.validate()
    # Ensure ranges do not overlap
    text_end = layout.text_size - 1
    image_end = layout.image_start + layout.image_size - 1
    video_end = layout.video_start + layout.video_size - 1
    audio_end = layout.audio_start + layout.audio_size - 1
    assert layout.image_start > text_end
    assert layout.video_start > image_end
    assert layout.audio_start > video_end
    assert audio_end > layout.audio_start


def test_mapping_offsets():
    layout = VocabLayout()
    img = map_image_tokens([0, 1, layout.image_size - 1], layout)
    vid = map_video_tokens([0, 1, layout.video_size - 1], layout)
    aud = map_audio_tokens([0, 1, layout.audio_size - 1], layout)
    assert img[0] == layout.image_start
    assert vid[0] == layout.video_start
    assert aud[0] == layout.audio_start

from omnicoder.config import MultiModalConfig


def test_vocab_ranges_do_not_overlap():
    mmc = MultiModalConfig()
    img_start = mmc.image_vocab_start
    vid_start = mmc.video_vocab_start
    aud_start = mmc.audio_vocab_start
    assert img_start < vid_start < aud_start
    assert (img_start + mmc.image_codebook_size) <= vid_start
    assert (vid_start + mmc.video_codebook_size) <= aud_start


def test_mapping_functions_offsets():
    mmc = MultiModalConfig()
    img = map_image_tokens([0, 1, mmc.image_codebook_size - 1], mmc)
    vid = map_video_tokens([0, 1, mmc.video_codebook_size - 1], mmc)
    aud = map_audio_tokens([0, 1, mmc.audio_codebook_size - 1], mmc)
    assert img[0] == mmc.image_vocab_start
    assert img[-1] == mmc.image_vocab_start + mmc.image_codebook_size - 1
    assert vid[0] == mmc.video_vocab_start
    assert vid[-1] == mmc.video_vocab_start + mmc.video_codebook_size - 1
    assert aud[0] == mmc.audio_vocab_start
    assert aud[-1] == mmc.audio_vocab_start + mmc.audio_codebook_size - 1


def test_concatenation_order_is_monotonic():
    mmc = MultiModalConfig()
    seq = []
    seq += map_image_tokens([0, 1, 2], mmc)
    seq += map_video_tokens([0, 1, 2], mmc)
    seq += map_audio_tokens([0, 1, 2], mmc)
    assert seq[0] < seq[3] < seq[6]


