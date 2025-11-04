import os
from pathlib import Path

from omnicoder.eval.reward_metrics import clip_score, fid


def test_clip_score_handles_missing_deps(tmp_path: Path):
    # Should return None if CLIP or PIL missing in this environment
    p = tmp_path / 'pairs.jsonl'
    p.write_text('')
    res = clip_score(str(p))
    assert res is None or isinstance(res, float)


def test_fid_handles_empty(tmp_path: Path):
    pred = tmp_path / 'pred'
    ref = tmp_path / 'ref'
    pred.mkdir(); ref.mkdir()
    assert fid(str(pred), str(ref)) in (None,)


