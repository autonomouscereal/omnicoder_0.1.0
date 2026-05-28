from __future__ import annotations

from omnicoder.eval import audio_eval


def test_fad_clap_and_mos_fail_closed_until_official_scorers_are_wired(
    capsys,
) -> None:
    assert audio_eval._compute_fad("generated", "reference") == -1.0
    assert audio_eval._compute_clap("pairs.jsonl") == -1.0
    assert audio_eval._compute_mos("pairs.jsonl") == -1.0

    out = capsys.readouterr().out
    assert "unavailable_for_official_scoring" in out
    assert "fad" in out
    assert "CLAPScore" in out
    assert "MOS" in out
