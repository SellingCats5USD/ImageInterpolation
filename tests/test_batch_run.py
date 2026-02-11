from pathlib import Path

from scripts.batch_run import IDEA_PAIRS, parse_prompt_file


def test_parse_prompt_file_accepts_valid_lines(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.txt"
    prompts.write_text(
        "# comment\n"
        "skyline at dawn ||| portrait of a pilot\n"
        "forest temple ||| portrait of a druid\n"
    )

    pairs = parse_prompt_file(prompts)

    assert pairs == [
        ("skyline at dawn", "portrait of a pilot"),
        ("forest temple", "portrait of a druid"),
    ]


def test_parse_prompt_file_rejects_missing_separator(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("bad line\n")

    try:
        parse_prompt_file(prompts)
    except ValueError as exc:
        assert "missing '|||" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_idea_pairs_match_requested_style_mix() -> None:
    assert len(IDEA_PAIRS) == 50
    non_oil = [pair for pair in IDEA_PAIRS if "oil painting" not in pair[0].lower() and "oil painting" not in pair[1].lower()]
    assert len(non_oil) <= 10
