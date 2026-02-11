from pathlib import Path

from scripts.batch_run import parse_prompt_file


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
