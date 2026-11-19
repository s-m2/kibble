from __future__ import annotations

from pathlib import Path

from kibble_engine.core.session import environment_for
from kibble_engine.dotenv import load_dotenv_file, parse_dotenv


def test_parse_handles_comments_quotes_and_export() -> None:
    text = (
        "# a comment\n"
        "OPENROUTER_API_KEY=sk-or-abc   # trailing comment\n"
        'export QUOTED="value with # hash"\n'
        "SINGLE='x y'\n"
        "EMPTY=\n"
        "not a pair\n"
        "1BAD=ignored\n"
        "SPACED = padded value \n"
    )
    assert parse_dotenv(text) == {
        "OPENROUTER_API_KEY": "sk-or-abc",
        "QUOTED": "value with # hash",
        "SINGLE": "x y",
        "EMPTY": "",
        "SPACED": "padded value",
    }


def test_missing_file_is_empty(tmp_path: Path) -> None:
    assert load_dotenv_file(tmp_path / ".env") == {}


def test_real_environment_wins_over_dotenv(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("A=from-file\nB=from-file\n", encoding="utf-8")
    env = environment_for(tmp_path, {"B": "from-environment", "C": "only-environment"})
    assert env == {"A": "from-file", "B": "from-environment", "C": "only-environment"}
