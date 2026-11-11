import pytest

from kibble_engine.parsing.identifiers import split_identifier, symbol_terms


@pytest.mark.parametrize(
    ("name", "parts"),
    [
        ("parseConfigV2", ["parse", "config", "v2"]),
        ("HTTPServerError", ["http", "server", "error"]),
        ("snake_case_name", ["snake", "case", "name"]),
        ("user-id", ["user", "id"]),
        ("XMLParser", ["xml", "parser"]),
        ("getHTTPResponse", ["get", "http", "response"]),
        ("utf8Decode", ["utf8", "decode"]),
        ("a", ["a"]),
        ("", []),
    ],
)
def test_split_identifier(name: str, parts: list[str]) -> None:
    assert split_identifier(name) == parts


def test_symbol_terms_keeps_original_and_parts_without_duplicates() -> None:
    assert symbol_terms("validateToken", None, "validateToken") == "validateToken validate token"
    assert symbol_terms("SessionManager", "get") == "SessionManager session manager get"
    assert symbol_terms() == ""
