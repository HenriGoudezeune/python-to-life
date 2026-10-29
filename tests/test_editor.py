"""The editor's server API (without the HTTP layer)."""
import pytest

from gol import editor
from gol.compiler import CompileError

SRC = "def xor2(a: bool, b: bool) -> bool:\n    return a ^ b\n"


def test_compile_and_run():
    c = editor.api_compile({"src": SRC, "inputs": {"a": "1", "b": "0"}})
    assert c["func"] == "xor2" and c["values"] == {"a": 1, "b": 0}
    assert c["rle"].count("!") >= 1 and c["t_end"] > 0
    r = editor.api_run({"src": SRC, "inputs": {"a": 1, "b": 0}})
    assert r["ok"] and r["life"] == r["python"] == 1


def test_errors():
    with pytest.raises(ValueError):
        editor.api_compile({"src": SRC, "inputs": {"a": 2}})
    with pytest.raises(CompileError):
        editor.api_compile({"src": "def f(a: u2) -> u2:\n    return [a]\n"})


def test_examples_listed():
    assert "add4" in editor.api_examples()
