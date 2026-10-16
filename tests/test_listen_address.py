"""
HOST / PORT resolution shared by start.sh, start.bat and the Docker image.

Run with pytest:  pytest tests/test_listen_address.py
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

import app  # noqa: E402


def test_defaults(monkeypatch):
    monkeypatch.delenv("HOST", raising=False)
    monkeypatch.delenv("PORT", raising=False)
    assert app.listen_address() == ("0.0.0.0", 8000)


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("HOST", "127.0.0.1")
    monkeypatch.setenv("PORT", " 9000 ")
    assert app.listen_address() == ("127.0.0.1", 9000)


def test_blank_values_fall_back_to_defaults(monkeypatch):
    monkeypatch.setenv("HOST", "")
    monkeypatch.setenv("PORT", "")
    assert app.listen_address() == ("0.0.0.0", 8000)


@pytest.mark.parametrize("bad", ["abc", "0", "65536", "-1", "80.5"])
def test_invalid_port_exits_with_message(monkeypatch, bad):
    monkeypatch.setenv("PORT", bad)
    with pytest.raises(SystemExit) as exc:
        app.listen_address()
    assert "Invalid PORT" in str(exc.value)
