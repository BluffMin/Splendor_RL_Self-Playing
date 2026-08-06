import pytest

from splendor_rl.progress import ProgressConfig, ProgressMode, progress_enabled


class Stream:
    def __init__(self, tty):
        self.tty = tty

    def isatty(self):
        return self.tty


def test_progress_modes(monkeypatch):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    assert not progress_enabled(ProgressConfig(ProgressMode.NEVER), Stream(True))
    assert progress_enabled(ProgressConfig(ProgressMode.ALWAYS), Stream(False))
    assert not progress_enabled(ProgressConfig(ProgressMode.AUTO), Stream(False))
    assert progress_enabled(ProgressConfig(ProgressMode.AUTO), Stream(True))


def test_refresh_must_be_positive():
    with pytest.raises(ValueError, match="must be positive"):
        ProgressConfig(refresh_seconds=0)
