import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from fixtures import build
from vsharness import PLUGIN, core


@pytest.fixture(scope="session", autouse=True)
def plugin():
    """The compiled filter, loaded once. Skips the suite when it is not built."""
    if not PLUGIN.exists():
        pytest.skip(f"{PLUGIN.name} is not built")
    if not hasattr(core, "tonemap"):
        core.std.LoadPlugin(path=str(PLUGIN))
    return core.tonemap


@pytest.fixture(scope="session")
def fixtures():
    """The oracle's synthetic inputs and expected outputs, built once."""
    return build()
