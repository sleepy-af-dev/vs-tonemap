import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402
import vapoursynth as vs  # noqa: E402
from fixtures import build  # noqa: E402
from vsharness import PLUGIN  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def plugin():
    """The compiled filter, loaded once. Skips the suite when it is not built."""
    if not PLUGIN.exists():
        pytest.skip(f"{PLUGIN.name} is not built")
    if not hasattr(vs.core, "tonemapper"):
        vs.core.std.LoadPlugin(path=str(PLUGIN))
    return vs.core.tonemapper


@pytest.fixture(scope="session")
def fixtures():
    """The oracle's synthetic inputs and expected outputs, built once."""
    return build()
