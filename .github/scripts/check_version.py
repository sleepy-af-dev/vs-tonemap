"""Check that the built plugin reports the version its tag claims.

The check loads the DLL the release is about to upload rather than reading the
version out of the source, so a stale or mismatched build cannot pass.

It reads Info(), not the plugin version VapourSynth exposes: that one is packed
into a single int as (major << 16) | minor and carries no patch, so it cannot
tell v0.1.0 from v0.1.1. Info() reports all three components.
"""

import sys
from pathlib import Path

import vapoursynth as vs


def check(tag: str, dll: str) -> int:
    core = vs.core
    core.std.LoadPlugin(str(Path(dll).resolve()))
    built = core.tonemap.Info()["version"]
    claimed = tag.removeprefix("v")
    if built != claimed:
        print(f"::error::the plugin reports {built}, tag {tag} claims {claimed}")
        return 1
    print(f"the plugin reports {built}, which matches tag {tag}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: check_version.py <tag> <path to vs-tonemap.dll>")
    raise SystemExit(check(sys.argv[1], sys.argv[2]))
