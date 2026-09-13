"""Extract a release's notes from CHANGELOG.md.

The notes are written when the change is made and reviewed in the pull request
that makes it, rather than assembled at tag time from a list of pull request
titles. This lives in a script so the test beside it covers the extraction,
instead of the release it is meant to produce being the first thing to run it.
"""

import re
import sys
from pathlib import Path


def extract(changelog: str, version: str) -> str:
    """Return the notes under a version's heading, without the heading itself.

    Reads from the version's own `## [x.y.z]` line to whichever comes first: the
    next `## ` heading, the link definitions at the foot of the file, or the end
    of the file.

    Raises:
        LookupError: if the file has no section for that version, or has one
            with nothing under it.
    """
    match = re.search(
        rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=\n## |\n\[|\Z)",
        changelog,
        re.DOTALL | re.MULTILINE,
    )
    if match is None:
        raise LookupError(f"CHANGELOG.md has no section for {version}")
    notes = match.group(1).strip()
    if not notes:
        raise LookupError(f"the CHANGELOG.md section for {version} is empty")
    return notes


def main(tag: str, destination: str) -> int:
    try:
        notes = extract(
            Path("CHANGELOG.md").read_text(encoding="utf-8"), tag.removeprefix("v")
        )
    except LookupError as error:
        print(f"::error::{error}")
        return 1
    Path(destination).write_text(notes + "\n", encoding="utf-8")
    print(notes)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: release_notes.py <tag> <output path>")
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
