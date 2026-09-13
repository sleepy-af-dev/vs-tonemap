"""The release notes come out of CHANGELOG.md, so the extraction is checked here.

A release runs this extraction once, at tag time, when getting it wrong costs a
version number. These cases run on every push instead.
"""

from pathlib import Path

import pytest
from release_notes import extract

CHANGELOG = """# Changelog

## [Unreleased]

## [0.2.0] - 2026-10-01

### Added

- A second thing.

## [0.1.0] - 2026-09-13

### Added

- A first thing.

[Unreleased]: https://example.com/compare/v0.2.0...HEAD
[0.1.0]: https://example.com/releases/tag/v0.1.0
"""


def test_it_stops_at_the_next_heading():
    assert extract(CHANGELOG, "0.2.0") == "### Added\n\n- A second thing."


def test_it_stops_at_the_link_definitions():
    assert extract(CHANGELOG, "0.1.0") == "### Added\n\n- A first thing."


def test_a_missing_version_is_an_error():
    with pytest.raises(LookupError, match="no section for 9.9.9"):
        extract(CHANGELOG, "9.9.9")


def test_an_empty_section_is_an_error():
    with pytest.raises(LookupError, match="is empty"):
        extract(CHANGELOG, "Unreleased")


def test_the_real_changelog_has_notes_for_its_latest_release():
    """Guards the format the extractor depends on, in the file it reads."""
    text = (
        Path(__file__).parents[2].joinpath("CHANGELOG.md").read_text(encoding="utf-8")
    )
    assert extract(text, "0.1.0").startswith("### Added")
