## Summary

<!-- What does this change, and why? -->

## Checklist

- [ ] Tests pass (`uv run --project reference pytest reference .github/scripts -q`)
- [ ] Lint and types pass (`uv run --project reference ruff format --check reference bench .github/scripts`, `ruff check` the same paths, `basedpyright`)
- [ ] Added anything user-visible to the `Unreleased` section of `CHANGELOG.md`, since the release notes are read from there
- [ ] Updated the README if behaviour, arguments or accuracy changed
- [ ] Checked NOTICE if a dependency or its licence moved
