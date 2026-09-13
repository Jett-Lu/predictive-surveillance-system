# Review fixes implementation plan

**Goal:** Apply the actionable review findings, preserve per-person isolation, and verify the resulting code with regression tests and an independent review.

**Architecture:** Keep the existing Python modules and public defaults where possible. Use source time for recordings, explicit wave events, and validated activity cache metadata. Avoid model downloads during tests.

**Scope:** User requested all remaining fixes and a fresh code-quality review. Earlier identity-expiry and prediction-expiry fixes are already present.

## Tasks

- [ ] Video timing: update `camera.py` and `detection.py`; test recorded timestamps and live-clock behavior using controlled captures. Ensure capture resources close on failure.
- [ ] Export names: update `media_export.py` to distinguish source formats; test same-stem videos and preserve image extensions.
- [ ] Wave events: pass `wave_detected` through `TrackSnapshot`; update `events.py` to use it; test replacement of an expired wave and demo overrides.
- [ ] Activity caches: validate preprocessing metadata and invalidate dependent backbone vectors for train, validation, and test. Add regression tests for changed frame counts, corrupt caches, and changed sources. Propagate overwrite to evaluation.
- [ ] Temporal sampling: define a fixed source-time sampling contract for new activity training and live checkpoints; preserve explicit legacy compatibility with a warning. Test the same timestamp spacing offline and live, missing poses, and track gaps.
- [ ] Review and verification: run the available complete test suite with real dependencies, compile source, check diff whitespace, and obtain an independent review. Fix substantiated findings before reporting completion.

## Rules

Write regression tests before implementation; confirm the relevant failure then run the tests after the fix. Keep tests independent of pretrained weights. Use typed, focused helpers and existing naming conventions. Update documentation for changed output names and checkpoint compatibility. Do not retrain models or change published benchmark numbers as part of this code fix.

## Verification environment

Use a repository-local `.venv` with Python 3.12 and repository requirements. If dependency availability blocks execution, record the precise limitation and distinguish isolated checks from integration tests.
