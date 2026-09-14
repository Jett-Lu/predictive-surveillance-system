# Final production audit — 2026-09-13

**Verdict: NOT READY for production deployment.** The source-checkout POC
passes the checks below, but operational validation and
data controls remain unresolved. This is not a claim that the demonstrated
review levels reliably identify real-world risk.

## Scope

Review covered runtime modules, activity preparation/training/evaluation,
scripts, tests, dependency manifests, CI, setup instructions and supporting
documentation across the repository, including unchanged files. Generated
datasets, caches and model binaries were not treated as source code. Model
provisioning was verified against the configured hashes. No new model training
or representative human-subject accuracy evaluation was performed.

## Findings and fixes

| Finding | Change |
| --- | --- |
| YuNet/SFace provisioning downloaded Git LFS pointers and failed checksum validation | Use binary-content URLs at the existing pinned revision; retain hashes and verify real downloads |
| Body translation could be mistaken for wrist wave motion | Measure wrist movement relative to the shoulder; add regression coverage |
| Anonymous fallback label could trigger the name-based demo override | Apply override only after identity confirmation; add regression coverage |
| Failed capture/export could return successful CLI status | Propagate capture failure and use strict CLI export failure handling; preserve interactive best effort |
| Same-directory exports could overwrite another input | Reject collisions before processor initialization; add regression coverage |
| Invalid activity CLI settings accepted zero/negative intervals and non-finite confidence | Validate at argument parsing; test invalid inputs; validate environment log levels |
| Cleanup could skip the expression detector after an activity-metrics exception | Guarantee detector and event-recorder cleanup with nested finally blocks; regression test |
| Pyright reported 19 type errors | Correct OpenCV API references, tensor-classifier typing, tuple shapes and optional-value narrowing |
| Dependency audit reported Torch and setuptools advisories | Upgrade local environment; require Torch >=2.13 and build setuptools >=83; align activity requirements and historical snapshot |
| Formatting drift and incomplete CI checks | Apply configured formatter; add formatting, type and distribution-build checks to CI |
| Documentation overstated tracker isolation and omitted runtime settings | Explain ID-switch risk, document configuration and deployment limitations |

No speculative model architecture changes were made. Regression tests were
observed failing before the corresponding behavior fixes, then passing.

## Executed verification

Environment: Windows 11, Python 3.12.14; Torch 2.13.0, torchvision 0.28.0,
setuptools 84.0.0. Checks below were rerun after implementation changes.

| Check | Result / evidence |
| --- | --- |
| `python -m unittest discover -s tests -v` | **125 passed**, `.tmp/audit-final-tests.log`; unit tests use mocks for model/hardware boundaries |
| `ruff check --config pyproject.toml src scripts tests` | Passed |
| `ruff format --check --config pyproject.toml src scripts tests` | All 49 Python files formatted |
| `pyright --pythonpath .venv/Scripts/python.exe src scripts` | 0 errors, 0 warnings |
| `python -m compileall -q src scripts tests` | Passed |
| `uv --cache-dir .tmp/uv-cache pip check` | 80 installed packages compatible |
| `python -m build --no-isolation --outdir .tmp/audit-dist` | sdist and wheel built, `.tmp/audit-final-build.log` |
| `python -m pip_audit` with explicit cache/output options | No known vulnerabilities in installed environment, `.tmp/dependency-audit-final.json`; database result is not a security guarantee |
| `python src/main.py --doctor --no-model-downloads` | All readiness checks passed, `.tmp/audit-final-doctor.log` |
| Real model initialization and inference | Pose, identity and expression models loaded; blank-frame pose/face inference and classifier inference passed, `.tmp/audit-real-smoke.log` |
| Real CLI image/video export | 2/2 synthetic inputs exported; output image and all 3 video frames decoded, `.tmp/audit-media-smoke.log` |
| CLI missing-input failure | Exit status 1, `.tmp/audit-cli-failure.log` |
| Tracked text secret-pattern scan | 67 files scanned, 0 high-confidence pattern matches; not a full Git-history or general-purpose secret audit |
| `git diff --check` | Passed; Git emitted Windows line-ending conversion notices |

Evidence logs and generated media are ignored local artifacts. Cross-platform
CI changes have not been executed on hosted Linux/macOS runners in this audit.

## Remaining blockers and manual verification

1. **Standalone packaging — resolved in follow-up:** the wheel now includes
   the MediaPipe detector, a launcher, and user-writable runtime defaults with
   `MONITOR_HOME`. Installed outside the checkout in a temporary environment;
   launcher, import origins, runtime paths and real detector loading passed.
   Dependencies were reused from the tested environment. Source and wheel
   builds, 127 tests, lint/format and type checks are covered by the follow-up;
   fresh dependency resolution and hosted cross-platform execution remain
   deployment checks. Evidence: `.tmp/packaging-install.log`.
2. **Dependency network behavior:** MediaPipe attempted a Clearcut upload
   during real export; it failed under network restrictions. The application
   model-download switch does not enforce offline operation. Assess the
   dependency's telemetry and deployment network controls before release.
3. **Real-world behavior:** verify enrollment and recognition with consented
   representative data, multiple people crossing/occluding, tracker ID switches,
   gesture false positives, lighting variation and identity expiration. Blank
   frames validate execution, not recognition accuracy. Historical benchmark
   metrics were not reproduced with the updated dependencies.
4. **Operational reliability:** test physical cameras, GUI controls, stream
   loss, codec combinations and long-duration CPU/GPU/memory behavior. Network
   capture lacks an application-level reconnect policy. Concurrent exporters
   or cache writers sharing destination paths are unsupported. Test disk-full
   and permission failures on the target deployment filesystem.
5. **Data controls — implemented locally:** private directory creation,
   explicit existing-data permission migration, and configurable preview-first
   retention maintenance now cover managed exports, enrollment images, events
   and application logs. Current managed directories were restricted under
   the operator's Windows account; no existing data was deleted. See
   [data governance](data-governance.md) for scope and commands. Scheduling,
   backups, external input media and deployment account policies remain an
   operator responsibility; there is no application authentication/encryption.
6. **Coverage/platform gaps:** automated tests do not establish performance,
   fairness or safety; optional activity training, deployed checkpoints and
   hardware paths need representative integration tests. Run hosted CI and
   target-platform installation checks, including fresh dependency resolution.

MediaPipe also logged that feedback tensors are disabled for the face model;
inference and exports still succeeded. Windows Git line-ending notices are
nonfunctional. Dependency security was checked against the installed Windows
resolution, not every possible version allowed by the dependency ranges.
