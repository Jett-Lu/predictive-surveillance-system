# Runtime configuration

Run from a source checkout or install the wheel using the README instructions.
The wheel includes the MediaPipe face detector and the `integrated-monitoring`
launcher. Installed runtime data defaults to `~/.integrated-monitoring-poc`;
source runs default to the checkout root. Set `MONITOR_HOME` to override this
root (relative values resolve from the working directory). Read-only bundled
assets stay in the installed package; runtime data does not go in site-packages.

Set these environment variables before starting Python. Unset or malformed
values use defaults. Numeric environment values are clamped to bounds; CLI
activity overrides reject invalid values. Boolean values accept 1/0,
true/false, yes/no and on/off. CLI overrides take precedence.

| Variable | Default | Type or bounds |
| --- | --- | --- |
| `MONITOR_HOME` | user data directory for wheels; checkout for source runs | Writable runtime root |
| `MONITOR_RECORDING_RETENTION_DAYS` | 30 | Positive integer; annotated exports |
| `MONITOR_IDENTITY_RETENTION_DAYS` | 365 | Positive integer; enrollment images |
| `MONITOR_LOG_RETENTION_DAYS` | 30 | Positive integer; events/application logs |
| `MONITOR_ACTIVITY_CHECKPOINT` | `PROJECT_ROOT / 'data' / 'activity_models' / 'mlp.pt'` | path |
| `MONITOR_ACTIVITY_CONFIDENCE` | `0.5` | maximum=1.0 |
| `MONITOR_ACTIVITY_INTERVAL` | `5` | int |
| `MONITOR_ACTIVITY_MODEL` | `'none'` | choice |
| `MONITOR_ACTIVITY_SEQUENCE_LENGTH` | `16` | int |
| `MONITOR_ACTIVITY_SMOOTHING_WINDOW` | `5` | int |
| `MONITOR_ALLOW_MODEL_DOWNLOADS` | `True` | bool |
| `MONITOR_DEBUG_TIMING` | `False` | bool |
| `MONITOR_EVENT_LOGGING` | `True` | bool |
| `MONITOR_EXPRESSION_CONFIDENCE` | `0.65` | maximum=1.0 |
| `MONITOR_EXPRESSION_DISPLAY_CONFIDENCE` | `0.45` | maximum=1.0 |
| `MONITOR_EXPRESSION_INTERVAL` | `5` | int |
| `MONITOR_EXPRESSION_SMOOTHING_SECONDS` | `0.8` | minimum=0.1 |
| `MONITOR_IDENTITY_INTERVAL` | `8` | int |
| `MONITOR_IDENTITY_MARGIN` | `0.03` | maximum=1.0 |
| `MONITOR_IDENTITY_MATCHES` | `3` | int |
| `MONITOR_IDENTITY_THRESHOLD` | `0.363` | maximum=1.0 |
| `MONITOR_IDENTITY_TTL_FRAMES` | `90` | int |
| `MONITOR_IDENTITY_WINDOW` | `5` | int |
| `MONITOR_KEYPOINT_CONFIDENCE` | `0.3` | maximum=1.0 |
| `MONITOR_LOG_LEVEL` | `'info'` | choice |
| `MONITOR_MIN_FACE_CONFIDENCE` | `0.85` | maximum=1.0 |
| `MONITOR_MIN_FACE_SIZE` | `64` | int |
| `MONITOR_MODEL_DOWNLOAD_ATTEMPTS` | `3` | int |
| `MONITOR_MODEL_DOWNLOAD_TIMEOUT` | `60.0` | minimum=5.0 |
| `MONITOR_POSE_CONFIDENCE` | `0.3` | maximum=1.0 |
| `MONITOR_POSE_IOU` | `0.45` | maximum=1.0 |
| `MONITOR_STALE_TRACK_FRAMES` | `90` | int |
| `DEMO_HIGH_REVIEW_NAMES` | empty | Comma/semicolon separated enrolled names; forces a visible demo HIGH override. Keep empty outside demonstrations. |

`MONITOR_ACTIVITY_MODEL` accepts `none` or `mlp`. Logging accepts DEBUG, INFO,
WARNING, ERROR, CRITICAL and NOTSET through the environment. Relative activity
checkpoint paths resolve under the runtime root. Defaults for data,
enrollments and log/event directories are also under the runtime root.
Explicit CLI input/output/manifest/report paths resolve from the working directory.

Model preparation needs outbound HTTPS and writable data/log directories.
Expression initialization copies the verified classifier into the dependency's
standard `~/.emotiefflib` cache, which must also be writable. Other library
caches use `.tmp` under the runtime root. Run `integrated-monitoring --doctor
--prepare-models`, then verify offline with `--doctor --no-model-downloads`.

`--no-model-downloads` only controls this application's model provisioning;
it is not a network-isolation switch. Ultralytics analytics are disabled before
model loading (`sync=False`, its existing event collector disabled, and
`YOLO_OFFLINE=true` before import). ONNX Runtime's supported
`disable_telemetry_events()` is called before the expression session is created.
These controls also apply when model classes are used without the CLI.

MediaPipe remains unresolved: the real export smoke test observed a
MediaPipe Clearcut upload attempt that failed under network restrictions.
Its maintainers confirm that the published SDK has no telemetry opt-out;
they recommend a source build or host-level network blocking. No MediaPipe
network control has been applied by these Python settings. See the
[maintainer discussion](https://github.com/google-ai-edge/mediapipe/issues/6291)
and [ONNX Runtime telemetry API](https://onnxruntime.ai/docs/api/python/api_summary.html#onnxruntime.disable_telemetry_events).

Enrollment images, JSONL events, logs and annotated media can contain personal
information. Private directories and explicit retention maintenance are
available; see [data governance](data-governance.md). There is no application
authentication or encryption layer. Only use trusted model/checkpoint and manifest files.
Use one writer per export/cache destination; concurrent jobs sharing paths
are not supported. Network capture has no application-level reconnect policy.

CLI detection and media processing return failure status for unavailable input
or failed processing. The interactive media workflow remains best effort.
Annotated video exports contain video only; original audio is not preserved.
A successful decode cannot prove a source file is complete or uncorrupted.
