# Activity sampling and cache compatibility

New activity caches sample every **0.1 seconds of source time** by default. A
16-frame sequence covers 1.5 seconds between its first and last samples, starting
at the beginning of the clip. Preprocessing selects the most recent source frame
at each sample time using the video's presentation timestamps. It decodes only the
required window and retains the selected frames. Short clips repeat their final
readable frame until the sequence is full. Missing or non-increasing timestamps
fall back to the reported frame rate (24 FPS if that rate is invalid). Videos
with no readable frames fail preprocessing.

Use `--frames` and `--sampling-interval-seconds` to change this contract. The
manifest and new checkpoints record the interval. Live MLP inference uses that
checkpoint interval with source timestamps, maintaining a separate sampling grid
and history for each tracked person. A backwards timestamp or an observation gap
longer than `max(0.5 seconds, 3 × sampling interval)` clears that track's history
and prediction. Processing more frames per second does not shorten the motion
window. Inference throttling and probability smoothing still apply.

Legacy checkpoints without sampling metadata remain loadable and emit an explicit
warning. They retain their original adjacent-observation behavior; adding an
interval to an old checkpoint does not retrain it. Rebuild the caches and retrain
to obtain a model trained with the new timing contract. The configured live
`--activity-sequence-length` must match the checkpoint's frame count.

## Rebuild and retrain

Run these commands from the repository root, adjusting the dataset and annotation
paths to your local HMDB51 files. Separate output directories preserve existing
caches, checkpoints, and historical benchmark results:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_activity_data.py `
  --dataset-root "data\hmdb51\hmdb51_org" `
  --annotations-root "data\hmdb51\splits" `
  --cache-dir "data\activity_cache\timed" `
  --frames 16 --sampling-interval-seconds 0.1

.\.venv\Scripts\python.exe scripts\train_activity_models.py `
  --manifest "data\activity_cache\timed\manifest.json" `
  --feature-dir "data\activity_cache\timed\features" `
  --model-dir "data\activity_models\timed" `
  --results-dir "output\activity_results_timed"
```

The training command evaluates the test split unless `--skip-evaluation` is set.
Standalone evaluation of the newly trained checkpoints uses the same paths:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_activity_models.py `
  --manifest "data\activity_cache\timed\manifest.json" `
  --feature-dir "data\activity_cache\timed\features" `
  --model-dir "data\activity_models\timed" `
  --results-dir "output\activity_results_timed"
```

Pose caches are reused only when source content, preprocessing metadata, labels,
split, and array shapes match. Changing the frame count, interval, or source video
regenerates incompatible caches automatically. The preparation command publishes
the manifest only after preprocessing succeeds. Training rejects incompatible
cache shapes or frame counts before creating checkpoints.

Backbone vectors carry provenance sidecars that identify the source NPZ content,
extractor, and tensor preprocessing settings. Changed or corrupt caches trigger
regeneration for the affected samples. `--overwrite` forces pose preprocessing;
`--overwrite-features` forces feature extraction and is available in both training
and standalone evaluation. Training passes this option through to its evaluation
step.

Historical benchmark results are unchanged. The new sampling window changes the
training inputs, so new metrics must come from a fresh training and evaluation
run. Existing checkpoints and their original manifests can still be evaluated
together; checkpoints and manifests with different sampling intervals are
rejected.
