# Artifact Contract

## Purpose

This document defines the files that each pipeline run must produce.

Artifacts must be reproducible, machine-readable where appropriate, and traceable to one pipeline RUN_ID.

---

## RUN_ID

Each invocation of `submit_pipeline.sh` creates exactly one:

```text
RUN_ID
```

and one:

```text
timestamp
```

The run directory is:

```text
data/outputs/<timestamp>-<RUN_ID>/
```

The same RUN_ID must be used for:

```text
preprocess
train
test
```

Slurm job IDs are separate identifiers.

---

## Required run structure

```text
data/outputs/<timestamp>-<RUN_ID>/
├── <RUN_ID>_model.tar.gz
├── <RUN_ID>_parameters.csv
├── <RUN_ID>_results.csv
├── <RUN_ID>_spatial.mp4
├── <RUN_ID>_object.mp4
├── <RUN_ID>_goal.mp4
├── <RUN_ID>_libero10.mp4
│
├── model/
│   └── <RUN_ID>_smolvla/
│
├── eval/
│   ├── libero_spatial/
│   ├── libero_object/
│   ├── libero_goal/
│   └── libero_10/
│
└── manifests/
    ├── run.json
    ├── preprocess.json
    ├── train.json
    └── test.json
```

Raw LeRobot evaluation outputs may be retained inside each suite directory.

---

## run.json

Written when the pipeline is submitted.

At minimum:

```json
{
  "run_id": "...",
  "timestamp": "...",
  "run_dir": "...",
  "preprocess_job_id": null,
  "train_job_id": null,
  "test_job_id": null
}
```

Update job IDs after submission.

Also record environment/revision information when available.

---

## preprocess.json

Must include at minimum:

```text
run_id
dataset_repo
dataset_revision
selected_episode_indices
selected_episode_count
base_model_path
dataset_path
vlm_path
libero_assets_path
```

The train stage must use this persisted selection.

---

## train.json

Must include at minimum:

```text
run_id
training_started_at
training_finished_at
training_exit_code
checkpoint_path
merged_model_path
model_archive_path
parameters_csv_path
```

Include the exact training command or structured command arguments for provenance.

---

## test.json

Must include at minimum:

```text
run_id
evaluation_started_at
evaluation_finished_at
suites
episodes_per_task
results_csv_path
video_paths
```

Include the exact evaluation configuration used.

---

## Final model

Required model directory:

```text
model/<RUN_ID>_smolvla/
```

It must be a standalone merged policy artifact.

It must contain all files required to load the policy for later evaluation/reuse.

Create:

```text
<RUN_ID>_model.tar.gz
```

from this directory.

The archive must be usable without the original LoRA training checkpoint.

---

## Parameter CSV

Required:

```text
<RUN_ID>_parameters.csv
```

Recommended format:

```csv
run_id,parameter,value
abc123,steps,3000
abc123,batch_size,1
abc123,learning_rate,0.0003
abc123,final_learning_rate,0.00003
abc123,warmup_steps,100
abc123,lora_r,16
abc123,lora_alpha,16
```

Also include provenance values such as:

```text
seed
base_model_repo
base_model_revision
vlm_repo
resolved_vlm_revision
dataset_repo
dataset_revision
training_episode_count
```

The exact schema may use either:

```text
one row per parameter
```

or:

```text
one experiment per row
```

but it must be deterministic and documented in code.

---

## Evaluation CSV

Produce exactly one required evaluation result CSV:

```text
<RUN_ID>_results.csv
```

Minimum columns:

```text
run_id
suite
task_id
level
n_trials
n_success
success_rate
```

### Task rows

Example:

```text
run_id = abc123
suite = libero_spatial
task_id = 0
level = task
n_trials = 5
n_success = 3
success_rate = 0.6
```

### Suite aggregate row

Example:

```text
run_id = abc123
suite = libero_spatial
task_id =
level = suite
n_trials = 15
n_success = 9
success_rate = 0.6
```

The suite total must be calculated from actual individual episode successes.

Do not derive `n_success` from a rounded percentage.

---

## Raw evaluation provenance

Retain LeRobot-generated:

```text
eval_info.json
```

files underneath:

```text
eval/<suite>/
```

These raw files are not additional user-facing CSV results.

They exist for debugging and reproducibility.

---

## Videos

Required:

```text
<RUN_ID>_spatial.mp4
<RUN_ID>_object.mp4
<RUN_ID>_goal.mp4
<RUN_ID>_libero10.mp4
```

Each is a rollout of task ID 0 of the corresponding suite.

Only one requested final video per suite is required.

---

## Completion validation

A run is considered complete only if all required files exist and are non-empty.

Validate:

```text
model archive
parameter CSV
result CSV
4 videos
merged model directory
run.json
preprocess.json
train.json
test.json
```

Additionally validate that:

* RUN_ID matches across manifests and user-facing filenames;
* all four suites occur in the result CSV;
* trial counts are greater than zero;
* success counts are between zero and trial counts;
* success rates equal `n_success / n_trials`;
* the merged model has all files required for later loading.

A partially complete run must not be reported as successfully completed.
