# TASK.md

## Goal

Convert the experiment in `final_homework_Advanced.ipynb` into the existing:

```text
submit_pipeline.sh
-> optional build
-> preprocess
-> train
-> test
```

pipeline without modifying the dependency environment.

The final pipeline must run non-interactively inside the existing Singularity + uv environment.

## Out of scope

Do NOT reproduce Notebook-only presentation or environment setup.

The following are explicitly out of scope:

* Colab setup;
* Google Drive;
* apt installation;
* pip/uv dependency modification;
* Jupyter widgets;
* display-only progress UI;
* intermediate visual checks;
* intermediate Spatial baseline-vs-finetuned evaluation;
* notebook download buttons.

Do not reproduce code merely because it exists in the notebook if it does not contribute to the required final artifacts.

---

## Source revisions

Use the following Notebook-defined experiment sources.

### Base SmolVLA model

```text
repo:
lerobot/smolvla_libero_plus

revision:
7bb70aa5bc92b82c9239142775d3a173103567ff
```

### VLM backbone

```text
HuggingFaceTB/SmolVLM2-500M-Video-Instruct
```

The original notebook does not pin this VLM to a commit.

During login-node asset preparation, resolve the concrete downloaded snapshot revision and record it in:

```text
data/manifests/assets.lock.json
```

Subsequent compute jobs must reuse that exact local snapshot.

### Dataset

```text
repo:
lerobot/libero_plus

revision:
f3f49f426d75030177b18778374005bc12ccd588
```

### Source packages

Use the existing project environment:

```text
LeRobot v0.6.0
LIBERO-plus 4976dc3
```

Do not clone or reinstall these packages at runtime.

---

## Stage 0: submit / prepare assets

Asset preparation happens on the Internet-connected login node BEFORE any Slurm compute job is submitted.

`submit_pipeline.sh` must:

1. create one RUN_ID;
2. create one timestamp;
3. determine the run output directory;
4. check every required offline asset;
5. download only missing assets;
6. validate that every required asset exists locally;
7. write/update the asset lock manifest;
8. only after successful validation, submit Slurm jobs.

If preparation fails, submit no preprocess/train/test jobs.

Do not modify Python dependencies during preparation.

### Required offline assets

At minimum determine and prepare all artifacts required by:

* pretrained SmolVLA policy;
* SmolVLM2 VLM backbone;
* `lerobot/libero_plus` dataset metadata and training data;
* LIBERO-plus simulation assets;
* any Hugging Face files that the train/eval commands would otherwise attempt to fetch.

Use:

```text
data/hf_cache/
data/models/
data/datasets/
data/assets/
```

as appropriate.

Compute jobs must work with networking unavailable.

Set offline environment flags where appropriate so accidental Internet fallback fails immediately rather than hanging.

---

## RUN_ID

Generate one opaque short RUN_ID for each pipeline invocation.

Use one UTC or local timestamp string suitable for paths.

Create:

```text
data/outputs/<timestamp>-<RUN_ID>/
```

Export RUN_ID and output path through `sbatch --export` so every stage refers to the same run.

Slurm job IDs are separate from RUN_ID and must be recorded in the run manifest.

All user-facing final artifact filenames must include RUN_ID.

---

## Stage 1: preprocess

Entry point:

```text
uv run python -m vla_simulation_project.main preprocess
```

Responsibilities:

1. verify compute node offline asset availability;
2. verify expected source/data revisions;
3. load local dataset metadata;
4. identify LIBERO-Spatial training episodes;
5. select training episodes deterministically;
6. write preprocessing manifest.

Use the notebook's training selection:

```text
10 LIBERO-Spatial tasks
x 5 episodes per task
= 50 training episodes
```

Preserve the notebook's task-name normalization and deterministic/even episode selection behavior.

Output:

```text
<run-dir>/manifests/preprocess.json
```

The manifest must contain enough information for train to run in a new process without Notebook globals.

---

## Stage 2: train

Entry point:

```text
uv run python -m vla_simulation_project.main train
```

Read the preprocess manifest.

Use `lerobot-train`.

Default notebook hyperparameters:

```text
steps               = 3000
batch_size          = 1
learning_rate       = 3e-4
final_learning_rate = 3e-5
warmup_steps        = 100
lora_r              = 16
lora_alpha          = 16
log_freq            = 100
```

Preserve notebook behavior including:

```text
freeze_vision_encoder = true
train_expert_only     = true
use_imagenet_stats    = false
video_backend         = torchcodec
num_workers           = 0
persistent_workers    = false
wandb                 = disabled
```

Experiment parameters must be configurable independently of `pyproject.toml`.

Use:

```text
config/experiment.toml
```

for tunable experiment settings.

### Model merge

After successful LoRA training:

1. find the final checkpoint;
2. load the SmolVLA base policy;
3. attach LoRA adapter;
4. merge with `merge_and_unload`;
5. save through the appropriate LeRobot policy `save_pretrained` API;
6. preserve required preprocessor/postprocessor statistics;
7. verify that the resulting model can be identified as a complete policy artifact;
8. verify no LoRA adapter weights remain in the merged weight artifact when applicable.

The merged model is the model evaluated by `test`.

A separate intermediate baseline model is not a required final artifact.

### Train outputs

At minimum:

```text
<run-dir>/model/<RUN_ID>_smolvla/
<run-dir>/<RUN_ID>_model.tar.gz
<run-dir>/<RUN_ID>_parameters.csv
<run-dir>/manifests/train.json
```

The uncompressed model directory may be retained in addition to the archive.

---

## Parameter CSV

Generate exactly one parameter CSV for the run:

```text
<RUN_ID>_parameters.csv
```

At minimum include:

```text
run_id
steps
batch_size
learning_rate
final_learning_rate
warmup_steps
lora_r
lora_alpha
seed
dataset_repo
dataset_revision
base_model_repo
base_model_revision
vlm_repo
resolved_vlm_revision
training_episode_count
```

Additional relevant parameters are welcome.

---

## Stage 3: test

Entry point:

```text
uv run python -m vla_simulation_project.main test
```

Evaluate the merged finetuned model.

Do not perform the notebook's earlier intermediate Spatial comparison.

Required final suites:

```text
libero_spatial
libero_object
libero_goal
libero_10
```

Default task IDs from the notebook:

```text
[0, 4, 8]
```

for each suite.

Default episodes per task:

```text
1
```

This value must be configurable so future runs can use more trials without source-code changes.

Use the Notebook evaluation settings unless incompatible with the installed fixed LeRobot version:

```text
policy.device=cuda
policy.use_amp=false
env.type=libero
env.is_libero_plus=true
observation_height=256
observation_width=256
control_mode=relative
max_parallel_tasks=1
eval.batch_size=1
eval.use_async_envs=false
```

Use the Notebook camera mapping:

```text
agentview_image -> front
robot0_eye_in_hand_image -> wrist
```

---

## Evaluation CSV

Produce one and only one required evaluation-results CSV:

```text
<run-dir>/<RUN_ID>_results.csv
```

It must contain detailed task-level data and suite-level totals.

At minimum support these fields:

```text
run_id
suite
task_id
level
n_trials
n_success
success_rate
```

Use:

```text
level=task
```

for individual task rows and:

```text
level=suite
```

for suite aggregate rows.

For a suite row, `task_id` may be empty.

The CSV therefore contains both detailed task results and the requested suite success rates without generating a second summary CSV.

Success counts must be derived from actual `eval_info.json` episode success values, not reconstructed from a rounded percentage.

Keep raw LeRobot `eval_info.json` outputs under the run directory for provenance even though they are not an additional user-facing CSV.

---

## Rollout videos

For each suite:

```text
libero_spatial
libero_object
libero_goal
libero_10
```

record the first task, task ID 0.

Generate exactly one requested video per suite.

Final names:

```text
<RUN_ID>_spatial.mp4
<RUN_ID>_object.mp4
<RUN_ID>_goal.mp4
<RUN_ID>_libero10.mp4
```

Do not create videos for every evaluation task.

Reuse an evaluation rollout video when possible rather than running an unnecessary duplicate simulation.

---

## Final run layout

Target structure:

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

All final user-facing artifact filenames must contain RUN_ID.

---

## Shell integration

Keep the existing dependency ordering:

```text
optional build
     |
preprocess
     |
train
     |
test
```

Only make minimal shell modifications.

Change stage entry points to:

```text
preprocess.sh:
uv run python -m vla_simulation_project.main preprocess

train.sh:
uv run python -m vla_simulation_project.main train

test.sh:
uv run python -m vla_simulation_project.main test
```

`submit_pipeline.sh` additionally handles:

* RUN_ID/timestamp generation;
* login-node offline-asset preparation;
* export of run metadata;
* existing sbatch dependency submission.

Do not change Slurm time limits as part of this task.

---

## Python architecture

Implement small modules rather than placing all extracted Notebook code in `main.py`.

Expected responsibilities:

```text
main.py
    CLI/stage dispatcher only

config.py
    experiment configuration

paths.py
    PROJECT/data/run path resolution

prepare_assets.py
    login-node local-cache validation/download

preprocess.py
    dataset metadata + training episode selection

train.py
    lerobot-train command/build/run

merge.py
    LoRA merge and model artifact validation

evaluate.py
    lerobot-eval + result aggregation + video handling

artifacts.py
    manifests, CSV, archive validation

subprocess_utils.py
    subprocess execution/error handling
```

Do not preserve Notebook cell boundaries when a cleaner module boundary is available.

---

## Tests

Add GPU-independent tests for at least:

1. RUN_ID/run-directory path construction;
2. asset manifest validation;
3. missing asset failure;
4. task-name normalization;
5. deterministic episode selection;
6. training command generation;
7. evaluation command generation;
8. results CSV aggregation from representative `eval_info.json`;
9. final artifact contract validation;
10. dependency files remain unchanged.

Tests must not:

* download models;
* modify dependencies;
* require Internet;
* start a full training run;
* require a GPU unless explicitly marked as an integration test.

---

## Acceptance criteria

The task is complete when all of the following are true.

### Repository integrity

Unchanged unless explicitly required:

```text
final_homework_Advanced.ipynb
pyproject.toml
uv.lock
singularity/ubuntu24.04.def
```

### Offline preparation

On the login node, missing required assets can be discovered and downloaded before `sbatch`.

When all assets already exist, preparation does not redownload them.

### Compute isolation

After submission, preprocess/train/test require no Internet connection.

A missing required asset produces a clear failure instead of a network attempt.

### Stage isolation

Each of:

```text
preprocess
train
test
```

can start in a fresh Python process using persisted manifests from the previous stage.

### Training

The train stage invokes `lerobot-train`, produces the LoRA checkpoint, merges LoRA into a standalone SmolVLA model, and creates the model archive and parameter CSV.

### Evaluation

The test stage evaluates:

```text
4 suites x 3 tasks x configured episodes
```

and creates one detailed results CSV containing trial counts, success counts, and success rates.

### Video

One task-0 video exists for each of the four suites.

### Traceability

The same RUN_ID is present in:

* run directory;
* final artifact names;
* parameter CSV;
* result CSV;
* manifests.

Slurm job IDs are recorded separately.

### Validation report

At completion, report:

* files changed;
* tests executed;
* test results;
* anything not tested because it requires actual Slurm/GPU execution;
* any remaining incompatibility found in the fixed environment.

The LIBERO-plus version and Git revision must not be changed.

The LIBERO-plus Python dependency must remain pinned to the revision recorded in `uv.lock`.

However, benchmark resources required by LIBERO-plus at runtime, including `bddl_files`, `init_files`, and other data contained in the repository, may be fetched from the same pinned Git revision on the login node and placed in shared storage.

After fetching the repository, the checked-out revision must be verified using `git rev-parse HEAD`. Compute nodes must not perform any network access.

External assets must be prepared on the login node from the designated fixed source. Compute nodes must use only the local copies stored in shared storage.
