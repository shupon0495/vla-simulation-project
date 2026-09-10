# Architecture

## Purpose

This document defines the runtime architecture of the SmolVLA / LeRobot experiment pipeline.

The behavioral reference is:

```text
final_homework_Advanced.ipynb
```

The notebook itself must not be modified.

The notebook workflow is converted into a non-interactive Python pipeline executed through the existing Slurm, Singularity, and uv infrastructure.

---

## High-level architecture

```text
Login node
  |
  | Internet available
  |
  +-- submit_pipeline.sh
        |
        +-- create RUN_ID and timestamp
        |
        +-- build/reuse login.sif
        |
        +-- prepare/check offline assets in login.sif
        |
        +-- optional build.sh
        |
        +-- sbatch preprocess.sh
                |
                v
          preprocess stage
                |
                v
          preprocess.json
                |
        +-- sbatch train.sh --dependency=afterok
                |
                v
             train
                |
                +-- lerobot-train
                +-- LoRA merge
                |
                v
            merged model
                |
        +-- sbatch test.sh --dependency=afterok
                |
                v
              test
                |
                +-- LIBERO evaluation
                +-- result aggregation
                +-- rollout videos
```

---

## Environment boundary

There are two execution environments.

### Login node

Internet access is available.

Responsibilities:

* generate RUN_ID;
* generate timestamp;
* inspect required models/datasets/assets;
* download missing Hugging Face artifacts;
* validate that all compute-time resources are available locally;
* submit Slurm jobs.

The login-node preparation phase must complete before preprocess/train/test jobs are submitted.

Asset preparation runs in the dedicated `singularity/login.sif`, built from
`singularity/login.def`.  This keeps it independent of both the login node's
system Python version and the compute image. The project uv environment supplies
`huggingface_hub`, whose snapshot progress indicators are shown while assets are
downloaded.

### Compute node

Internet access is unavailable.

The compute stages are:

```text
preprocess
train
test
```

All required resources must already exist locally.

Compute-stage code must never depend on a network fallback.

Use offline Hugging Face configuration where applicable.

---

## Execution infrastructure

The existing pipeline is authoritative:

```text
submit_pipeline.sh
  -> build.sh if required
  -> preprocess.sh
  -> train.sh
  -> test.sh
```

The shell pipeline must not be redesigned.

`build.sh` is responsible only for creating/updating the Singularity image when necessary.

The application is executed in the container using:

```text
uv run python -m vla_simulation_project.main <stage>
```

Supported stages:

```text
preprocess
train
test
```

---

## Python architecture

Expected modules:

```text
src/vla_simulation_project/
├── __init__.py
├── main.py
├── config.py
├── paths.py
├── prepare_assets.py
├── preprocess.py
├── train.py
├── merge.py
├── evaluate.py
├── artifacts.py
└── subprocess_utils.py
```

### main.py

Responsibilities:

* CLI parsing;
* stage dispatch.

It must not contain experiment implementation logic.

Expected usage:

```text
python -m vla_simulation_project.main preprocess
python -m vla_simulation_project.main train
python -m vla_simulation_project.main test
```

### config.py

Responsibilities:

* load `config/experiment.toml`;
* validate experiment parameters;
* provide typed configuration to the stages.

Dependency versions must not be configured here.

### paths.py

Responsibilities:

* resolve PROJECT;
* resolve persistent data directories;
* resolve current RUN_ID directory;
* prevent accidental writes outside managed project directories.

### prepare_assets.py

Executed on the login node.

Responsibilities:

* inspect required offline resources;
* download only missing resources;
* resolve and record Hugging Face revisions;
* validate resources before Slurm submission.

It must not install Python packages.

### preprocess.py

Responsibilities:

* validate offline assets;
* inspect local dataset metadata;
* select the training episodes;
* write `preprocess.json`.

### train.py

Responsibilities:

* read preprocessing state;
* construct the `lerobot-train` command;
* execute training;
* identify the final checkpoint;
* invoke LoRA merge;
* write training manifest and parameter CSV.

### merge.py

Responsibilities:

* merge LoRA adapter into SmolVLA;
* save a standalone LeRobot policy;
* preserve processor configuration/statistics;
* validate final model artifacts.

### evaluate.py

Responsibilities:

* evaluate the final merged model;
* aggregate successes;
* generate final result CSV;
* create required rollout videos.

### artifacts.py

Responsibilities:

* JSON manifest read/write;
* CSV generation;
* archive creation;
* final artifact validation.

### subprocess_utils.py

Responsibilities:

* safe subprocess execution;
* stdout/stderr handling;
* command failure propagation.

---

## Stage isolation

Stages execute as different processes.

Python globals cannot carry state between:

```text
preprocess -> train -> test
```

Persistent JSON files must be used instead.

Expected files:

```text
<run-dir>/manifests/run.json
<run-dir>/manifests/preprocess.json
<run-dir>/manifests/train.json
<run-dir>/manifests/test.json
```

A stage must be restartable using only:

* environment variables;
* configuration;
* files created by previous stages.

---

## Configuration boundary

Dependency configuration:

```text
pyproject.toml
uv.lock
```

is immutable.

Experiment configuration belongs in:

```text
config/experiment.toml
```

Example tunable values:

```text
steps
batch_size
learning_rate
final_learning_rate
warmup_steps
lora_r
lora_alpha
seed
evaluation task IDs
evaluation episodes per task
```

---

## Storage layout

Persistent storage is located below the repository:

```text
data/
├── hf_cache/
├── models/
├── datasets/
├── assets/
├── manifests/
└── outputs/
```

Each experiment has exactly one run directory:

```text
data/outputs/<timestamp>-<RUN_ID>/
```

The same RUN_ID is used through the whole pipeline.

Slurm job IDs are recorded separately.

---

## Failure policy

Fail fast.

Do not:

* silently skip missing artifacts;
* download resources from compute nodes;
* continue after failed `lerobot-train`;
* continue after failed `lerobot-eval`;
* modify dependencies to solve runtime errors.

A failure must propagate as a non-zero process exit code.

---


---
