# Offline Assets

## Purpose

Slurm compute nodes have no Internet access.

This document defines every external artifact that must be available locally before compute jobs are submitted.

Asset preparation occurs on the Internet-connected login node.

`submit_pipeline.sh` executes preparation in the dedicated login-node image
`singularity/login.sif`.  It does not use the login node's system Python and it
does not launch the compute image.

Hugging Face repositories are staged with `huggingface_hub.snapshot_download`.
Its file and byte progress indicators are kept enabled so long dataset transfers
remain observable from the submission terminal.

---

## Principle

The required sequence is:

```text
submit_pipeline.sh
        |
        v
check local assets
        |
        +-- complete --> validate
        |
        +-- missing --> download missing assets
                              |
                              v
                           validate
                              |
                              v
                       submit Slurm jobs
```

No compute job may depend on an Internet connection.

---

## Persistent locations

Use:

```text
data/
├── hf_cache/
├── models/
├── datasets/
├── assets/
└── manifests/
```

Do not store persistent data below `/tmp` or notebook-style `/content`.

---

## Required external resources

### SmolVLA pretrained model

```text
repo:
lerobot/smolvla_libero_plus

revision:
7bb70aa5bc92b82c9239142775d3a173103567ff
```

Required content includes the model and all policy processor/configuration files needed by LeRobot training.

Do not download irrelevant evaluation videos or repository documentation.

---

## SmolVLM backbone

```text
repo:
HuggingFaceTB/SmolVLM2-500M-Video-Instruct
```

The source notebook does not specify a commit revision.

The first successful preparation must resolve the concrete Hugging Face snapshot revision.

Record the resolved revision so later runs reproduce the same model.

---

## LeRobot dataset

```text
repo:
lerobot/libero_plus

revision:
f3f49f426d75030177b18778374005bc12ccd588
```

The locally available dataset must contain everything needed for:

* metadata inspection;
* episode selection;
* training video/data decoding.

A metadata-only download is insufficient if training later requires files that are not cached.

---

## LIBERO-plus assets

All LIBERO-plus environment assets required for evaluation must exist locally before `test`.

Evaluation must not attempt to retrieve them from the Internet.

---

## Source packages

The following are NOT downloaded by `prepare_assets.py`:

```text
LeRobot v0.6.0
LIBERO-plus 4976dc3
```

They are provided by the existing uv environment.

Do not clone or reinstall them during the pipeline.

---

## Asset lock file

Use:

```text
data/manifests/assets.lock.json
```

Suggested format:

```json
{
  "base_model": {
    "repo": "lerobot/smolvla_libero_plus",
    "revision": "7bb70aa5bc92b82c9239142775d3a173103567ff",
    "local_path": "data/models/..."
  },
  "dataset": {
    "repo": "lerobot/libero_plus",
    "revision": "f3f49f426d75030177b18778374005bc12ccd588",
    "local_path": "data/datasets/..."
  },
  "vlm": {
    "repo": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    "revision": "<resolved-commit>",
    "local_path": "data/models/..."
  },
  "libero_assets": {
    "local_path": "data/assets/..."
  }
}
```

Paths should preferably be project-relative where practical.

---

## Validation requirements

Before submitting Slurm jobs, verify at minimum:

* required directories exist;
* model configuration exists;
* model weights exist;
* processor configuration/statistics exist;
* dataset metadata exists;
* required training dataset files exist;
* VLM snapshot exists;
* LIBERO assets exist;
* the lock manifest matches expected fixed revisions.

Do not treat an empty directory as a valid cached artifact.

---

## Download policy

Download only missing resources.

Do not repeatedly download an already valid artifact.

Do not automatically switch to a different revision when a download fails.

If the fixed revision cannot be prepared, stop before submitting Slurm jobs.

---

## Compute-node offline configuration

Compute jobs should enable explicit offline behavior where supported.

Expected environment configuration includes:

```text
HF_HOME=<PROJECT>/data/hf_cache
HF_HUB_OFFLINE=1
HF_DATASETS_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

Any library that still tries to reach the network should fail rather than hang indefinitely.

---

## Missing resources on compute node

If a resource is missing during:

```text
preprocess
train
test
```

the stage must fail with a message identifying:

* missing resource;
* expected local path;
* relevant repository/revision;
* instruction that asset preparation must be run from the login node.

Compute-stage code must not attempt to repair the problem by downloading the resource.

---
