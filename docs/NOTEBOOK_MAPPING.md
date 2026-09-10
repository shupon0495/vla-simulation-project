# Notebook Mapping

## Purpose

This document records how `final_homework_Advanced.ipynb` is translated into the non-interactive project pipeline.

The goal is not to execute every notebook cell.

Notebook-specific environment setup and presentation logic must be removed while preserving the required experiment semantics.

---

## Mapping table

| Notebook section | Purpose                         | Pipeline mapping                   | Action                             |
| ---------------- | ------------------------------- | ---------------------------------- | ---------------------------------- |
| Section 1        | Colab/runtime setup             | none                               | Do not port                        |
| Section 2        | apt/system packages             | existing Singularity image         | Do not port                        |
| Section 3        | LeRobot installation/patching   | existing uv environment            | Do not reinstall                   |
| Section 4        | Hugging Face helper             | `prepare_assets.py`                | Extract only reusable asset logic  |
| Section 5        | LIBERO-plus setup/assets        | `prepare_assets.py` / local assets | Do not reinstall package           |
| Section 6        | experiment constants            | `config.py` / `experiment.toml`    | Port                               |
| Section 7        | Spatial episode selection       | `preprocess.py`                    | Port                               |
| Section 8.0      | hyperparameters                 | `config/experiment.toml`           | Port                               |
| Section 8.1      | base model download             | `prepare_assets.py`                | Adapt for offline runtime          |
| Section 8.2      | LoRA training                   | `train.py`                         | Port                               |
| Section 8.3      | LoRA merge                      | `merge.py`                         | Port                               |
| Section 8.4      | baseline preparation            | none                               | Omit unless later required         |
| Section 8.5      | intermediate Spatial comparison | none                               | Omit                               |
| Section 8.6      | intermediate comparison CSV     | none                               | Omit                               |
| Section 8.7      | model archive                   | `artifacts.py`                     | Port                               |
| Section 8.8      | rollout video                   | `evaluate.py`                      | Reimplement for final suite videos |
| Section 9        | advanced evaluation             | `evaluate.py`                      | Port                               |

---

## Explicitly excluded notebook behavior

Do not copy:

```text
google.colab
Google Drive mount
/content paths
apt install
pip install
pip uninstall
editable installation
IPython display
ipywidgets
notebook download UI
interactive progress UI
```

The target environment is not Colab.

---

## Fixed experiment sources

### Base SmolVLA model

```text
repo:
lerobot/smolvla_libero_plus

revision:
7bb70aa5bc92b82c9239142775d3a173103567ff
```

### Dataset

```text
repo:
lerobot/libero_plus

revision:
f3f49f426d75030177b18778374005bc12ccd588
```

### VLM

```text
HuggingFaceTB/SmolVLM2-500M-Video-Instruct
```

The notebook does not pin the VLM commit.

Asset preparation must resolve the concrete revision used and persist it in `assets.lock.json`.

### Python source dependencies

```text
LeRobot:
v0.6.0

LIBERO-plus:
4976dc3
```

These are already managed through the existing uv project configuration.

Do not clone or reinstall them.

---

## Training dataset selection

Preserve the notebook behavior:

```text
LIBERO-Spatial
10 tasks
5 episodes per task
50 episodes total
```

Selection must be deterministic.

Task-name normalization behavior from the notebook must be preserved.

The selected episode IDs must be persisted in:

```text
<run-dir>/manifests/preprocess.json
```

Train must not recompute a different selection.

---

## Default training parameters

Initial defaults come from the notebook:

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

These are experiment defaults, not dependency versions.

They must be configurable through:

```text
config/experiment.toml
```

---

## Required training behavior

Preserve:

```text
freeze_vision_encoder = true
train_expert_only = true
use_imagenet_stats = false
video_backend = torchcodec
num_workers = 0
persistent_workers = false
wandb = disabled
```

Use the existing:

```text
lerobot-train
```

CLI rather than implementing a custom trainer.

---

## LoRA merge

The training output is not considered final until the LoRA adapter has been merged into a standalone SmolVLA policy.

Use the appropriate SmolVLA / PEFT APIs.

The saved model must preserve all configuration and processor files required by `lerobot-eval`.

The final model must be loadable independently from the training checkpoint.

---

## Evaluation

Only the final finetuned model is required.

Do not reproduce the notebook's intermediate baseline-vs-finetuned Spatial comparison.

Final evaluation suites:

```text
libero_spatial
libero_object
libero_goal
libero_10
```

Default selected task IDs:

```text
[0, 4, 8]
```

Default number of episodes:

```text
1 episode per task
```

The episode count must remain configurable.

Use the notebook's evaluation configuration:

```text
device=cuda
use_amp=false
is_libero_plus=true
observation size=256x256
control_mode=relative
max_parallel_tasks=1
batch_size=1
async environments=false
```

Camera mapping:

```text
agentview_image -> front
robot0_eye_in_hand_image -> wrist
```

---

## Videos

Produce one final video for task ID 0 of every suite:

```text
libero_spatial -> <RUN_ID>_spatial.mp4
libero_object  -> <RUN_ID>_object.mp4
libero_goal    -> <RUN_ID>_goal.mp4
libero_10      -> <RUN_ID>_libero10.mp4
```

Avoid additional simulation solely for video generation if a usable rollout video from evaluation already exists.
