# AGENTS.md

## Project purpose

This repository converts the workflow in `final_homework_Advanced.ipynb` into a reproducible, non-interactive Python/Slurm pipeline for training and evaluating SmolVLA with LeRobot and LIBERO-plus.

The existing execution infrastructure is already prepared.

The expected pipeline is:

```text
submit_pipeline.sh
  -> build.sh, only when necessary
  -> preprocess.sh
  -> train.sh
  -> test.sh
```

Preserve this architecture.

## Primary source

`final_homework_Advanced.ipynb` is the behavioral reference for the LeRobot / LIBERO-plus experiment.

Do not modify the notebook.

When extracting logic from the notebook:

* preserve experiment semantics unless `TASK.md` explicitly changes them;
* do not blindly copy Colab-specific setup;
* separate environment/bootstrap code from experiment logic;
* prefer small testable Python functions over notebook-style global state.

## Existing environment contract

The Python environment is managed by `uv`.

The project already contains all required Python dependencies.

The Singularity image already contains required system packages.

Treat the existing environment as immutable.

### Dependency modifications are prohibited

Unless the user explicitly requests a dependency change, DO NOT run:

```text
uv add
uv remove
uv lock
pip install
pip uninstall
python -m pip install
python -m pip uninstall
apt install
apt update
conda install
```

DO NOT modify for dependency resolution:

```text
pyproject.toml
uv.lock
singularity/ubuntu24.04.def
```

If a package is missing or incompatible, stop and report:

* package/module name;
* code requiring it;
* expected version/API when known;
* installed version when detectable;
* command that failed.

Do not solve dependency errors by upgrading, downgrading, adding, or removing packages.

## Fixed package revisions

The existing project configuration is authoritative.

In particular:

```text
LeRobot: v0.6.0
LIBERO-plus: 4976dc3
Python: >=3.12
```

Do not change these revisions.

## Network contract

The login node has Internet access.

Slurm compute jobs do NOT have Internet access.

All model, dataset, and asset downloads must therefore finish BEFORE Slurm compute jobs are submitted.

Compute-stage Python must never depend on successful Internet access.

During compute stages, prefer explicit offline behavior and fail clearly when a required local artifact is missing.

Do not silently attempt network fallback from:

```text
preprocess
train
test
```

## Existing shell pipeline

Preserve:

```text
submit_pipeline.sh
slurm/build.sh
slurm/config.sh
slurm/preprocess.sh
slurm/train.sh
slurm/test.sh
```

Do not redesign the Slurm dependency chain.

Minimal edits required to select the Python stage are allowed.

The intended Python stage interface is:

```text
uv run python -m vla_simulation_project.main preprocess
uv run python -m vla_simulation_project.main train
uv run python -m vla_simulation_project.main test
```

Asset preparation runs on the login node before `sbatch`.

## Writable implementation area

Normal implementation work should be limited primarily to:

```text
src/vla_simulation_project/**
tests/**
config/**
docs/**
```

Minimal modifications are allowed to:

```text
submit_pipeline.sh
slurm/preprocess.sh
slurm/train.sh
slurm/test.sh
```

Do not make unrelated modifications.

## Runtime identifiers

Each pipeline execution has one `RUN_ID`.

Create one timestamp at pipeline submission time.

The run directory is:

```text
data/outputs/<timestamp>-<RUN_ID>/
```

The same RUN_ID must be propagated to preprocess, train, and test jobs.

Do not independently generate a new RUN_ID inside each stage.

Record the individual Slurm job IDs separately in the run manifest.

## Paths

Resolve repository paths relative to `PROJECT`.

Do not use Colab paths such as:

```text
/content
/content/workdir
/content/drive
```

Do not require Google Drive or `google.colab`.

Persistent project data belongs under:

```text
data/models/
data/datasets/
data/assets/
data/hf_cache/
data/manifests/
data/outputs/
```

## Pipeline state

Do not depend on Python globals surviving between stages.

Each Slurm stage executes in a separate process.

Information that must cross a stage boundary must be persisted to a file.

Use machine-readable JSON manifests for pipeline state.

At minimum record:

* RUN_ID;
* timestamp;
* stage;
* Slurm job ID when available;
* model/dataset revisions;
* local artifact paths;
* selected training episodes;
* experiment configuration;
* output paths.

## Notebook-specific exclusions

Do not port these Notebook behaviors into runtime Python:

* Google Drive mounting;
* Colab APIs;
* apt installation;
* pip installation/uninstallation;
* editable package installation;
* package upgrade/downgrade;
* notebook widgets;
* IPython-only display logic;
* manual download UI.

Replace interactive visualization with normal artifact files where required.

## Training behavior

Use LeRobot's existing training interface rather than implementing a replacement training framework.

Preserve use of:

```text
lerobot-train
```

through subprocess unless there is a concrete compatibility reason not to.

Keep command construction separate from command execution so command generation is unit-testable without a GPU.

Training must produce a merged, directly loadable SmolVLA policy artifact.

## Evaluation behavior

Use:

```text
lerobot-eval
```

for LIBERO-plus evaluation.

The required suites are:

```text
libero_spatial
libero_object
libero_goal
libero_10
```

Do not add intermediate evaluation runs that are not required by `TASK.md`.

## Errors

Shell and Python failures must propagate as non-zero exit codes.

Do not catch an exception merely to continue with incomplete artifacts.

Error messages should identify:

* failing stage;
* failing command;
* relevant path;
* missing artifact or invalid state.

## Safety around filesystem operations

Be conservative with destructive operations.

Never recursively delete paths outside the current run directory or explicitly managed generated directories.

Before deleting a directory, verify that it resolves underneath an approved project data/output location.

## Validation

Prefer tests that do not require a GPU:

* configuration parsing;
* path construction;
* RUN_ID handling;
* asset-manifest validation;
* episode selection;
* train command construction;
* eval command construction;
* CSV aggregation;
* artifact validation.

GPU-heavy training/evaluation should not be run merely to validate a small code edit.

Do not run a full 3000-step training job as a routine code test.

## Source control checks

Before finishing work, inspect the diff.

Confirm that these files have not changed unless explicitly authorized:

```text
pyproject.toml
uv.lock
singularity/ubuntu24.04.def
final_homework_Advanced.ipynb
```

Report:

* files changed;
* tests/checks run;
* checks that could not be executed;
* remaining assumptions.
