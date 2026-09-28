# Notebook Mapping

## 目的

このドキュメントは、`final_homework_Advanced.ipynb` の内容を、非対話型のプロジェクトパイプラインへどのように移植するかを記録するものです。

目的は、Notebook のすべてのセルをそのまま実行することではありません。

必要な実験の意味や条件を維持しつつ、Notebook 固有の環境構築処理や表示ロジックは除去してください。

---

## 対応表

| Notebook section | 目的                              | Pipeline 上の対応先                     | 対応方針                  |
| ---------------- | ------------------------------- | ---------------------------------- | --------------------- |
| Section 1        | Colab / runtime setup           | なし                                 | 移植しない                 |
| Section 2        | apt / system packages           | 既存 Singularity image               | 移植しない                 |
| Section 3        | LeRobot installation / patching | 既存 uv environment                  | 再インストールしない            |
| Section 4        | Hugging Face helper             | `prepare_assets.py`                | 再利用可能な asset 処理だけ抽出   |
| Section 5        | LIBERO-plus setup / assets      | `prepare_assets.py` / local assets | package は再インストールしない   |
| Section 6        | experiment constants            | `config.py` / `experiment.toml`    | 移植する                  |
| Section 7        | Spatial episode selection       | `preprocess.py`                    | 移植する                  |
| Section 8.0      | hyperparameters                 | `config/experiment.toml`           | 移植する                  |
| Section 8.1      | base model download             | `prepare_assets.py`                | offline runtime 向けに適応 |
| Section 8.2      | LoRA training                   | `train.py`                         | 移植する                  |
| Section 8.3      | LoRA merge                      | `merge.py`                         | 移植する                  |
| Section 8.4      | baseline preparation            | なし                                 | 後から必要にならない限り省略        |
| Section 8.5      | intermediate Spatial comparison | なし                                 | 省略                    |
| Section 8.6      | intermediate comparison CSV     | なし                                 | 省略                    |
| Section 8.7      | model archive                   | `artifacts.py`                     | 移植する                  |
| Section 8.8      | rollout video                   | `evaluate.py`                      | 最終 suite 動画用として再実装    |
| Section 9        | advanced evaluation             | `evaluate.py`                      | 移植する                  |

---

## 明示的に除外する Notebook の処理

以下はコピーしないでください。

```text id="04ki7p"
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

対象環境は Colab ではありません。

---

## 固定された実験ソース

### Base SmolVLA model

```text id="sv35vf"
repo:
lerobot/smolvla_libero_plus

revision:
7bb70aa5bc92b82c9239142775d3a173103567ff
```

### Dataset

```text id="f77nhn"
repo:
lerobot/libero_plus

revision:
f3f49f426d75030177b18778374005bc12ccd588
```

### VLM

```text id="wx61u8"
HuggingFaceTB/SmolVLM2-500M-Video-Instruct
```

Notebook では、VLM の commit は固定されていません。

Asset preparation の際に、実際に使用する具体的な revision を解決し、`assets.lock.json` に永続化してください。

### Python source dependencies

```text id="n5exg0"
LeRobot:
v0.6.0

LIBERO-plus:
4976dc3
```

これらは、既存の uv project configuration によってすでに管理されています。

clone や再インストールを行ってはいけません。

---

## 学習データセットの選択

Notebook の以下の動作を維持してください。

```text id="3u4jzr"
LIBERO-Spatial
10 tasks
5 episodes per task
50 episodes total
```

選択は決定論的でなければなりません。

Notebook の task-name normalization の動作も維持してください。

選択された episode ID は以下へ永続化してください。

```text id="2qn5ns"
<run-dir>/manifests/preprocess.json
```

train stage で、異なる episode 選択を再計算してはいけません。

---

## デフォルト学習パラメータ

初期デフォルト値は Notebook の以下を使用します。

```text id="y65z4k"
steps                = 3000
batch_size           = 1
learning_rate        = 3e-4
final_learning_rate  = 3e-5
warmup_steps         = 100
lora_r               = 16
lora_alpha           = 16
log_freq             = 100
```

これらは実験用のデフォルト値であり、dependency version ではありません。

以下から変更可能にしてください。

```text id="b9t7mg"
config/experiment.toml
```

---

## 必須の学習動作

以下の設定を維持してください。

```text id="givt3w"
freeze_vision_encoder = true
train_expert_only = true
use_imagenet_stats = false
video_backend = torchcodec
num_workers = 0
persistent_workers = false
wandb = disabled
```

独自の trainer を実装するのではなく、既存の以下の CLI を使用してください。

```text id="hscm3p"
lerobot-train
```

---

## LoRA merge

LoRA adapter が standalone な SmolVLA policy に merge されるまでは、学習出力を最終成果物とみなしません。

適切な SmolVLA / PEFT API を使用してください。

保存された model は、`lerobot-eval` で必要となる configuration file および processor file をすべて保持している必要があります。

最終 model は、training checkpoint とは独立して読み込めなければなりません。

---

## Evaluation

必要なのは、最終的な finetuned model の評価だけです。

Notebook で行われている中間的な baseline と finetuned の Spatial 比較は再現しないでください。

最終 evaluation suite:

```text id="oz46o8"
libero_spatial
libero_object
libero_goal
libero_10
```

デフォルトで選択する task ID:

```text id="fd6ic0"
[0, 4, 8]
```

デフォルト episode 数:

```text id="2r4jtz"
1 episode per task
```

episode 数は設定可能なままにしてください。

Notebook の以下の evaluation configuration を使用してください。

```text id="77qj5e"
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

```text id="sp94tc"
agentview_image -> front
robot0_eye_in_hand_image -> wrist
```

---

## Videos

各 suite について、task ID 0 の最終動画を1つずつ生成してください。

```text id="n3i8v7"
libero_spatial -> <RUN_ID>_spatial.mp4
libero_object  -> <RUN_ID>_object.mp4
libero_goal    -> <RUN_ID>_goal.mp4
libero_10      -> <RUN_ID>_libero10.mp4
```

evaluation ですでに利用可能な rollout video が存在する場合、動画生成だけを目的として追加の simulation を実行することは避けてください。
