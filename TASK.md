# TASK.md

## 目的

`final_homework_Advanced.ipynb` にある実験を、既存の以下のパイプラインへ移植してください。

```text
submit_pipeline.sh
  -> 必要な場合のみ build
  -> preprocess
  -> train
  -> test
```

依存関係の環境は変更しないでください。

最終的なパイプラインは、既存の Singularity + uv 環境内で、非対話的に実行できる必要があります。

## 対象外

Notebook 固有の表示処理や環境構築処理は再現しないでください。

以下は明示的に対象外です。

* Colab のセットアップ
* Google Drive
* apt によるインストール
* pip / uv による依存関係の変更
* Jupyter widget
* 表示目的だけの進捗 UI
* 中間的な目視確認
* 中間段階の Spatial baseline と finetuned の比較評価
* Notebook のダウンロードボタン

Notebook に存在するという理由だけでコードを再現してはいけません。

必要な最終アーティファクトの生成に寄与しない処理は移植しないでください。

---

## ソースのリビジョン

Notebook で定義されている以下の実験用ソースを使用してください。

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

元の Notebook では、この VLM は特定の commit に固定されていません。

ログインノード上でアセットを準備するときに、実際にダウンロードした snapshot の具体的な revision を解決し、以下へ記録してください。

```text
data/manifests/assets.lock.json
```

以降の計算ジョブでは、その同じローカル snapshot を再利用してください。

### Dataset

```text
repo:
lerobot/libero_plus

revision:
f3f49f426d75030177b18778374005bc12ccd588
```

### ソースパッケージ

既存のプロジェクト環境を使用してください。

```text
LeRobot v0.6.0
LIBERO-plus 4976dc3
```

実行時にこれらのパッケージを clone または再インストールしてはいけません。

---

## Stage 0: submit / prepare assets

アセット準備は、Slurm の計算ジョブを投入する前に、インターネットへ接続可能なログインノード上で行います。

`submit_pipeline.sh` は以下を行う必要があります。

1. 1つの `RUN_ID` を作成する
2. 1つのタイムスタンプを作成する
3. 実行出力ディレクトリを決定する
4. 必要なオフラインアセットをすべて確認する
5. 不足しているアセットだけをダウンロードする
6. 必要なアセットがすべてローカルに存在することを検証する
7. asset lock manifest を書き込み、または更新する
8. 検証が成功した後にのみ Slurm ジョブを投入する

準備に失敗した場合、preprocess / train / test の各ジョブを投入してはいけません。

準備中に Python の依存関係を変更してはいけません。

### 必要なオフラインアセット

最低限、以下に必要なアーティファクトをすべて特定し、準備してください。

* pretrained SmolVLA policy
* SmolVLM2 VLM backbone
* `lerobot/libero_plus` の dataset metadata および学習データ
* LIBERO-plus simulation assets
* train / eval コマンドが実行時に Hugging Face から取得しようとする可能性のあるその他のファイル

必要に応じて、以下を使用してください。

```text
data/hf_cache/
data/models/
data/datasets/
data/assets/
```

計算ジョブは、ネットワークが利用できない状態で動作する必要があります。

計算ステージでは、必要に応じてオフライン用の環境変数を設定し、誤ってインターネットへフォールバックしようとした場合に、ハングするのではなく即座に失敗するようにしてください。

---

## RUN_ID

パイプラインを1回実行するたびに、短く不透明な `RUN_ID` を1つ生成してください。

パスに使用可能な UTC またはローカルタイムのタイムスタンプ文字列を1つ使用してください。

以下を作成します。

```text
data/outputs/<timestamp>-<RUN_ID>/
```

`RUN_ID` と出力パスを `sbatch --export` によって渡し、すべてのステージが同じ run を参照するようにしてください。

Slurm job ID は `RUN_ID` とは別のものです。

Slurm job ID は run manifest に記録してください。

ユーザー向けの最終アーティファクトのファイル名には、すべて `RUN_ID` を含めてください。

---

## Stage 1: preprocess

エントリーポイント:

```text
uv run python -m vla_simulation_project.main preprocess
```

責務:

1. 計算ノード上で必要なオフラインアセットが利用可能か確認する
2. 想定している source / data revision を確認する
3. ローカルの dataset metadata を読み込む
4. LIBERO-Spatial の学習用 episode を特定する
5. 学習 episode を決定論的に選択する
6. preprocessing manifest を書き出す

Notebook の学習データ選択条件を使用してください。

```text
10 LIBERO-Spatial tasks
x 5 episodes per task
= 50 training episodes
```

Notebook の task name normalization と、決定論的かつ等間隔な episode 選択の動作を維持してください。

出力:

```text
<run-dir>/manifests/preprocess.json
```

この manifest には、Notebook のグローバル変数を使用せず、新しいプロセスから train を実行するために十分な情報を含めてください。

---

## Stage 2: train

エントリーポイント:

```text
uv run python -m vla_simulation_project.main train
```

preprocess manifest を読み込んでください。

`lerobot-train` を使用してください。

Notebook のデフォルトハイパーパラメータ:

```text
steps                = 3000
batch_size           = 1
learning_rate        = 3e-4
final_learning_rate  = 3e-5
warmup_steps         = 100
lora_r               = 16
lora_alpha           = 16
log_freq             = 100
```

以下を含む Notebook の動作を維持してください。

```text
freeze_vision_encoder = true
train_expert_only     = true
use_imagenet_stats    = false
video_backend         = torchcodec
num_workers           = 0
persistent_workers    = false
wandb                 = disabled
```

実験パラメータは `pyproject.toml` とは独立して設定可能にしてください。

変更可能な実験設定には以下を使用してください。

```text
config/experiment.toml
```

### Model merge

LoRA 学習が正常に完了した後、以下を行ってください。

1. 最終 checkpoint を見つける
2. SmolVLA の base policy を読み込む
3. LoRA adapter を取り付ける
4. `merge_and_unload` でマージする
5. 適切な LeRobot policy の `save_pretrained` API を使用して保存する
6. 必要な preprocessor / postprocessor の統計情報を保持する
7. 生成されたモデルが完全な policy artifact として認識できることを確認する
8. 該当する場合、マージ済み weight artifact に LoRA adapter の weight が残っていないことを確認する

`test` で評価するモデルは、このマージ済みモデルです。

独立した中間 baseline model は、必須の最終アーティファクトではありません。

### Train outputs

最低限、以下を生成してください。

```text
<run-dir>/model/<RUN_ID>_smolvla/
<run-dir>/<RUN_ID>_model.tar.gz
<run-dir>/<RUN_ID>_parameters.csv
<run-dir>/manifests/train.json
```

圧縮されていないモデルディレクトリは、アーカイブに加えて保持しても構いません。

---

## Parameter CSV

run ごとに、parameter CSV を正確に1つ生成してください。

```text
<RUN_ID>_parameters.csv
```

最低限、以下を含めてください。

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

その他の関連パラメータを追加しても構いません。

---

## Stage 3: test

エントリーポイント:

```text
uv run python -m vla_simulation_project.main test
```

マージ済みの finetuned model を評価してください。

Notebook で行われている、それ以前の中間 Spatial comparison は実行しないでください。

必要な最終 suite:

```text
libero_spatial
libero_object
libero_goal
libero_10
```

Notebook のデフォルト task ID:

```text
[0, 4, 8]
```

を各 suite で使用してください。

task ごとのデフォルト episode 数:

```text
1
```

この値は設定可能にし、将来 source code を変更することなく trial 数を増やせるようにしてください。

固定されている LeRobot のバージョンと互換性がない場合を除き、Notebook の以下の評価設定を使用してください。

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

Notebook の camera mapping を使用してください。

```text
agentview_image -> front
robot0_eye_in_hand_image -> wrist
```

---

## Evaluation CSV

必須の evaluation results CSV は、1つだけ生成してください。

```text
<run-dir>/<RUN_ID>_results.csv
```

この CSV には、task 単位の詳細結果と suite 単位の集計結果の両方を含める必要があります。

最低限、以下のフィールドをサポートしてください。

```text
run_id
suite
task_id
level
n_trials
n_success
success_rate
```

各 task の行では、

```text
level=task
```

を使用してください。

suite 集計行では、

```text
level=suite
```

を使用してください。

suite 行では `task_id` は空欄でも構いません。

この構成により、詳細な task 結果と、要求される suite success rate の両方を1つの CSV に含め、2つ目の summary CSV を作らないようにしてください。

成功数は、丸められた成功率から逆算してはいけません。

実際の `eval_info.json` に含まれる episode success 値から計算してください。

ユーザー向けの追加 CSV にはしませんが、追跡可能性のため、LeRobot の生の `eval_info.json` 出力は run directory 内に保持してください。

---

## Rollout videos

以下の各 suite について、

```text
libero_spatial
libero_object
libero_goal
libero_10
```

最初の task、つまり task ID 0 を録画してください。

suite ごとに、要求される動画を正確に1つ生成してください。

最終ファイル名:

```text
<RUN_ID>_spatial.mp4
<RUN_ID>_object.mp4
<RUN_ID>_goal.mp4
<RUN_ID>_libero10.mp4
```

すべての evaluation task に対して動画を作成してはいけません。

不要な重複 simulation を実行するのではなく、可能な場合は evaluation rollout の動画を再利用してください。

---

## 最終 run layout

目標となる構造:

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

ユーザー向けの最終アーティファクトのファイル名には、すべて `RUN_ID` を含めてください。

---

## Shell integration

既存の依存順序を維持してください。

```text
optional build
     |
preprocess
     |
train
     |
test
```

Shell に対する変更は最小限にしてください。

各 stage のエントリーポイントを以下に変更してください。

```text
preprocess.sh:
uv run python -m vla_simulation_project.main preprocess

train.sh:
uv run python -m vla_simulation_project.main train

test.sh:
uv run python -m vla_simulation_project.main test
```

`submit_pipeline.sh` は追加で以下を担当します。

* `RUN_ID` / timestamp の生成
* ログインノード上での offline asset preparation
* run metadata の export
* 既存の sbatch dependency に基づくジョブ投入

このタスクの一部として Slurm の time limit を変更してはいけません。

---

## Python architecture

Notebook から抽出したコードをすべて `main.py` に入れるのではなく、小さな module に分割してください。

想定される責務:

```text
main.py
    CLI / stage dispatcher のみ

config.py
    experiment configuration

paths.py
    PROJECT / data / run path resolution

prepare_assets.py
    login node 上での local cache validation / download

preprocess.py
    dataset metadata + training episode selection

train.py
    lerobot-train command の生成 / 実行

merge.py
    LoRA merge と model artifact validation

evaluate.py
    lerobot-eval + result aggregation + video handling

artifacts.py
    manifests, CSV, archive validation

subprocess_utils.py
    subprocess execution / error handling
```

より自然な module boundary がある場合、Notebook の cell 境界をそのまま維持する必要はありません。

---

## Tests

最低限、以下について GPU を必要としないテストを追加してください。

1. `RUN_ID` / run directory path の構築
2. asset manifest validation
3. asset 不足時の failure
4. task name normalization
5. deterministic episode selection
6. training command generation
7. evaluation command generation
8. 代表的な `eval_info.json` からの results CSV aggregation
9. final artifact contract validation
10. dependency files が変更されていないこと

テストでは以下を行ってはいけません。

* model をダウンロードする
* dependency を変更する
* Internet を必要とする
* 完全な training run を開始する
* 明示的に integration test として指定されていない限り GPU を必要とする

---

## Acceptance criteria

以下のすべてを満たした場合、このタスクは完了です。

### Repository integrity

明示的に必要とされない限り、以下は変更しないでください。

```text
final_homework_Advanced.ipynb
pyproject.toml
uv.lock
singularity/ubuntu24.04.def
```

### Offline preparation

ログインノード上で、必要なアセットが不足している場合、それらを `sbatch` の前に検出してダウンロードできること。

必要なアセットがすべて存在する場合、再ダウンロードしないこと。

### Compute isolation

ジョブ投入後、preprocess / train / test は Internet 接続を必要としないこと。

必要なアセットが不足している場合、ネットワークアクセスを試みるのではなく、原因が分かる形で失敗すること。

### Stage isolation

以下の各 stage が、

```text
preprocess
train
test
```

前の stage から永続化された manifest を使用し、新しい Python process から開始できること。

### Training

train stage が `lerobot-train` を呼び出し、LoRA checkpoint を生成し、LoRA を standalone SmolVLA model に merge し、model archive と parameter CSV を生成すること。

### Evaluation

test stage が以下を評価すること。

```text
4 suites x 3 tasks x configured episodes
```

また、trial 数、成功数、成功率を含む詳細な results CSV を1つ生成すること。

### Video

4つの suite それぞれについて、task 0 の動画が1つ存在すること。

### Traceability

同じ `RUN_ID` が以下すべてに存在すること。

* run directory
* final artifact filenames
* parameter CSV
* result CSV
* manifests

Slurm job ID は別途記録すること。

### Validation report

完了時に以下を報告してください。

* 変更したファイル
* 実行したテスト
* テスト結果
* 実際の Slurm / GPU 実行が必要なため実行できなかった検証
* 固定された環境内で見つかった未解決の互換性問題

LIBERO-plus のバージョンおよび Git revision を変更してはいけません。

LIBERO-plus の Python dependency は、`uv.lock` に記録されている revision に固定されたままにしてください。

ただし、実行時に LIBERO-plus が必要とする benchmark resource、具体的には `bddl_files`、`init_files`、およびリポジトリ内に含まれるその他のデータについては、ログインノード上で同じ固定済み Git revision から取得し、共有ストレージへ配置して構いません。

リポジトリを取得した後、checkout された revision を以下で検証してください。

```text
git rev-parse HEAD
```

計算ノードからネットワークアクセスを行ってはいけません。

外部アセットは、指定された固定 source からログインノード上で準備してください。

計算ノードでは、共有ストレージに保存されたローカルコピーのみを使用してください。
