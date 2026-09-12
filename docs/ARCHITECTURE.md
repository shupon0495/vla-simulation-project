# Architecture

## 目的

このドキュメントは、SmolVLA / LeRobot 実験パイプラインの実行時アーキテクチャを定義します。

動作上の参照元は以下です。

```text
final_homework_Advanced.ipynb
```

Notebook 自体は変更してはいけません。

Notebook のワークフローを、既存の Slurm、Singularity、uv 基盤を通して実行される、非対話型の Python パイプラインへ変換します。

---

## 全体アーキテクチャ

```text
Login node
  |
  | Internet available
  |
  +-- submit_pipeline.sh
        |
        +-- RUN_ID と timestamp を生成
        |
        +-- login.sif を build / 再利用
        |
        +-- login.sif 内で offline asset を準備 / 確認
        |
        +-- build.sh（compute node 上で image の鮮度と CPU architecture を確認）
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

## 実行環境の境界

実行環境は2つあります。

### Login node

インターネット接続が利用可能です。

責務:

* `RUN_ID` を生成する
* timestamp を生成する
* 必要な model / dataset / asset を確認する
* 不足している Hugging Face artifact をダウンロードする
* 計算時に必要なリソースがすべてローカルに存在することを検証する
* Slurm job を投入する

ログインノード上での準備処理は、preprocess / train / test job を投入する前に完了していなければなりません。

Asset preparation は、`singularity/login.def` から作成された専用の `singularity/login.sif` 内で実行します。

これにより、ログインノードの system Python のバージョン、および compute image の両方から独立させます。

プロジェクトの uv 環境から `huggingface_hub` を利用し、asset のダウンロード中には snapshot の進捗表示を行います。

### Compute node

インターネット接続は利用できません。

Compute stage は以下です。

```text
preprocess
train
test
```

必要なリソースは、すべて事前にローカルへ配置されている必要があります。

Compute stage のコードは、ネットワークへのフォールバックに依存してはいけません。

必要に応じて Hugging Face の offline 設定を使用してください。

---

## 実行基盤

既存のパイプラインを正とします。

```text
submit_pipeline.sh
  -> 必要な場合のみ build.sh
  -> preprocess.sh
  -> train.sh
  -> test.sh
```

Shell pipeline を再設計してはいけません。

`build.sh` は compute node 上で Singularity image を確認します。`ubuntu24.04.sif.arch`
に記録した `uname -m`、definition file の更新時刻、および image の有無を照合し、
必要な場合だけ Singularity image を作成または更新します。architecture は allocation
された compute node でのみ確定できるため、`build.sh` 自体は各 pipeline run で投入されます。

同じ build job は、`<PROJECT>/.venv-compute-<architecture>` を compute SIF 内で
offline に同期し、`torch._C` を import して native extension が完全であることを確認します。
login image の `.venv-login` と compute environment は共有しません。preprocess、train、test
は `uv run --frozen --offline --no-sync` で実行するため、compute stage が package を
install、uninstall、download することはありません。

アプリケーションは container 内で以下の形式で実行します。

```text
uv run python -m vla_simulation_project.main <stage>
```

対応する stage:

```text
preprocess
train
test
```

---

## Python アーキテクチャ

想定される module:

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

責務:

* CLI parsing
* stage dispatch

実験の実装ロジックを含めてはいけません。

想定される使用方法:

```text
python -m vla_simulation_project.main preprocess
python -m vla_simulation_project.main train
python -m vla_simulation_project.main test
```

### config.py

責務:

* `config/experiment.toml` の読み込み
* 実験パラメータの検証
* 各 stage へ型付けされた configuration を提供する

Dependency version をここで設定してはいけません。

### paths.py

責務:

* `PROJECT` を解決する
* persistent data directory を解決する
* 現在の `RUN_ID` に対応する directory を解決する
* 管理対象外の project directory への誤った書き込みを防止する

### prepare_assets.py

ログインノード上で実行します。

責務:

* 必要な offline resource を確認する
* 不足している resource のみをダウンロードする
* Hugging Face revision を解決して記録する
* Slurm job の投入前に resource を検証する

Python package をインストールしてはいけません。

### preprocess.py

責務:

* offline asset を検証する
* local dataset metadata を確認する
* training episode を選択する
* `preprocess.json` を書き出す

### train.py

責務:

* preprocessing state を読み込む
* `lerobot-train` command を構築する
* training を実行する
* final checkpoint を特定する
* LoRA merge を呼び出す
* training manifest と parameter CSV を書き出す

### merge.py

責務:

* LoRA adapter を SmolVLA に merge する
* standalone な LeRobot policy として保存する
* processor configuration / statistics を保持する
* final model artifact を検証する

### evaluate.py

責務:

* 最終的な merged model を評価する
* success を集計する
* final result CSV を生成する
* 必要な rollout video を生成する

### artifacts.py

責務:

* JSON manifest の読み書き
* CSV の生成
* archive の作成
* final artifact の検証

### subprocess_utils.py

責務:

* 安全な subprocess execution
* stdout / stderr の処理
* command failure の伝播

---

## Stage の分離

各 stage は別々の process として実行されます。

Python の global state を、以下の stage 間で引き継ぐことはできません。

```text
preprocess -> train -> test
```

代わりに、永続化された JSON file を使用してください。

想定される file:

```text
<run-dir>/manifests/run.json
<run-dir>/manifests/preprocess.json
<run-dir>/manifests/train.json
<run-dir>/manifests/test.json
```

各 stage は、以下だけを使用して再実行可能でなければなりません。

* environment variable
* configuration
* 前の stage が生成した file

---

## 設定の境界

Dependency configuration:

```text
pyproject.toml
uv.lock
```

は変更不可とします。

Experiment configuration は以下に配置します。

```text
config/experiment.toml
```

変更可能な値の例:

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

Persistent storage は repository 配下に配置します。

```text
data/
├── hf_cache/
├── models/
├── datasets/
├── assets/
├── manifests/
└── outputs/
```

各 experiment には、正確に1つの run directory を使用します。

```text
data/outputs/<timestamp>-<RUN_ID>/
```

パイプライン全体を通して、同じ `RUN_ID` を使用します。

Slurm job ID は別途記録します。

---

## Failure policy

問題が発生した場合は、早期に失敗してください。

以下を行ってはいけません。

* 不足している artifact を黙って無視する
* compute node から resource をダウンロードする
* `lerobot-train` が失敗した後も処理を続行する
* `lerobot-eval` が失敗した後も処理を続行する
* runtime error を解決するために dependency を変更する

失敗は、0 以外の process exit code として伝播させる必要があります。

---

---
