# Artifact Contract

## 目的

このドキュメントは、パイプラインを1回実行するごとに生成しなければならないファイルを定義します。

アーティファクトは再現可能であり、適切なものについては機械可読であり、1つのパイプライン `RUN_ID` に追跡可能でなければなりません。

---

## RUN_ID

`submit_pipeline.sh` を1回実行するたびに、正確に1つの以下を生成します。

```text id="yjc1le"
RUN_ID
```

また、正確に1つの以下を生成します。

```text id="ur9yau"
timestamp
```

timestamp は JST（UTC+09:00）で生成し、UTC オフセットを含む
`YYYYMMDDTHHMMSS+0900` 形式とします。

run directory は以下です。

```text id="0f5afh"
data/outputs/<timestamp>-<RUN_ID>/
```

同じ `RUN_ID` を以下すべてで使用してください。

```text id="n2e8vj"
preprocess
train
test
```

Slurm job ID は、`RUN_ID` とは別の識別子です。

---

## 必須の run 構造

```text id="bsc528"
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

LeRobot が生成する生の evaluation output は、各 suite directory 内に保持して構いません。

---

## run.json

パイプライン投入時に書き込みます。

最低限、以下を含めてください。

```json id="shf6gc"
{
  "run_id": "...",
  "timestamp": "...",
  "run_dir": "...",
  "preprocess_job_id": null,
  "train_job_id": null,
  "test_job_id": null
}
```

ジョブ投入後に job ID を更新してください。

利用可能な場合は、environment / revision に関する情報も記録してください。

---

## preprocess.json

最低限、以下を含める必要があります。

```text id="2d0gs6"
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

train stage は、この永続化された episode 選択結果を使用しなければなりません。

---

## train.json

最低限、以下を含める必要があります。

```text id="arkjcv"
run_id
training_started_at
training_finished_at
training_exit_code
checkpoint_path
merged_model_path
model_archive_path
parameters_csv_path
```

追跡可能性を確保するため、実際に使用した training command、または構造化された command arguments も含めてください。

---

## test.json

最低限、以下を含める必要があります。

```text id="qaqdy6"
run_id
evaluation_started_at
evaluation_finished_at
suites
episodes_per_task
results_csv_path
video_paths
```

実際に使用した evaluation configuration も含めてください。

---

## Final model

必須の model directory:

```text id="xumvqw"
model/<RUN_ID>_smolvla/
```

これは、standalone な merged policy artifact でなければなりません。

後から evaluation や再利用のために policy を読み込む際に必要となるファイルを、すべて含んでいなければなりません。

この directory から以下を作成してください。

```text id="ljyk53"
<RUN_ID>_model.tar.gz
```

この archive は、元の LoRA training checkpoint がなくても使用可能でなければなりません。

---

## Parameter CSV

必須:

```text id="vb4izw"
<RUN_ID>_parameters.csv
```

推奨形式:

```csv id="teom0p"
run_id,parameter,value
abc123,steps,3000
abc123,batch_size,1
abc123,learning_rate,0.0003
abc123,final_learning_rate,0.00003
abc123,warmup_steps,100
abc123,lora_r,16
abc123,lora_alpha,16
```

さらに、以下のような provenance 情報も含めてください。

```text id="659gxq"
seed
base_model_repo
base_model_revision
vlm_repo
resolved_vlm_revision
dataset_repo
dataset_revision
training_episode_count
```

正確な schema は、以下のどちらでも構いません。

```text id="l52ls1"
parameter ごとに1行
```

または、

```text id="cjjioa"
experiment ごとに1行
```

ただし、決定論的であり、コード上で文書化されていなければなりません。

---

## Evaluation CSV

必須の evaluation result CSV は、正確に1つだけ生成してください。

```text id="hbju3n"
<RUN_ID>_results.csv
```

最低限必要な column:

```text id="bfs8pi"
run_id
suite
task_id
level
n_trials
n_success
success_rate
```

### Task row

例:

```text id="k48qei"
run_id = abc123
suite = libero_spatial
task_id = 0
level = task
n_trials = 5
n_success = 3
success_rate = 0.6
```

### Suite aggregate row

例:

```text id="3x0zva"
run_id = abc123
suite = libero_spatial
task_id =
level = suite
n_trials = 15
n_success = 9
success_rate = 0.6
```

suite 全体の集計値は、実際の各 episode の成功結果から計算してください。

丸められた percentage から `n_success` を逆算してはいけません。

---

## 生の evaluation provenance

LeRobot が生成する以下を保持してください。

```text id="yool65"
eval_info.json
```

配置先:

```text id="vfzyvq"
eval/<suite>/
```

これらの生ファイルは、追加のユーザー向け CSV result ではありません。

debugging と reproducibility のために保持します。

---

## Videos

必須:

```text id="u9ktjf"
<RUN_ID>_spatial.mp4
<RUN_ID>_object.mp4
<RUN_ID>_goal.mp4
<RUN_ID>_libero10.mp4
```

各動画は、対応する suite の task ID 0 の rollout でなければなりません。

suite ごとに、要求される最終動画は1つだけです。

---

## 完了時の検証

必要なファイルがすべて存在し、かつ空でない場合にのみ、その run を完了とみなします。

以下を検証してください。

```text id="faux8z"
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

さらに、以下も検証してください。

* manifest とユーザー向けファイル名の間で `RUN_ID` が一致していること
* result CSV に4つすべての suite が含まれていること
* trial 数が 0 より大きいこと
* success 数が 0 以上かつ trial 数以下であること
* success rate が `n_success / n_trials` と一致すること
* merged model に、後から読み込むために必要なファイルがすべて含まれていること

一部だけ完成している run を、正常完了として報告してはいけません。
