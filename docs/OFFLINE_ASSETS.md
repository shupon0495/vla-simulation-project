# Offline Assets

## 目的

Slurm の計算ノードはインターネットへ接続できません。

このドキュメントでは、計算ジョブを投入する前にローカルで利用可能になっていなければならない、すべての外部アーティファクトを定義します。

アセットの準備は、インターネットへ接続可能なログインノード上で行います。

`submit_pipeline.sh` は、専用のログインノード用イメージである

```text
singularity/login.sif
```

の内部で準備処理を実行します。

ログインノードの system Python は使用せず、compute image も起動しません。

Hugging Face の repository は、`huggingface_hub.snapshot_download` を使用してローカルへ配置します。

長時間の dataset 転送中でも、submission terminal から進捗を確認できるように、file および byte 単位の progress indicator は有効なままにしてください。

---

## 基本方針

必要な処理の順序は以下です。

```text
submit_pipeline.sh
        |
        v
ローカルアセットを確認
        |
        +-- すべて揃っている --> 検証
        |
        +-- 不足している --> 不足アセットをダウンロード
                                  |
                                  v
                                検証
                                  |
                                  v
                          Slurm job を投入
```

計算ジョブは、インターネット接続に依存してはいけません。

---

## 永続化する配置先

以下を使用してください。

```text
data/
├── hf_cache/
├── models/
├── datasets/
├── assets/
└── manifests/
```

永続化するデータを `/tmp` や Notebook 形式の `/content` 配下へ保存してはいけません。

---

## 必要な外部リソース

### SmolVLA pretrained model

```text
repo:
lerobot/smolvla_libero_plus

revision:
7bb70aa5bc92b82c9239142775d3a173103567ff
```

必要な内容には、model 本体だけでなく、LeRobot の training に必要となるすべての policy processor / configuration file を含めてください。

関係のない evaluation video や repository documentation はダウンロードしないでください。

---

## SmolVLM backbone

```text
repo:
HuggingFaceTB/SmolVLM2-500M-Video-Instruct
```

元の Notebook では、commit revision は指定されていません。

最初に正常な preparation が完了したとき、実際に使用した Hugging Face snapshot の具体的な revision を解決してください。

後の run でも同じ model を再現できるように、解決した revision を記録してください。

---

## LeRobot dataset

```text
repo:
lerobot/libero_plus

revision:
f3f49f426d75030177b18778374005bc12ccd588
```

ローカルに配置された dataset は、以下すべてに必要な内容を含んでいなければなりません。

* metadata の確認
* episode の選択
* training 用 video / data の decode

後の training で未キャッシュのファイルが必要になる場合、metadata だけをダウンロードした状態では不十分です。

---

## LIBERO-plus assets

evaluation に必要なすべての LIBERO-plus environment asset は、`test` 実行前にローカルに存在していなければなりません。

evaluation 中にインターネットから取得しようとしてはいけません。

---

## LIBERO-plus benchmark resources

LIBERO-plus の Python package 自体は、`uv.lock` によって管理されたままとします。

一方、benchmark resource は別途配置します。

これは、package wheel に evaluation で必要となる Python 以外の directory が含まれていない可能性があるためです。

インターネットへ接続可能なログインノード上で、`prepare-assets` は以下を clone します。

```text
https://github.com/sylvestf/LIBERO-plus.git

4976dc30028e805ff8094b55501d532c48fec182
```

配置先:

```text
data/assets/libero_plus/source/
```

その後、以下を行います。

* `git rev-parse HEAD` の結果が上記の完全な commit hash と完全一致することを検証する
* `libero/libero/bddl_files` が存在することを検証する
* `libero/libero/init_files` が存在することを検証する
* source tree 内の従来の `assets` path を、別途配置済みの external assets へ link する

Compute stage では、このローカル tree を確認するだけにしてください。

Compute node 上では、clone、fetch、install を行ってはいけません。

## Source packages

以下は `prepare_assets.py` ではダウンロードしません。

```text
LeRobot v0.6.0
LIBERO-plus Python package 4976dc3
```

これらは、既存の uv environment によって提供されます。

pipeline 実行中に clone や再インストールを行ってはいけません。

---

## Asset lock file

以下を使用してください。

```text
data/manifests/assets.lock.json
```

推奨形式:

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
  },
  "libero_source": {
    "repo": "https://github.com/sylvestf/LIBERO-plus.git",
    "revision": "4976dc30028e805ff8094b55501d532c48fec182",
    "local_path": "data/assets/libero_plus/source"
  }
}
```

可能な場合、path は project-relative にしてください。

---

## 検証要件

Slurm job を投入する前に、最低限以下を確認してください。

* 必要な directory が存在すること
* model configuration が存在すること
* model weight が存在すること
* processor configuration / statistics が存在すること
* dataset metadata が存在すること
* 必要な training dataset file が存在すること
* VLM の weight、configuration、tokenizer file が存在すること
* pretrained policy processor の `tokenizer_name` が、ローカルに配置された VLM path を指していること
* LIBERO asset が存在すること
* LIBERO-plus benchmark source が lock された commit で存在し、BDDL および init-state directory が存在すること
* lock manifest が、想定されている固定 revision と一致すること

空の directory を、有効な cached artifact とみなしてはいけません。

---

## Download policy

不足している resource だけをダウンロードしてください。

すでに有効な artifact を繰り返しダウンロードしてはいけません。

ダウンロードに失敗した場合、自動的に別の revision へ切り替えてはいけません。

固定された revision を準備できない場合は、Slurm job を投入する前に処理を停止してください。

---

## Compute node の offline configuration

Compute job では、対応している場合、明示的な offline 動作を有効にしてください。

想定される environment configuration:

```text
HF_HOME=<PROJECT>/data/hf_cache
HF_HUB_OFFLINE=1
HF_DATASETS_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

それでもネットワークへ接続しようとする library が存在する場合、無期限に待機するのではなく失敗するようにしてください。

---

## Compute node 上で resource が不足している場合

以下のいずれかの stage で resource が不足している場合、

```text
preprocess
train
test
```

その stage は失敗し、以下を特定できる message を表示してください。

* 不足している resource
* 想定される local path
* 関連する repository / revision
* ログインノードから asset preparation を実行する必要があること

Compute stage のコードは、resource をダウンロードして問題を修復しようとしてはいけません。

---
