# AGENTS.md

## プロジェクトの目的

このリポジトリは、`final_homework_Advanced.ipynb` にあるワークフローを、SmolVLA を LeRobot および LIBERO-plus と組み合わせて学習・評価するための、再現可能かつ非対話型の Python / Slurm パイプラインへ変換することを目的としています。

既存の実行基盤はすでに準備されています。

想定されるパイプラインは次のとおりです。

```text
submit_pipeline.sh
  -> build.sh（必要な場合のみ）
  -> preprocess.sh
  -> train.sh
  -> test.sh
```

このアーキテクチャを維持してください。

## 主要な参照元

`final_homework_Advanced.ipynb` は、LeRobot / LIBERO-plus 実験の動作上の基準です。

Notebook は変更しないでください。

Notebook からロジックを抽出するときは、次の方針に従ってください。

* `TASK.md` に明示的な変更指示がない限り、実験の意味や条件を維持すること。
* Colab 固有のセットアップ処理をそのままコピーしないこと。
* 環境構築・ブートストラップ処理と、実験ロジックを分離すること。
* Notebook のようなグローバル状態に依存する実装より、小さくテスト可能な Python 関数を優先すること。

## 既存環境に関する契約

Python 環境は `uv` で管理されています。

このプロジェクトには、必要な Python 依存関係がすでにすべて含まれています。

Singularity イメージには、必要なシステムパッケージがすでに含まれています。

既存環境は変更不可のものとして扱ってください。

### 依存関係の変更は禁止

ユーザーが依存関係の変更を明示的に依頼しない限り、以下を実行してはいけません。

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

依存関係の解決を目的として、以下を変更してはいけません。

```text
pyproject.toml
uv.lock
singularity/ubuntu24.04.def
```

パッケージが不足している、または互換性がない場合は、処理を停止し、以下を報告してください。

* パッケージ名またはモジュール名
* そのパッケージを必要としているコード
* 分かる場合は、想定されるバージョンまたは API
* 検出可能な場合は、現在インストールされているバージョン
* 失敗したコマンド

依存関係のエラーを、パッケージのアップグレード、ダウングレード、追加、削除によって解決してはいけません。

## 固定されたパッケージリビジョン

既存のプロジェクト設定を正としてください。

特に、以下は固定されています。

```text
LeRobot: v0.6.0
LIBERO-plus: 4976dc3
Python: >=3.12
```

これらのリビジョンを変更してはいけません。

## ネットワークに関する契約

ログインノードはインターネットへ接続できます。

Slurm の計算ジョブはインターネットへ接続できません。

したがって、モデル、データセット、アセットのダウンロードは、Slurm の計算ジョブを投入する前にすべて完了させる必要があります。

計算ステージの Python コードは、インターネット接続の成功に依存してはいけません。

計算ステージでは、明示的なオフライン動作を優先し、必要なローカルアーティファクトが存在しない場合は、原因が分かる形で失敗してください。

以下の処理から、暗黙的にネットワークへフォールバックしてはいけません。

```text
preprocess
train
test
```

## 既存のシェルパイプライン

以下を維持してください。

```text
submit_pipeline.sh
slurm/build.sh
slurm/config.sh
slurm/preprocess.sh
slurm/train.sh
slurm/test.sh
```

Slurm の依存関係チェーンを再設計してはいけません。

Python の実行ステージを選択するために必要な最小限の変更は許可します。

想定される Python ステージのインターフェースは次のとおりです。

```text
uv run python -m vla_simulation_project.main preprocess
uv run python -m vla_simulation_project.main train
uv run python -m vla_simulation_project.main test
```

アセット準備は、`sbatch` を実行する前にログインノード上で行います。

## 通常変更してよい実装領域

通常の実装作業は、主として以下の範囲に限定してください。

```text
src/vla_simulation_project/**
tests/**
config/**
docs/**
```

以下については、必要最小限の変更を許可します。

```text
submit_pipeline.sh
slurm/preprocess.sh
slurm/train.sh
slurm/test.sh
```

関係のない変更を行ってはいけません。

## 実行時識別子

パイプラインの各実行には、1つの `RUN_ID` を使用します。

パイプライン投入時に、タイムスタンプを1つだけ生成してください。

実行ディレクトリは次の形式です。

```text
data/outputs/<timestamp>-<RUN_ID>/
```

同じ `RUN_ID` を preprocess、train、test の各ジョブへ引き継いでください。

各ステージの内部で、それぞれ新しい `RUN_ID` を生成してはいけません。

各 Slurm ジョブ ID は、実行マニフェスト内に個別に記録してください。

## パス

リポジトリ内のパスは `PROJECT` を基準に解決してください。

以下のような Colab 固有のパスを使用してはいけません。

```text
/content
/content/workdir
/content/drive
```

Google Drive や `google.colab` を必要としてはいけません。

永続化するプロジェクトデータは、以下に配置してください。

```text
data/models/
data/datasets/
data/assets/
data/hf_cache/
data/manifests/
data/outputs/
```

## パイプライン状態

ステージ間で Python のグローバル変数が維持されることを前提にしてはいけません。

各 Slurm ステージは別々のプロセスとして実行されます。

ステージをまたいで必要になる情報は、必ずファイルへ永続化してください。

パイプライン状態には、機械可読な JSON マニフェストを使用してください。

最低限、以下を記録してください。

* `RUN_ID`
* タイムスタンプ
* ステージ
* 取得可能な場合は Slurm ジョブ ID
* モデルおよびデータセットのリビジョン
* ローカルアーティファクトのパス
* 選択された学習エピソード
* 実験設定
* 出力パス

## Notebook 固有処理の除外

以下の Notebook 固有の処理を、実行時 Python コードへ移植してはいけません。

* Google Drive のマウント
* Colab API
* apt によるインストール
* pip によるインストール・アンインストール
* editable install
* パッケージのアップグレード・ダウングレード
* Notebook widget
* IPython 専用の表示ロジック
* 手動操作を必要とするダウンロード UI

必要な対話的可視化は、通常のアーティファクトファイル出力へ置き換えてください。

## 学習動作

独自の学習フレームワークを実装するのではなく、LeRobot の既存学習インターフェースを使用してください。

以下の使用を維持してください。

```text
lerobot-train
```

具体的な互換性上の理由がない限り、subprocess 経由で使用してください。

コマンド生成とコマンド実行は分離し、GPU がなくてもコマンド生成をユニットテストできるようにしてください。

学習後には、マージ済みで直接ロード可能な SmolVLA policy アーティファクトを生成する必要があります。

## 評価動作

LIBERO-plus の評価には以下を使用してください。

```text
lerobot-eval
```

必要な suite は以下です。

```text
libero_spatial
libero_object
libero_goal
libero_10
```

`TASK.md` で要求されていない中間評価を追加してはいけません。

## エラー処理

Shell および Python の失敗は、0 以外の終了コードとして伝播させてください。

不完全なアーティファクトのまま処理を継続する目的で、例外を単に捕捉して無視してはいけません。

エラーメッセージには、以下を特定できる情報を含めてください。

* 失敗したステージ
* 失敗したコマンド
* 関係するパス
* 不足しているアーティファクト、または不正な状態

## ファイルシステム操作に関する安全性

破壊的操作は慎重に行ってください。

現在の実行ディレクトリ、または明示的に管理対象とされている生成ディレクトリの外側を、再帰的に削除してはいけません。

ディレクトリを削除する前に、そのパスを解決し、許可されたプロジェクトのデータ領域または出力領域の配下にあることを確認してください。

## 検証

GPU を必要としないテストを優先してください。

* 設定のパース
* パス生成
* `RUN_ID` の処理
* アセットマニフェストの検証
* エピソード選択
* train コマンド生成
* eval コマンド生成
* CSV 集計
* アーティファクト検証

小さなコード変更の検証だけを目的として、GPU を大量に使う学習や評価を実行してはいけません。

通常のコードテストとして、3000 step の完全な学習ジョブを実行してはいけません。

## ソース管理上の確認

作業を終了する前に diff を確認してください。

明示的な許可がない限り、以下のファイルが変更されていないことを確認してください。

```text
pyproject.toml
uv.lock
singularity/ubuntu24.04.def
final_homework_Advanced.ipynb
```

以下を報告してください。

* 変更したファイル
* 実行したテストまたは確認
* 実行できなかった確認
* 残っている前提条件や仮定

## 詳細ドキュメント

サブシステムを変更する前に、関連する仕様書を確認してください。

* `docs/ARCHITECTURE.md` — 実行構成およびモジュール構成
* `docs/NOTEBOOK_MAPPING.md` — Notebook の各セクションと実装の対応関係
* `docs/OFFLINE_ASSETS.md` — オフラインで必要となるモデル、データセット、アセット
* `docs/ARTIFACT_CONTRACT.md` — 必須の出力ファイルおよびマニフェスト

ドキュメント間で内容が矛盾する場合は、`AGENTS.md` と `TASK.md` を優先してください。

## prepare-assetsの制約
`prepare-assets` はログインノード上で行うアセットのステージング処理であり、計算ノード実行時の処理ではありません。

その実装は、計算用 Singularity イメージの起動に依存してはいけません。

`prepare-assets` は、`singularity/login.sif` 内蔵の最小限の Python 環境（`huggingface-hub` と `tqdm`）で直接実行し、プロジェクトの `.venv`（CUDA を含む）は読み込みません。

`login.sif` のシステム Python は Debian の externally-managed 配下にあるため、イメージビルド時の `uv pip install --system` には `--break-system-packages` を付けて明示的にイメージ内へインストールしてください。イメージは隔離された使い捨て環境であり、ホストの Python には影響しません。

計算ノード用の `.venv` は、`uv sync --frozen --check` で差分があるときだけ同期することで、ログインノードでの起動を高速化しています。
