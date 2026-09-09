## Slurm pipeline

Singularity の fakeroot 環境で `uv` を実行し、Slurm の `afterok` 依存関係で
`preprocess → train → test` を順番に実行します。

```bash
uv sync
chmod +x slurm/*.sh
PARTITION=your_partition ./slurm/submit_pipeline.sh
```

ジョブ ID は標準出力に表示され、ログは `log/<job-name>-<job-id>.{out,err}` に保存されます。
`PROJECT`, `DATA_DIR`, `TIME_LIMIT`, `CPUS_PER_TASK`, `MEMORY` は環境変数で上書きできます。

`submit_pipeline.sh` は `singularity build --fakeroot` で
`singularity/ubuntu24.04.def` から `singularity/ubuntu24.04.sif` を作成し、
各段階を `singularity exec --fakeroot` で実行します。
`DEF`, `SIF`, `CONTAINER` は環境変数で上書きできます。

各 Slurm ジョブが呼び出す Python ファイルは次の通りです。

- `src/vla_simulation_source/preprocess.py`
- `src/vla_simulation_source/train.py`
- `src/vla_simulation_source/test.py`

各ファイルはひな形なので、`main()` 内の TODO を実際の処理へ置き換えてください。
