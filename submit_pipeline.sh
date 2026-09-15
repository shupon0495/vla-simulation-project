# submit_pipeline.sh
# loginノードでのHFダウンロードと計算ノードへのjob作成までを行う
#!/bin/bash
# エラーの時に止める
# 未定義の変数を触ったらエラーにする
# パイプラインの途中でエラーになったら止める
set -euo pipefail

# このソースファイルの場所に移動してからpwdしたものを受け取ることでこのソースファイルの絶対パスを手に入れる
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# もしPROJECTが何も入っていないならSCRIPT_DIRをPROJECTに入れる
export PROJECT=${PROJECT:-$SCRIPT_DIR}
# パスや環境変数を導入
source "$PROJECT/slurm/config.sh"
# uv のキャッシュをプロジェクト内に配置し、ログインノードと計算ノード間で共有する。
# 計算ノードは --offline で uv run するため、キャッシュがプロジェクトの
# バインドマウント配下にないと wheel が見つからず sdist からビルドされてしまう。
export UV_CACHE_DIR="$PROJECT/.uv-cache"
mkdir -p "$UV_CACHE_DIR"

# RUN_IDを作成して公開
RUN_ID=$(date +%s%N | sha256sum | cut -c1-10)
# Keep the run directory, manifest, and Slurm log names on Japan Standard Time.
# Include the numeric UTC offset so the timestamp remains unambiguous.
TIMESTAMP=$(TZ=Asia/Tokyo date +%Y%m%dT%H%M%S%z)
RUN_DIR="$PROJECT/data/outputs/${TIMESTAMP}-${RUN_ID}"
# 同じRUN_IDのSlurmログは log/<RUN_ID>/ にまとめて保存する
RUN_LOG_DIR="$LOG/$RUN_ID"
export RUN_ID TIMESTAMP RUN_DIR RUN_LOG_DIR
mkdir -p "$RUN_DIR/manifests" "$RUN_LOG_DIR"

>>>>>>> 476d2f52c13fc15e131624dc5ef7a4a19f6a88cc
# ログインノードの環境によらずに環境構築をするためにコンテナ内でprepare-assetsを実行するようにする
# もしSIFファイルが古い・存在しないならDEFファイルから作成する. もう既に存在するならスキップ
if [ ! -f "$LOGIN_SIF" ] || [ "$LOGIN_DEF" -nt "$LOGIN_SIF" ]; then
    echo "Building login-node asset preparation image..."
    singularity build --fakeroot --force "$LOGIN_SIF" "$LOGIN_DEF"
else
    echo "Login-node asset preparation image is up to date."
fi

# 計算ノード用 .venv は差分があるときだけ同期する
# uv sync --check は環境が lockfile と一致していれば即座に成功終了するため、
# 定常状態では CUDA を含む大きな wheel の再展開は起こらない
if ! singularity exec \
        --bind "$PROJECT:$PROJECT" \
        --pwd "$PROJECT" \
        "$LOGIN_SIF" \
        uv sync --frozen --check; then
    singularity exec \
        --bind "$PROJECT:$PROJECT" \
        --pwd "$PROJECT" \
        "$LOGIN_SIF" \
        uv sync --frozen
fi

# prepare-assets は login.sif 内蔵の最小Python環境( huggingface-hub + tqdm )で
# 直接実行し、CUDA を含むプロジェクトの .venv を読み込まない
singularity exec \
    --bind "$PROJECT:$PROJECT" \
    --pwd "$PROJECT" \
    --env "PROJECT=$PROJECT,RUN_ID=$RUN_ID,RUN_DIR=$RUN_DIR,PYTHONPATH=$PROJECT/src,HF_HOME=$PROJECT/data/hf_cache" \
    "$LOGIN_SIF" \
    python3.12 -m vla_simulation_project.main prepare-assets
printf '{\n  "run_id": "%s",\n  "timestamp": "%s",\n  "run_dir": "%s",\n  "build_job_id": null,\n  "preprocess_job_id": null,\n  "train_job_id": null,\n  "test_job_id": null\n}\n' \
    "$RUN_ID" "$TIMESTAMP" "$RUN_DIR" > "$RUN_DIR/manifests/run.json"

# もしcompute用のイメージのビルドが必要ならpreprocessに依存として追加
# 大体login用のイメージの所と同じ
BUILD_JOB=""
BUILD_DEPENDENCY=()
if [ ! -f "$SIF" ] || [ "$DEF" -nt "$SIF" ]; then
    echo "Building Singularity image..."

    BUILD_JOB=$(sbatch --parsable \
        --export="ALL,PROJECT=$PROJECT,LOG=$RUN_LOG_DIR,RUN_ID=$RUN_ID,RUN_DIR=$RUN_DIR" \
        --partition="$PPC_PARTITION" \
        --output="$RUN_LOG_DIR/build-${TIMESTAMP}-%j.out" \
        --error="$RUN_LOG_DIR/build-${TIMESTAMP}-%j.err" \
        "$PROJECT/slurm/build.sh"
    )
    BUILD_DEPENDENCY=(--dependency="afterok:$BUILD_JOB")
else
    echo "Singularity image is up to date."
fi

JOB1=$(sbatch --parsable \
    --export="ALL,PROJECT=$PROJECT,LOG=$RUN_LOG_DIR,RUN_ID=$RUN_ID,RUN_DIR=$RUN_DIR,HF_HOME=$PROJECT/data/hf_cache,HF_HUB_OFFLINE=1,HF_DATASETS_OFFLINE=1,TRANSFORMERS_OFFLINE=1" \
    --partition="$PPC_PARTITION" \
    --output="$RUN_LOG_DIR/preprocess-${TIMESTAMP}-%j.out" \
    --error="$RUN_LOG_DIR/preprocess-${TIMESTAMP}-%j.err" \
    "${BUILD_DEPENDENCY[@]}" \
    "$PROJECT/slurm/preprocess.sh"
)

JOB2=$(sbatch --parsable \
    --export="ALL,PROJECT=$PROJECT,LOG=$RUN_LOG_DIR,RUN_ID=$RUN_ID,RUN_DIR=$RUN_DIR,HF_HOME=$PROJECT/data/hf_cache,HF_HUB_OFFLINE=1,HF_DATASETS_OFFLINE=1,TRANSFORMERS_OFFLINE=1" \
    --partition="$TRAIN_PARTITION" \
    --output="$RUN_LOG_DIR/train-${TIMESTAMP}-%j.out" \
    --error="$RUN_LOG_DIR/train-${TIMESTAMP}-%j.err" \
    --dependency="afterok:$JOB1" \
    "$PROJECT/slurm/train.sh"
)

JOB3=$(sbatch --parsable \
    --export="ALL,PROJECT=$PROJECT,LOG=$RUN_LOG_DIR,RUN_ID=$RUN_ID,RUN_DIR=$RUN_DIR,HF_HOME=$PROJECT/data/hf_cache,HF_HUB_OFFLINE=1,HF_DATASETS_OFFLINE=1,TRANSFORMERS_OFFLINE=1" \
    --partition="$TEST_PARTITION" \
    --output="$RUN_LOG_DIR/test-${TIMESTAMP}-%j.out" \
    --error="$RUN_LOG_DIR/test-${TIMESTAMP}-%j.err" \
    --dependency="afterok:$JOB2" \
    "$PROJECT/slurm/test.sh"
)

BUILD_JSON=null
if [ -n "$BUILD_JOB" ]; then BUILD_JSON="\"$BUILD_JOB\""; fi
printf '{\n  "run_id": "%s",\n  "timestamp": "%s",\n  "run_dir": "%s",\n  "build_job_id": %s,\n  "preprocess_job_id": "%s",\n  "train_job_id": "%s",\n  "test_job_id": "%s"\n}\n' \
    "$RUN_ID" "$TIMESTAMP" "$RUN_DIR" "$BUILD_JSON" "$JOB1" "$JOB2" "$JOB3" \
    > "$RUN_DIR/manifests/run.json"

echo "build:      ${BUILD_JOB:-skipped}"
echo "preprocess: $JOB1"
echo "train:      $JOB2"
echo "test:       $JOB3"
