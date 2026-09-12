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

# The login image and the compute image must not share a virtual environment:
# native packages such as torch contain architecture-specific extensions.
LOGIN_VENV="$PROJECT/.venv-login"
STAGING_ROOT="$PROJECT/data/assets/uv_env_staging"

# ログディレクトリがないなら作成
mkdir -p "$LOG"

# RUN_IDを作成して公開
RUN_ID=$(date +%s%N | sha256sum | cut -c1-10)
# Keep the run directory, manifest, and Slurm log names on Japan Standard Time.
# Include the numeric UTC offset so the timestamp remains unambiguous.
TIMESTAMP=$(TZ=Asia/Tokyo date +%Y%m%dT%H%M%S%z)
RUN_DIR="$PROJECT/data/outputs/${TIMESTAMP}-${RUN_ID}"
export RUN_ID TIMESTAMP RUN_DIR
mkdir -p "$RUN_DIR/manifests"

# ログインノードの環境によらずに環境構築をするためにコンテナ内でprepare-assetsを実行するようにする
# もしSIFファイルが古い・存在しないならDEFファイルから作成する. もう既に存在するならスキップ
if [ ! -f "$LOGIN_SIF" ] || [ "$LOGIN_DEF" -nt "$LOGIN_SIF" ]; then
    echo "Building login-node asset preparation image..."
    singularity build --fakeroot --force "$LOGIN_SIF" "$LOGIN_DEF"
else
    echo "Login-node asset preparation image is up to date."
fi

# Asset preparation needs its own login-image environment.  Its wheel cache is
# persisted under PROJECT so build.sh can create the compute environment using
# only local files after it has been allocated on the target architecture.
mkdir -p "$UV_CACHE_DIR"
singularity exec \
    --bind "$PROJECT:$PROJECT" \
    --pwd "$PROJECT" \
    --env "UV_PROJECT_ENVIRONMENT=$LOGIN_VENV" \
    --env "UV_CACHE_DIR=$UV_CACHE_DIR" \
    "$PROJECT/singularity/login.sif" \
    uv sync --frozen

# build.sh discovers the actual architecture only after it receives a Slurm
# allocation.  Cache the locked wheels for every architecture supported by the
# configured compute partitions now, while the login node still has network
# access.  The temporary environments are never used to execute Python.
mkdir -p "$STAGING_ROOT"
for COMPUTE_PLATFORM in x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu; do
    STAGING_VENV=$(mktemp -d "$STAGING_ROOT/$COMPUTE_PLATFORM.XXXXXX")
    trap 'rm -rf "$STAGING_VENV"' EXIT
    singularity exec \
        --bind "$PROJECT:$PROJECT" \
        --pwd "$PROJECT" \
        --env "UV_PROJECT_ENVIRONMENT=$STAGING_VENV" \
        --env "UV_CACHE_DIR=$UV_CACHE_DIR" \
        --env "UV_LINK_MODE=hardlink" \
        "$LOGIN_SIF" \
        uv sync --frozen --no-install-project --python-platform "$COMPUTE_PLATFORM"
    rm -rf "$STAGING_VENV"
    trap - EXIT
done

# jobを投げる
singularity exec \
    --bind "$PROJECT:$PROJECT" \
    --pwd "$PROJECT" \
    --env "PROJECT=$PROJECT,RUN_ID=$RUN_ID,RUN_DIR=$RUN_DIR,PYTHONPATH=$PROJECT/src" \
    --env "UV_PROJECT_ENVIRONMENT=$LOGIN_VENV" \
    --env "UV_CACHE_DIR=$UV_CACHE_DIR" \
    "$LOGIN_SIF" \
    uv run --frozen --no-sync python -m vla_simulation_project.main prepare-assets
printf '{\n  "run_id": "%s",\n  "timestamp": "%s",\n  "run_dir": "%s",\n  "build_job_id": null,\n  "preprocess_job_id": null,\n  "train_job_id": null,\n  "test_job_id": null\n}\n' \
    "$RUN_ID" "$TIMESTAMP" "$RUN_DIR" > "$RUN_DIR/manifests/run.json"

# Compute node architecture must be determined on the allocated node, rather
# than on the login node.  build.sh is therefore submitted for every run, but
# only calls `singularity build` when the definition or target architecture
# requires a new image.
echo "Checking compute Singularity image on a compute node..."
BUILD_JOB=$(sbatch --parsable \
    --export="ALL,PROJECT=$PROJECT,LOG=$LOG,RUN_ID=$RUN_ID,RUN_DIR=$RUN_DIR" \
    --partition="$PPC_PARTITION" \
    --output="$LOG/build-${TIMESTAMP}-%j.out" \
    --error="$LOG/build-${TIMESTAMP}-%j.err" \
    "$PROJECT/slurm/build.sh"
)
BUILD_DEPENDENCY=(--dependency="afterok:$BUILD_JOB")

JOB1=$(sbatch --parsable \
    --export="ALL,PROJECT=$PROJECT,LOG=$LOG,RUN_ID=$RUN_ID,RUN_DIR=$RUN_DIR,HF_HOME=$PROJECT/data/hf_cache,HF_HUB_OFFLINE=1,HF_DATASETS_OFFLINE=1,TRANSFORMERS_OFFLINE=1" \
    --partition="$PPC_PARTITION" \
    --output="$LOG/preprocess-${TIMESTAMP}-%j.out" \
    --error="$LOG/preprocess-${TIMESTAMP}-%j.err" \
    "${BUILD_DEPENDENCY[@]}" \
    "$PROJECT/slurm/preprocess.sh"
)

JOB2=$(sbatch --parsable \
    --export="ALL,PROJECT=$PROJECT,LOG=$LOG,RUN_ID=$RUN_ID,RUN_DIR=$RUN_DIR,HF_HOME=$PROJECT/data/hf_cache,HF_HUB_OFFLINE=1,HF_DATASETS_OFFLINE=1,TRANSFORMERS_OFFLINE=1" \
    --partition="$TRAIN_PARTITION" \
    --output="$LOG/train-${TIMESTAMP}-%j.out" \
    --error="$LOG/train-${TIMESTAMP}-%j.err" \
    --dependency="afterok:$JOB1" \
    "$PROJECT/slurm/train.sh"
)

JOB3=$(sbatch --parsable \
    --export="ALL,PROJECT=$PROJECT,LOG=$LOG,RUN_ID=$RUN_ID,RUN_DIR=$RUN_DIR,HF_HOME=$PROJECT/data/hf_cache,HF_HUB_OFFLINE=1,HF_DATASETS_OFFLINE=1,TRANSFORMERS_OFFLINE=1" \
    --partition="$TEST_PARTITION" \
    --output="$LOG/test-${TIMESTAMP}-%j.out" \
    --error="$LOG/test-${TIMESTAMP}-%j.err" \
    --dependency="afterok:$JOB2" \
    "$PROJECT/slurm/test.sh"
)

printf '{\n  "run_id": "%s",\n  "timestamp": "%s",\n  "run_dir": "%s",\n  "build_job_id": %s,\n  "preprocess_job_id": "%s",\n  "train_job_id": "%s",\n  "test_job_id": "%s"\n}\n' \
    "$RUN_ID" "$TIMESTAMP" "$RUN_DIR" "\"$BUILD_JOB\"" "$JOB1" "$JOB2" "$JOB3" \
    > "$RUN_DIR/manifests/run.json"

echo "build:      $BUILD_JOB"
echo "preprocess: $JOB1"
echo "train:      $JOB2"
echo "test:       $JOB3"
