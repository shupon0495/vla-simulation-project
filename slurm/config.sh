export DEF=$PROJECT/singularity/ubuntu24.04.def
export SIF=$PROJECT/singularity/ubuntu24.04.sif
export LOGIN_DEF=$PROJECT/singularity/login.def
export LOGIN_SIF=$PROJECT/singularity/login.sif
export LOG=$PROJECT/log
export PPC_PARTITION=ng-dgx-m2
export TRAIN_PARTITION=ng-dgx-m2
export TEST_PARTITION=ng-dgx-m2
# Keep the wheel cache under the project so it is available to every
# containerized stage.  Compute jobs may read it offline, but must never use
# it to resolve/download packages.
export UV_CACHE_DIR=${UV_CACHE_DIR:-$PROJECT/data/assets/uv_cache}
export UV_LINK_MODE=copy
