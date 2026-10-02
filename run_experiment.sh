#!/bin/bash
set -e

# ========================= Configuration =========================
# Where the model weights live.  Point MODEL_ROOT at a directory holding the
# three model folders below, or replace the entries with Hugging Face model
# ids (e.g. "mistralai/Mistral-7B-Instruct-v0.2") to download on first use.
MODEL_ROOT=${MODEL_ROOT:-$HOME/Models}
MODEL_LIST=(
    "$MODEL_ROOT/mistral-7b-instruct"
    "$MODEL_ROOT/qwen2.5-7b-instruct"
    "$MODEL_ROOT/llama-2-13b"
)
TASKS=(
    boolq
    openbookqa
    hellaswag
    winogrande
    xsum
)
RUN_BASELINE=${RUN_BASELINE:-1}  # 1=运行baseline, 0=跳过baseline
RUN_COMPRESSED=${RUN_COMPRESSED:-1}  # 1=运行compressed, 0=跳过compressed
SVD_METHOD=${SVD_METHOD:-shared_basis}  # 可选: shared_basis 或 independent
STRATEGY_MODE=${STRATEGY_MODE:-adaptive}  # 可选: adaptive 或 uniform
UNIFORM_TAU=${UNIFORM_TAU:-0.9}  # uniform 模式的 tau 值
LAYER_LEVEL=${LAYER_LEVEL:-}  # tau按head计算; 留空=per-head(论文默认), mean/median/max=将同layer的tau聚合后共享
REUSE_TAU=${REUSE_TAU:-0}  # 1=tau文件已存在则跳过重新生成(只查存在性,不校验样本覆盖); 0=每次重算(默认)
NO_DETAILS=${NO_DETAILS:-0}  # 1=不保存per-head明细(大任务防OOM); 0=保存(默认,小任务ablation用)
DECOMPOSITION_BACKEND=${DECOMPOSITION_BACKEND:-batched_eigh}  # batched_eigh=整层批量eigh(默认,论文数据出自它); reference_svd=逐head SVD参考实现,等价但慢
CACHE_BACKEND=${CACHE_BACKEND:-factored}  # factored=cache只存因子、attention直接吃因子不重构K/V(默认,须配 shared_basis+batched_eigh); dense=重构K̂/V̂存入cache
EIGH_COMPUTE_DEVICE=${EIGH_COMPUTE_DEVICE:-cpu}  # cpu=小Gram矩阵搬CPU做eigh(默认,本机空闲时约快10x,但对CPU争用敏感); input=留在张量所在设备,较慢但稳定
VALUE_RESIDUAL_BITS=${VALUE_RESIDUAL_BITS:-1}  # 主方法默认对V residual使用1-bit QJL；0=ablation
QJL_SEED=${QJL_SEED:-0}  # QJL随机投影的种子,任意整数,数值本身无含义;论文所有结果用0。换成别的值=换一组随机投影,结果另存(文件名带_seedN),用来验证精度不依赖某一组投影

# ========================= Main pipeline =========================
main() {
    SAMPLES=${1:-100}  # 每个任务取多少样本,按原序取前N条: 整数N(默认100); all=整个dev集
    shift || true

    prepare_args "$@"
    validate_config

    for CUR_MODEL in "${MODEL_LIST[@]}"; do
        MODEL_NAME=$(basename "$CUR_MODEL")
        echo "==> Using model: $CUR_MODEL"

        for TASK in "${TASKS[@]}"; do
            echo "==> Running pipeline for task: $TASK ($SAMPLES samples)"

            build_run_args "$MODEL_NAME" "$TASK"
            generate_tau_if_needed "$CUR_MODEL" "$MODEL_NAME" "$TASK"

            python experiment.py \
                "${STRATEGY_ARG[@]}" \
                "${TAU_ARG[@]}" \
                --task "$TASK" \
                "${SAMPLE_ARG_EXP[@]}" \
                --svd-method "$SVD_METHOD" \
                --decomposition-backend "$DECOMPOSITION_BACKEND" \
                --cache-backend "$CACHE_BACKEND" \
                --eigh-compute-device "$EIGH_COMPUTE_DEVICE" \
                --value-residual-bits "$VALUE_RESIDUAL_BITS" \
                --qjl-seed "$QJL_SEED" \
                "${NAME_ARG[@]}" \
                "${BASELINE_ARG[@]}" \
                "${COMPRESSED_ARG[@]}" \
                "${DETAILS_ARG[@]}" \
                --model "$CUR_MODEL" \
                --output-dir "$OUTPUT_DIR"

            echo "==> Finished task: $TASK"
            echo
        done
    done

    echo "==> All tasks done!"
}

# =========================== Helpers =============================
# 每个 task 开始时无条件重算：结果文件名、strategy 参数、输出目录。
# 全部先清空再赋值，避免继承上一轮。
build_run_args() {
    local model_name=$1 task=$2

    STRATEGY_ARG=()
    TAU_ARG=()
    NAME_ARG=()
    OUTPUT_DIR=""

    LEVEL_SUFFIX=""
    if [ "$STRATEGY_MODE" = "adaptive" ]; then
        LEVEL_SUFFIX="${LAYER_LEVEL:+_$LAYER_LEVEL}"
    fi

    BACKEND_SUFFIX=""
    if [ "$DECOMPOSITION_BACKEND" != "batched_eigh" ]; then
        BACKEND_SUFFIX="_${DECOMPOSITION_BACKEND}"
    elif [ "$EIGH_COMPUTE_DEVICE" != "cpu" ]; then
        BACKEND_SUFFIX="_eigh_${EIGH_COMPUTE_DEVICE}"
    fi
    if [ "$CACHE_BACKEND" != "factored" ]; then
        BACKEND_SUFFIX="${BACKEND_SUFFIX}_${CACHE_BACKEND}"
    fi

    EXPERIMENT_NAME="${STRATEGY_MODE}_${task}"
    if [ "$VALUE_RESIDUAL_BITS" != "0" ]; then
        EXPERIMENT_NAME="${EXPERIMENT_NAME}_qjl${VALUE_RESIDUAL_BITS}_seed${QJL_SEED}"
    elif [ "$SVD_METHOD" != "shared_basis" ]; then
        EXPERIMENT_NAME="${EXPERIMENT_NAME}_${SVD_METHOD}"
    fi
    NAME_ARG=(--name "${EXPERIMENT_NAME}${LEVEL_SUFFIX}${BACKEND_SUFFIX}")

    if [ "$STRATEGY_MODE" = "adaptive" ]; then
        STRATEGY_ARG=(--mode adaptive)
        OUTPUT_DIR="results/post_prefill/${model_name}"
    elif [ "$STRATEGY_MODE" = "uniform" ]; then
        STRATEGY_ARG=(--mode uniform --uniform_tau "$UNIFORM_TAU")
        OUTPUT_DIR="results/post_prefill/${model_name}/uniform_tau${UNIFORM_TAU}"
    else
        echo "Error: Unknown STRATEGY_MODE: $STRATEGY_MODE"
        exit 1
    fi
}

# adaptive 模式下先跑一遍 tau 生成，把 --adaptive-tau 写进 TAU_ARG。
# tau 只被 compressed 评估消费，baseline-only 运行不该为它付出这趟预处理。
generate_tau_if_needed() {
    local cur_model=$1 model_name=$2 task=$3
    if [ "$STRATEGY_MODE" != "adaptive" ] || [ "$RUN_COMPRESSED" != "1" ]; then
        return
    fi

    # generate_adaptive_tau.py 固定把文件命名为 ${task}_adaptive_tau.json，不带
    # layer-level 标记；共用一个目录的话 layer-level 的 tau 会覆盖 per-head 的那份
    # (主结果全靠它)。所以按 LAYER_LEVEL 分目录存放。
    TAU_DIR="results/post_prefill/${model_name}/adaptiveTau${LEVEL_SUFFIX}"
    TAU_FILE="${TAU_DIR}/${task}_adaptive_tau.json"
    if [ "$REUSE_TAU" = "1" ] && [ -f "$TAU_FILE" ]; then
        echo "==> Reusing existing tau (REUSE_TAU=1): $TAU_FILE"
        echo "    注意：只检查了文件存在，未校验样本覆盖；样本数超过已存tau会在运行时报错。"
    else
        python utils/generate_adaptive_tau.py \
            --task "$task" \
            "${SAMPLE_ARG_TAU[@]}" \
            --model "$cur_model" \
            --output "$TAU_DIR" \
            "${LAYER_LEVEL_ARG[@]}"
    fi

    TAU_ARG=(--adaptive-tau "$TAU_FILE")
}

die() {
    echo "Error: $*" >&2
    exit 1
}

# 位置参数 -> TASKS，环境变量 -> 传给 Python 的 flag 数组。
# 用数组而非字符串，避免依赖 shell 的隐式拆词。
prepare_args() {
    # 命令行给了任务就覆盖顶部的 TASKS，没给就沿用顶部的默认列表
    if [ "$#" -gt 0 ]; then
        TASKS=("$@")
    fi

    LAYER_LEVEL_ARG=()
    if [ -n "$LAYER_LEVEL" ]; then
        LAYER_LEVEL_ARG=(--layer-level "$LAYER_LEVEL")
    fi

    SAMPLE_ARG_TAU=()
    SAMPLE_ARG_EXP=()
    if [ "$SAMPLES" != "all" ]; then
        SAMPLE_ARG_TAU=(--samples "$SAMPLES")
        SAMPLE_ARG_EXP=(--max-samples "$SAMPLES")
    fi

    BASELINE_ARG=()
    COMPRESSED_ARG=()
    DETAILS_ARG=()
    # 用 if/fi 而非 [ ] && x=y：后者作为函数最后一条语句时，
    # 条件不成立会让函数返回 1，set -e 会当场终止整个脚本。
    if [ "$RUN_BASELINE" = "0" ]; then BASELINE_ARG=(--no-baseline); fi
    if [ "$RUN_COMPRESSED" = "0" ]; then COMPRESSED_ARG=(--no-compressed); fi
    if [ "$NO_DETAILS" = "1" ]; then DETAILS_ARG=(--no-details); fi
}

# 在加载模型和生成 tau 之前拦下非法组合，免得几分钟后才报错。
validate_config() {
    if [ "$RUN_BASELINE" = "0" ] && [ "$RUN_COMPRESSED" = "0" ]; then
        die "RUN_BASELINE and RUN_COMPRESSED cannot both be 0."
    fi

    # independent 路径不消费 value_residual_bits(见 strategies/base.py)，带默认
    # VALUE_RESIDUAL_BITS=1 跑它会产出名字带 qjl1、内容却没有 QJL 的文件，
    # 正好与 shared_basis 主结果同名并覆盖它。
    if [ "$SVD_METHOD" != "shared_basis" ] && [ "$VALUE_RESIDUAL_BITS" != "0" ]; then
        die "SVD_METHOD=$SVD_METHOD does not support the QJL residual.
       Set VALUE_RESIDUAL_BITS=0 explicitly, e.g.
       VALUE_RESIDUAL_BITS=0 SVD_METHOD=$SVD_METHOD ./run_experiment.sh 100 xsum"
    fi

    # 因子表示只在 shared_basis + batched_eigh 这条路径上实现(见 core/config.py)。
    if [ "$CACHE_BACKEND" = "factored" ] && \
       { [ "$SVD_METHOD" != "shared_basis" ] || [ "$DECOMPOSITION_BACKEND" != "batched_eigh" ]; }; then
        die "CACHE_BACKEND=factored requires shared_basis + batched_eigh.
       Add CACHE_BACKEND=dense for this configuration, e.g.
       CACHE_BACKEND=dense DECOMPOSITION_BACKEND=$DECOMPOSITION_BACKEND SVD_METHOD=$SVD_METHOD ./run_experiment.sh 100 boolq"
    fi
}

main "$@"

# Usage: bash run_experiment.sh [sample_count|all] [task ...]
# Defaults: first 100 examples per task; configuration is defined above.
