#!/bin/bash

# Usage:
#   bash scripts/launchers/run-deepmath-1.7b_Base-4b-opd.sh
#
# Student: Qwen3-1.7B-Base
# Teacher: Qwen3-4B
# Data: DeepMath level6
#
# Useful overrides:
#   NUM_ROLLOUT=2 bash scripts/launchers/run-deepmath-1.7b_Base-4b-opd.sh
#   TOTAL_GPUS=8 ACTOR_NUM_GPUS_PER_NODE=6 bash scripts/launchers/run-deepmath-1.7b_Base-4b-opd.sh
#   ROLLOUT_BATCH_SIZE=8 GLOBAL_BATCH_SIZE=32 bash scripts/launchers/run-deepmath-1.7b_Base-4b-opd.sh

set -euo pipefail



ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"
SCRIPT_DIR="${ROOT_DIR}/scripts"
DRY_RUN=0
case "${1:-}" in
   --dry-run) DRY_RUN=1 ;;
   "") ;;
   *) echo "Usage: bash $0 [--dry-run]" >&2; exit 2 ;;
esac
source "${SCRIPT_DIR}/runtime.sh"
if [ -n "${SLIME_ENV_BIN:-}" ]; then
   resolve_project_paths SLIME_ENV_BIN
   export SLIME_ENV_BIN PATH="${SLIME_ENV_BIN}:${PATH}"
fi
SLIME_UPSTREAM_DIR="${SLIME_UPSTREAM_DIR:-../slime}"
MODEL_ROOT="${MODEL_ROOT:-models}"
DATA_ROOT="${DATA_ROOT:-data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
resolve_project_paths SLIME_UPSTREAM_DIR MODEL_ROOT DATA_ROOT OUTPUT_ROOT
MEGATRON_PATH="${MEGATRON_PATH:-${SLIME_UPSTREAM_DIR}/Megatron-LM}"
SGLANG_PATH="${SGLANG_PATH:-${SLIME_UPSTREAM_DIR}/sglang/python}"
resolve_project_paths MEGATRON_PATH SGLANG_PATH
export SLIME_UPSTREAM_DIR MEGATRON_PATH SGLANG_PATH

DATASET_DIR="${DATASET_DIR:-${DATA_ROOT}/DeepMath-103K}"
PROMPT_DATA="${PROMPT_DATA:-${DATASET_DIR}/math_train_level6.jsonl}"
STUDENT_MODEL="${STUDENT_MODEL:-${MODEL_ROOT}/Qwen3-1.7B-Base}"
TEACHER_MODEL="${TEACHER_MODEL:-${MODEL_ROOT}/Qwen3-4B}"
REF_LOAD="${REF_LOAD:-${MODEL_ROOT}/Qwen3-1.7B-Base_torch_dist}"
EXP_NAME="${EXP_NAME:-opd_stu-Q3-1.7B-Base_teacher-Q3-4B_data-DM-103k_ref_kl_1e-3_$(date +%Y%m%d-%H%M%S)}"
SAVE_DIR="${SAVE_DIR:-${OUTPUT_ROOT}/checkpoints/${EXP_NAME}}"
DUMP_DIR="${DUMP_DIR:-${OUTPUT_ROOT}/runs/${EXP_NAME}}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${OUTPUT_ROOT}/logs/${EXP_NAME}}"
ENABLE_DEBUG_DUMPS="${ENABLE_DEBUG_DUMPS:-0}"
EVAL_DATA_ROOT="${EVAL_DATA_ROOT:-${DATA_ROOT}/test_data}"
EVAL_AIME24_JSONL="${EVAL_AIME24_JSONL:-${EVAL_DATA_ROOT}/AIME24/test.slime.jsonl}"
EVAL_AIME25_JSONL="${EVAL_AIME25_JSONL:-${EVAL_DATA_ROOT}/AIME25/test.slime.jsonl}"
EVAL_AMC23_JSONL="${EVAL_AMC23_JSONL:-${EVAL_DATA_ROOT}/AMC23/test.slime.jsonl}"
EVAL_AIME26_JSONL="${EVAL_AIME26_JSONL:-${EVAL_DATA_ROOT}/AIME26/test.slime.jsonl}"
EVAL_DATA_SLICE="${EVAL_DATA_SLICE:-}"

TEACHER_LOG_FILE="${TEACHER_LOG_FILE:-${RUN_LOG_DIR}/teacher_sglang.log}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-${OUTPUT_ROOT}/ray}"
FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${OUTPUT_ROOT}/cache/flashinfer}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${OUTPUT_ROOT}/cache/triton}"
TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${OUTPUT_ROOT}/cache/torchinductor}"
VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${OUTPUT_ROOT}/cache/vllm}"
resolve_project_paths DATASET_DIR PROMPT_DATA STUDENT_MODEL TEACHER_MODEL REF_LOAD \
   SAVE_DIR DUMP_DIR RUN_LOG_DIR EVAL_DATA_ROOT EVAL_AIME24_JSONL EVAL_AIME25_JSONL \
   EVAL_AMC23_JSONL EVAL_AIME26_JSONL TEACHER_LOG_FILE RAY_TEMP_DIR \
   FLASHINFER_WORKSPACE_BASE TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR VLLM_CACHE_ROOT
export FLASHINFER_WORKSPACE_BASE TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR VLLM_CACHE_ROOT

TOTAL_GPUS="${TOTAL_GPUS:-8}"
# Split: GPUs 0-5 = actor + student rollout DP; GPUs 6-7 = teacher SGLang DP.
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-6}"
TEACHER_NUM_GPUS="${TEACHER_NUM_GPUS:-$((TOTAL_GPUS - ACTOR_NUM_GPUS_PER_NODE))}"
TEACHER_TENSOR_PARALLEL_SIZE="${TEACHER_TENSOR_PARALLEL_SIZE:-1}"
TEACHER_DATA_PARALLEL_SIZE="${TEACHER_DATA_PARALLEL_SIZE:-${TEACHER_NUM_GPUS}}"
RAY_CUDA_VISIBLE_DEVICES="${RAY_CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((ACTOR_NUM_GPUS_PER_NODE - 1)))}"
TEACHER_CUDA_VISIBLE_DEVICES="${TEACHER_CUDA_VISIBLE_DEVICES:-$(seq -s, ${ACTOR_NUM_GPUS_PER_NODE} $((TOTAL_GPUS - 1)))}"
RAY_NUM_GPUS="${RAY_NUM_GPUS:-${ACTOR_NUM_GPUS_PER_NODE}}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-${ACTOR_NUM_GPUS_PER_NODE}}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"

TEACHER_IP="${TEACHER_IP:-127.0.0.1}"
TEACHER_PORT="${TEACHER_PORT:-$(if [ "${DRY_RUN}" = "1" ]; then echo 13141; else find_free_port 13141; fi)}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-$(if [ "${DRY_RUN}" = "1" ]; then echo 8265; else find_free_port 8265; fi)}"
RAY_HEAD_PORT="${RAY_HEAD_PORT:-$(if [ "${DRY_RUN}" = "1" ]; then echo 6379; else find_free_port 6379; fi)}"
RAY_CLIENT_SERVER_PORT="${RAY_CLIENT_SERVER_PORT:-$(if [ "${DRY_RUN}" = "1" ]; then echo 20001; else find_free_port 20001; fi)}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_ADDR
STOP_EXISTING_RAY="${STOP_EXISTING_RAY:-0}"
TEACHER_READY_TIMEOUT="${TEACHER_READY_TIMEOUT:-900}"
NUM_EPOCH="${NUM_EPOCH:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-40}"
EVAL_INTERVAL="${EVAL_INTERVAL:-20}"
N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-8}"
EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-16384}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-21}" # (num of prompts)
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-6}" # one round rollout (num of rollouts): 21*6=126
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-16384}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
ROLLOUT_LABEL_KEY="${ROLLOUT_LABEL_KEY-label}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-126}" # one round training (num of rollouts)
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-8192}"
TEACHER_MEM_FRACTION_STATIC="${TEACHER_MEM_FRACTION_STATIC:-0.7}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.5}"

export PYTHONBUFFERED=16
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
EXTRA_PYTHONPATH=""
IFS=: read -r -a python_paths <<< "${PYTHONPATH:-}"
for python_path in "${python_paths[@]}"; do
   [ -n "${python_path}" ] || continue
   resolve_project_paths python_path
   EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH}:${python_path}"
done
export PYTHONPATH="${ROOT_DIR}:${SLIME_UPSTREAM_DIR}:${SGLANG_PATH}:${MEGATRON_PATH}${EXTRA_PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

if [ -n "${http_proxy:-}" ] && [ -z "${HTTP_PROXY:-}" ]; then
   export HTTP_PROXY="${http_proxy}"
fi
if [ -n "${HTTP_PROXY:-}" ] && [ -z "${http_proxy:-}" ]; then
   export http_proxy="${HTTP_PROXY}"
fi
if [ -n "${https_proxy:-}" ] && [ -z "${HTTPS_PROXY:-}" ]; then
   export HTTPS_PROXY="${https_proxy}"
fi
if [ -n "${HTTPS_PROXY:-}" ] && [ -z "${https_proxy:-}" ]; then
   export https_proxy="${HTTPS_PROXY}"
fi
if [ -n "${all_proxy:-}" ] && [ -z "${ALL_PROXY:-}" ]; then
   export ALL_PROXY="${all_proxy}"
fi
if [ -n "${ALL_PROXY:-}" ] && [ -z "${all_proxy:-}" ]; then
   export all_proxy="${ALL_PROXY}"
fi
if [ "${DRY_RUN}" = "0" ]; then
   NO_PROXY="$(build_no_proxy "${MASTER_ADDR}" "${TEACHER_IP}")"
else
   NO_PROXY="${NO_PROXY:-localhost,127.0.0.1}"
fi
export NO_PROXY
export no_proxy="${NO_PROXY}"
unset WANDB_DISABLED

cd "${ROOT_DIR}"
MODEL_CONFIG="${SLIME_UPSTREAM_DIR}/scripts/models/qwen3-1.7B.sh"
if [ -f "${MODEL_CONFIG}" ]; then
   source "${MODEL_CONFIG}"
elif [ "${DRY_RUN}" = "1" ]; then
   MODEL_ARGS=()
   echo "MODEL_CONFIG=${MODEL_CONFIG} (not resolved in dry-run)"
else
   echo "Missing upstream model config: ${MODEL_CONFIG}" >&2
   exit 1
fi

# Resume root, defaulting to SAVE_DIR (the in-place resume). Set STUDENT_LOAD to
# an existing run's checkpoints and STUDENT_CKPT_STEP to an iteration to restart
# from that iteration while writing somewhere else.
STUDENT_LOAD="${STUDENT_LOAD:-${SAVE_DIR}}"
resolve_project_paths STUDENT_LOAD
STUDENT_CKPT_STEP="${STUDENT_CKPT_STEP:-}"

if [ -n "${STUDENT_CKPT_STEP}" ] && \
   [ "$(realpath -m -- "${SAVE_DIR}")" = "$(realpath -m -- "${STUDENT_LOAD}")" ]; then
   echo "ERROR: SAVE_DIR must differ from STUDENT_LOAD when STUDENT_CKPT_STEP is set," >&2
   echo "       otherwise this run overwrites the checkpoints it resumes from." >&2
   exit 2
fi

CKPT_ARGS=(
   --hf-checkpoint "${STUDENT_MODEL}"
   --ref-load "${REF_LOAD}"
   --load "${STUDENT_LOAD}"
   --save "${SAVE_DIR}"
   --save-interval "${SAVE_INTERVAL}"
)

if [ -n "${STUDENT_CKPT_STEP}" ]; then
   CKPT_ARGS+=(--ckpt-step "${STUDENT_CKPT_STEP}")
fi

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --apply-chat-template
   --rollout-shuffle
   --num-epoch "${NUM_EPOCH}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
   --rollout-temperature "${ROLLOUT_TEMPERATURE}"
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --balance-data
)

if [ -n "${ROLLOUT_LABEL_KEY}" ]; then
   ROLLOUT_ARGS+=(--label-key "${ROLLOUT_LABEL_KEY}")
fi

if [ -n "${NUM_ROLLOUT:-}" ]; then
   ROLLOUT_ARGS+=(--num-rollout "${NUM_ROLLOUT}")
fi
if [ -n "${APPLY_CHAT_TEMPLATE_KWARGS:-}" ]; then
   ROLLOUT_ARGS+=(--apply-chat-template-kwargs "${APPLY_CHAT_TEMPLATE_KWARGS}")
fi
if [ -n "${ROLLOUT_STOP_TOKEN_IDS:-}" ]; then
   read -r -a rollout_stop_token_ids <<< "${ROLLOUT_STOP_TOKEN_IDS}"
   ROLLOUT_ARGS+=(--rollout-stop-token-ids "${rollout_stop_token_ids[@]}")
fi
if [ -n "${ROLLOUT_SAMPLE_FILTER_PATH:-}" ]; then
   ROLLOUT_ARGS+=(--rollout-sample-filter-path "${ROLLOUT_SAMPLE_FILTER_PATH}")
fi
if [ -n "${DYNAMIC_SAMPLING_SAMPLE_FILTER_PATH:-}" ]; then
   ROLLOUT_ARGS+=(--dynamic-sampling-sample-filter-path "${DYNAMIC_SAMPLING_SAMPLE_FILTER_PATH}")
fi
if [ -n "${DYNAMIC_SAMPLING_MAX_GENERATED_SAMPLES:-}" ]; then
   ROLLOUT_ARGS+=(--dynamic-sampling-max-generated-samples "${DYNAMIC_SAMPLING_MAX_GENERATED_SAMPLES}")
fi

RM_ARGS=(
   --custom-rm-path scripts.on_policy_distillation.reward_func
   --custom-reward-post-process-path scripts.on_policy_distillation.post_process_rewards
   --rm-url "http://${TEACHER_IP}:${TEACHER_PORT}/generate"
)

EVAL_ARGS=(
   --eval-interval "${EVAL_INTERVAL}"
   --eval-prompt-data AIME24 "${EVAL_AIME24_JSONL}${EVAL_DATA_SLICE}" AIME25 "${EVAL_AIME25_JSONL}${EVAL_DATA_SLICE}" AMC23 "${EVAL_AMC23_JSONL}${EVAL_DATA_SLICE}" AIME26 "${EVAL_AIME26_JSONL}${EVAL_DATA_SLICE}"
   --eval-input-key prompt
   --eval-label-key label
   --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT}"
   --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
   --eval-temperature 1.0
   --eval-top-p 1.0
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size "${CONTEXT_PARALLEL_SIZE}"
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"

   --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE:-2048}"
)

OPD_ARGS=(
   --advantage-estimator on_policy_distillation
   --teacher-top-logprobs-num 0
   --use-kl-loss
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LEARNING_RATE}"
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

if [ "${ENABLE_DEBUG_DUMPS}" = "1" ]; then
   MISC_ARGS+=(--dump-details "${DUMP_DIR}/details")
fi

WANDB_PROJECT="${WANDB_PROJECT:-slime_opd}"
WANDB_GROUP="${WANDB_GROUP:-${EXP_NAME}}"
WANDB_DIR="${WANDB_DIR:-${DUMP_DIR}/wandb}"
resolve_project_paths WANDB_DIR
WANDB_ARGS=(
   --use-wandb
   --wandb-project "${WANDB_PROJECT}"
   --wandb-group "${WANDB_GROUP}"
   --disable-wandb-random-suffix
   --wandb-dir "${WANDB_DIR}"
)

WANDB_KEY_VALUE="${WANDB_KEY:-${WANDB_API_KEY:-}}"
if [ -n "${WANDB_KEY_VALUE}" ]; then
   WANDB_ARGS+=(--wandb-key "${WANDB_KEY_VALUE}")
fi
if [ -n "${WANDB_MODE:-}" ]; then
   WANDB_ARGS+=(--wandb-mode "${WANDB_MODE}")
fi
if [ -n "${WANDB_HOST:-}" ]; then
   WANDB_ARGS+=(--wandb-host "${WANDB_HOST}")
fi
if [ -n "${WANDB_TEAM:-}" ]; then
   WANDB_ARGS+=(--wandb-team "${WANDB_TEAM}")
fi

JOB_ID="${JOB_ID:-deepmath_qwen3_1p7b_base_4b_opd_$(date +%Y%m%d-%H%M%S)_$$}"
if [ "${DRY_RUN}" = "1" ]; then
   printf 'UPSTREAM_ENTRY=%s/train.py\nMODEL_CONFIG=%s\n' "${SLIME_UPSTREAM_DIR}" "${MODEL_CONFIG}"
   printf 'PYTHONPATH=%s\n' "${PYTHONPATH}"
   for path_key in PROMPT_DATA STUDENT_MODEL TEACHER_MODEL REF_LOAD SLIME_UPSTREAM_DIR MODEL_ROOT DATA_ROOT OUTPUT_ROOT RUN_LOG_DIR TEACHER_LOG_FILE RAY_TEMP_DIR FLASHINFER_WORKSPACE_BASE TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR VLLM_CACHE_ROOT WANDB_DIR; do
      printf '%s=%s\n' "${path_key}" "${!path_key}"
   done
   printf 'TRAIN_ARGS:'
   printf ' %q' "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${ROLLOUT_ARGS[@]}" "${OPTIMIZER_ARGS[@]}" "${OPD_ARGS[@]}" "${EVAL_ARGS[@]}" "${PERF_ARGS[@]}" "${SGLANG_ARGS[@]}" "${MISC_ARGS[@]}" "${RM_ARGS[@]}"
   printf '\n'
   exit 0
fi
if [ ! -f "${SGLANG_PATH}/sglang/__init__.py" ]; then
   echo "Missing patched SGLang checkout: ${SGLANG_PATH}" >&2
   exit 1
fi
ulimit -n "$(ulimit -Hn)"
for path in "${PROMPT_DATA}" "${STUDENT_MODEL}" "${TEACHER_MODEL}" "${EVAL_AIME24_JSONL}" "${EVAL_AIME25_JSONL}" "${EVAL_AMC23_JSONL}" "${EVAL_AIME26_JSONL}"; do
   if [ ! -e "${path}" ]; then
      echo "Required path does not exist: ${path}" >&2
      exit 1
   fi
done


if [ ! -f "${REF_LOAD}/latest_checkpointed_iteration.txt" ]; then
   mkdir -p "${REF_LOAD}"
   python3 "${SCRIPT_DIR}/upstream_entry.py" tools/convert_hf_to_torch_dist.py \
      "${MODEL_ARGS[@]}" \
      --hf-checkpoint "${STUDENT_MODEL}" \
      --save "${REF_LOAD}"
fi

TEACHER_LOG_FILE="${TEACHER_LOG_FILE:-${RUN_LOG_DIR}/teacher_sglang.log}"
mkdir -p "${SAVE_DIR}" "${DUMP_DIR}" "$(dirname "${TEACHER_LOG_FILE}")"
RAY_STARTED_BY_LAUNCHER=0

cleanup() {
   set +e
   if [ -n "${TAIL_PID:-}" ]; then
      kill "${TAIL_PID}" 2>/dev/null || true
   fi
   if [ -n "${TEACHER_PID:-}" ]; then
      kill "${TEACHER_PID}" 2>/dev/null || true
   fi
   if [ "${RAY_STARTED_BY_LAUNCHER}" = "1" ]; then
      ray stop --force >/dev/null 2>&1 || true
   fi
}
trap cleanup EXIT

if [ "${STOP_EXISTING_RAY}" = "1" ]; then
   ray stop --force >/dev/null 2>&1 || true
fi

CUDA_VISIBLE_DEVICES="${TEACHER_CUDA_VISIBLE_DEVICES}" SGLANG_TEMP_SCALED_LOGPROBS=1 python3 -m sglang.launch_server \
   --model-path "${TEACHER_MODEL}" \
   --host 0.0.0.0 \
   --port "${TEACHER_PORT}" \
   --tp "${TEACHER_TENSOR_PARALLEL_SIZE}" \
   --dp-size "${TEACHER_DATA_PARALLEL_SIZE}" \
   --chunked-prefill-size 4096 \
   --mem-fraction-static "${TEACHER_MEM_FRACTION_STATIC}" \
   > "${TEACHER_LOG_FILE}" 2>&1 &
TEACHER_PID=$!

wait_for_http_health "http://${TEACHER_IP}:${TEACHER_PORT}/health_generate" "${TEACHER_READY_TIMEOUT}" || {
   tail -n 200 "${TEACHER_LOG_FILE}" >&2 || true
   exit 1
}

export WANDB_DIR
RUNTIME_ENV_JSON="$(build_runtime_env_json)"

CUDA_VISIBLE_DEVICES="${RAY_CUDA_VISIBLE_DEVICES}" ray start \
   --head \
   --temp-dir="${RAY_TEMP_DIR}" \
   --node-ip-address "${MASTER_ADDR}" \
   --port="${RAY_HEAD_PORT}" \
   --num-gpus "${RAY_NUM_GPUS}" \
   --ray-client-server-port="${RAY_CLIENT_SERVER_PORT}" \
   --disable-usage-stats \
   --dashboard-host=0.0.0.0 \
   --dashboard-port="${RAY_DASHBOARD_PORT}"
RAY_STARTED_BY_LAUNCHER=1

ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
   --submission-id "${JOB_ID}" \
   --no-wait \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 "${SCRIPT_DIR}/upstream_entry.py" train.py \
   --colocate \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
   --rollout-num-gpus "${ROLLOUT_NUM_GPUS}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${OPD_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${RM_ARGS[@]}"

echo "Submitted Ray job: ${JOB_ID}"


DRIVER_LOG=""
for _ in $(seq 1 60); do
   DRIVER_LOG=$(ls "${RAY_TEMP_DIR}"/session_*/logs/job-driver-"${JOB_ID}".log 2>/dev/null | head -n 1 || true)
   [ -n "${DRIVER_LOG}" ] && break
   sleep 1
done

if [ -n "${DRIVER_LOG}" ]; then
   (tail -n 0 -F "${DRIVER_LOG}" 2>/dev/null \
       | grep --line-buffered -E "rollout [0-9]+: \{|eval [0-9]+: \{|example data:|saving checkpoint at iteration") &
   TAIL_PID=$!
else
   echo "[warn] driver log not found for ${JOB_ID}; running without filtered stream" >&2
fi

PREV_STATUS=""
while true; do
   if ! JOB_STATUS="$(get_ray_job_status "http://127.0.0.1:${RAY_DASHBOARD_PORT}" "${JOB_ID}" 2>&1 | tr -d '\r' | tail -n 1)"; then
      echo "Failed to query Ray job status for ${JOB_ID}:" >&2
      echo "${JOB_STATUS}" >&2
      exit 1
   fi
   if [ "${JOB_STATUS}" != "${PREV_STATUS}" ]; then
      echo "[job status] ${JOB_STATUS}" >&2
      PREV_STATUS="${JOB_STATUS}"
   fi
   if [ "${JOB_STATUS}" = "SUCCEEDED" ]; then
      break
   fi
   if [ "${JOB_STATUS}" = "FAILED" ] || [ "${JOB_STATUS}" = "STOPPED" ]; then
      ray job logs "${JOB_ID}" --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" --log-style=record || true
      exit 1
   fi
   case "${JOB_STATUS}" in
      PENDING|RUNNING)
         ;;
      *)
         echo "Unexpected Ray job status output: ${JOB_STATUS}" >&2
         exit 1
         ;;
   esac
   sleep 10
done
