#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

GRID_NUM_ROLLOUT="${GRID_NUM_ROLLOUT:-100}"
TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-1.0}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
SKIP_EVAL_BEFORE_TRAIN="${SKIP_EVAL_BEFORE_TRAIN:-0}"
# Set RUN_NAME to override the auto-generated EXP_NAME used for checkpoints,
# local run dumps, and W&B.
RUN_NAME="${RUN_NAME:-deepmath-JRL-t_1p0}"
EXP_NAME="${RUN_NAME}"
export GRID_NUM_ROLLOUT TEACHER_TEMPERATURE EVAL_INTERVAL SAVE_INTERVAL SKIP_EVAL_BEFORE_TRAIN RUN_NAME EXP_NAME

source "${SCRIPT_DIR}/launchers/common.sh"

DATA_LABEL="DeepMath-103k(level>=6)"
DATA_ID="deepmath-103k-level6"
PROMPT_DATA_VALUE="${PROMPT_DATA:-${DATA_ROOT}/DeepMath-103K/math_train_level6.jsonl}"

TEACHER_LABEL="JustRL-1.5B"
TEACHER_ID="justrl-1.5b"
TEACHER_MODEL_VALUE="${TEACHER_MODEL:-${MODEL_ROOT}/JustRL-DeepSeek-1.5B}"

STUDENT_LABEL="Deepseek-Distill-1.5B"
STUDENT_ID="deepseek-distill-1.5b"
STUDENT_MODEL_VALUE="${STUDENT_MODEL:-${MODEL_ROOT}/DeepSeek-R1-Distill-Qwen-1.5B}"
REF_LOAD_VALUE="${REF_LOAD:-${MODEL_ROOT}/DeepSeek-R1-Distill-Qwen-1.5B_torch_dist}"
BASE_SCRIPT="${ROOT_DIR}/scripts/launchers/run-deepmath-1.5b-opd.sh"
GRID_EXP_INDEX=6

grid_main "$@"
