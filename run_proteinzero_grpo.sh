#!/bin/bash
# GRPO training launcher for protein sequence design.
#
# Usage:
#   bash run_proteinzero_grpo.sh             # Standard eight-GPU configuration
#
# Use the software environment documented in INSTALL.md.

export ACCELERATE_MIXED_PRECISION=no
export TMPDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.scratch"
mkdir -p "$TMPDIR"
NUM_GPUS=${1:-8}
if [ "$NUM_GPUS" -lt 1 ]; then
    echo "ERROR: NUM_GPUS must be positive." >&2
    exit 2
fi
GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-$((1024 / NUM_GPUS))}
NUM_GEN=${NUM_GEN:-64}
GENERATION_BATCH_SIZE=${GENERATION_BATCH_SIZE:-64}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-20}
MAX_STEPS=${MAX_STEPS:--1}
SAVE_STEPS=${SAVE_STEPS:-10}
VALIDATE_EVERY_STEPS=${VALIDATE_EVERY_STEPS:-10}
LR=${LR:-1e-5}
VAL_NUM_GEN=${VAL_NUM_GEN:-1}
if [ "$NUM_GPUS" -lt 1 ] || [ "$GRAD_ACC_STEPS" -lt 1 ]; then
    echo "ERROR: NUM_GPUS and GRAD_ACC_STEPS must be positive." >&2
    exit 2
fi
if [ "$VAL_NUM_GEN" -lt 1 ]; then
    echo "ERROR: VAL_NUM_GEN must be positive." >&2
    exit 2
fi

# Dataset paths.
# Paths are derived from CATH_VERSION.
CATH_VERSION=${CATH_VERSION:-4.3}
MODEL_NAME="./MPNN-ProGen2-xlarge-CATH42"
DATA_DIR="cath${CATH_VERSION//./}"
PDB_TRAIN_DIR="${DATA_DIR}/pdbs/train"
PDB_VAL_DIR="${DATA_DIR}/pdbs/val"
TRAIN_SPLIT=${TRAIN_SPLIT:-le100}
TRAIN_CSV="${DATA_DIR}/train_data_${TRAIN_SPLIT}.csv"
VAL_CSV="${DATA_DIR}/val_data_${TRAIN_SPLIT}.csv"
STRUCTURE_EMB_PREFIX="./structure_embeddings_${DATA_DIR}"
BASE_OUTPUT_DIR="./grpo_protein_output"

# Run identification and sampling configuration.
# Set RUN_NAME explicitly to override the generated identifier.
NUM_ITER=${NUM_ITER:-1}
# These rollout defaults correspond to gen_temperature and gen_top_p in
# train_proteinzero_grpo.py.
GEN_TEMP=${GEN_TEMP:-1.0}
GEN_TOP_P=${GEN_TOP_P:-1.0}
EVAL_TEMP=${EVAL_TEMP:-0.7}
EVAL_TOP_P=${EVAL_TOP_P:-1.0}
KL_BETA=${KL_BETA:-0.1}
EPSILON=${EPSILON:-0.2}
RW_TM=${RW_TM:-1.0}
RW_DDG=${RW_DDG:-1.0}
RW_PLDDT=${RW_PLDDT:-0.0}
RW_REC=${RW_REC:-0.0}
RW_LEN=${RW_LEN:-0.0}
ALPHA_DIV=${ALPHA_DIV:-0.05}
# Standard GRPO ("group") divides advantages by the group standard deviation;
# Dr.GRPO ("none") omits this normalization. A common positive reward scale is
# absorbed only in group mode and changes the effective update scale in none mode.
SCALE_REWARDS=${SCALE_REWARDS:-group}

PER_DEVICE_TRAIN_BATCH_SIZE=1
LOCAL_GENERATION_BATCH=$((PER_DEVICE_TRAIN_BATCH_SIZE * NUM_GPUS))
if [ $((GENERATION_BATCH_SIZE % LOCAL_GENERATION_BATCH)) -ne 0 ]; then
    echo "ERROR: GENERATION_BATCH_SIZE=$GENERATION_BATCH_SIZE must be divisible by per_device_bs*NUM_GPUS=$LOCAL_GENERATION_BATCH." >&2
    exit 2
fi
if [ $((GENERATION_BATCH_SIZE % NUM_GEN)) -ne 0 ]; then
    echo "ERROR: GENERATION_BATCH_SIZE=$GENERATION_BATCH_SIZE must be divisible by NUM_GEN=$NUM_GEN." >&2
    exit 2
fi
STEPS_PER_GENERATION=$((GENERATION_BATCH_SIZE / LOCAL_GENERATION_BATCH))
GACC_WINDOW=$((STEPS_PER_GENERATION * NUM_ITER))
GACC_MOD_SPG=$((GRAD_ACC_STEPS % GACC_WINDOW))
if [ "$GACC_MOD_SPG" -ne 0 ]; then
    echo "ERROR: GRAD_ACC_STEPS=$GRAD_ACC_STEPS must be divisible by steps_per_generation*num_iterations=$GACC_WINDOW." >&2
    exit 2
fi
EXPECTED_GROUPS=$((GENERATION_BATCH_SIZE / NUM_GEN))

RUN_NAME="${RUN_NAME:-proteinzero_grpo_cath${CATH_VERSION//./}_${TRAIN_SPLIT}_ngen${NUM_GEN}_mu${NUM_ITER}_lr${LR}_kl${KL_BETA}_eps${EPSILON}_sr${SCALE_REWARDS}_tm${RW_TM}_ddg${RW_DDG}_div${ALPHA_DIV}_gpu${NUM_GPUS}_gacc${GRAD_ACC_STEPS}}"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${RUN_NAME}"

# Training arguments.
COMMON_ARGS="
    --model_name $MODEL_NAME
    --cath_version $CATH_VERSION
    --pdb_train_dir $PDB_TRAIN_DIR
    --pdb_val_dir $PDB_VAL_DIR
    --train_csv $TRAIN_CSV
    --val_csv $VAL_CSV
    --structure_emb_path_prefix $STRUCTURE_EMB_PREFIX
    --output_dir $OUTPUT_DIR

    --num_generations $NUM_GEN
    --val_num_generations $VAL_NUM_GEN
    --num_iterations $NUM_ITER
    --gen_temperature $GEN_TEMP
    --gen_top_p $GEN_TOP_P
    --top_k 0
    --repetition_penalty 1.0
    --use_vllm False
    --use_transformers_paged False
    --use_liger_kernel False
    --eval_temperature $EVAL_TEMP
    --eval_top_p $EVAL_TOP_P

    --loss_type grpo
    --kl_beta $KL_BETA
    --epsilon $EPSILON
    --scale_rewards $SCALE_REWARDS

    --lora_rank ${LORA_RANK:-16}
    --lora_alpha ${LORA_ALPHA:-16}
    --lora_dropout 0.05
    --disable_dropout True

    --reward_weight_tm $RW_TM
    --reward_weight_ddg $RW_DDG
    --reward_weight_plddt $RW_PLDDT
    --reward_weight_recovery $RW_REC
    --reward_weight_length $RW_LEN

    --alpha_diversity $ALPHA_DIV
    --force_exact_length True

    --per_device_train_batch_size 1
    --gradient_accumulation_steps $GRAD_ACC_STEPS
    --generation_batch_size $GENERATION_BATCH_SIZE
    --num_train_epochs $NUM_TRAIN_EPOCHS
    --max_steps $MAX_STEPS
    --learning_rate $LR
    --warmup_steps 30
    --weight_decay 0.0
    --adam_beta1 0.9
    --adam_beta2 0.999
    --adam_epsilon 1e-8
    --max_grad_norm 1.0

    --fp16 False
    --bf16 False
    --tf32 False
    --torch_compile False

    --logging_steps 1
    --save_strategy steps
    --save_steps $SAVE_STEPS

    --report_to none
    --seed 42

    --validate_after_training True
    --validate_every_steps $VALIDATE_EVERY_STEPS
"

# Checkpoints under the generated output hierarchy are accepted directly.
# Other checkpoint locations require explicit opt-in through RUN_NAME.
if [ -n "${RESUME_FROM:-}" ]; then
    case "$RESUME_FROM" in
        *proteinzero_grpo_*/checkpoints/checkpoint-*)
            ;;
        *)
            case "$RUN_NAME" in
                *allow_incompatible_resume*)
                    ;;
                *)
                    echo "ERROR: refusing an external or incompatible checkpoint." >&2
                    echo "Use RUN_NAME=allow_incompatible_resume... to opt in explicitly." >&2
                    exit 2
                    ;;
            esac
            ;;
    esac
    COMMON_ARGS="$COMMON_ARGS --resume_from_checkpoint $RESUME_FROM"
    echo "Resuming from: $RESUME_FROM"
fi

# Launch training.
echo "Run: $RUN_NAME"
echo "Output: $OUTPUT_DIR"
echo "Derived: NUM_GPUS=$NUM_GPUS GRAD_ACC_STEPS=$GRAD_ACC_STEPS GENERATION_BATCH_SIZE=$GENERATION_BATCH_SIZE NUM_GENERATIONS=$NUM_GEN VAL_NUM_GEN=$VAL_NUM_GEN STEPS_PER_GENERATION=$STEPS_PER_GENERATION GACC_MOD_SPG=$GACC_MOD_SPG EXPECTED_GROUPS=$EXPECTED_GROUPS"

DRY_RUN=${DRY_RUN:-0}
if [ "$DRY_RUN" = "1" ]; then
    echo "DRY_RUN=1: configuration validated; launcher will not start Python."
    exit 0
elif [ "$DRY_RUN" != "0" ]; then
    echo "ERROR: DRY_RUN must be 0 or 1." >&2
    exit 2
fi

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Launching GRPO training on $NUM_GPUS GPUs with DDP..."
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29500 \
        train_proteinzero_grpo.py \
        $COMMON_ARGS
else
    echo "Launching GRPO training on single GPU..."
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python train_proteinzero_grpo.py \
        $COMMON_ARGS
fi
