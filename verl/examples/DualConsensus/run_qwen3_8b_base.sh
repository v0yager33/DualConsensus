#!/usr/bin/env bash
set -euxo pipefail
unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1


MODEL="Qwen3-8B-Base"

MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=3072


EPISODE=2
DATA_TRAIN_BATCH_SIZE=128
MINI_BATCH_SIZE=128
MICRO_BATCH_SIZE=8


N_VOTES_ANCHOR=16
N_VOTES_EXPLORER=16
N_VOTES_PER_PROMPT=32
N_SAMPLES_ANCHOR=8
N_SAMPLES_EXPLORER=8
N_SAMPLES_PER_PROMPT=16
UNLEARN_LR=3e-7

OUTPUT_BASE_DIR="/path/to/output"
TRAIN_DATA="[$data/train/dapo14k.parquet]"
TEST_DATA="[$data/test/math.parquet]"
MODEL_PATH="/path/to/models"

EXPERIMENT_GROUP="${MODEL}/DCRL"


RUN_ID="$(date +%m%d_%H%M%S)"
LOG_NAME="${EXPERIMENT_GROUP}_${RUN_ID}" 


OUTPUT_DIR="${OUTPUT_BASE_DIR}/checkpoints/${EXPERIMENT_GROUP}/${RUN_ID}"
mkdir -p "$OUTPUT_DIR" && chmod 775 "$OUTPUT_DIR"

export TENSORBOARD_DIR="${OUTPUT_BASE_DIR}/tensorboard/${EXPERIMENT_GROUP}/${RUN_ID}"
mkdir -p "$TENSORBOARD_DIR" && chmod 775 "$TENSORBOARD_DIR"


# ========================= Training =========================
python3 -m verl.trainer.main_ppo \
--config-path="verl/trainer/config" \
--config-name='ppo_trainer_dcrl.yaml' \
data.train_files=$TRAIN_DATA \
data.val_files=$TEST_DATA \
data.max_prompt_length=$MAX_PROMPT_LENGTH \
data.max_response_length=$MAX_RESPONSE_LENGTH \
data.train_batch_size=$DATA_TRAIN_BATCH_SIZE \
data.filter_overlong_prompts=True \
+data.suffix_prompt='"\nPlease reason step by step, and put your final answer within \boxed{}."' \
data.truncation='error' \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.model.enable_gradient_checkpointing=True \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
actor_rollout_ref.actor.use_kl_loss=True \
actor_rollout_ref.actor.optim.lr=1e-6 \
actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
actor_rollout_ref.actor.optim.warmup_style='cosine' \
actor_rollout_ref.actor.fsdp_config.param_offload=False \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
actor_rollout_ref.ref.fsdp_config.param_offload=True \
actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.rollout.temperature=1.0 \
actor_rollout_ref.rollout.enforce_eager=False \
actor_rollout_ref.rollout.free_cache_engine=False \
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
actor_rollout_ref.rollout.n=$N_SAMPLES_PER_PROMPT \
actor_rollout_ref.rollout.val_kwargs.do_sample=True \
actor_rollout_ref.rollout.val_kwargs.n=$N \
actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
critic.optim.lr=9e-6 \
critic.model.use_remove_padding=True \
critic.model.path=$MODEL_PATH \
critic.model.enable_gradient_checkpointing=True \
critic.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
critic.model.fsdp_config.param_offload=False \
critic.model.fsdp_config.optimizer_offload=False \
algorithm.kl_ctrl.kl_coef=0.00 \
algorithm.adv_estimator="grpo" \
custom_reward_function.path="verl/utils/reward_score/dcrl_math/__init__.py" \
custom_reward_function.name=reward_func \
dcrl.enable=True \
dcrl.n_votes_anchor=$N_VOTES_ANCHOR \
dcrl.n_votes_explorer=$N_VOTES_EXPLORER \
dcrl.n_votes_per_prompt=$N_VOTES_PER_PROMPT \
dcrl.n_samples_anchor=$N_SAMPLES_ANCHOR \
dcrl.n_samples_explorer=$N_SAMPLES_EXPLORER \
dcrl.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
dcrl.harmonic_vote=True \
actor_rollout_ref.actor.dcrl.unlearn_lr=$UNLEARN_LR \
actor_rollout_ref.dcrl.unlearn_lr=$UNLEARN_LR \
actor_rollout_ref.actor.dcrl.epsilon_low=0.01 \
actor_rollout_ref.actor.dcrl.epsilon_high=0.01 \
trainer.logger=['console','tensorboard'] \
trainer.project_name="dcrl_math" \
trainer.experiment_name="$EXPERIMENT_GROUP" \
trainer.run_id="$RUN_ID" \
trainer.n_gpus_per_node=8 \
trainer.nnodes=1 \
trainer.save_freq=2000000 \
trainer.test_freq=5 \
trainer.max_actor_ckpt_to_keep=0 \
trainer.max_critic_ckpt_to_keep=0 \
trainer.default_local_dir="$OUTPUT_DIR" \
trainer.total_epochs=$EPISODE "$@"