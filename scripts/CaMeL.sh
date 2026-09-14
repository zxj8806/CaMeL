set -euo pipefail

DEVICE="${1:-cuda}"

python experiments/scaling.py \
  general.exp_name=CaMeL \
  general.base_dir=$(pwd)/CaMeL \
  general.seed=42 \
  data.dim_z=3 \
  data.nature.seed=42 \
  data.always_generate_new_data=false \
  data.lifelong.enabled=true \
  data.lifelong.sequential=true \
  data.lifelong.stage=0 \
  data.lifelong.num_stages=3 \
  data.lifelong.mode=overlap_pairs \
  data.lifelong.targets_per_stage=2 \
  data.lifelong.evaluate_all_stages=true \
  data.lifelong.eval_seen_only=true \
  data.lifelong.save_stage_checkpoints=true \
  data.lifelong.replay.enabled=true \
  data.lifelong.replay.samples_per_stage=4096 \
  data.lifelong.replay.seed=42 \
  "data.lifelong.apply_to=[train,val]" \
  training=scaling_fast \
  training.device=${DEVICE} \
  training.batchsize=512 \
  eval.batchsize=512 \
  eval.causal_pruning.enabled=true \
  eval.causal_pruning.partition=test \
  eval.causal_pruning.relative_threshold=0.5 \
  eval.causal_pruning.use_stage_support=true \
  eval.causal_pruning.min_stage_support=1 \
  model.scm.three_d.enabled=true \
  model.scm.three_d.layout=cube \
  model.scm.three_d.seed=42 \
  training.three_d_edge_regularization_amount=1e-3 \
  model.three_d_spiking.enabled=true \
  model.three_d_spiking.neurons=27 \
  model.three_d_spiking.time_steps=6 \
  model.three_d_spiking.seed=42 \
  model.three_d_spiking.posterior_blend=0.0 \
  model.three_d_spiking.enable_causal_plasticity=true \
  model.three_d_spiking.plasticity_drive_scale=0.05 \
  training.three_d_spiking_consistency_amount=1e-2 \
  training.three_d_spiking_spike_regularization_amount=1e-3 \
  training.three_d_spiking_distance_regularization_amount=1e-4 \
  training.three_d_spiking_causal_plasticity_amount=1e-2 \
  training.three_d_spiking_causal_distance_amount=1e-3 \
  training.three_d_spiking_causal_l1_amount=1e-4 \
  plotting.save_three_d_diagnostics=true \
  plotting.save_three_d_spiking_diagnostics=true \
  2>&1 | tee CaMeL.txt
