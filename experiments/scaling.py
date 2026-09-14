
import hydra
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import trange
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import mlflow
import pandas as pd
from collections import defaultdict
from CaMeL.encoder import SONEncoder, GaussianEncoder
from CaMeL.lcm import FlowLCM
from CaMeL.training import VAEMetrics
from CaMeL.metrics import compute_dci
from CaMeL.causal.graph import create_graph
from CaMeL.causal.scm import (
    MLPFixedOrderSCM,
    MLPVariableOrderCausalModel,
    UnstructuredPrior,
    FixedGraphLinearANM,
)
from CaMeL.causal.implicit_scm import MLPImplicitSCM
from CaMeL.lcm import ELCM, ILCM
from CaMeL.posthoc_graph_learning import (
    compute_implicit_causal_effects,
    find_topological_order,
    run_enco,
)
from CaMeL.causal.three_d_diagnostics import save_three_d_diagnostics
from CaMeL.causal.three_d_spiking_memory import ThreeDSpikingMemory
from CaMeL.causal.three_d_spiking_diagnostics import save_three_d_spiking_diagnostics
from experiments.experiment_utils import (
    initialize_experiment,
    save_config,
    save_model,
    logger,
    create_optimizer_and_scheduler,
    set_manifold_thickness,
    compute_metrics_on_dataset,
    reset_optimizer_state,
    create_intervention_encoder,
    update_dict,
    log_training_step,
    optimizer_step,
    step_schedules,
    determine_graph_learning_settings,
    frequency_check,
)
@hydra.main(config_path="../config", config_name="scaling_ilcm")
def main(cfg):
    experiment(cfg)
def experiment(cfg):
    experiment_id = initialize_experiment(cfg)
    with mlflow.start_run(experiment_id=experiment_id, run_name=cfg.general.run_name):
        save_config(cfg)
        model = create_model(cfg)
        train(cfg, model)
        save_model(cfg, model)
        metrics = evaluate(cfg, model)
    logger.info("Anders nog iets?")
    return metrics
def create_model(cfg):
    logger.info(f"Creating {cfg.model.type} model")
    scm = create_scm(cfg)
    encoder, decoder = create_encoder_decoder(cfg)
    three_d_spiking_memory = create_three_d_spiking_memory(cfg)
    if cfg.model.type == "causal_vae":
        model = ELCM(
            scm, encoder=encoder, decoder=decoder, intervention_prior=None, dim_z=cfg.model.dim_z
        )
    elif cfg.model.type == "intervention_noise_vae":
        intervention_encoder = create_intervention_encoder(cfg)
        model = ILCM(
            scm,
            encoder=encoder,
            decoder=decoder,
            intervention_encoder=intervention_encoder,
            intervention_prior=None,
            averaging_strategy=cfg.model.averaging_strategy,
            dim_z=cfg.model.dim_z,
            three_d_spiking_memory=three_d_spiking_memory,
        )
    else:
        raise ValueError(f"Unknown value for cfg.model.type: {cfg.model.type}")
    if "load" in cfg.model and cfg.model.load is not None:
        filename = cfg.model.load
        logger.info(f"Loading model checkpoint from {filename}")
        state_dict = torch.load(filename, map_location="cpu")
        model.load_state_dict(state_dict)
    return model

def create_three_d_spiking_memory(cfg):
    three_d_spiking_cfg = cfg.model.get("three_d_spiking", {})
    if not three_d_spiking_cfg.get("enabled", False):
        return None
    logger.info("Creating 3dSpiking spiking memory")
    return ThreeDSpikingMemory(
        dim_z=cfg.model.dim_z,
        n_interventions=cfg.model.dim_z + 1,
        neurons=three_d_spiking_cfg.get("neurons", 27),
        time_steps=three_d_spiking_cfg.get("time_steps", 8),
        layout=three_d_spiking_cfg.get("layout", "cube"),
        seed=three_d_spiking_cfg.get("seed", 0),
        threshold=three_d_spiking_cfg.get("threshold", 0.5),
        decay=three_d_spiking_cfg.get("decay", 0.85),
        input_scale=three_d_spiking_cfg.get("input_scale", 0.2),
        recurrent_scale=three_d_spiking_cfg.get("recurrent_scale", 0.05),
        distance_power=three_d_spiking_cfg.get("distance_power", 1.0),
        train_recurrent=three_d_spiking_cfg.get("train_recurrent", False),
        target_spike_rate=three_d_spiking_cfg.get("target_spike_rate", 0.08),
        surrogate_slope=three_d_spiking_cfg.get("surrogate_slope", 8.0),
        posterior_blend=three_d_spiking_cfg.get("posterior_blend", 0.0),
        enable_causal_plasticity=three_d_spiking_cfg.get("enable_causal_plasticity", True),
        causal_synapse_scale=three_d_spiking_cfg.get("causal_synapse_scale", 0.03),
        plasticity_drive_scale=three_d_spiking_cfg.get("plasticity_drive_scale", 0.05),
        plasticity_effect_power=three_d_spiking_cfg.get("plasticity_effect_power", 1.0),
        detach_plasticity_teacher=three_d_spiking_cfg.get("detach_plasticity_teacher", True),
    )
def create_scm(cfg):
    logger.info(f"Creating {cfg.model.scm.type} SCM")
    noise_centric = cfg.model.type in {
        "noise_vae",
        "intervention_noise_vae",
        "alt_intervention_noise_vae",
    }
    if cfg.model.scm.type == "ground_truth":
        raise NotImplementedError
    elif cfg.model.scm.type == "unstructured":
        scm = UnstructuredPrior(dim_z=cfg.model.dim_z)
    elif noise_centric and cfg.model.scm.type == "mlp":
        logger.info(
            f"Graph parameterization for noise-centric learning: {cfg.model.scm.adjacency_matrix}"
        )
        three_d_cfg = cfg.model.scm.get("three_d", {})
        scm = MLPImplicitSCM(
            graph_parameterization=cfg.model.scm.adjacency_matrix,
            manifold_thickness=cfg.model.scm.manifold_thickness,
            hidden_units=cfg.model.scm.hidden_units,
            hidden_layers=cfg.model.scm.hidden_layers,
            homoskedastic=cfg.model.scm.homoskedastic,
            dim_z=cfg.model.dim_z,
            min_std=cfg.model.scm.min_std,
            three_d_enabled=three_d_cfg.get("enabled", False),
            three_d_layout=three_d_cfg.get("layout", "cube"),
            three_d_seed=three_d_cfg.get("seed", 0),
            three_d_distance_power=three_d_cfg.get("distance_power", 1.0),
        )
    elif (
        not noise_centric
        and cfg.model.scm.type == "mlp"
        and cfg.model.scm.adjacency_matrix in {"enco", "dds"}
    ):
        logger.info(
            f"Adjacency matrix: learnable, {cfg.model.scm.adjacency_matrix} parameterization"
        )
        scm = MLPVariableOrderCausalModel(
            graph_parameterization=cfg.model.scm.adjacency_matrix,
            manifold_thickness=cfg.model.scm.manifold_thickness,
            hidden_units=cfg.model.scm.hidden_units,
            hidden_layers=cfg.model.scm.hidden_layers,
            homoskedastic=cfg.model.scm.homoskedastic,
            dim_z=cfg.model.dim_z,
            enhance_causal_effects_at_init=False,
            min_std=cfg.model.scm.min_std,
        )
    elif (
        not noise_centric
        and cfg.model.scm.type == "mlp"
        and cfg.model.scm.adjacency_matrix == "fixed_order"
    ):
        logger.info(f"Adjacency matrix: learnable, fixed topological order")
        scm = MLPFixedOrderSCM(
            manifold_thickness=cfg.model.scm.manifold_thickness,
            hidden_units=cfg.model.scm.hidden_units,
            hidden_layers=cfg.model.scm.hidden_layers,
            homoskedastic=cfg.model.scm.homoskedastic,
            dim_z=cfg.model.dim_z,
            enhance_causal_effects_at_init=False,
            min_std=cfg.model.scm.min_std,
        )
    else:
        raise ValueError(f"Unknown value for cfg.model.scm.type: {cfg.model.scm.type}")
    return scm
def create_encoder_decoder(cfg):
    logger.info(f"Creating {cfg.model.encoder.type} encoder / decoder")
    if cfg.model.encoder.type == "mlp":
        encoder_hidden_layers = cfg.model.encoder.hidden_layers
        encoder_hidden = [cfg.model.encoder.hidden_units for _ in range(encoder_hidden_layers)]
        decoder_hidden_layers = cfg.model.decoder.hidden_layers
        decoder_hidden = [cfg.model.decoder.hidden_units for _ in range(decoder_hidden_layers)]
        encoder = GaussianEncoder(
            hidden=encoder_hidden,
            input_features=cfg.model.dim_x,
            output_features=cfg.model.dim_z,
            fix_std=cfg.model.encoder.fix_std,
            init_std=cfg.model.encoder.std,
            min_std=cfg.model.encoder.min_std,
        )
        decoder = GaussianEncoder(
            hidden=decoder_hidden,
            input_features=cfg.model.dim_z,
            output_features=cfg.model.dim_x,
            fix_std=cfg.model.decoder.fix_std,
            init_std=cfg.model.decoder.std,
            min_std=cfg.model.decoder.min_std,
        )
    else:
        raise ValueError(f"Unknown value for encoder_cfg.type: {cfg.model.encoder.type}")
    return encoder, decoder
def train(cfg, model):
    if "skip" in cfg.training and cfg.training.skip and cfg.training.skip != "None":
        return {}, {}
    if _lifelong_sequential_enabled(cfg):
        return train_lifelong_sequential(cfg, model)
    return train_single_stage(cfg, model)

def train_single_stage(cfg, model):
    logger.info("Starting training")
    logger.info(f"Training on {cfg.training.device}")
    device = torch.device(cfg.training.device)
    criteria = VAEMetrics(dim_z=cfg.data.dim_z)
    optim, scheduler = create_optimizer_and_scheduler(cfg, model, separate_param_groups=True)
    train_metrics = defaultdict(list)
    val_metrics = defaultdict(list)
    best_state = {"state_dict": None, "loss": None, "step": None}
    train_data = load_dataset(cfg, "train")
    val_data = load_dataset(cfg, "val")
    train_loader = DataLoader(train_data, batch_size=cfg.training.batchsize, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=cfg.training.batchsize, shuffle=False)
    model = model.to(device)
    step, nan_counter = _run_training_epochs(
        cfg,
        model,
        train_loader,
        val_loader,
        criteria,
        optim,
        scheduler,
        train_metrics,
        val_metrics,
        best_state,
        device,
        start_step=0,
        start_epoch=0,
        stage=None,
    )
    if cfg.training.validate_every_n_steps is not None and cfg.training.validate_every_n_steps > 0:
        validation_loop(cfg, model, criteria, val_loader, best_state, val_metrics, step, device)
        if cfg.training.early_stopping and best_state["step"] < step:
            logger.info(
                f'Early stopping after step {best_state["step"]} with validation loss '
                f'{best_state["loss"]}'
            )
            model.load_state_dict(best_state["state_dict"])
    set_manifold_thickness(cfg, model, None)
    return train_metrics, val_metrics

def train_lifelong_sequential(cfg, model):
    logger.info("Starting sequential same-graph lifelong training with replay and intervention-contrast pruning")
    logger.info(f"Training on {cfg.training.device}")
    device = torch.device(cfg.training.device)
    criteria = VAEMetrics(dim_z=cfg.data.dim_z)
    optim, scheduler = create_optimizer_and_scheduler(cfg, model, separate_param_groups=True)
    train_metrics = defaultdict(list)
    val_metrics = defaultdict(list)
    best_state = {"state_dict": None, "loss": None, "step": None}
    model = model.to(device)
    n_stages = _lifelong_num_stages(cfg)
    lifelong_rows = []
    replay_summary_rows = []
    replay_buffers = []
    step = 0
    nan_counter = 0
    original_stage = _lifelong_active_stage(cfg)
    replay_enabled = _lifelong_replay_enabled(cfg)
    if replay_enabled:
        logger.info(
            "Lifelong replay enabled: samples_per_stage=%s, seed=%s",
            _lifelong_replay_samples_per_stage(cfg),
            _lifelong_replay_seed(cfg),
        )
    for stage in range(n_stages):
        cfg.data.lifelong.stage = stage
        labels = _lifelong_stage_labels(cfg, stage=stage)
        logger.info(
            "=== Lifelong train stage %s/%s: intervention labels %s ===",
            stage,
            n_stages - 1,
            labels,
        )
        current_train_data = load_dataset(cfg, "train")
        train_data = _make_replay_augmented_dataset(
            cfg, current_train_data, replay_buffers, stage=stage
        )
        if train_data is not current_train_data:
            replay_summary_rows.append(
                {
                    "stage": int(stage),
                    "current_samples": len(current_train_data),
                    "replay_samples": len(train_data) - len(current_train_data),
                    "total_train_samples": len(train_data),
                    "replay_buffers": len(replay_buffers),
                }
            )
        else:
            replay_summary_rows.append(
                {
                    "stage": int(stage),
                    "current_samples": len(current_train_data),
                    "replay_samples": 0,
                    "total_train_samples": len(train_data),
                    "replay_buffers": len(replay_buffers),
                }
            )
        val_data = load_dataset(cfg, "val")
        train_loader = DataLoader(train_data, batch_size=cfg.training.batchsize, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=cfg.training.batchsize, shuffle=False)
        stage_start_epoch = stage * int(cfg.training.epochs)
        step, nan_counter = _run_training_epochs(
            cfg,
            model,
            train_loader,
            val_loader,
            criteria,
            optim,
            scheduler,
            train_metrics,
            val_metrics,
            best_state,
            device,
            start_step=step,
            start_epoch=stage_start_epoch,
            stage=stage,
            nan_counter=nan_counter,
        )
        if replay_enabled:
            replay_buffer = _sample_replay_buffer_from_dataset(cfg, current_train_data, stage)
            replay_buffers.append(replay_buffer)
            logger.info(
                "Stored replay buffer for stage %s: %s samples",
                stage,
                len(replay_buffer),
            )
        if cfg.data.lifelong.get("save_stage_checkpoints", True):
            save_model(cfg, model, f"model_after_stage{stage}.pt")
        if cfg.data.lifelong.get("evaluate_all_stages", True):
            lifelong_rows.extend(evaluate_lifelong_after_stage(cfg, model, train_stage=stage))
    cfg.data.lifelong.stage = n_stages - 1
    metrics_dir = Path(cfg.general.exp_dir) / "metrics"
    if lifelong_rows:
        df = pd.DataFrame(lifelong_rows)
        output = metrics_dir / "lifelong_metrics.csv"
        df.to_csv(output, index=False)
        logger.info(f"Saved lifelong metrics at {output}")
        summary = summarize_lifelong_metrics(df)
        summary_output = metrics_dir / "lifelong_forgetting_summary.csv"
        summary.to_csv(summary_output, index=False)
        logger.info(f"Saved lifelong forgetting summary at {summary_output}")
    if replay_summary_rows:
        replay_output = metrics_dir / "lifelong_replay_summary.csv"
        pd.DataFrame(replay_summary_rows).to_csv(replay_output, index=False)
        logger.info(f"Saved lifelong replay summary at {replay_output}")
    set_manifold_thickness(cfg, model, None)
    return train_metrics, val_metrics

def _run_training_epochs(
    cfg,
    model,
    train_loader,
    val_loader,
    criteria,
    optim,
    scheduler,
    train_metrics,
    val_metrics,
    best_state,
    device,
    start_step=0,
    start_epoch=0,
    stage=None,
    nan_counter=0,
):
    step = start_step
    steps_per_epoch = len(train_loader)
    epoch_generator = trange(cfg.training.epochs, disable=not cfg.general.verbose)
    if stage is not None:
        epoch_generator.set_description(f"Stage {stage}")
    for local_epoch in epoch_generator:
        epoch = start_epoch + local_epoch
        mlflow.log_metric("train.epoch", epoch, step=step)
        if stage is not None:
            mlflow.log_metric("train.lifelong_stage", stage, step=step)
        graph_kwargs = determine_graph_learning_settings(cfg, epoch, model)
        model_interventions, pretrain, deterministic_intervention_encoder = epoch_schedules(
            cfg, model, epoch, optim
        )
        for x1, x2, z1, z2, intervention_labels, true_interventions in train_loader:
            fractional_epoch = step / steps_per_epoch
            model.train()
            (
                beta,
                beta_intervention,
                consistency_regularization_amount,
                cyclicity_regularization_amount,
                edge_regularization_amount,
                inverse_consistency_regularization_amount,
                z_regularization_amount,
                intervention_entropy_regularization_amount,
                intervention_encoder_offset,
            ) = step_schedules(cfg, model, fractional_epoch)
            three_d_edge_regularization_amount = cfg.training.get(
                "three_d_edge_regularization_amount", 0.0
            )
            three_d_spiking_consistency_amount = cfg.training.get(
                "three_d_spiking_consistency_amount", 0.0
            )
            three_d_spiking_spike_regularization_amount = cfg.training.get(
                "three_d_spiking_spike_regularization_amount", 0.0
            )
            three_d_spiking_distance_regularization_amount = cfg.training.get(
                "three_d_spiking_distance_regularization_amount", 0.0
            )
            three_d_spiking_causal_plasticity_amount = cfg.training.get(
                "three_d_spiking_causal_plasticity_amount", 0.0
            )
            three_d_spiking_causal_distance_amount = cfg.training.get(
                "three_d_spiking_causal_distance_amount", 0.0
            )
            three_d_spiking_causal_l1_amount = cfg.training.get(
                "three_d_spiking_causal_l1_amount", 0.0
            )
            x1, x2, z1, z2, intervention_labels, true_interventions = (
                x1.to(device),
                x2.to(device),
                z1.to(device),
                z2.to(device),
                intervention_labels.to(device),
                true_interventions.to(device),
            )
            log_prob, model_outputs = model(
                x1,
                x2,
                beta=beta,
                beta_intervention_target=beta_intervention,
                pretrain_beta=cfg.training.pretrain_beta,
                full_likelihood=cfg.training.full_likelihood,
                likelihood_reduction=cfg.training.likelihood_reduction,
                pretrain=pretrain,
                model_interventions=model_interventions,
                deterministic_intervention_encoder=deterministic_intervention_encoder,
                intervention_encoder_offset=intervention_encoder_offset,
                **graph_kwargs,
            )
            loss, metrics = criteria(
                log_prob,
                true_intervention_labels=intervention_labels,
                z_regularization_amount=z_regularization_amount,
                edge_regularization_amount=edge_regularization_amount,
                cyclicity_regularization_amount=cyclicity_regularization_amount,
                consistency_regularization_amount=consistency_regularization_amount,
                inverse_consistency_regularization_amount=inverse_consistency_regularization_amount,
                intervention_entropy_regularization_amount=intervention_entropy_regularization_amount,
                three_d_edge_regularization_amount=three_d_edge_regularization_amount,
                three_d_spiking_consistency_amount=three_d_spiking_consistency_amount,
                three_d_spiking_spike_regularization_amount=three_d_spiking_spike_regularization_amount,
                three_d_spiking_distance_regularization_amount=three_d_spiking_distance_regularization_amount,
                three_d_spiking_causal_plasticity_amount=three_d_spiking_causal_plasticity_amount,
                three_d_spiking_causal_distance_amount=three_d_spiking_causal_distance_amount,
                three_d_spiking_causal_l1_amount=three_d_spiking_causal_l1_amount,
                **model_outputs,
            )
            finite, grad_norm = optimizer_step(cfg, loss, model, model_outputs, optim, x1, x2)
            if not finite:
                nan_counter += 1
            step += 1
            log_training_step(
                cfg,
                beta,
                epoch_generator,
                finite,
                grad_norm,
                metrics,
                model,
                step,
                train_metrics,
                nan_counter,
            )
            if frequency_check(step, cfg.training.validate_every_n_steps):
                validation_loop(
                    cfg, model, criteria, val_loader, best_state, val_metrics, step, device
                )
            if frequency_check(step, cfg.training.save_model_every_n_steps):
                suffix = f"stage{stage}_" if stage is not None else ""
                save_model(cfg, model, f"model_{suffix}step_{step}.pt")
        if scheduler is not None and local_epoch < cfg.training.epochs - 1:
            scheduler.step()
            mlflow.log_metric("train.lr", scheduler.get_last_lr()[0], step=step)
            if (
                cfg.training.lr_schedule.type == "cosine_restarts_reset"
                and (local_epoch + 1) % cfg.training.lr_schedule.restart_every_epochs == 0
                and local_epoch + 1 < cfg.training.epochs
            ):
                logger.info(f"Resetting optimizer at local epoch {local_epoch + 1}")
                reset_optimizer_state(optim)
    return step, nan_counter

def _get_lifelong_cfg(cfg):
    return cfg.data.get("lifelong", {})

def _lifelong_enabled(cfg):
    return bool(_get_lifelong_cfg(cfg).get("enabled", False))

def _lifelong_sequential_enabled(cfg):
    return bool(_lifelong_enabled(cfg) and _get_lifelong_cfg(cfg).get("sequential", False))

def _lifelong_num_stages(cfg):
    lifelong_cfg = _get_lifelong_cfg(cfg)
    return int(lifelong_cfg.get("num_stages", cfg.data.dim_z))

def _lifelong_active_stage(cfg):
    return int(_get_lifelong_cfg(cfg).get("stage", 0))

def _lifelong_apply_to(cfg):
    lifelong_cfg = _get_lifelong_cfg(cfg)
    apply_to = lifelong_cfg.get("apply_to", ["train", "val"])
    if isinstance(apply_to, str):
        apply_to = [part.strip() for part in apply_to.split(",") if part.strip()]
    return set(apply_to)

def _lifelong_replay_cfg(cfg):
    lifelong_cfg = _get_lifelong_cfg(cfg)
    return lifelong_cfg.get("replay", {})

def _lifelong_replay_enabled(cfg):
    replay_cfg = _lifelong_replay_cfg(cfg)
    return bool(_lifelong_sequential_enabled(cfg) and replay_cfg.get("enabled", False))

def _lifelong_replay_samples_per_stage(cfg):
    replay_cfg = _lifelong_replay_cfg(cfg)
    return int(replay_cfg.get("samples_per_stage", 4096))

def _lifelong_replay_seed(cfg):
    replay_cfg = _lifelong_replay_cfg(cfg)
    return int(replay_cfg.get("seed", cfg.general.seed))

def _sample_replay_buffer_from_dataset(cfg, dataset, stage):
    n = len(dataset)
    samples_per_stage = min(_lifelong_replay_samples_per_stage(cfg), n)
    generator = torch.Generator()
    generator.manual_seed(_lifelong_replay_seed(cfg) + int(stage))
    indices = torch.randperm(n, generator=generator)[:samples_per_stage]
    tensors = tuple(tensor[indices].clone() for tensor in dataset.tensors)
    return TensorDataset(*tensors)

def _concat_tensor_datasets(datasets):
    datasets = [dataset for dataset in datasets if dataset is not None and len(dataset) > 0]
    if len(datasets) == 1:
        return datasets[0]
    n_components = len(datasets[0].tensors)
    for dataset in datasets:
        if len(dataset.tensors) != n_components:
            raise RuntimeError("Cannot concatenate replay datasets with different tensor tuples")
    tensors = []
    for component_idx in range(n_components):
        tensors.append(torch.cat([dataset.tensors[component_idx] for dataset in datasets], dim=0))
    return TensorDataset(*tensors)

def _make_replay_augmented_dataset(cfg, current_dataset, replay_buffers, stage):
    if not _lifelong_replay_enabled(cfg) or not replay_buffers:
        return current_dataset
    replay_samples = sum(len(buffer) for buffer in replay_buffers)
    logger.info(
        "Augmenting stage %s training data with %s replay samples from %s previous stages",
        stage,
        replay_samples,
        len(replay_buffers),
    )
    return _concat_tensor_datasets([current_dataset] + list(replay_buffers))

def summarize_lifelong_metrics(df):
    rows = []
    numeric_df = df.copy()
    for metric in [
        "causal_disentanglement",
        "intervention_accuracy",
        "intervention_correct_posterior",
        "nll",
        "loss",
    ]:
        if metric not in numeric_df.columns:
            continue
        for eval_stage in sorted(numeric_df["eval_stage"].astype(str).unique()):
            stage_df = numeric_df[numeric_df["eval_stage"].astype(str) == eval_stage]
            vals = pd.to_numeric(stage_df[metric], errors="coerce").dropna()
            if vals.empty:
                continue
            train_stages = pd.to_numeric(stage_df.loc[vals.index, "train_stage"], errors="coerce")
            final_idx = train_stages.idxmax()
            final_value = float(stage_df.loc[final_idx, metric])
            if metric in {"nll", "loss"}:
                best_value = float(vals.min())
                forgetting = final_value - best_value
            else:
                best_value = float(vals.max())
                forgetting = best_value - final_value
            rows.append(
                {
                    "metric": metric,
                    "eval_stage": eval_stage,
                    "best_value": best_value,
                    "final_value": final_value,
                    "forgetting": forgetting,
                    "n_points": int(vals.shape[0]),
                }
            )
    return pd.DataFrame(rows)

def _lifelong_stage_labels(cfg, stage=None):
    lifelong_cfg = _get_lifelong_cfg(cfg)
    dim_z = int(cfg.data.dim_z)
    n_stages = _lifelong_num_stages(cfg)
    if stage is None:
        stage = _lifelong_active_stage(cfg)
    stage = int(stage)
    if stage < 0:
        return list(range(dim_z + 1))
    if stage >= n_stages:
        raise ValueError(f"Requested lifelong stage {stage}, but num_stages={n_stages}")
    include_no_intervention = bool(lifelong_cfg.get("include_no_intervention", True))
    if "stage_labels" in lifelong_cfg and lifelong_cfg.stage_labels is not None:
        labels = [int(label) for label in lifelong_cfg.stage_labels[stage]]
    else:
        mode = lifelong_cfg.get("mode", "overlap_pairs")
        if mode == "single_target":
            labels = [1 + (stage % dim_z)]
        elif mode == "overlap_pairs":
            targets_per_stage = int(lifelong_cfg.get("targets_per_stage", min(2, dim_z)))
            targets_per_stage = max(1, min(targets_per_stage, dim_z))
            labels = [1 + ((stage + offset) % dim_z) for offset in range(targets_per_stage)]
        elif mode == "prefix":
            labels = list(range(1, min(dim_z, stage + 1) + 1))
        elif mode == "all":
            labels = list(range(1, dim_z + 1))
        else:
            raise ValueError(
                f"Unknown data.lifelong.mode={mode}. "
                "Use one of: single_target, overlap_pairs, prefix, all, or provide stage_labels."
            )
        if include_no_intervention:
            labels = [0] + labels
    labels = sorted(set(int(label) for label in labels))
    valid_labels = set(range(dim_z + 1))
    invalid_labels = [label for label in labels if label not in valid_labels]
    if invalid_labels:
        raise ValueError(
            f"Invalid lifelong intervention labels {invalid_labels}; valid labels are 0..{dim_z}"
        )
    return labels

def _filter_tensor_data_by_intervention_labels(data, allowed_labels):
    intervention_labels = data[4].to(torch.long)
    allowed = torch.as_tensor(allowed_labels, dtype=intervention_labels.dtype, device=intervention_labels.device)
    mask = torch.zeros_like(intervention_labels, dtype=torch.bool)
    for label in allowed:
        mask = mask | (intervention_labels == label)
    n_selected = int(mask.sum().item())
    if n_selected == 0:
        raise RuntimeError(
            f"Lifelong stage filter selected zero samples for labels {allowed_labels}."
        )
    filtered = tuple(component[mask] for component in data)
    counts = {
        int(label): int((intervention_labels == int(label)).sum().item())
        for label in sorted(torch.unique(intervention_labels).cpu().tolist())
    }
    selected_counts = {label: counts.get(label, 0) for label in allowed_labels}
    return filtered, n_selected, counts, selected_counts

def _materialize_lifelong_stage_files(cfg, data_dir=None):
    if not _lifelong_enabled(cfg):
        return False
    if data_dir is None:
        data_dir = Path(cfg.data.data_dir)
    data_dir = Path(data_dir)
    tags = ["train", "dci_train", "val", "test"]
    n_stages = _lifelong_num_stages(cfg)
    summary_rows = []
    for tag in tags:
        base_filename = data_dir / f"{tag}.pt"
        if not base_filename.exists():
            continue
        data = torch.load(base_filename)
        n_before = int(data[0].shape[0])
        for stage in range(n_stages):
            allowed_labels = _lifelong_stage_labels(cfg, stage=stage)
            stage_data, n_after, full_counts, selected_counts = _filter_tensor_data_by_intervention_labels(
                data, allowed_labels
            )
            stage_filename = data_dir / f"{tag}_stage{stage}.pt"
            torch.save(stage_data, stage_filename)
            logger.info(
                "Materialized lifelong %s stage %s at %s: labels=%s, samples=%s/%s",
                tag, stage, stage_filename, allowed_labels, n_after, n_before,
            )
            summary_rows.append(
                {
                    "partition": tag,
                    "stage": stage,
                    "allowed_labels": " ".join(map(str, allowed_labels)),
                    "n_before": n_before,
                    "n_after": n_after,
                    "full_label_counts": str(full_counts),
                    "selected_label_counts": str(selected_counts),
                }
            )
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(data_dir / "lifelong_stage_summary.csv", index=False)
        return True
    return False

def _dataset_filename(cfg, tag):
    data_dir = Path(cfg.data.data_dir)
    if _lifelong_enabled(cfg) and tag in _lifelong_apply_to(cfg):
        stage = _lifelong_active_stage(cfg)
        if stage >= 0:
            return data_dir / f"{tag}_stage{stage}.pt"
    return data_dir / f"{tag}.pt"

def _lifelong_stage_dataset(cfg, tag, stage):
    data_dir = Path(cfg.data.data_dir)
    filename = data_dir / f"{tag}_stage{stage}.pt"
    if not filename.exists():
        if data_dir.exists():
            _materialize_lifelong_stage_files(cfg, data_dir)
        if not filename.exists():
            generate_datasets(cfg)
    logger.info(
        "Loading lifelong eval stage %s %s data with intervention labels %s from %s",
        stage, tag, _lifelong_stage_labels(cfg, stage=stage), filename,
    )
    data = torch.load(filename)
    return TensorDataset(*data)

def load_dataset(cfg, tag):
    assert tag in {"train", "dci_train", "test", "val"}
    data_dir = Path(cfg.data.data_dir)
    filename = _dataset_filename(cfg, tag)
    if _lifelong_enabled(cfg) and not filename.exists() and data_dir.exists():
        _materialize_lifelong_stage_files(cfg, data_dir)
    if cfg.data.always_generate_new_data or not data_dir.exists() or not filename.exists():
        generate_datasets(cfg)
        cfg.data.always_generate_new_data = False
        filename = _dataset_filename(cfg, tag)
    if _lifelong_enabled(cfg) and tag in _lifelong_apply_to(cfg):
        logger.info(
            "Loading lifelong stage %s %s data with intervention labels %s from %s",
            _lifelong_active_stage(cfg), tag, _lifelong_stage_labels(cfg), filename,
        )
    else:
        logger.debug(f"Loading data from {filename}")
    data = torch.load(filename)
    dataset = TensorDataset(*data)
    return dataset
def create_true_model(cfg):
    graph = create_graph(
        cfg.data.dim_z, cfg.data.nature.mode, cfg.data.nature.seed, permutation=None
    )
    logger.info(f"Created ground-truth latent graph:\n{graph.adjacency_matrix}")
    scm = FixedGraphLinearANM(
        graph,
        cfg.data.dim_z,
        manifold_thickness=cfg.data.nature.manifold_thickness,
        initialization=cfg.data.nature.causal_effects,
    )
    encoder = SONEncoder(
        coeff_std=1.0, input_features=cfg.data.dim_z, output_features=cfg.data.dim_z
    )
    nature = FlowLCM(scm, encoder, dim_z=cfg.data.dim_z)
    for param in nature.parameters():
        param.requires_grad = False
    return nature
def load_true_model(cfg):
    if cfg.data.always_generate_new_data or not Path(cfg.data.data_dir).exists():
        generate_datasets(cfg)
        cfg.data.always_generate_new_data = (
            False
        )
    nature = create_true_model(cfg)
    nature.load_state_dict(torch.load(Path(cfg.data.data_dir) / "nature.pt"))
    return nature
def generate_datasets(cfg):
    logger.info(f"Generating dataset at {cfg.data.data_dir}")
    data_dir = Path(cfg.data.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    old_seed = torch.random.seed()
    torch.random.manual_seed(cfg.data.nature.seed)
    nature = create_true_model(cfg)
    torch.save(nature.state_dict(), Path(cfg.data.data_dir) / "nature.pt")
    torch.random.manual_seed(old_seed)
    tags_samples = {
        "train": cfg.data.samples.train,
        "dci_train": cfg.data.samples.train,
        "val": cfg.data.samples.val,
        "test": cfg.data.samples.test,
    }
    for tag, n_samples in tags_samples.items():
        filename = data_dir / f"{tag}.pt"
        data = nature.sample(n_samples, additional_noise=cfg.data.nature.observation_noise)
        torch.save(data, filename)
    _materialize_lifelong_stage_files(cfg, data_dir)
@torch.no_grad()
def validation_loop(cfg, model, criteria, val_loader, best_state, val_metrics, step, device):
    loss, nll, metrics = compute_metrics_on_dataset(cfg, model, criteria, val_loader, device)
    metrics.update(eval_dci_scores(cfg, model, partition="val"))
    metrics.update(eval_implicit_graph(cfg, model, partition="val"))
    update_dict(val_metrics, metrics)
    for key, value in metrics.items():
        mlflow.log_metric(f"val.{key}", value, step=step)
    logger.info(f"Step {step}: causal disentanglement = {metrics['causal_disentanglement']:.2f}")
    new_val_loss = metrics["nll"] if cfg.training.early_stopping_var == "nll" else loss.item()
    if best_state["loss"] is None or new_val_loss < best_state["loss"]:
        best_state["loss"] = new_val_loss
        best_state["state_dict"] = model.state_dict().copy()
        best_state["step"] = step
def _metric_value(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().numpy().tolist()
    try:
        import numpy as _np
        if isinstance(value, (_np.floating, _np.integer)):
            return value.item()
    except Exception:
        pass
    return value

@torch.no_grad()
def _eval_dci_scores_on_dataset(cfg, model, test_data, full_importance_matrix=False):
    model.eval()
    device = torch.device(cfg.training.device)
    model = model.to(device)
    x_train, _, train_true_z, *_ = load_dataset(cfg, "dci_train").tensors
    x_test, _, test_true_z, *_ = test_data.tensors
    train_model_z = model.encode_to_causal(x_train.to(device), deterministic=True)
    test_model_z = model.encode_to_causal(x_test.to(device), deterministic=True)
    causal_dci_metrics = compute_dci(
        train_true_z,
        train_model_z,
        test_true_z,
        test_model_z,
        return_full_importance_matrix=full_importance_matrix,
    )
    return {f"causal_{key}": val for key, val in causal_dci_metrics.items()}

@torch.no_grad()
def _eval_metrics_on_dataset(cfg, model, dataset):
    device = torch.device(cfg.training.device)
    model = model.to(device)
    criteria = VAEMetrics(dim_z=cfg.data.dim_z)
    batch_size = min(int(cfg.eval.batchsize), len(dataset))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    _, _, metrics = compute_metrics_on_dataset(cfg, model, criteria, loader, device=device)
    metrics.update(_eval_dci_scores_on_dataset(cfg, model, dataset, full_importance_matrix=False))
    return {key: _metric_value(val) for key, val in metrics.items()}

def evaluate_lifelong_after_stage(cfg, model, train_stage):
    rows = []
    n_stages = _lifelong_num_stages(cfg)
    if cfg.data.lifelong.get("eval_seen_only", True):
        eval_stages = list(range(train_stage + 1))
    else:
        eval_stages = list(range(n_stages))
    for eval_stage in eval_stages:
        dataset = _lifelong_stage_dataset(cfg, "test", eval_stage)
        metrics = _eval_metrics_on_dataset(cfg, model, dataset)
        row = {
            "train_stage": int(train_stage),
            "eval_stage": int(eval_stage),
            "eval_partition": f"test_stage{eval_stage}",
            "allowed_labels": " ".join(map(str, _lifelong_stage_labels(cfg, stage=eval_stage))),
            "n_eval_samples": len(dataset),
        }
        row.update(metrics)
        rows.append(row)
        for key, val in metrics.items():
            if isinstance(val, (int, float)):
                mlflow.log_metric(f"lifelong.after_stage{train_stage}.eval_stage{eval_stage}.{key}", val)
        logger.info(
            "Lifelong eval after train_stage=%s on eval_stage=%s: causal disentanglement = %.4f",
            train_stage,
            eval_stage,
            float(metrics.get("causal_disentanglement", float("nan"))),
        )
    full_dataset = TensorDataset(*torch.load(Path(cfg.data.data_dir) / "test.pt"))
    full_metrics = _eval_metrics_on_dataset(cfg, model, full_dataset)
    full_row = {
        "train_stage": int(train_stage),
        "eval_stage": "full",
        "eval_partition": "test_full",
        "allowed_labels": "all",
        "n_eval_samples": len(full_dataset),
    }
    full_row.update(full_metrics)
    rows.append(full_row)
    for key, val in full_metrics.items():
        if isinstance(val, (int, float)):
            mlflow.log_metric(f"lifelong.after_stage{train_stage}.eval_full.{key}", val)
    logger.info(
        "Lifelong eval after train_stage=%s on full test: causal disentanglement = %.4f",
        train_stage,
        float(full_metrics.get("causal_disentanglement", float("nan"))),
    )
    return rows

def evaluate(cfg, model):
    logger.info("Starting evaluation")
    test_metrics = eval_dci_scores(cfg, model, partition=cfg.eval.eval_partition)
    raw_enco_adjacency = _compute_enco_adjacency(cfg, model, partition=cfg.eval.eval_partition)
    test_metrics.update(_adjacency_matrix_to_metric_dict("enco_graph", raw_enco_adjacency))
    if cfg.eval.get("causal_pruning", {}).get("enabled", False):
        test_metrics.update(
            eval_intervention_contrast_pruned_graph(
                cfg,
                model,
                raw_enco_adjacency=raw_enco_adjacency,
                partition=cfg.eval.causal_pruning.get("partition", cfg.eval.eval_partition),
            )
        )
    test_metrics.update(eval_implicit_graph(cfg, model, partition=cfg.eval.eval_partition))
    test_metrics.update(eval_test_metrics(cfg, model))
    if _lifelong_enabled(cfg):
        test_metrics["lifelong_stage"] = _lifelong_active_stage(cfg)
        test_metrics["lifelong_num_stages"] = _lifelong_num_stages(cfg)
        test_metrics["lifelong_sequential"] = float(_lifelong_sequential_enabled(cfg))
        test_metrics["lifelong_allowed_label_count"] = len(_lifelong_stage_labels(cfg))
    if cfg.plotting.get("save_three_d_diagnostics", True):
        three_d_stats = save_three_d_diagnostics(model, Path(cfg.general.exp_dir), prefix="three_d")
        for key, val in three_d_stats.items():
            if isinstance(val, (int, float)):
                test_metrics[f"three_d_{key}"] = val
    if cfg.plotting.get("save_three_d_spiking_diagnostics", True):
        three_d_spiking_stats = save_three_d_spiking_diagnostics(model, Path(cfg.general.exp_dir), prefix="three_d_spiking")
        for key, val in three_d_spiking_stats.items():
            if isinstance(val, (int, float)):
                test_metrics[f"three_d_spiking_{key}"] = val
    for key, val in test_metrics.items():
        mlflow.log_metric(f"eval.{key}", val)
    logger.info(
        f"Final evaluation: causal disentanglement = {test_metrics['causal_disentanglement']:.2f}"
    )
    test_metrics_ = {key: [val] for key, val in test_metrics.items()}
    df = pd.DataFrame.from_dict(test_metrics_)
    df.to_csv(Path(cfg.general.exp_dir) / "metrics" / "test_metrics.csv")
    return test_metrics
@torch.no_grad()
def eval_test_metrics(cfg, model):
    device = torch.device(cfg.training.device)
    model = model.to(device)
    criteria = VAEMetrics(dim_z=cfg.data.dim_z)
    test_data = load_dataset(cfg, cfg.eval.eval_partition)
    test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)
    _, _, metrics = compute_metrics_on_dataset(
        cfg, model, criteria, test_loader, device=torch.device(cfg.training.device)
    )
    return metrics
@torch.no_grad()
def eval_dci_scores(cfg, model, partition="test", full_importance_matrix=True):
    model.eval()
    device = torch.device(cfg.training.device)
    model = model.to(device)
    x_train, _, train_true_z, *_ = load_dataset(cfg, "dci_train").tensors
    x_test, _, test_true_z, *_ = load_dataset(cfg, partition).tensors
    train_model_z = model.encode_to_causal(x_train.to(device), deterministic=True)
    test_model_z = model.encode_to_causal(x_test.to(device), deterministic=True)
    causal_dci_metrics = compute_dci(
        train_true_z,
        train_model_z,
        test_true_z,
        test_model_z,
        return_full_importance_matrix=full_importance_matrix,
    )
    renamed_metrics = {}
    for key, val in causal_dci_metrics.items():
        renamed_metrics[f"causal_{key}"] = val
    return renamed_metrics
@torch.no_grad()
def eval_implicit_graph(cfg, model, partition="val"):
    if cfg.model.type not in ["intervention_noise_vae", "alt_intervention_noise_vae"]:
        return {}
    if cfg.model.dim_z > 5:
        return {}
    model.eval()
    device = torch.device(cfg.training.device)
    x, *_ = load_dataset(cfg, partition).tensors
    noise = model.encode_to_noise(x.to(device), deterministic=True).detach()
    causal_effects, topological_order = compute_implicit_causal_effects(model, noise)
    results = {
        f"implicit_graph_{i}_{j}": causal_effects[i, j].item()
        for i in range(model.dim_z)
        for j in range(model.dim_z)
    }
    return results
def _compute_enco_adjacency(cfg, model, partition="train"):
    if cfg.model.type not in ["intervention_noise_vae", "alt_intervention_noise_vae"]:
        return None
    logger.info("Evaluating learned graph")
    model.eval()
    device = torch.device(cfg.training.device)
    with torch.no_grad():
        x0, x1, *_ = load_dataset(cfg, partition).tensors
        _, _, _, _, e0, e1, _, _, intervention = model.encode_decode_pair(
            x0.to(device), x1.to(device)
        )
        z0 = model.scm.noise_to_causal(e0)
        z1 = model.scm.noise_to_causal(e1)
    adjacency_matrix = (
        run_enco(z0, z1, intervention, lambda_sparse=cfg.eval.enco_lambda, device=device)
        .cpu()
        .detach()
    )
    logger.info(f"Adjacency matrix: {adjacency_matrix}")
    return adjacency_matrix

def _adjacency_matrix_to_metric_dict(prefix, adjacency_matrix):
    if adjacency_matrix is None:
        return {}
    dim_z = int(adjacency_matrix.shape[0])
    return {
        f"{prefix}_{i}_{j}": adjacency_matrix[i, j].item()
        for i in range(dim_z)
        for j in range(dim_z)
    }

def eval_enco_graph(cfg, model, partition="train"):
    adjacency_matrix = _compute_enco_adjacency(cfg, model, partition=partition)
    return _adjacency_matrix_to_metric_dict("enco_graph", adjacency_matrix)

@torch.no_grad()
def _compute_intervention_effect_matrix(cfg, model, dataset, use_true_targets=False):
    device = torch.device(cfg.training.device)
    model.eval()
    model = model.to(device)
    dim_z = int(cfg.model.dim_z)
    batch_size = min(int(cfg.eval.batchsize), len(dataset))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    effect_sum = torch.zeros(dim_z, dim_z, device=device)
    weight_sum = torch.zeros(dim_z, device=device)
    hard_counts = torch.zeros(dim_z, device=device)
    for batch in loader:
        x0, x1 = batch[0].to(device), batch[1].to(device)
        true_labels = batch[4].to(device).long()
        (
            _,
            _,
            e0,
            e1,
            _,
            _,
            intervention_posterior,
            most_likely_intervention_idx,
            _,
        ) = model.encode_decode_pair(x0, x1)
        z0 = model.scm.noise_to_causal(e0)
        z1 = model.scm.noise_to_causal(e1)
        delta = torch.abs(z1 - z0)
        if use_true_targets:
            target_weights = torch.zeros(x0.shape[0], dim_z, device=device)
            mask = (true_labels > 0) & (true_labels <= dim_z)
            if torch.any(mask):
                target_weights[torch.arange(x0.shape[0], device=device)[mask], true_labels[mask] - 1] = 1.0
        else:
            target_weights = intervention_posterior[:, 1 : dim_z + 1]
            hard_targets = most_likely_intervention_idx - 1
            valid = (hard_targets >= 0) & (hard_targets < dim_z)
            if torch.any(valid):
                hard_counts += torch.bincount(hard_targets[valid], minlength=dim_z).to(device).float()
        effect_sum += target_weights.transpose(0, 1) @ delta
        weight_sum += target_weights.sum(dim=0)
    effect = effect_sum / torch.clamp(weight_sum.unsqueeze(1), min=1.0)
    effect.fill_diagonal_(0.0)
    row_max = effect.max(dim=1, keepdim=True).values
    normalized = effect / torch.clamp(row_max, min=1.0e-12)
    normalized.fill_diagonal_(0.0)
    return effect.detach().cpu(), normalized.detach().cpu(), weight_sum.detach().cpu(), hard_counts.detach().cpu()

def _true_adjacency_matrix_from_cfg(cfg):
    graph = create_graph(
        cfg.data.dim_z, cfg.data.nature.mode, cfg.data.nature.seed, permutation=None
    )
    return graph.adjacency_matrix.to(torch.float32).cpu()

def _graph_scores(predicted, target, prefix):
    predicted = predicted.to(torch.float32).cpu()
    target = target.to(torch.float32).cpu()
    mask = 1.0 - torch.eye(predicted.shape[0])
    pred = (predicted * mask) > 0.5
    truth = (target * mask) > 0.5
    tp = int((pred & truth).sum().item())
    fp = int((pred & (~truth)).sum().item())
    fn = int(((~pred) & truth).sum().item())
    tn = int(((~pred) & (~truth) & (mask > 0)).sum().item())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
    return {
        f"{prefix}_tp": tp,
        f"{prefix}_fp": fp,
        f"{prefix}_fn": fn,
        f"{prefix}_tn": tn,
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_f1": f1,
        f"{prefix}_shd": fp + fn,
        f"{prefix}_edge_count": int(pred.sum().item()),
    }

def _prune_adjacency_with_effects(cfg, adjacency_matrix, normalized_effect, stage_support=None):
    pruning_cfg = cfg.eval.get("causal_pruning", {})
    relative_threshold = float(pruning_cfg.get("relative_threshold", 0.35))
    absolute_threshold = float(pruning_cfg.get("absolute_threshold", 0.0))
    min_stage_support = int(pruning_cfg.get("min_stage_support", 1))
    effect = normalized_effect.clone().to(torch.float32)
    raw = (adjacency_matrix.to(torch.float32).cpu() > 0.5).to(torch.float32)
    keep = (effect >= relative_threshold).to(torch.float32)
    if absolute_threshold > 0.0:
        keep = keep * (effect >= absolute_threshold).to(torch.float32)
    if stage_support is not None and min_stage_support > 0:
        keep = keep * (stage_support.to(torch.float32) >= float(min_stage_support)).to(torch.float32)
    keep.fill_diagonal_(0.0)
    return raw * keep

def _save_matrix_csv(matrix, path, index_prefix="z", columns_prefix="z"):
    matrix = matrix.detach().cpu() if isinstance(matrix, torch.Tensor) else torch.as_tensor(matrix)
    dim = matrix.shape[0]
    df = pd.DataFrame(
        matrix.numpy(),
        index=[f"{index_prefix}{i}" for i in range(dim)],
        columns=[f"{columns_prefix}{j}" for j in range(dim)],
    )
    df.to_csv(path)

def eval_intervention_contrast_pruned_graph(cfg, model, raw_enco_adjacency=None, partition="test"):
    pruning_cfg = cfg.eval.get("causal_pruning", {})
    if not pruning_cfg.get("enabled", False):
        return {}
    metrics_dir = Path(cfg.general.exp_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    if raw_enco_adjacency is None:
        raw_enco_adjacency = _compute_enco_adjacency(cfg, model, partition=partition)
    dataset = load_dataset(cfg, partition)
    effect, normalized_effect, weight_sum, hard_counts = _compute_intervention_effect_matrix(
        cfg, model, dataset, use_true_targets=False
    )
    stage_support = None
    stage_rows = []
    if bool(pruning_cfg.get("use_stage_support", True)) and _lifelong_enabled(cfg):
        n_stages = _lifelong_num_stages(cfg)
        relative_threshold = float(pruning_cfg.get("relative_threshold", 0.35))
        supports = []
        for stage in range(n_stages):
            stage_dataset = _lifelong_stage_dataset(cfg, "test", stage)
            _, stage_norm, stage_weight_sum, stage_hard_counts = _compute_intervention_effect_matrix(
                cfg, model, stage_dataset, use_true_targets=False
            )
            stage_keep = (stage_norm >= relative_threshold).to(torch.float32)
            stage_keep.fill_diagonal_(0.0)
            supports.append(stage_keep)
            for i in range(int(cfg.model.dim_z)):
                stage_rows.append(
                    {
                        "stage": int(stage),
                        "source": int(i),
                        "posterior_weight_sum": float(stage_weight_sum[i].item()),
                        "hard_target_count": float(stage_hard_counts[i].item()),
                    }
                )
        stage_support = torch.stack(supports, dim=0).sum(dim=0)
    pruned_adjacency = _prune_adjacency_with_effects(
        cfg, raw_enco_adjacency, normalized_effect, stage_support=stage_support
    )
    true_adjacency = _true_adjacency_matrix_from_cfg(cfg)
    _save_matrix_csv(raw_enco_adjacency, metrics_dir / "raw_enco_adjacency.csv")
    _save_matrix_csv(effect, metrics_dir / "intervention_effect_matrix.csv")
    _save_matrix_csv(normalized_effect, metrics_dir / "intervention_effect_matrix_normalized.csv")
    _save_matrix_csv(pruned_adjacency, metrics_dir / "pruned_causal_adjacency.csv")
    _save_matrix_csv(true_adjacency, metrics_dir / "true_adjacency.csv")
    if stage_support is not None:
        _save_matrix_csv(stage_support, metrics_dir / "causal_pruning_stage_support.csv")
    pd.DataFrame(
        {
            "source": list(range(int(cfg.model.dim_z))),
            "posterior_weight_sum": [float(x) for x in weight_sum.tolist()],
            "hard_target_count": [float(x) for x in hard_counts.tolist()],
        }
    ).to_csv(metrics_dir / "intervention_effect_target_support.csv", index=False)
    if stage_rows:
        pd.DataFrame(stage_rows).to_csv(metrics_dir / "intervention_effect_stage_target_support.csv", index=False)
    logger.info(f"Intervention-effect normalized matrix:\n{normalized_effect}")
    logger.info(f"Pruned causal adjacency matrix:\n{pruned_adjacency}")
    results = {}
    results.update(_adjacency_matrix_to_metric_dict("pruned_causal_graph", pruned_adjacency))
    results.update(_graph_scores(raw_enco_adjacency, true_adjacency, "raw_enco_graph"))
    results.update(_graph_scores(pruned_adjacency, true_adjacency, "pruned_causal_graph"))
    for i in range(int(cfg.model.dim_z)):
        for j in range(int(cfg.model.dim_z)):
            results[f"intervention_effect_{i}_{j}"] = float(effect[i, j].item())
            results[f"intervention_effect_normalized_{i}_{j}"] = float(normalized_effect[i, j].item())
            if stage_support is not None:
                results[f"causal_pruning_stage_support_{i}_{j}"] = float(stage_support[i, j].item())
    pd.DataFrame([results]).to_csv(metrics_dir / "causal_pruning_summary.csv", index=False)
    logger.info(
        "Raw ENCO graph SHD=%s, pruned graph SHD=%s",
        results.get("raw_enco_graph_shd"),
        results.get("pruned_causal_graph_shd"),
    )
    return results

def epoch_schedules(cfg, model, epoch, optim):
    pretrain = cfg.training.pretrain_epochs is not None and epoch < cfg.training.pretrain_epochs
    if epoch == cfg.training.pretrain_epochs:
        logger.info(f"Stopping pretraining at epoch {epoch}")
    model_interventions = (
        cfg.training.model_interventions_after_epoch is None
        or epoch >= cfg.training.model_interventions_after_epoch
    )
    if epoch == cfg.training.model_interventions_after_epoch:
        logger.info(f"Beginning to model intervention distributions at epoch {epoch}")
    if cfg.training.freeze_encoder_epoch is not None and epoch == cfg.training.freeze_encoder_epoch:
        logger.info(f"Freezing encoder and decoder at epoch {epoch}")
        optim.param_groups[0]["lr"] = 0.0
    if (
        "fix_topological_order_epoch" in cfg.training
        and cfg.training.fix_topological_order_epoch is not None
        and epoch == cfg.training.fix_topological_order_epoch
    ):
        logger.info(f"Determining topological order at epoch {epoch}")
        fix_topological_order(cfg, model, partition="val")
    if cfg.training.deterministic_intervention_encoder_after_epoch is None:
        deterministic_intervention_encoder = False
    else:
        deterministic_intervention_encoder = (
            epoch >= cfg.training.deterministic_intervention_encoder_after_epoch
        )
    if epoch == cfg.training.deterministic_intervention_encoder_after_epoch:
        logger.info(f"Switching to deterministic intervention encoder at epoch {epoch}")
    return model_interventions, pretrain, deterministic_intervention_encoder
@torch.no_grad()
def fix_topological_order(cfg, model, partition="val", dataloader=None):
    assert cfg.model.type == "intervention_noise_vae"
    model.eval()
    device = torch.device(cfg.training.device)
    cpu = torch.device("cpu")
    model.to(device)
    if dataloader is None:
        dataset = load_dataset(cfg, partition)
        dataloader = DataLoader(dataset, batch_size=cfg.eval.batchsize, shuffle=False)
    noise = []
    for x_batch, *_ in dataloader:
        x_batch = x_batch.to(device)
        noise.append(model.encode_to_noise(x_batch, deterministic=True).to(cpu))
    noise = torch.cat(noise, dim=0).detach()
    dummy_values = torch.median(noise, dim=0).values
    logger.info(f"Dummy noise encodings: {dummy_values}")
    model = model.to(cpu)
    topological_order = find_topological_order(model, noise)
    logger.info(f"Topological order: {topological_order}")
    model.scm.set_causal_structure(
        None, "fixed_order", topological_order=topological_order, mask_values=dummy_values
    )
    model.to(device)
if __name__ == "__main__":
    main()




