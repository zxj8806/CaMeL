
import itertools
import torch
from torch import nn
LOG_MEAN_VARS = {
    "elbo",
    "kl",
    "kl_epsilon",
    "kl_intervention_target",
    "mse",
    "consistency_mse",
    "inverse_consistency_mse",
    "log_prior",
    "log_prior_observed",
    "log_prior_intervened",
    "log_prior_nonintervened",
    "log_likelihood",
    "log_posterior",
    "z_regularization",
    "edges",
    "cyclicity",
    "encoder_std",
    "three_d_implicit_edge_regularization",
    "three_d_graph_edge_regularization",
    "three_d_spiking_consistency_kl",
    "three_d_spiking_spike_rate",
    "three_d_spiking_spike_sparsity",
    "three_d_spiking_spike_rate_loss",
    "three_d_spiking_distance_regularization",
    "three_d_spiking_causal_plasticity_loss",
    "three_d_spiking_causal_alignment",
    "three_d_spiking_causal_distance_regularization",
    "three_d_spiking_causal_synapse_l1",
    "three_d_spiking_causal_evidence_mass",
    "three_d_spiking_causal_mean_effect",
}
class VAEMetrics(nn.Module):
    def __init__(self, dim_z=2):
        super().__init__()
        self.dim_z = dim_z
    def forward(
        self,
        loss,
        true_intervention_labels,
        solution_std=None,
        intervention_posterior=None,
        eps=1.0e-9,
        z_regularization_amount=0.0,
        consistency_regularization_amount=0.0,
        inverse_consistency_regularization_amount=0.0,
        edge_regularization_amount=0.0,
        cyclicity_regularization_amount=0.0,
        intervention_entropy_regularization_amount=0.0,
        three_d_edge_regularization_amount=0.0,
        three_d_spiking_consistency_amount=0.0,
        three_d_spiking_spike_regularization_amount=0.0,
        three_d_spiking_distance_regularization_amount=0.0,
        three_d_spiking_causal_plasticity_amount=0.0,
        three_d_spiking_causal_distance_amount=0.0,
        three_d_spiking_causal_l1_amount=0.0,
        **model_outputs,
    ):
        metrics = {}
        batchsize = loss.shape[0]
        loss = torch.mean(loss)
        loss = self._regulate(
            batchsize,
            consistency_regularization_amount,
            eps,
            intervention_entropy_regularization_amount,
            intervention_posterior,
            inverse_consistency_regularization_amount,
            loss,
            metrics,
            model_outputs,
            z_regularization_amount,
            edge_regularization_amount,
            cyclicity_regularization_amount,
            three_d_edge_regularization_amount,
            three_d_spiking_consistency_amount,
            three_d_spiking_spike_regularization_amount,
            three_d_spiking_distance_regularization_amount,
            three_d_spiking_causal_plasticity_amount,
            three_d_spiking_causal_distance_amount,
            three_d_spiking_causal_l1_amount,
        )
        assert torch.isfinite(loss)
        metrics["loss"] = loss.item()
        with torch.no_grad():
            self._evaluate_intervention_posterior(
                eps, metrics, true_intervention_labels, intervention_posterior
            )
            if "three_d_spiking_intervention_posterior" in model_outputs:
                self._evaluate_intervention_posterior(
                    eps,
                    metrics,
                    true_intervention_labels,
                    model_outputs["three_d_spiking_intervention_posterior"],
                    prefix="three_d_spiking_",
                )
            if solution_std is not None:
                for i in range(solution_std.shape[-1]):
                    metrics[f"solution_std_{i}"] = torch.mean(solution_std[..., i]).item()
            for key in LOG_MEAN_VARS:
                if key in model_outputs:
                    try:
                        metrics[key] = torch.mean(model_outputs[key].to(torch.float)).item()
                    except AttributeError:
                        metrics[key] = float(model_outputs[key])
        return loss, metrics
    def _regulate(
        self,
        batchsize,
        consistency_regularization_amount,
        eps,
        intervention_entropy_regularization_amount,
        intervention_posterior,
        inverse_consistency_regularization_amount,
        loss,
        metrics,
        model_outputs,
        z_regularization_amount,
        edge_regularization_amount,
        cyclicity_regularization_amount,
        three_d_edge_regularization_amount,
        three_d_spiking_consistency_amount,
        three_d_spiking_spike_regularization_amount,
        three_d_spiking_distance_regularization_amount,
        three_d_spiking_causal_plasticity_amount,
        three_d_spiking_causal_distance_amount,
        three_d_spiking_causal_l1_amount,
    ):
        if edge_regularization_amount is not None and "edges" in model_outputs:
            loss += edge_regularization_amount * torch.mean(model_outputs["edges"])
        if cyclicity_regularization_amount is not None and "cyclicity" in model_outputs:
            try:
                loss += cyclicity_regularization_amount * torch.mean(model_outputs["cyclicity"])
            except TypeError:
                loss += cyclicity_regularization_amount * model_outputs["cyclicity"]
        if z_regularization_amount is not None and "z_regularization" in model_outputs:
            loss += z_regularization_amount * torch.mean(model_outputs["z_regularization"])
        if three_d_edge_regularization_amount is not None:
            if "three_d_implicit_edge_regularization" in model_outputs:
                loss += three_d_edge_regularization_amount * torch.mean(
                    model_outputs["three_d_implicit_edge_regularization"]
                )
            if "three_d_graph_edge_regularization" in model_outputs:
                loss += three_d_edge_regularization_amount * torch.mean(
                    model_outputs["three_d_graph_edge_regularization"]
                )

        if three_d_spiking_consistency_amount is not None and "three_d_spiking_consistency_kl" in model_outputs:
            loss += three_d_spiking_consistency_amount * torch.mean(
                model_outputs["three_d_spiking_consistency_kl"]
            )
        if three_d_spiking_spike_regularization_amount is not None and "three_d_spiking_spike_rate_loss" in model_outputs:
            loss += three_d_spiking_spike_regularization_amount * torch.mean(
                model_outputs["three_d_spiking_spike_rate_loss"]
            )
        if three_d_spiking_distance_regularization_amount is not None and "three_d_spiking_distance_regularization" in model_outputs:
            loss += three_d_spiking_distance_regularization_amount * torch.mean(
                model_outputs["three_d_spiking_distance_regularization"]
            )
        if three_d_spiking_causal_plasticity_amount is not None and "three_d_spiking_causal_plasticity_loss" in model_outputs:
            loss += three_d_spiking_causal_plasticity_amount * torch.mean(
                model_outputs["three_d_spiking_causal_plasticity_loss"]
            )
        if three_d_spiking_causal_distance_amount is not None and "three_d_spiking_causal_distance_regularization" in model_outputs:
            loss += three_d_spiking_causal_distance_amount * torch.mean(
                model_outputs["three_d_spiking_causal_distance_regularization"]
            )
        if three_d_spiking_causal_l1_amount is not None and "three_d_spiking_causal_synapse_l1" in model_outputs:
            loss += three_d_spiking_causal_l1_amount * torch.mean(
                model_outputs["three_d_spiking_causal_synapse_l1"]
            )
        if consistency_regularization_amount is not None and "consistency_mse" in model_outputs:
            loss += consistency_regularization_amount * torch.mean(model_outputs["consistency_mse"])
        if (
            inverse_consistency_regularization_amount is not None
            and "inverse_consistency_mse" in model_outputs
        ):
            loss += inverse_consistency_regularization_amount * torch.mean(
                model_outputs["inverse_consistency_mse"]
            )
        if (
            inverse_consistency_regularization_amount is not None
            and "inverse_consistency_mse" in model_outputs
        ):
            loss += inverse_consistency_regularization_amount * torch.mean(
                model_outputs["inverse_consistency_mse"]
            )
        if (
            intervention_entropy_regularization_amount is not None
            and intervention_posterior is not None
        ):
            aggregate_posterior = torch.mean(intervention_posterior, dim=0)
            intervention_entropy = -torch.sum(
                aggregate_posterior * torch.log(aggregate_posterior + eps)
            )
            loss -= (
                intervention_entropy_regularization_amount * intervention_entropy
            )
            metrics["intervention_entropy"] = intervention_entropy.item()
            most_likely_intervention = torch.argmax(intervention_posterior, dim=1)
            det_posterior = torch.zeros_like(intervention_posterior)
            det_posterior[torch.arange(batchsize), most_likely_intervention] = 1.0
            aggregate_det_posterior = torch.mean(det_posterior, dim=0)
            det_intervention_entropy = -torch.sum(
                aggregate_det_posterior * torch.log(aggregate_det_posterior + eps)
            )
            metrics["intervention_entropy_deterministic"] = det_intervention_entropy.item()
        return loss
    @torch.no_grad()
    def _evaluate_intervention_posterior(
        self, eps, metrics, true_intervention_labels, intervention_posterior, prefix=""
    ):
        if self.dim_z > 5:
            return
        if intervention_posterior is None:
            return
        batchsize = true_intervention_labels.shape[0]
        idx = torch.arange(batchsize)
        for i in range(intervention_posterior.shape[1]):
            metrics[f"{prefix}intervention_posterior_{i}"] = torch.mean(intervention_posterior[:, i]).item()
        true_int_prob, log_true_int_prob, int_accuracy = -float("inf"), -float("inf"), -float("inf")
        for permutation in itertools.permutations(list(range(1, self.dim_z + 1))):
            permutation = [0] + list(permutation)
            intervention_probs_permuted = intervention_posterior[:, permutation]
            predicted_intervention_permuted = torch.zeros_like(intervention_probs_permuted)
            predicted_intervention_permuted[
                idx, torch.argmax(intervention_probs_permuted, dim=1)
            ] = 1.0
            log_true_int_prob_ = torch.mean(
                torch.log(
                    intervention_probs_permuted[idx, true_intervention_labels.flatten()] + eps
                )
            ).item()
            log_true_int_prob = max(log_true_int_prob, log_true_int_prob_)
            true_int_prob_ = torch.mean(
                intervention_probs_permuted[idx, true_intervention_labels.flatten()]
            ).item()
            true_int_prob = max(true_int_prob, true_int_prob_)
            int_accuracy_ = torch.mean(
                predicted_intervention_permuted[idx, true_intervention_labels.flatten()]
            ).item()
            int_accuracy = max(int_accuracy, int_accuracy_)
        metrics[f"{prefix}intervention_correct_log_posterior"] = log_true_int_prob
        metrics[f"{prefix}intervention_correct_posterior"] = true_int_prob
        metrics[f"{prefix}intervention_accuracy"] = int_accuracy






