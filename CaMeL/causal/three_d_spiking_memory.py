
import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class ThreeDSpikingMemory(nn.Module):
    def __init__(
        self,
        dim_z: int,
        n_interventions: Optional[int] = None,
        neurons: int = 27,
        time_steps: int = 8,
        layout: str = "cube",
        seed: int = 0,
        threshold: float = 0.5,
        decay: float = 0.85,
        input_scale: float = 0.2,
        recurrent_scale: float = 0.05,
        distance_power: float = 1.0,
        train_recurrent: bool = False,
        target_spike_rate: float = 0.08,
        surrogate_slope: float = 8.0,
        posterior_blend: float = 0.0,
        enable_causal_plasticity: bool = True,
        causal_synapse_scale: float = 0.03,
        plasticity_drive_scale: float = 0.05,
        plasticity_effect_power: float = 1.0,
        detach_plasticity_teacher: bool = True,
    ):
        super().__init__()
        self.dim_z = int(dim_z)
        self.n_interventions = int(n_interventions or (self.dim_z + 1))
        self.neurons = int(neurons)
        self.time_steps = int(time_steps)
        self.layout = str(layout)
        self.threshold = float(threshold)
        self.decay = float(decay)
        self.distance_power = float(distance_power)
        self.target_spike_rate = float(target_spike_rate)
        self.surrogate_slope = float(surrogate_slope)
        self.posterior_blend = float(posterior_blend)
        self.enable_causal_plasticity = bool(enable_causal_plasticity)
        self.plasticity_drive_scale = float(plasticity_drive_scale)
        self.plasticity_effect_power = float(plasticity_effect_power)
        self.detach_plasticity_teacher = bool(detach_plasticity_teacher)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))

        positions = self._make_positions(self.neurons, self.layout, generator)
        distances = torch.cdist(positions, positions)
        distances = distances / (distances.max() + 1.0e-8)
        distance_bias = torch.exp(-torch.pow(distances + 1.0e-8, self.distance_power))
        distance_bias.fill_diagonal_(0.0)
        mask = (distance_bias > distance_bias.mean()).to(torch.float32)
        mask.fill_diagonal_(0.0)

        recurrent = torch.randn(self.neurons, self.neurons, generator=generator) * recurrent_scale
        recurrent = recurrent * mask * distance_bias
        input_features = 3 * self.dim_z

        variable_to_neuron, variable_positions = self._make_variable_assemblies(
            positions, self.dim_z, self.layout, generator
        )
        variable_distances = torch.cdist(variable_positions, variable_positions)
        variable_distances = variable_distances / (variable_distances.max() + 1.0e-8)
        causal_mask = torch.ones(self.dim_z, self.dim_z, dtype=torch.float32)
        causal_mask.fill_diagonal_(0.0)

        self.register_buffer("positions", positions)
        self.register_buffer("distances", distances)
        self.register_buffer("distance_bias", distance_bias)
        self.register_buffer("recurrent_mask", mask)
        self.register_buffer("variable_to_neuron", variable_to_neuron)
        self.register_buffer("variable_positions", variable_positions)
        self.register_buffer("variable_distances", variable_distances)
        self.register_buffer("causal_mask", causal_mask)

        self.input_projection = nn.Linear(input_features, self.neurons)
        with torch.no_grad():
            self.input_projection.weight.mul_(input_scale)
            self.input_projection.bias.zero_()

        if train_recurrent:
            self.recurrent_weight = nn.Parameter(recurrent)
        else:
            self.register_buffer("recurrent_weight", recurrent)

        if self.enable_causal_plasticity:
            init = -4.0 + torch.randn(self.dim_z, self.dim_z, generator=generator) * float(causal_synapse_scale)
            init = init * causal_mask
            self.causal_synapses = nn.Parameter(init)
        else:
            self.register_buffer("causal_synapses", torch.zeros(self.dim_z, self.dim_z))

        self.readout = nn.Sequential(
            nn.Linear(2 * self.neurons, max(16, self.neurons // 2)),
            nn.SiLU(),
            nn.Linear(max(16, self.neurons // 2), self.n_interventions),
        )

    @staticmethod
    def _make_positions(neurons: int, layout: str, generator: torch.Generator) -> torch.Tensor:
        if layout == "line":
            x = torch.linspace(0.0, 1.0, neurons)
            return torch.stack([x, torch.zeros_like(x), torch.zeros_like(x)], dim=1)
        if layout == "random":
            return torch.rand((neurons, 3), generator=generator)
        side = int(math.ceil(neurons ** (1.0 / 3.0)))
        xs, ys, zs = torch.meshgrid(
            torch.linspace(0.0, 1.0, side),
            torch.linspace(0.0, 1.0, side),
            torch.linspace(0.0, 1.0, side),
            indexing="ij",
        )
        positions = torch.stack([xs.flatten(), ys.flatten(), zs.flatten()], dim=1)
        return positions[:neurons].contiguous()

    @classmethod
    def _make_variable_assemblies(
        cls,
        positions: torch.Tensor,
        dim_z: int,
        layout: str,
        generator: torch.Generator,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        anchors = cls._make_positions(int(dim_z), layout, generator)
        if anchors.shape[0] < dim_z:
            pad = torch.rand((dim_z - anchors.shape[0], 3), generator=generator)
            anchors = torch.cat([anchors, pad], dim=0)
        distances = torch.cdist(anchors[:dim_z], positions)
        assignment = torch.argmin(distances, dim=0)
        membership = torch.zeros(dim_z, positions.shape[0], dtype=torch.float32)
        membership[assignment, torch.arange(positions.shape[0])] = 1.0

        for k in range(dim_z):
            if membership[k].sum() == 0:
                nearest = torch.argmin(distances[k])
                membership[:, nearest] = 0.0
                membership[k, nearest] = 1.0
        membership = membership / membership.sum(dim=1, keepdim=True).clamp_min(1.0)
        variable_positions = membership @ positions
        return membership, variable_positions

    def _surrogate_spike(self, membrane_minus_threshold: torch.Tensor) -> torch.Tensor:
        soft = torch.sigmoid(self.surrogate_slope * membrane_minus_threshold)
        hard = (soft > 0.5).to(soft.dtype)
        return hard.detach() - soft.detach() + soft

    def _target_probs(self, intervention_posterior: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if intervention_posterior is None:
            return None
        if intervention_posterior.shape[1] >= self.dim_z + 1:
            target_probs = intervention_posterior[:, 1 : self.dim_z + 1]
        else:
            target_probs = intervention_posterior[:, : self.dim_z]
            if target_probs.shape[1] < self.dim_z:
                pad = torch.zeros(
                    target_probs.shape[0],
                    self.dim_z - target_probs.shape[1],
                    device=target_probs.device,
                    dtype=target_probs.dtype,
                )
                target_probs = torch.cat([target_probs, pad], dim=1)
        return target_probs

    def causal_strength(self) -> torch.Tensor:
        if not self.enable_causal_plasticity:
            return torch.zeros_like(self.causal_synapses) * self.causal_mask
        return F.softplus(self.causal_synapses) * self.causal_mask

    def _causal_drive(
        self,
        intervention_teacher: Optional[torch.Tensor],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        target_probs = self._target_probs(intervention_teacher)
        if target_probs is None or self.plasticity_drive_scale <= 0.0:
            return None
        if self.detach_plasticity_teacher:
            target_probs = target_probs.detach()
        target_probs = target_probs.to(device=device, dtype=dtype)
        strength = self.causal_strength().to(device=device, dtype=dtype)
        variable_drive = target_probs @ strength.t()
        neuron_drive = variable_drive @ self.variable_to_neuron.to(device=device, dtype=dtype)
        return self.plasticity_drive_scale * neuron_drive

    def intervention_gated_plasticity_outputs(
        self,
        e1: torch.Tensor,
        e2: torch.Tensor,
        intervention_teacher: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}
        strength = self.causal_strength().to(device=e1.device, dtype=e1.dtype)
        strength_sum = strength.sum().clamp_min(1.0e-8)
        strength_norm = strength / strength_sum
        distance_reg = (strength * self.variable_distances.to(e1.device, e1.dtype)).sum() / strength_sum
        synapse_l1 = strength.mean()

        outputs["three_d_spiking_causal_distance_regularization"] = distance_reg.expand(e1.shape[0], 1)
        outputs["three_d_spiking_causal_synapse_l1"] = synapse_l1.expand(e1.shape[0], 1)

        target_probs = self._target_probs(intervention_teacher)
        if target_probs is None:
            return outputs
        if self.detach_plasticity_teacher:
            target_probs = target_probs.detach()
        target_probs = target_probs.to(device=e1.device, dtype=e1.dtype)

        effects = torch.abs(e2 - e1).clamp_min(1.0e-8)
        if self.plasticity_effect_power != 1.0:
            effects = torch.pow(effects, self.plasticity_effect_power)
        effects = effects / effects.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        evidence = effects.t() @ target_probs / float(max(1, e1.shape[0]))
        evidence = evidence * self.causal_mask.to(e1.device, e1.dtype)
        evidence_mass = evidence.sum().clamp_min(1.0e-8)
        evidence_norm = evidence / evidence_mass

        mse_loss = torch.mean((strength_norm - evidence_norm.detach()) ** 2)
        dot = torch.sum(strength_norm * evidence_norm.detach())
        denom = torch.sqrt(torch.sum(strength_norm ** 2) * torch.sum(evidence_norm.detach() ** 2)).clamp_min(1.0e-8)
        alignment = dot / denom

        outputs["three_d_spiking_causal_plasticity_loss"] = mse_loss.expand(e1.shape[0], 1)
        outputs["three_d_spiking_causal_alignment"] = alignment.expand(e1.shape[0], 1)
        outputs["three_d_spiking_causal_evidence_mass"] = evidence_mass.expand(e1.shape[0], 1)
        outputs["three_d_spiking_causal_mean_effect"] = effects.mean().expand(e1.shape[0], 1)
        return outputs

    def _simulate_spiking(
        self,
        e1: torch.Tensor,
        e2: torch.Tensor,
        intervention_teacher: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        features = torch.cat([e1, e2, e2 - e1], dim=1)
        base_drive = self.input_projection(features)
        causal_drive = self._causal_drive(intervention_teacher, dtype=e1.dtype, device=e1.device)
        drive = base_drive if causal_drive is None else base_drive + causal_drive

        membrane = torch.zeros(e1.shape[0], self.neurons, device=e1.device, dtype=e1.dtype)
        spikes = torch.zeros_like(membrane)
        spike_trace = []
        membrane_trace = []

        recurrent = self.recurrent_weight * self.recurrent_mask
        for _ in range(self.time_steps):
            recurrent_current = torch.matmul(spikes, recurrent.t())
            membrane = self.decay * membrane + drive + recurrent_current
            spikes = self._surrogate_spike(membrane - self.threshold)
            membrane = membrane * (1.0 - spikes.detach())
            spike_trace.append(spikes)
            membrane_trace.append(membrane)

        spike_trace = torch.stack(spike_trace, dim=1)
        membrane_trace = torch.stack(membrane_trace, dim=1)
        mean_spikes = torch.mean(spike_trace, dim=1)
        last_membrane = membrane_trace[:, -1, :]
        logits = self.readout(torch.cat([mean_spikes, last_membrane], dim=1))
        posterior = torch.softmax(logits, dim=1)
        return {
            "features": features,
            "base_drive": base_drive,
            "causal_drive": causal_drive,
            "total_drive": drive,
            "spike_trace": spike_trace,
            "membrane_trace": membrane_trace,
            "mean_spikes": mean_spikes,
            "last_membrane": last_membrane,
            "logits": logits,
            "posterior": posterior,
        }

    @torch.no_grad()
    def analyze_pair(
        self,
        e1: torch.Tensor,
        e2: torch.Tensor,
        intervention_teacher: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = self._simulate_spiking(e1, e2, intervention_teacher=intervention_teacher)
        return outputs

    def forward(
        self,
        e1: torch.Tensor,
        e2: torch.Tensor,
        intervention_teacher: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        sim = self._simulate_spiking(e1, e2, intervention_teacher=intervention_teacher)
        spike_trace = sim["spike_trace"]
        logits = sim["logits"]
        posterior = sim["posterior"]

        outputs: Dict[str, torch.Tensor] = {
            "three_d_spiking_logits": logits,
            "three_d_spiking_intervention_posterior": posterior,
            "three_d_spiking_spike_rate": torch.mean(spike_trace, dim=(1, 2), keepdim=False).unsqueeze(1),
            "three_d_spiking_spike_sparsity": torch.mean((spike_trace < 0.5).to(e1.dtype), dim=(1, 2), keepdim=False).unsqueeze(1),
            "three_d_spiking_distance_regularization": self.distance_regularization().expand(e1.shape[0], 1),
        }
        spike_rate = outputs["three_d_spiking_spike_rate"]
        outputs["three_d_spiking_spike_rate_loss"] = (spike_rate - self.target_spike_rate) ** 2
        outputs.update(self.intervention_gated_plasticity_outputs(e1, e2, intervention_teacher))

        if intervention_teacher is not None:
            teacher = intervention_teacher.detach().clamp_min(1.0e-8)
            teacher = teacher / teacher.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
            log_posterior = torch.log(posterior.clamp_min(1.0e-8))
            kl = torch.sum(teacher * (torch.log(teacher) - log_posterior), dim=1, keepdim=True)
            outputs["three_d_spiking_consistency_kl"] = kl
        return outputs

    def blended_posterior(
        self,
        ilcm_posterior: torch.Tensor,
        three_d_spiking_posterior: torch.Tensor,
    ) -> torch.Tensor:
        alpha = max(0.0, min(1.0, self.posterior_blend))
        if alpha <= 0.0:
            return ilcm_posterior
        posterior = (1.0 - alpha) * ilcm_posterior + alpha * three_d_spiking_posterior
        return posterior / posterior.sum(dim=1, keepdim=True).clamp_min(1.0e-8)

    def distance_regularization(self) -> torch.Tensor:
        W = torch.abs(self.recurrent_weight * self.recurrent_mask)
        denom = torch.sum(W).clamp_min(1.0e-8)
        return torch.sum(W * self.distances) / denom

    @torch.no_grad()
    def get_summary(self) -> Dict[str, float]:
        W = torch.abs(self.recurrent_weight * self.recurrent_mask).detach()
        active = (W > 0).to(torch.float32)
        causal = self.causal_strength().detach()
        causal_active = (causal > 1.0e-8).to(torch.float32)
        causal_sum = causal.sum().clamp_min(1.0e-8)
        return {
            "neurons": float(self.neurons),
            "time_steps": float(self.time_steps),
            "active_recurrent_edges": float(active.sum().item()),
            "mean_recurrent_distance": float((active * self.distances).sum().item() / active.sum().clamp_min(1.0).item()),
            "weighted_recurrent_distance": float(self.distance_regularization().item()),
            "posterior_blend": float(self.posterior_blend),
            "causal_plasticity_enabled": float(1.0 if self.enable_causal_plasticity else 0.0),
            "active_causal_synapses": float(causal_active.sum().item()),
            "causal_synapse_l1": float(causal.mean().item()),
            "weighted_causal_distance": float((causal * self.variable_distances).sum().item() / causal_sum.item()),
        }

    @torch.no_grad()
    def export_matrices(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.positions.detach().cpu(),
            self.distances.detach().cpu(),
            (self.recurrent_weight * self.recurrent_mask).detach().cpu(),
        )

    @torch.no_grad()
    def export_causal_matrices(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.variable_positions.detach().cpu(),
            self.variable_distances.detach().cpu(),
            self.variable_to_neuron.detach().cpu(),
            self.causal_strength().detach().cpu(),
        )

