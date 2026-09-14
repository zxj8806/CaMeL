
import itertools
import numpy as np
import torch
from torch import nn
class InterventionPrior(nn.Module):
    def __init__(self, permutation=0, dim_z=2, intervention_set="atomic_or_none"):
        super().__init__()
        assert intervention_set in {"atomic_or_none", "atomic", "all"}
        self.intervention_set = intervention_set
        self.dim_z = dim_z
        _masks, self.n_interventions = self._generate_interventions()
        self.register_buffer("_masks", _masks)
        _permutation, _inverse_permutation = self._generate_permutation(permutation)
        self.register_buffer("_permutation", _permutation)
        self.register_buffer("_inverse_permutation", _inverse_permutation)
    def forward(self, intervention_label, convert_to_int=False):
        assert len(intervention_label.shape) == 1
        assert torch.all(intervention_label >= 0)
        assert torch.all(intervention_label < self.n_interventions)
        intervention_idx = torch.index_select(self._permutation, 0, intervention_label)
        intervention = torch.index_select(self._masks, 0, intervention_idx)
        if convert_to_int:
            intervention = intervention.to(torch.int)
        return intervention
    def inverse(self, intervention):
        assert len(intervention.shape) == 2
        assert intervention.shape[1] == self.dim_z
        if self.intervention_set == "all":
            intervention_idx = 0
            for i, intervention_i in enumerate(intervention.T):
                intervention_idx = intervention_idx + 2**i * intervention_i
        else:
            intervention_idx = 0
            for i, intervention_i in enumerate(intervention.T):
                intervention_idx = intervention_idx + intervention_i
        assert torch.all(intervention_idx < self.n_interventions)
        intervention_label = torch.index_select(self._inverse_permutation, 0, intervention_idx)
        return intervention_label
    def sample(self, n, convert_to_int=True):
        intervention_labels = torch.randint(
            self.n_interventions, size=(n,), device=self._masks.device
        )
        interventions = self(intervention_labels, convert_to_int=convert_to_int)
        return interventions, intervention_labels
    def _generate_interventions(self):
        if self.intervention_set == "all":
            n_interventions = 2**self.dim_z
            masks = []
            for idx in range(n_interventions):
                masks.append(
                    torch.BoolTensor([(idx >> k) & 1 for k in range(0, self.dim_z)]).unsqueeze(0)
                )
        elif self.intervention_set == "atomic_or_none":
            n_interventions = self.dim_z + 1
            masks = [torch.BoolTensor([False for _ in range(self.dim_z)]).unsqueeze(0)]
            for idx in range(self.dim_z):
                masks.append(torch.BoolTensor([(idx == k) for k in range(self.dim_z)]).unsqueeze(0))
        elif self.intervention_set == "atomic":
            n_interventions = self.dim_z
            masks = []
            for idx in range(n_interventions):
                masks.append(torch.BoolTensor([(idx == k) for k in range(self.dim_z)]).unsqueeze(0))
        else:
            raise ValueError(f"Unknown intervention set {self.intervention_set}")
        assert len(masks) == n_interventions
        return torch.cat(masks, 0), n_interventions
    def _generate_permutation(self, permutation):
        idx = list(range(self.n_interventions))
        permutation_ = None
        for i, perm in enumerate(itertools.permutations(idx)):
            if i == permutation:
                permutation_ = perm
                break
        assert permutation_ is not None
        permutation_tensor = torch.IntTensor(permutation_)
        inverse_permutation_tensor = torch.IntTensor(np.argsort(permutation_))
        return permutation_tensor, inverse_permutation_tensor
class HeuristicInterventionEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(-3.0))
        self.beta = torch.nn.Parameter(torch.tensor(15.0))
        self.gamma = torch.nn.Parameter(torch.tensor(15.0))
    def forward(self, inputs, eps=1.0e-4):
        dim_z = inputs.shape[1] // 2
        delta_epsilon = inputs[:, dim_z:]
        intervention_logits = (
            self.alpha + self.beta * torch.abs(delta_epsilon) + self.gamma * delta_epsilon**2
        )
        no_intervention_logit = torch.zeros((inputs.shape[0], 1), device=inputs.device)
        logits = torch.cat((no_intervention_logit, intervention_logits), dim=1)
        probs = torch.nn.functional.softmax(logits, dim=1)
        probs = probs + eps
        probs = probs / torch.sum(probs, dim=1, keepdim=True)
        return probs
    def get_parameters(self):
        return dict(alpha=self.alpha, beta=self.beta, gamma=self.gamma)






