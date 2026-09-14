
from collections import defaultdict
import nflows.distributions
import nflows.nn.nets
import torch
from torch import nn
from CaMeL.causal.graph import ENCOLearnedGraph, DDSLearnedGraph, FixedOrderLearnedGraph
from CaMeL.causal.three_d_memory import ThreeDCausalMemory
from CaMeL.transforms import make_mlp_structure_transform, MaskedSolutionTransform
from CaMeL.utils import mask, clean_and_clamp
DEFAULT_BASE_DENSITY = nflows.distributions.StandardNormal((1,))
class ImplicitSCM(nn.Module):
    def __init__(
        self,
        graph,
        solution_functions,
        base_density,
        manifold_thickness,
        dim_z,
        causal_structure,
        three_d_memory=None,
    ):
        super().__init__()
        self.dim_z = dim_z
        self.solution_functions = torch.nn.ModuleList(solution_functions)
        self.base_density = base_density
        self.three_d_memory = three_d_memory
        self.register_buffer("_manifold_thickness", torch.tensor(manifold_thickness))
        self.register_buffer("_mask_values", torch.zeros(dim_z))
        self.register_buffer("topological_order", torch.zeros(dim_z, dtype=torch.long))
        self.set_causal_structure(graph, causal_structure)
    def sample(self, n, intervention=None, graph_mode="hard", graph_temperature=1.0):
        raise NotImplementedError
    def sample_weakly_supervised(self, n, intervention, graph_mode="hard", graph_temperature=1.0):
        raise NotImplementedError
    def sample_noise_weakly_supervised(self, n, intervention, adjacency_matrix=None):
        intervention = self._sanitize_intervention(intervention, n)
        epsilon1 = self._sample_noise(n)
        intervention_noise = self._sample_noise(n)
        epsilon2 = (
            intervention
            * self._inverse(intervention_noise, epsilon1, adjacency_matrix=adjacency_matrix)[0]
        )
        cf_noise = self._sample_noise(n, True)
        epsilon2 += (1.0 - intervention) * (epsilon1 + cf_noise)
        return epsilon1, epsilon2
    def log_prob_weakly_supervised(self, z1, z2, intervention, adjacency_matrix):
        raise NotImplementedError
    def log_prob_noise_weakly_supervised(
        self,
        epsilon1,
        epsilon2,
        intervention,
        adjacency_matrix,
        include_intervened=True,
        include_nonintervened=True,
    ):
        intervention = self._sanitize_intervention(intervention, epsilon1.shape[0])
        assert torch.all(torch.isfinite(epsilon1))
        assert torch.all(torch.isfinite(epsilon2))
        logprob_observed = self._compute_logprob_observed(epsilon1)
        logprob = logprob_observed
        if include_intervened:
            log_det, logprob_intervened = self._compute_logprob_intervened(
                adjacency_matrix, epsilon1, epsilon2, intervention
            )
            logprob = logprob + logprob_intervened
        else:
            logprob_intervened = torch.zeros_like(logprob_observed)
            log_det = torch.zeros((epsilon1.shape[0], 1), device=epsilon1.device)
        if include_nonintervened:
            logprob_nonintervened = self._compute_logprob_nonintervened(
                epsilon1, epsilon2, intervention
            )
            logprob = logprob + logprob_nonintervened
        else:
            logprob_nonintervened = torch.zeros_like(logprob_intervened)
        assert torch.all(torch.isfinite(logprob))
        outputs = dict(
            log_prior_observed=logprob_observed,
            log_prior_intervened=logprob_intervened,
            log_prior_nonintervened=logprob_nonintervened,
            solution_std=torch.exp(
                -log_det
            ),
        )
        return logprob, outputs
    def _compute_logprob_nonintervened(self, epsilon1, epsilon2, intervention):
        cf_noise = (epsilon2 - epsilon1) / self.manifold_thickness
        assert torch.all(torch.isfinite(cf_noise))
        logprob_nonintervened = self.base_density.log_prob(cf_noise.reshape((-1, 1))).reshape(
            (-1, self.dim_z)
        )
        logprob_nonintervened -= torch.log(self.manifold_thickness)
        logprob_nonintervened = clean_and_clamp(logprob_nonintervened)
        logprob_nonintervened = (
            1.0 - intervention
        ) * logprob_nonintervened
        logprob_nonintervened = torch.sum(logprob_nonintervened, 1, keepdim=True)
        return logprob_nonintervened
    def _compute_logprob_intervened(self, adjacency_matrix, epsilon1, epsilon2, intervention):
        z_intervened, log_det = self._solve(
            epsilon=epsilon2, conditioning_epsilon=epsilon1, adjacency_matrix=adjacency_matrix
        )
        assert torch.all(torch.isfinite(z_intervened))
        logprob_intervened = self.base_density.log_prob(z_intervened.reshape((-1, 1))).reshape(
            (-1, self.dim_z)
        )
        logprob_intervened += log_det
        logprob_intervened = intervention * logprob_intervened
        logprob_intervened = clean_and_clamp(logprob_intervened)
        logprob_intervened = torch.sum(logprob_intervened, 1, keepdim=True)
        return log_det, logprob_intervened
    def _compute_logprob_observed(self, epsilon1):
        logprob_observed = self.base_density.log_prob(epsilon1.reshape((-1, 1))).reshape(
            (-1, self.dim_z)
        )
        logprob_observed = clean_and_clamp(logprob_observed)
        logprob_observed = torch.sum(logprob_observed, 1, keepdim=True)
        return logprob_observed
    def noise_to_causal(self, epsilon, adjacency_matrix=None):
        return self._solve(epsilon, epsilon, adjacency_matrix=adjacency_matrix)[0]
    def causal_to_noise(self, z, adjacency_matrix=None):
        assert self.topological_order is not None
        conditioning_epsilon = z.clone()
        epsilons = {}
        for i in self.topological_order:
            i = i.item()
            masked_epsilon = self.get_masked_context(i, conditioning_epsilon, adjacency_matrix)
            epsilon, _ = self.solution_functions[i](z[:, i : i + 1], context=masked_epsilon)
            epsilons[i] = epsilon
            conditioning_epsilon[:, i : i + 1] = epsilon
        epsilon = torch.cat([epsilons[i] for i in range(self.dim_z)], 1)
        return epsilon
    @property
    def manifold_thickness(self):
        return self._manifold_thickness
    @manifold_thickness.setter
    @torch.no_grad()
    def manifold_thickness(self, value):
        self._manifold_thickness.copy_(torch.as_tensor(value).to(self._manifold_thickness.device))
    @torch.no_grad()
    def get_scm_parameters(self):
        parameters = {"manifold_thickness": self.manifold_thickness}
        if self.three_d_memory is not None:
            parameters["three_d_mean_pairwise_distance"] = self.three_d_memory.mean_pairwise_distance()
        return parameters
    def three_d_regularizers(self, adjacency_matrix=None):
        if self.three_d_memory is None:
            return {}
        implicit_reg = self.three_d_memory.implicit_first_layer_regularizer(
            self.solution_functions
        )
        if adjacency_matrix is None and self.graph is not None:
            adjacency_matrix = self.graph.adjacency_matrix
        graph_reg = self.three_d_memory.graph_distance_regularizer(adjacency_matrix)
        return {
            "three_d_implicit_edge_regularization": implicit_reg,
            "three_d_graph_edge_regularization": graph_reg,
        }
    def generate_similar_intervention(
        self, z1, z2_example, intervention, adjacency_matrix, sharp_manifold=True
    ):
        raise NotImplementedError
    @staticmethod
    def _sanitize_intervention(intervention, n):
        if intervention is not None:
            assert len(intervention.shape) == 2
            assert intervention.shape[0] == n
            intervention = intervention.to(torch.float)
        return intervention
    @torch.no_grad()
    def get_masked_solution_function(self, i):
        return MaskedSolutionTransform(self, i)
    def _solve(self, epsilon, conditioning_epsilon, adjacency_matrix):
        zs = []
        logdets = []
        for i, transform in enumerate(self.solution_functions):
            masked_epsilon = self.get_masked_context(i, conditioning_epsilon, adjacency_matrix)
            z, logdet = transform.inverse(epsilon[:, i : i + 1], context=masked_epsilon)
            zs.append(z)
            logdets.append(logdet)
        z = torch.cat(zs, 1)
        logdet = torch.cat(logdets, 1)
        return z, logdet
    def _inverse(self, z, conditioning_epsilon, adjacency_matrix=None, order=None):
        if order is None:
            assert self.topological_order is not None
            order = self.topological_order
        epsilons = {}
        logdets = {}
        for i in order:
            masked_epsilon = self.get_masked_context(i, conditioning_epsilon, adjacency_matrix)
            epsilon, logdet = self.solution_functions[i](z[:, i : i + 1], context=masked_epsilon)
            epsilons[i] = epsilon
            logdets[i] = logdet
        epsilon = torch.cat([epsilons[i] for i in range(self.dim_z)], 1)
        logdet = torch.cat([logdets[i] for i in range(self.dim_z)], 1)
        return epsilon, logdet
    def get_masked_context(self, i, epsilon, adjacency_matrix):
        mask_ = self._get_ancestor_mask(
            i, adjacency_matrix, device=epsilon.device, n=epsilon.shape[0]
        )
        dummy_data = self._mask_values.unsqueeze(0)
        dummy_data[:, i] = 0.0
        masked_epsilon = mask(epsilon, mask_, mask_data=dummy_data)
        return masked_epsilon
    def _get_ancestor_mask(self, i, adjacency_matrix, device, n=1):
        if self.graph is None:
            if self.causal_structure == "fixed_order":
                ancestor_mask = torch.zeros((n, self.dim_z), device=device)
                ancestor_mask[..., self.ancestor_idx[i]] = 1.0
            elif self.causal_structure == "trivial":
                ancestor_mask = torch.zeros((n, self.dim_z), device=device)
            else:
                ancestor_mask = torch.ones((n, self.dim_z), device=device)
                ancestor_mask[..., i] = 0.0
        else:
            non_ancestor_matrix = torch.ones_like(adjacency_matrix)
            for n in range(1, self.dim_z):
                non_ancestor_matrix *= 1.0 - torch.linalg.matrix_power(adjacency_matrix, n)
            ancestor_mask = 1.0 - non_ancestor_matrix[..., i]
        return ancestor_mask
    def _sample_noise(self, n, sample_consistency_noise=False):
        if sample_consistency_noise:
            return self.manifold_thickness * self.base_density.sample(n * self.dim_z).reshape(
                n, self.dim_z
            )
        else:
            return self.base_density.sample(n * self.dim_z).reshape(n, self.dim_z)
    def set_causal_structure(
        self, graph, causal_structure, topological_order=None, mask_values=None
    ):
        if graph is None:
            assert causal_structure in ["none", "fixed_order", "trivial"]
        if topological_order is None:
            topological_order = list(range(self.dim_z))
        if mask_values is None:
            mask_values = torch.zeros(self.dim_z, device=self._manifold_thickness.device)
        self.graph = graph
        self.causal_structure = causal_structure
        self.topological_order.copy_(torch.LongTensor(topological_order))
        self._mask_values.copy_(mask_values)
        self._compute_ancestors()
    def _compute_ancestors(self):
        ancestor_idx = defaultdict(list)
        descendants = set(range(self.dim_z))
        for i in self.topological_order:
            i = i.item()
            descendants.remove(i)
            for j in descendants:
                ancestor_idx[j].append(i)
        self.ancestor_idx = ancestor_idx
    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict)
        self._compute_ancestors()
class MLPImplicitSCM(ImplicitSCM):
    def __init__(
        self,
        graph_parameterization,
        manifold_thickness,
        dim_z,
        hidden_layers=1,
        hidden_units=100,
        base_density=DEFAULT_BASE_DENSITY,
        homoskedastic=True,
        min_std=None,
        three_d_enabled=False,
        three_d_layout="cube",
        three_d_seed=0,
        three_d_distance_power=1.0,
    ):
        solution_functions = []
        causal_structure = None
        assert graph_parameterization in {
            "enco",
            "dds",
            "fixed_order",
            None,
            "none",
            "none_fixed_order",
            "none_trivial",
        }
        if graph_parameterization == "enco":
            graph = ENCOLearnedGraph(dim_z)
        elif graph_parameterization == "dds":
            graph = DDSLearnedGraph(dim_z)
        elif graph_parameterization == "fixed_order":
            graph = FixedOrderLearnedGraph(dim_z)
        elif graph_parameterization == "none_fixed_order":
            graph = None
            causal_structure = "fixed_order"
        elif graph_parameterization == "none_trivial":
            graph = None
            causal_structure = "trivial"
        else:
            graph = None
            causal_structure = "none"
        for _ in range(dim_z):
            solution_functions.append(
                make_mlp_structure_transform(
                    dim_z,
                    hidden_layers,
                    hidden_units,
                    homoskedastic,
                    min_std=min_std,
                    initialization="broad",
                )
            )
        three_d_memory = None
        if three_d_enabled:
            three_d_memory = ThreeDCausalMemory(
                dim_z=dim_z,
                layout=three_d_layout,
                seed=three_d_seed,
                distance_power=three_d_distance_power,
            )
        super().__init__(
            graph,
            solution_functions,
            base_density,
            manifold_thickness,
            dim_z=dim_z,
            causal_structure=causal_structure,
            three_d_memory=three_d_memory,
        )





