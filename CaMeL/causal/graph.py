
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from CaMeL.gumbel import sample_permutation, gumbel_bernouilli
from CaMeL.utils import upper_triangularize, topological_sort
class LearnedGraph(nn.Module):
    def __init__(self, dim_z):
        super().__init__()
        self.dim_z = dim_z
    @property
    def adjacency_matrix(self):
        raise NotImplementedError
    @property
    def num_edges(self):
        return torch.sum(self.adjacency_matrix)
    @property
    def acyclicity_regularizer(self):
        adj = self.adjacency_matrix
        return (torch.trace(torch.matrix_exp(adj)) - adj.shape[0]) ** 2
    def sample_adjacency_matrices(self, n, mode="hard", temperature=1.0):
        raise NotImplementedError
    def descendant_masks(self, adjacency_matrix, intervention, eps=1.0e-9):
        return 1.0 - self.non_descendant_mask(adjacency_matrix, intervention, eps=eps)
    def non_descendant_mask(self, adjacency_matrix, intervention, eps=1.0e-9):
        assert adjacency_matrix.shape[-2:] == (self.dim_z, self.dim_z)
        assert intervention.shape[-1] == self.dim_z
        nd = self._nondescendancy_matrix(adjacency_matrix)
        intervention_nd = torch.exp(
            torch.sum(intervention.to(torch.float).unsqueeze(-1) * torch.log(nd + eps), dim=-2)
        )
        zeros_with_gradients = intervention_nd - intervention_nd.detach()
        intervention_nd = torch.where(intervention_nd >= eps, intervention_nd, zeros_with_gradients)
        return intervention_nd
    @torch.no_grad()
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
    @torch.no_grad()
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
    def get_graph_parameters(self):
        raise NotImplementedError
    def _nondescendancy_matrix(self, adjacency_matrix):
        nondescendancy_matrix = torch.ones_like(adjacency_matrix)
        for n in range(0, self.dim_z):
            nondescendancy_matrix *= 1.0 - torch.linalg.matrix_power(adjacency_matrix, n)
        return nondescendancy_matrix
class ENCOLearnedGraph(LearnedGraph):
    def __init__(self, dim_z):
        super().__init__(dim_z)
        self.order = list(range(self.dim_z))
        self.n_edges = self.dim_z * (self.dim_z - 1) // 2
        self.edge_existence_logits = torch.nn.Parameter(torch.empty(self.n_edges))
        self.edge_orientation_logits = torch.nn.Parameter(torch.empty(self.n_edges))
        self._initialize_edges()
    @property
    def edge_existence_matrix(self):
        edge_probs = torch.sigmoid(self.edge_existence_logits)
        edge_matrix = upper_triangularize(edge_probs, self.dim_z)
        edge_matrix = edge_matrix + edge_matrix.T
        return edge_matrix
    @property
    def edge_orientation_matrix(self):
        orientation_probs = torch.sigmoid(self.edge_orientation_logits)
        orientation_matrix = (
            upper_triangularize(orientation_probs, self.dim_z)
            + upper_triangularize(1.0 - orientation_probs, self.dim_z).T
        )
        return orientation_matrix
    @property
    def adjacency_matrix(self):
        return self.edge_existence_matrix * self.edge_orientation_matrix
    @property
    def hard_adjacency_matrix(self):
        return ((self.edge_existence_matrix >= 0.5) * (self.edge_orientation_matrix >= 0.5)).to(
            torch.float
        )
    def sample_adjacency_matrices(self, n, mode="hard", temperature=1.0):
        assert mode in {"deterministic", "hard", "soft"}, f"Unknown graph sampling mode {mode}"
        if mode == "deterministic":
            hard_adjacency_matrices = self.hard_adjacency_matrix.unsqueeze(0).expand(
                (n, self.dim_z, self.dim_z)
            )
            soft_adjacency_matrices = self.sample_adjacency_matrices(n, "soft", temperature)
            return hard_adjacency_matrices.detach() + (
                soft_adjacency_matrices - soft_adjacency_matrices.detach()
            )
        existence_logits = self.edge_existence_logits.unsqueeze(0).broadcast_to((n, self.n_edges))
        edge_existence, log_prob_existence = gumbel_bernouilli(
            existence_logits, tau=temperature, hard=mode == "hard"
        )
        orientation_logits = self.edge_orientation_logits.unsqueeze(0).broadcast_to(
            (n, self.n_edges)
        )
        edge_orientation, log_prob_orientation = gumbel_bernouilli(
            orientation_logits, tau=temperature, hard=mode == "hard"
        )
        edge_matrix = upper_triangularize(edge_existence, self.dim_z)
        edge_matrix = edge_matrix + torch.transpose(edge_matrix, -2, -1)
        orientation_matrix = upper_triangularize(edge_orientation, self.dim_z) + torch.transpose(
            upper_triangularize(1.0 - edge_orientation, self.dim_z), -2, -1
        )
        adjacency_matrix = edge_matrix * orientation_matrix
        return adjacency_matrix
    @torch.no_grad()
    def get_graph_parameters(self):
        parameters = {}
        for i, val in enumerate(self.edge_existence_logits):
            parameters[f"edge_existence_logit_{i}"] = val.cpu().detach()
        for i, val in enumerate(self.edge_orientation_logits):
            parameters[f"edge_orientation_logit_{i}"] = val.cpu().detach()
        return parameters
    @torch.no_grad()
    def _initialize_edges(self):
        torch.nn.init.normal_(
            self.edge_existence_logits, mean=4, std=0.01
        )
        torch.nn.init.normal_(
            self.edge_orientation_logits, mean=0, std=0.01
        )
class DDSLearnedGraph(LearnedGraph):
    def __init__(self, dim_z):
        super().__init__(dim_z)
        self.order = list(range(self.dim_z))
        self.n_edges = self.dim_z * (self.dim_z - 1) // 2
        self.edge_existence_logits = torch.nn.Parameter(torch.empty(self.n_edges))
        self.permutation_logits = torch.nn.Parameter(torch.empty(self.dim_z))
        self._initialize_edges()
    @property
    def standard_adjacency_matrix(self):
        edge_probs = torch.sigmoid(self.edge_existence_logits)
        edge_matrix = upper_triangularize(edge_probs, self.dim_z)
        return edge_matrix
    @property
    def permutation_matrix(self):
        return sample_permutation(
            self.permutation_logits.unsqueeze(0), mode="deterministic"
        ).squeeze(0)
    @property
    def adjacency_matrix(self):
        permutation = self.permutation_matrix
        adjacency_matrix = permutation.T @ self.standard_adjacency_matrix @ permutation
        return adjacency_matrix
    @property
    def num_edges(self):
        return torch.sum(self.standard_adjacency_matrix)
    @property
    def acyclicity_regularizer(self):
        return 0.0
    @property
    def hard_adjacency_matrix(self):
        standard_adjacency_matrix = (self.standard_adjacency_matrix >= 0.5).to(torch.float)
        permutation = self.permutation_matrix
        adjacency_matrix = permutation.T @ standard_adjacency_matrix @ permutation
        return adjacency_matrix
    def sample_adjacency_matrices(self, n, mode="hard", temperature=1.0):
        assert mode in {"deterministic", "hard", "soft"}, f"Unknown graph sampling mode {mode}"
        if mode == "deterministic":
            hard_adjacency_matrices = self.hard_adjacency_matrix.unsqueeze(0).expand(
                (n, self.dim_z, self.dim_z)
            )
            soft_adjacency_matrices = self.sample_adjacency_matrices(n, "soft", temperature)
            return hard_adjacency_matrices.detach() + (
                soft_adjacency_matrices - soft_adjacency_matrices.detach()
            )
        existence_logits = (
            self.edge_existence_logits.unsqueeze(0).broadcast_to((n, self.n_edges)).unsqueeze(2)
        )
        existence_logits = torch.cat((existence_logits, torch.zeros_like(existence_logits)), dim=2)
        edge_existence = F.gumbel_softmax(existence_logits, tau=temperature, hard=mode == "hard")[
            ..., 0
        ]
        scores = self.permutation_logits.unsqueeze(0).expand(n, self.dim_z)
        permutation = sample_permutation(scores, tau=1.0, mode="hard")
        standard_adjacency_matrix = upper_triangularize(edge_existence, self.dim_z)
        adjacency_matrix = (
            torch.transpose(permutation, 1, 2) @ standard_adjacency_matrix @ permutation
        )
        return adjacency_matrix
    @torch.no_grad()
    def get_graph_parameters(self):
        parameters = {}
        for i, val in enumerate(self.edge_existence_logits):
            parameters[f"edge_existence_logit_{i}"] = val.cpu().detach()
        for i, val in enumerate(self.permutation_logits):
            parameters[f"permutation_score_{i}"] = val.cpu().detach()
        return parameters
    @torch.no_grad()
    def _initialize_edges(self):
        torch.nn.init.normal_(
            self.edge_existence_logits, mean=4, std=0.01
        )
        torch.nn.init.normal_(
            self.permutation_logits, mean=0, std=0.01
        )
class FixedOrderLearnedGraph(LearnedGraph):
    def __init__(self, dim_z):
        super().__init__(dim_z)
        self.order = list(range(self.dim_z))
        self.n_edges = self.dim_z * (self.dim_z - 1) // 2
        self.edge_logits = torch.nn.Parameter(torch.empty(self.n_edges))
        self._initialize_edges()
    @property
    def adjacency_matrix(self):
        edge_probs = torch.sigmoid(self.edge_logits)
        adjacency_matrix = upper_triangularize(edge_probs, self.dim_z)
        return adjacency_matrix
    @property
    def hard_adjacency_matrix(self):
        return (self.adjacency_matrix >= 0.5).to(torch.float)
    def sample_adjacency_matrices(self, n, mode="hard", temperature=1.0):
        assert mode in {"deterministic", "hard", "soft"}, f"Unknown graph sampling mode {mode}"
        if mode == "deterministic":
            hard_adjacency_matrices = self.hard_adjacency_matrix.unsqueeze(0).expand(
                (n, self.dim_z, self.dim_z)
            )
            soft_adjacency_matrices, log_prob = self.sample_adjacency_matrices(
                n, "soft", temperature
            )
            det_adjacency_matrices = (
                hard_adjacency_matrices.detach()
                + soft_adjacency_matrices
                - soft_adjacency_matrices.detach()
            )
            return det_adjacency_matrices, log_prob
        logits = self.edge_logits.unsqueeze(0).broadcast_to((n, self.n_edges))
        edges, log_prob = gumbel_bernouilli(logits, tau=temperature, hard=mode == "hard")
        adjacency_matrices = upper_triangularize(edges, self.dim_z)
        log_prob = torch.sum(log_prob, dim=1)
        return adjacency_matrices, log_prob
    @torch.no_grad()
    def get_graph_parameters(self):
        parameters = {}
        for i, val in enumerate(self.edge_logits):
            parameters[f"edge_existence_logit_{i}"] = val.cpu().detach()
        return parameters
    @property
    def acyclicity_regularizer(self):
        return 0.0
    @torch.no_grad()
    def _initialize_edges(self):
        torch.nn.init.normal_(
            self.edge_logits, mean=4, std=0.01
        )
class FixedGraph(nn.Module):
    def __init__(self, adjacency_matrix, dim_z=2, epsilon=1.0e-9):
        super().__init__()
        self.dim_z = dim_z
        self.register_buffer("adjacency_matrix", adjacency_matrix.to(torch.bool))
        self.order = topological_sort(self.adjacency_matrix)
        assert self.adjacency_matrix.shape == (self.dim_z, self.dim_z)
        assert (
            torch.abs(
                torch.trace(torch.matrix_exp(self.adjacency_matrix.to(torch.float))) - self.dim_z
            )
            < epsilon
        ), "Adjacency matrix not acyclical"
    @property
    def hard_adjacency_matrix(self):
        return self.adjacency_matrix
    @property
    def num_edges(self):
        return torch.sum(self.adjacency_matrix)
    @property
    def acyclicity_regularizer(self):
        return 0.0
    def sample_adjacency_matrices(self, n, **kwargs):
        return self.adjacency_matrix.unsqueeze(0).expand((n, *self.adjacency_matrix.shape)), None
    def descendant_masks(self, intervention, epsilon=1.0e-9, **kwargs):
        assert intervention.shape[-1] == self.dim_z
        descendants = torch.matrix_exp(
            self.adjacency_matrix.to(torch.float)
        )
        descended_from_intervention = torch.einsum(
            "...i,ij->...j", intervention.to(torch.float), descendants
        )
        descendant_mask = torch.abs(descended_from_intervention) > epsilon
        return descendant_mask
    @torch.no_grad()
    def freeze(self):
        pass
    @torch.no_grad()
    def unfreeze(self):
        pass
    @torch.no_grad()
    def get_graph_parameters(self):
        return {}
    def non_descendant_mask(self, intervention, epsilon=1.0e-9, **kwargs):
        return ~self.descendant_masks(intervention, epsilon)
    def parent_masks_from_intervention_masks(self, intervention, **kwargs):
        assert intervention.shape[-1] == self.dim_z
        parents = torch.einsum(
            "ij,...j->...i", self.adjacency_matrix.to(torch.float), intervention.to(torch.float)
        )
        parents -= intervention * self.dim_z
        parent_masks = parents > 0
        return parent_masks
    def parent_masks(self, indices, **kwargs):
        return self.adjacency_matrix.T[indices]
def bools_from_seed(seed, n_bools):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=n_bools).astype(bool)
def create_graph(dim_z, mode, edge_existence_seed, permutation=None):
    if permutation:
        raise NotImplementedError
    n_edges = dim_z * (dim_z - 1) // 2
    if mode == "full":
        edges = torch.ones((n_edges,), dtype=torch.bool)
        adjacency_matrix = upper_triangularize(edges, dim_z)
    elif mode == "empty":
        adjacency_matrix = torch.zeros((dim_z, dim_z), dtype=torch.bool)
    elif mode == "chain":
        adjacency_matrix = torch.diag_embed(torch.ones((dim_z - 1,), dtype=torch.bool), offset=1)
    elif mode == "random":
        edges = torch.BoolTensor(bools_from_seed(edge_existence_seed, n_edges))
        adjacency_matrix = upper_triangularize(edges, dim_z)
    else:
        raise NotImplementedError(f"Unknown mode {mode}")
    graph = FixedGraph(adjacency_matrix, dim_z=dim_z)
    return graph





