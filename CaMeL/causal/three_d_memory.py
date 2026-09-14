
import math
import torch
from torch import nn


class ThreeDCausalMemory(nn.Module):
    def __init__(
        self,
        dim_z: int,
        layout: str = "cube",
        seed: int = 0,
        distance_power: float = 1.0,
        normalize_distances: bool = True,
        coordinates=None,
    ):
        super().__init__()
        self.dim_z = int(dim_z)
        self.layout = str(layout)
        self.seed = int(seed)
        self.distance_power = float(distance_power)
        self.normalize_distances = bool(normalize_distances)

        if coordinates is None:
            positions = self._make_positions(self.dim_z, self.layout, self.seed)
        else:
            positions = torch.as_tensor(coordinates, dtype=torch.float32)
            if positions.shape != (self.dim_z, 3):
                raise ValueError(
                    f"coordinates must have shape ({self.dim_z}, 3), got {tuple(positions.shape)}"
                )

        distances = torch.cdist(positions, positions, p=2)
        if self.normalize_distances and distances.numel() > 1:
            max_distance = torch.max(distances)
            if max_distance > 0:
                distances = distances / max_distance
        if self.distance_power != 1.0:
            distances = distances.pow(self.distance_power)

        nonself = torch.ones(self.dim_z, self.dim_z, dtype=torch.float32)
        nonself.fill_diagonal_(0.0)
        distances = distances * nonself

        self.register_buffer("positions", positions)
        self.register_buffer("distances", distances)
        self.register_buffer("nonself_mask", nonself)

    @staticmethod
    def _make_positions(dim_z: int, layout: str, seed: int) -> torch.Tensor:
        if dim_z <= 0:
            raise ValueError("dim_z must be positive")

        if layout == "line":
            x = torch.linspace(0.0, 1.0, dim_z)
            return torch.stack([x, torch.zeros_like(x), torch.zeros_like(x)], dim=1)

        if layout == "circle":
            theta = torch.linspace(0.0, 2.0 * math.pi, dim_z + 1)[:-1]
            x = 0.5 + 0.5 * torch.cos(theta)
            y = 0.5 + 0.5 * torch.sin(theta)
            z = torch.zeros_like(x)
            return torch.stack([x, y, z], dim=1)

        if layout == "random":
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            return torch.rand((dim_z, 3), generator=generator, dtype=torch.float32)

        if layout == "cube":
            side = math.ceil(dim_z ** (1.0 / 3.0))
            coords = torch.linspace(0.0, 1.0, max(side, 2))
            grid = torch.stack(
                torch.meshgrid(coords, coords, coords, indexing="ij"), dim=-1
            ).reshape(-1, 3)
            return grid[:dim_z].to(torch.float32)

        raise ValueError(f"Unknown 3D layout: {layout}")

    def graph_distance_regularizer(self, adjacency_matrix):
        if adjacency_matrix is None:
            return self.distances.new_tensor(0.0)
        adj = adjacency_matrix.to(dtype=self.distances.dtype, device=self.distances.device)
        dist = self.distances
        while dist.dim() < adj.dim():
            dist = dist.unsqueeze(0)
        return torch.mean(adj * dist)

    def implicit_first_layer_regularizer(self, solution_functions):
        costs = []
        for target_i, transform in enumerate(solution_functions):
            param_net = getattr(transform, "param_net", None)
            if param_net is None:
                continue
            first_linear = None
            for module in param_net.modules():
                if isinstance(module, nn.Linear):
                    first_linear = module
                    break
            if first_linear is None:
                continue
            weight = first_linear.weight[:, : self.dim_z]
            sensitivity = torch.mean(torch.abs(weight), dim=0)
            distance_row = self.distances[target_i].to(sensitivity.device)
            nonself_row = self.nonself_mask[target_i].to(sensitivity.device)
            costs.append(torch.sum(sensitivity * distance_row * nonself_row))
        if not costs:
            return self.distances.new_tensor(0.0)
        return torch.stack(costs).mean()



    def implicit_dependency_matrix(self, solution_functions):
        rows = []
        for target_i, transform in enumerate(solution_functions):
            param_net = getattr(transform, "param_net", None)
            first_linear = None
            if param_net is not None:
                for module in param_net.modules():
                    if isinstance(module, nn.Linear):
                        first_linear = module
                        break
            if first_linear is None:
                rows.append(self.distances.new_zeros(self.dim_z))
                continue
            weight = first_linear.weight[:, : self.dim_z]
            sensitivity = torch.mean(torch.abs(weight), dim=0)
            sensitivity = sensitivity.to(self.distances.device)
            sensitivity = sensitivity * self.nonself_mask[target_i].to(sensitivity.device)
            rows.append(sensitivity)
        if not rows:
            return self.distances.new_zeros((self.dim_z, self.dim_z))
        return torch.stack(rows, dim=0)

    def implicit_distance_weighted_matrix(self, solution_functions):
        return self.implicit_dependency_matrix(solution_functions) * self.distances

    def edge_length_statistics(self, edge_strengths=None, eps: float = 1.0e-12):
        if edge_strengths is None:
            weights = self.nonself_mask
        else:
            weights = edge_strengths.to(device=self.distances.device, dtype=self.distances.dtype)
            weights = weights * self.nonself_mask
        total_weight = torch.sum(weights).clamp_min(eps)
        weighted_mean = torch.sum(weights * self.distances) / total_weight
        max_distance = torch.max(self.distances * (weights > 0).to(self.distances.dtype))
        return {
            "edge_weight_sum": torch.sum(weights),
            "edge_weighted_mean_distance": weighted_mean,
            "edge_max_distance": max_distance,
        }

    def summary_dict(self):
        return {
            "dim_z": int(self.dim_z),
            "layout": str(self.layout),
            "seed": int(self.seed),
            "distance_power": float(self.distance_power),
            "normalize_distances": bool(self.normalize_distances),
            "mean_pairwise_distance": float(self.mean_pairwise_distance().detach().cpu()),
        }

    def mean_pairwise_distance(self):
        denom = torch.sum(self.nonself_mask).clamp_min(1.0)
        return torch.sum(self.distances) / denom



