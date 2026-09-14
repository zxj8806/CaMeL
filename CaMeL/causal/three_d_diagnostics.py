
from pathlib import Path
import json
import math
import numpy as np
import torch


def _as_numpy(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().to(torch.float).numpy()
    return np.asarray(x)


def get_three_d_memory(model):
    scm = getattr(model, "scm", None)
    if scm is None:
        return None
    return getattr(scm, "three_d_memory", None)


def get_implicit_dependency_matrix(model):
    memory = get_three_d_memory(model)
    if memory is None:
        return None
    scm = getattr(model, "scm", None)
    solution_functions = getattr(scm, "solution_functions", None)
    if solution_functions is None:
        return None
    if hasattr(memory, "implicit_dependency_matrix"):
        return memory.implicit_dependency_matrix(solution_functions)
    rows = []
    dim_z = int(memory.dim_z)
    for target_i, transform in enumerate(solution_functions):
        param_net = getattr(transform, "param_net", None)
        first_linear = None
        if param_net is not None:
            for module in param_net.modules():
                if isinstance(module, torch.nn.Linear):
                    first_linear = module
                    break
        if first_linear is None:
            rows.append(torch.zeros(dim_z, device=memory.distances.device))
            continue
        weight = first_linear.weight[:, :dim_z]
        sensitivity = torch.mean(torch.abs(weight), dim=0)
        sensitivity = sensitivity.to(memory.distances.device)
        sensitivity[target_i] = 0.0
        rows.append(sensitivity)
    return torch.stack(rows, dim=0)


def get_graph_adjacency(model):
    scm = getattr(model, "scm", None)
    graph = getattr(scm, "graph", None)
    if graph is None:
        return None
    try:
        return graph.adjacency_matrix.detach()
    except Exception:
        return None


def _write_matrix_csv(path, matrix, row_name="target", col_name="source"):
    matrix = _as_numpy(matrix)
    if matrix is None:
        return
    path = Path(path)
    with path.open("w") as f:
        header = [row_name] + [f"{col_name}_{i}" for i in range(matrix.shape[1])]
        f.write(",".join(header) + "\n")
        for i, row in enumerate(matrix):
            f.write(str(i) + "," + ",".join(f"{float(v):.10g}" for v in row) + "\n")


def _write_positions_csv(path, positions):
    positions = _as_numpy(positions)
    with Path(path).open("w") as f:
        f.write("variable,x,y,z\n")
        for i, (x, y, z) in enumerate(positions):
            f.write(f"{i},{float(x):.10g},{float(y):.10g},{float(z):.10g}\n")


def _safe_import_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _plot_positions(path, positions, implicit_matrix=None, top_k=None):
    plt = _safe_import_matplotlib()
    positions = _as_numpy(positions)
    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=80)
    for i, p in enumerate(positions):
        ax.text(p[0], p[1], p[2], f" z{i}", fontsize=10)
    if implicit_matrix is not None and positions.shape[0] > 1:
        W = _as_numpy(implicit_matrix).copy()
        np.fill_diagonal(W, 0.0)
        flat = [(W[i, j], i, j) for i in range(W.shape[0]) for j in range(W.shape[1]) if i != j]
        flat = [x for x in flat if np.isfinite(x[0]) and x[0] > 0]
        flat.sort(reverse=True)
        if top_k is None:
            top_k = min(8, len(flat))
        max_w = max([x[0] for x in flat[:top_k]] + [1e-12])
        for w, target, source in flat[:top_k]:
            p0 = positions[source]
            p1 = positions[target]
            v = p1 - p0
            ax.quiver(
                p0[0], p0[1], p0[2], v[0], v[1], v[2],
                arrow_length_ratio=0.12,
                linewidth=0.5 + 2.5 * float(w / max_w),
                alpha=0.55,
            )
    ax.set_title("3D causal memory: latent variables and strongest implicit dependencies")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_heatmap(path, matrix, title, xlabel="source", ylabel="target"):
    plt = _safe_import_matplotlib()
    matrix = _as_numpy(matrix)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(matrix)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_scatter(path, distances, implicit_matrix):
    plt = _safe_import_matplotlib()
    D = _as_numpy(distances)
    W = _as_numpy(implicit_matrix)
    if D is None or W is None:
        return
    mask = ~np.eye(D.shape[0], dtype=bool)
    x = D[mask].reshape(-1)
    y = W[mask].reshape(-1)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.scatter(x, y, s=26, alpha=0.75)
    ax.set_title("Implicit dependency strength vs. 3D distance")
    ax.set_xlabel("normalized 3D distance")
    ax.set_ylabel("first-layer dependency proxy")
    if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0:
        corr = float(np.corrcoef(x, y)[0, 1])
        ax.text(0.02, 0.98, f"Pearson r = {corr:.3f}", transform=ax.transAxes, va="top")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_histogram(path, distances, implicit_matrix):
    plt = _safe_import_matplotlib()
    D = _as_numpy(distances)
    W = _as_numpy(implicit_matrix)
    if D is None or W is None:
        return
    mask = (~np.eye(D.shape[0], dtype=bool)) & (W > 0)
    vals = D[mask].reshape(-1)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    if vals.size:
        ax.hist(vals, bins=min(10, max(3, int(math.sqrt(vals.size)))))
    ax.set_title("3D lengths of nonzero implicit dependencies")
    ax.set_xlabel("normalized 3D distance")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_three_d_diagnostics(model, exp_dir, prefix="three_d"):
    memory = get_three_d_memory(model)
    if memory is None:
        return {}

    exp_dir = Path(exp_dir)
    metrics_dir = exp_dir / "metrics"
    figures_dir = exp_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    positions = memory.positions.detach().cpu()
    distances = memory.distances.detach().cpu()
    implicit = get_implicit_dependency_matrix(model)
    adjacency = get_graph_adjacency(model)

    _write_positions_csv(metrics_dir / f"{prefix}_positions.csv", positions)
    _write_matrix_csv(metrics_dir / f"{prefix}_distances.csv", distances, row_name="target", col_name="source")
    if implicit is not None:
        _write_matrix_csv(metrics_dir / f"{prefix}_implicit_dependency.csv", implicit, row_name="target", col_name="source")
    if adjacency is not None:
        _write_matrix_csv(metrics_dir / f"{prefix}_graph_adjacency.csv", adjacency, row_name="target", col_name="source")

    stats = {}
    if hasattr(memory, "summary_dict"):
        stats.update(memory.summary_dict())
    else:
        stats.update({
            "dim_z": int(memory.dim_z),
            "layout": str(memory.layout),
            "seed": int(memory.seed),
            "distance_power": float(memory.distance_power),
            "mean_pairwise_distance": float(memory.mean_pairwise_distance().detach().cpu()),
        })

    if implicit is not None:
        D = distances.detach().cpu().numpy()
        W = implicit.detach().cpu().to(torch.float).numpy()
        mask = ~np.eye(D.shape[0], dtype=bool)
        w = W[mask]
        d = D[mask]
        stats["implicit_dependency_mean"] = float(np.mean(w)) if w.size else 0.0
        stats["implicit_dependency_max"] = float(np.max(w)) if w.size else 0.0
        stats["implicit_weighted_mean_distance"] = float(np.sum(w * d) / (np.sum(w) + 1e-12)) if w.size else 0.0
        if w.size > 1 and np.std(w) > 0 and np.std(d) > 0:
            stats["implicit_distance_pearson"] = float(np.corrcoef(d, w)[0, 1])
        else:
            stats["implicit_distance_pearson"] = 0.0

    with (metrics_dir / f"{prefix}_summary.json").open("w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)

    try:
        _plot_positions(figures_dir / f"{prefix}_memory_3d.png", positions, implicit)
        _plot_heatmap(figures_dir / f"{prefix}_distance_matrix.png", distances, "3D distance matrix")
        if implicit is not None:
            _plot_heatmap(figures_dir / f"{prefix}_implicit_dependency_matrix.png", implicit, "Implicit dependency proxy")
            _plot_scatter(figures_dir / f"{prefix}_dependency_vs_distance.png", distances, implicit)
            _plot_histogram(figures_dir / f"{prefix}_implicit_edge_lengths.png", distances, implicit)
        if adjacency is not None:
            _plot_heatmap(figures_dir / f"{prefix}_graph_adjacency_matrix.png", adjacency, "Explicit graph adjacency")
    except Exception as exc:
        stats["plot_error"] = str(exc)
        with (metrics_dir / f"{prefix}_summary.json").open("w") as f:
            json.dump(stats, f, indent=2, sort_keys=True)

    return stats

