
from pathlib import Path
import json
import numpy as np
import torch


def _as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().to(torch.float).numpy()
    return np.asarray(x)


def get_three_d_spiking_memory(model):
    return getattr(model, "three_d_spiking_memory", None)


def _write_matrix_csv(path, matrix, row_name="row", col_name="col"):
    matrix = _as_numpy(matrix)
    with Path(path).open("w") as f:
        header = [row_name] + [f"{col_name}_{i}" for i in range(matrix.shape[1])]
        f.write(",".join(header) + "\n")
        for i, row in enumerate(matrix):
            f.write(str(i) + "," + ",".join(f"{float(v):.10g}" for v in row) + "\n")


def _write_positions_csv(path, positions, index_name="neuron"):
    positions = _as_numpy(positions)
    with Path(path).open("w") as f:
        f.write(f"{index_name},x,y,z\n")
        for i, (x, y, z) in enumerate(positions):
            f.write(f"{i},{float(x):.10g},{float(y):.10g},{float(z):.10g}\n")


def _safe_import_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _plot_positions(path, positions, weights=None, top_k=40):
    plt = _safe_import_matplotlib()
    positions = _as_numpy(positions)
    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=30)
    if weights is not None:
        W = np.abs(_as_numpy(weights)).copy()
        np.fill_diagonal(W, 0.0)
        flat = [(W[i, j], i, j) for i in range(W.shape[0]) for j in range(W.shape[1]) if i != j]
        flat = [x for x in flat if np.isfinite(x[0]) and x[0] > 0]
        flat.sort(reverse=True)
        max_w = max([x[0] for x in flat[:top_k]] + [1.0e-12])
        for w, target, source in flat[:top_k]:
            p0 = positions[source]
            p1 = positions[target]
            v = p1 - p0
            ax.quiver(
                p0[0], p0[1], p0[2], v[0], v[1], v[2],
                arrow_length_ratio=0.08,
                linewidth=0.3 + 1.8 * float(w / max_w),
                alpha=0.35,
            )
    ax.set_title("3D spiking memory")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_heatmap(path, matrix, title):
    plt = _safe_import_matplotlib()
    matrix = _as_numpy(matrix)
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    im = ax.imshow(matrix)
    ax.set_title(title)
    ax.set_xlabel("source")
    ax.set_ylabel("target")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_variable_causal_memory(path, positions, causal_weights, top_k=40):
    plt = _safe_import_matplotlib()
    positions = _as_numpy(positions)
    W = _as_numpy(causal_weights).copy()
    if W.size == 0:
        return
    np.fill_diagonal(W, 0.0)
    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=120)
    for i, p in enumerate(positions):
        ax.text(p[0], p[1], p[2], f"z{i}", fontsize=9)
    flat = [(abs(W[j, i]), i, j) for j in range(W.shape[0]) for i in range(W.shape[1]) if i != j]
    flat = [x for x in flat if np.isfinite(x[0]) and x[0] > 0]
    flat.sort(reverse=True)
    max_w = max([x[0] for x in flat[:top_k]] + [1.0e-12])
    for w, source, target in flat[:top_k]:
        p0 = positions[source]
        p1 = positions[target]
        v = p1 - p0
        ax.quiver(
            p0[0], p0[1], p0[2], v[0], v[1], v[2],
            arrow_length_ratio=0.10,
            linewidth=0.6 + 2.4 * float(w / max_w),
            alpha=0.45,
        )
    ax.set_title("Intervention-gated causal synapses")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_three_d_spiking_diagnostics(model, exp_dir, prefix="three_d_spiking"):
    memory = get_three_d_spiking_memory(model)
    if memory is None:
        return {}
    exp_dir = Path(exp_dir)
    metrics_dir = exp_dir / "metrics"
    figures_dir = exp_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    positions, distances, weights = memory.export_matrices()
    summary = memory.get_summary()
    _write_positions_csv(metrics_dir / f"{prefix}_positions.csv", positions, index_name="neuron")
    _write_matrix_csv(metrics_dir / f"{prefix}_distances.csv", distances, row_name="target", col_name="source")
    _write_matrix_csv(metrics_dir / f"{prefix}_recurrent_weights.csv", weights, row_name="target", col_name="source")

    if hasattr(memory, "export_causal_matrices"):
        variable_positions, variable_distances, variable_to_neuron, causal_synapses = memory.export_causal_matrices()
        _write_positions_csv(metrics_dir / f"{prefix}_variable_positions.csv", variable_positions, index_name="variable")
        _write_matrix_csv(metrics_dir / f"{prefix}_variable_distances.csv", variable_distances, row_name="target", col_name="source")
        _write_matrix_csv(metrics_dir / f"{prefix}_variable_to_neuron.csv", variable_to_neuron, row_name="variable", col_name="neuron")
        _write_matrix_csv(metrics_dir / f"{prefix}_causal_synapses.csv", causal_synapses, row_name="effect", col_name="target")
    else:
        variable_positions = variable_distances = causal_synapses = None

    with (metrics_dir / f"{prefix}_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    try:
        _plot_positions(figures_dir / f"{prefix}_memory_3d.png", positions, weights)
        _plot_heatmap(figures_dir / f"{prefix}_distance_matrix.png", distances, "3dSpiking 3D distance matrix")
        _plot_heatmap(figures_dir / f"{prefix}_recurrent_weight_matrix.png", weights, "3dSpiking recurrent weights")
        if variable_positions is not None:
            _plot_heatmap(figures_dir / f"{prefix}_causal_synapse_matrix.png", causal_synapses, "Intervention-gated causal synapses")
            _plot_heatmap(figures_dir / f"{prefix}_variable_distance_matrix.png", variable_distances, "Variable assembly distances")
            _plot_variable_causal_memory(figures_dir / f"{prefix}_causal_memory_3d.png", variable_positions, causal_synapses)
    except Exception as exc:
        summary["diagnostic_plot_error"] = str(exc)
    return summary


