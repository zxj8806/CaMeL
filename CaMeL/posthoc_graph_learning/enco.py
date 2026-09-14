
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import torch.optim as optim
from tqdm.auto import tqdm
import numpy as np
def gaussian_log_prob(mean, log_std, samples):
    if len(samples.shape) == len(mean.shape) + 1:
        mean = mean[..., None]
    if len(samples.shape) == len(log_std.shape) + 1:
        log_std = log_std[..., None]
    return -log_std - 0.5 * np.log(2 * np.pi) - 0.5 * ((samples - mean) / log_std.exp()) ** 2
class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters, min_factor=0.05, offset=0):
        self.warmup = warmup
        self.max_num_iters = max_iters
        self.min_factor = min_factor
        self.offset = offset
        super().__init__(optimizer)
        if isinstance(self.warmup, list) and not isinstance(self.offset, list):
            self.offset = [self.offset for _ in self.warmup]
    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        if isinstance(lr_factor, list):
            return [base_lr * f for base_lr, f in zip(self.base_lrs, lr_factor)]
        else:
            return [base_lr * lr_factor for base_lr in self.base_lrs]
    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        lr_factor = lr_factor * (1 - self.min_factor) + self.min_factor
        if isinstance(self.warmup, list):
            new_lr_factor = []
            for o, w in zip(self.offset, self.warmup):
                e = max(0, epoch - o)
                l = lr_factor * ((e * 1.0 / w) if e <= w and w > 0 else 1)
                new_lr_factor.append(l)
            lr_factor = new_lr_factor
        else:
            epoch = max(0, epoch - self.offset)
            if epoch <= self.warmup and self.warmup > 0:
                lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
class MultivarLinear(nn.Module):
    def __init__(self, input_dims, output_dims, extra_dims, bias=True):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.extra_dims = extra_dims
        self.weight = nn.Parameter(torch.zeros(*extra_dims, output_dims, input_dims))
        if bias:
            self.bias = nn.Parameter(torch.zeros(*extra_dims, output_dims))
        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
    def forward(self, x):
        x_extra_dims = x.shape[1:-1]
        if len(x_extra_dims) > 0:
            for i in range(len(x_extra_dims)):
                assert (
                    x_extra_dims[-(i + 1)] == self.extra_dims[-(i + 1)]
                ), "Shape mismatch: X=%s, Layer=%s" % (
                    str(x.shape),
                    str(self.extra_dims),
                )
        for _ in range(len(self.extra_dims) - len(x_extra_dims)):
            x = x.unsqueeze(dim=1)
        x = x.unsqueeze(dim=-1)
        weight = self.weight.unsqueeze(dim=0)
        out = torch.matmul(weight, x).squeeze(dim=-1)
        if hasattr(self, "bias"):
            bias = self.bias.unsqueeze(dim=0)
            out = out + bias
        return out
class ENCOGraphLearning:
    def __init__(
        self,
        num_causal_vars,
        verbose=True,
        num_graph_samples=100,
        lambda_sparse=0.01,
        debug=False,
        c_hid=64,
        device="cuda:0",
    ):
        self.debug = debug
        self.verbose = verbose
        self.num_graph_samples = num_graph_samples
        self.lambda_sparse = lambda_sparse
        self.num_causal_vars = num_causal_vars
        self.device = device
        self.net = nn.Sequential(
            MultivarLinear(self.num_causal_vars * 2, c_hid, [self.num_causal_vars]),
            nn.SiLU(),
            MultivarLinear(c_hid, c_hid, [self.num_causal_vars]),
            nn.SiLU(),
            MultivarLinear(c_hid, 2, [self.num_causal_vars]),
        ).to(self.device)
        self.gamma = nn.Parameter(torch.eye(self.num_causal_vars, device=self.device) * -9e15)
        self.theta = nn.Parameter(self.gamma.data.clone())
        self.model_optimizer = torch.optim.AdamW(self.net.parameters(), lr=2e-3, weight_decay=1e-4)
        self.model_scheduler = CosineWarmupScheduler(
            self.model_optimizer, warmup=100, max_iters=int(1e7)
        )
        self.gamma_optimizer = torch.optim.Adam([self.gamma], lr=5e-3, betas=(0.9, 0.9))
        self.theta_optimizer = torch.optim.Adam([self.theta], lr=1e-2, betas=(0.9, 0.999))
        self.latents_means = torch.zeros(self.num_causal_vars, device=self.gamma.device)
        self.latents_stds = torch.ones(self.num_causal_vars, device=self.gamma.device)
    def iterator(self, it, desc=None, leave=False):
        if self.verbose:
            return tqdm(it, desc=desc, leave=leave)
        else:
            return it
    def learn_graph(self, dist_dataset, graph_dataset=None, num_epochs=40):
        if not graph_dataset:
            graph_dataset = dist_dataset
        dist_data_loader = torch.utils.data.DataLoader(dist_dataset, batch_size=512, shuffle=True)
        graph_data_loader = torch.utils.data.DataLoader(graph_dataset, batch_size=256, shuffle=True)
        self.prepare_latent_statistics(dist_dataset)
        for _ in self.iterator(range(10 if not self.debug else 1), "Distribution pretraining"):
            self.distribution_fitting_epoch(dist_data_loader)
        for _ in self.iterator(range(num_epochs), desc="ENCO epochs"):
            if self.is_gamma_saturated():
                continue
            self.distribution_fitting_epoch(dist_data_loader)
            self.graph_fitting_epoch(graph_data_loader)
        return self.get_adj_matrix()
    @torch.no_grad()
    def prepare_latent_statistics(self, dataset):
        encodings = dataset.tensors[0]
        self.latents_means = encodings.mean(dim=0).detach()
        self.latents_stds = encodings.std(dim=0).detach()
    def distribution_fitting_epoch(self, data_loader, max_steps=1000):
        self.net.train()
        if self.debug:
            max_steps = 10
        data_iter = iter(data_loader)
        for _ in self.iterator(range(min(max_steps, len(data_iter))), "Distribution fitting"):
            latents, targets = next(data_iter)
            latents = latents.to(self.device)
            targets = targets.to(self.device)
            causal_graphs = self.sample_graphs(latents.shape[0])
            nll = self.run_priors(latents, causal_graphs)
            nll = nll * (1 - targets)
            loss = nll.mean()
            self.model_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 0.1, error_if_nonfinite=True)
            self.model_optimizer.step()
            self.model_scheduler.step()
    def sample_graphs(self, batch_size):
        gamma_sigm = torch.sigmoid(self.gamma.detach())
        theta_sigm = torch.sigmoid(self.theta.detach())
        edge_probs = gamma_sigm * theta_sigm
        causal_graphs = torch.bernoulli(edge_probs[None].expand(batch_size, -1, -1))
        return causal_graphs
    def graph_fitting_epoch(self, data_loader, max_steps=100):
        self.net.eval()
        if self.debug:
            max_steps = 10
        data_iter = iter(data_loader)
        for _ in self.iterator(range(min(max_steps, len(data_iter))), "Graph fitting"):
            latents, targets = next(data_iter)
            with torch.no_grad():
                latents = latents.to(self.device)
                targets = targets.to(self.device)
                causal_graphs = self.sample_graphs(self.num_graph_samples)
                causal_graphs_exp = (
                    causal_graphs[None].expand(latents.shape[0], -1, -1, -1).flatten(0, 1)
                )
                latents_exp = latents[:, None].expand(-1, self.num_graph_samples, -1).flatten(0, 1)
                nll = self.run_priors(latents_exp, causal_graphs_exp)
                nll = nll.unflatten(0, (-1, self.num_graph_samples))
                causal_graphs_exp = causal_graphs_exp.unflatten(
                    0, (latents.shape[0], self.num_graph_samples)
                )
                gamma_sigm = torch.sigmoid(self.gamma.detach())
                theta_sigm = torch.sigmoid(self.theta.detach())
                targets = targets.squeeze(dim=1)
                num_pos = causal_graphs_exp.sum(dim=1)
                num_neg = self.num_graph_samples - num_pos
                mask = ((num_pos > 0) * (num_neg > 0)).float()
                pos_grads = (nll[:, :, None] * causal_graphs_exp).sum(dim=1) / num_pos.clamp_(
                    min=1e-5
                )
                neg_grads = (nll[:, :, None] * (1 - causal_graphs_exp)).sum(dim=1) / num_neg.clamp_(
                    min=1e-5
                )
                gamma_grads = (
                    mask
                    * theta_sigm
                    * gamma_sigm
                    * (1 - gamma_sigm)
                    * (pos_grads - neg_grads + self.lambda_sparse)
                )
                gamma_grads = gamma_grads * (
                    1 - targets[:, None, :]
                )
                gamma_grads[
                    :, torch.arange(gamma_grads.shape[2]), torch.arange(gamma_grads.shape[2])
                ] = 0.0
                theta_grads = (
                    mask * gamma_sigm * theta_sigm * (1 - theta_sigm) * (pos_grads - neg_grads)
                )
                theta_grads = (
                    theta_grads * targets[:, :, None]
                )
                theta_grads = theta_grads * (
                    1 - targets[:, :, None] * targets[:, None, :]
                )
                theta_grads = theta_grads - theta_grads.transpose(
                    1, 2
                )
                gamma_grads = gamma_grads.mean(dim=0)
                theta_grads = theta_grads.mean(dim=0)
            self.gamma_optimizer.zero_grad()
            self.theta_optimizer.zero_grad()
            self.gamma.grad = gamma_grads
            self.theta.grad = theta_grads
            self.gamma_optimizer.step()
            self.theta_optimizer.step()
    def run_priors(self, latents, causal_graphs):
        latent_mask = causal_graphs.transpose(-2, -1)
        inp = torch.cat([latents[..., None, :] * latent_mask, latent_mask * 2 - 1], dim=-1)
        prior_mean, prior_logstd = self.net.forward(inp).unbind(dim=-1)
        nll = -gaussian_log_prob(prior_mean, prior_logstd, latents)
        return nll
    def get_adj_matrix(self):
        adj_matrix = ((self.theta.data > 0.0) * (self.gamma.data > 0.0)).long().detach().cpu()
        return adj_matrix
    @torch.no_grad()
    def is_gamma_saturated(self):
        gamma_sigm = torch.sigmoid(self.gamma)
        max_grad = (gamma_sigm * (1 - gamma_sigm)).max()
        return max_grad.item() < 1e-3
def run_enco(
    z1, z2, interventions, lambda_sparse=0.01, device="cuda:0", verbose=False, debug=False
):
    dim_z = z1.shape[1]
    dist_dataset = TensorDataset(z1, torch.zeros_like(z1))
    graph_dataset = TensorDataset(z2, interventions)
    enco = ENCOGraphLearning(
        dim_z, lambda_sparse=lambda_sparse, verbose=verbose, debug=debug, device=device
    )
    adj_matrix = enco.learn_graph(dist_dataset, graph_dataset)
    return adj_matrix
if __name__ == "__main__":
    import argparse
    import pandas as pd
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, required=True)
    parser.add_argument("--lambda_sparse", type=float, default=0.01)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    data_arr = pd.read_csv(args.csv_file).to_numpy()
    data_arr = torch.from_numpy(data_arr).float()
    assert (data_arr.shape[1] - 1) % 3 == 0, "Wrong data shape"
    num_causal_vars = (data_arr.shape[1] - 1) // 3
    print(f"Num causal vars: {num_causal_vars}")
    latents_before_intv = data_arr[:, 1 : 1 + num_causal_vars]
    latents_after_intv = data_arr[:, 1 + num_causal_vars : 1 + 2 * num_causal_vars]
    intv_targets = data_arr[:, 1 + 2 * num_causal_vars : 1 + 3 * num_causal_vars]
    dist_dataset = TensorDataset(latents_before_intv, torch.zeros_like(latents_before_intv))
    graph_dataset = TensorDataset(latents_after_intv, intv_targets)
    print(f"Dataset size: {len(dist_dataset)}")
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    enco = ENCOGraphLearning(
        num_causal_vars,
        lambda_sparse=args.lambda_sparse,
        verbose=args.verbose,
        debug=args.debug,
        device=device,
    )
    adj_matrix = enco.learn_graph(dist_dataset, graph_dataset)
    print("Learned adjacency matrix:")
    print(adj_matrix)






