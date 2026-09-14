
import torch
from torch import nn
from torch.nn import functional as F
from itertools import chain
from CaMeL.encoder.base import Encoder
from CaMeL.nets import make_mlp
from CaMeL.utils import inverse_softplus
class GaussianEncoder(Encoder):
    def __init__(
        self,
        hidden=(100,),
        input_features=2,
        output_features=2,
        fix_std=False,
        init_std=1.0,
        min_std=1.0e-3,
    ):
        super().__init__(input_features, output_features)
        assert not fix_std or init_std is not None
        self._min_std = min_std if not fix_std else 0.0
        self.net, self.mean_head, self.std_head = self._create_nets(
            hidden, fix_std=fix_std, init_std=init_std
        )
    def forward(
        self,
        inputs,
        eval_likelihood_at=None,
        deterministic=False,
        return_mean=False,
        return_std=False,
        full=True,
        reduction="sum",
    ):
        mean, std = self.mean_std(inputs)
        return gaussian_encode(
            mean,
            std,
            eval_likelihood_at,
            deterministic,
            return_mean=return_mean,
            return_std=return_std,
            full=full,
            reduction=reduction,
        )
    def _create_nets(self, hidden, fix_std=False, init_std=None):
        dims = [self.input_features] + hidden
        main_net = make_mlp(dims, final_activation="relu")
        mean_head = nn.Linear(dims[-1], self.output_features)
        if dims[-1] == self.output_features:
            mean_head.weight.data.copy_(
                torch.eye(self.output_features)
                + 0.1 * torch.randn(self.output_features, self.output_features)
            )
        if fix_std:
            std_head = nn.Linear(dims[-1], self.output_features)
            nn.init.constant_(std_head.weight, 0.0)
            if init_std is not None:
                assert init_std > 0.0
                init_value = inverse_softplus(init_std)
                nn.init.constant_(std_head.bias, init_value)
            for param in std_head.parameters():
                param.requires_grad = False
        else:
            linear_layer = nn.Linear(dims[-1], self.output_features)
            nn.init.normal_(linear_layer.weight, 0.0, 1.0e-3)
            if init_std is not None:
                assert init_std > 0.0
                init_value = inverse_softplus(init_std - self._min_std)
                nn.init.constant_(linear_layer.bias, init_value)
            std_head = nn.Sequential(linear_layer, nn.Softplus())
        return main_net, mean_head, std_head
    def mean_std(self, x):
        hidden = self.net(x)
        mean = self.mean_head(hidden)
        std = self._min_std + self.std_head(hidden)
        return mean, std
    def freezable_parameters(self):
        return chain(self.net.parameters(), self.mean_head.parameters(), self.std_head.parameters())
    def unfreezable_parameters(self):
        return []
def gaussian_encode(
    mean,
    std,
    eval_likelihood_at=None,
    deterministic=False,
    return_mean=False,
    return_std=False,
    full=True,
    reduction="sum",
):
    if deterministic:
        z = mean
    else:
        u = torch.randn_like(mean)
        z = mean + std * u
    if eval_likelihood_at is None:
        log_likelihood = gaussian_log_likelihood(z, mean, std, full=full, reduction=reduction)
    else:
        log_likelihood = gaussian_log_likelihood(
            eval_likelihood_at, mean, std, full=full, reduction=reduction
        )
    results = [z, log_likelihood]
    if return_mean:
        results.append(mean)
    if return_std:
        results.append(std)
    return tuple(results)
def gaussian_log_likelihood(x, mean, std, full=True, reduction="sum"):
    var = std**2
    log_likelihood = -F.gaussian_nll_loss(mean, x, var, full=full, reduction="none")
    if reduction == "sum":
        log_likelihood = torch.sum(
            log_likelihood, dim=tuple(range(1, len(log_likelihood.shape)))
        ).unsqueeze(1)
    elif reduction == "mean":
        log_likelihood = torch.mean(
            log_likelihood, dim=tuple(range(1, len(log_likelihood.shape)))
        ).unsqueeze(1)
    elif reduction == "none":
        pass
    else:
        raise ValueError(f"Unknown likelihood reduction {reduction}")
    return log_likelihood
class DeterministicVAEEncoderWrapper(GaussianEncoder):
    def __init__(self, base_model, hidden=None, fix_std=True, init_std=1.0, min_std=1.0e-3):
        super().__init__(
            hidden=hidden,
            input_features=base_model.input_features,
            output_features=base_model.output_features,
            fix_std=fix_std,
            min_std=min_std,
            init_std=init_std,
        )
        self.base_model = base_model
    def _create_nets(self, hidden, fix_std=False, init_std=None):
        if hidden is None:
            dims = [self.input_features + self.output_features]
        else:
            dims = [self.input_features + self.output_features] + hidden
        main_net = make_mlp(dims, final_activation="relu")
        if fix_std:
            std_head = nn.Linear(dims[-1], self.output_features)
            nn.init.constant_(std_head.weight, 0.0)
            if init_std is not None:
                assert init_std > 0.0
                init_value = inverse_softplus(init_std)
                nn.init.constant_(std_head.bias, init_value)
            for param in std_head.parameters():
                param.requires_grad = False
        else:
            linear_layer = nn.Linear(dims[-1], self.output_features)
            nn.init.normal_(linear_layer.weight, 0.0, 1.0e-3)
            if init_std is not None:
                assert init_std > 0.0
                init_value = inverse_softplus(init_std - self._min_std)
                nn.init.constant_(linear_layer.bias, init_value)
            std_head = nn.Sequential(linear_layer, nn.Softplus())
        std_net = nn.Sequential(main_net, std_head)
        return std_net, None, None
    def mean_std(self, x):
        mean, _ = self.base_model(x)
        x_mean = torch.cat((mean, x), dim=1)
        std = self._min_std + self.net(x_mean)
        return mean, std





