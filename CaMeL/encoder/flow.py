
import nflows
import torch
from torch import nn
from CaMeL.encoder.base import Encoder
from CaMeL.transforms import make_scalar_transform
class InvertibleEncoder(Encoder):
    def __init__(self, input_features=2, output_features=2, **kwargs):
        super().__init__(input_features, output_features)
        assert self.input_features == self.output_features
        self.transform = self._make_transform(**kwargs)
    def forward(self, inputs, deterministic=False):
        return self.transform(inputs)
    def inverse(self, inputs, deterministic=False):
        return self.transform.inverse(inputs)
    def _make_transform(self, **kwargs):
        raise NotImplementedError
class SONEncoder(Encoder):
    def __init__(self, coeffs=None, input_features=2, output_features=2, coeff_std=0.05):
        super().__init__(input_features, output_features)
        d = self.output_features * (self.output_features - 1) // 2
        if coeffs is None:
            self.coeffs = nn.Parameter(torch.zeros((d,)))
            nn.init.normal_(self.coeffs, std=coeff_std)
        else:
            assert coeffs.shape == (d,)
            self.coeffs = nn.Parameter(coeffs)
        self.generators = torch.zeros((d, self.output_features, self.output_features))
    def forward(self, inputs, deterministic=False):
        z = torch.einsum("ij,bj->bi", self._rotation_matrix(), inputs)
        logdet = torch.zeros([])
        return z, logdet
    def inverse(self, inputs, deterministic=False):
        x = torch.einsum("ij,bj->bi", self._rotation_matrix(inverse=True), inputs)
        logdet = torch.zeros([])
        return x, logdet
    def _rotation_matrix(self, inverse=False):
        o = torch.zeros(self.output_features, self.output_features, device=self.coeffs.device)
        i, j = torch.triu_indices(self.output_features, self.output_features, offset=1)
        if inverse:
            o[i, j] = -self.coeffs
            o.T[i, j] = self.coeffs
        else:
            o[i, j] = self.coeffs
            o.T[i, j] = -self.coeffs
        a = torch.matrix_exp(o)
        return a
class FlowEncoder(InvertibleEncoder):
    def __init__(
        self,
        layers=3,
        hidden=10,
        transform_blocks=1,
        sigmoid=False,
        input_features=2,
        output_features=2,
    ):
        super().__init__(
            input_features,
            output_features,
            layers=layers,
            hidden=hidden,
            transform_blocks=transform_blocks,
            sigmoid=sigmoid,
        )
    def _make_transform(self, layers=3, hidden=10, transform_blocks=1, sigmoid=False):
        return make_scalar_transform(
            self.output_features,
            layers=layers,
            hidden=hidden,
            transform_blocks=transform_blocks,
            sigmoid=sigmoid,
        )
class LULinearEncoder(InvertibleEncoder):
    def _make_transform(self, **kwargs):
        return nflows.transforms.LULinear(self.output_features, **kwargs)
class NaiveLinearEncoder(InvertibleEncoder):
    def __init__(self, input_features=2, output_features=2, matrix=None, **kwargs):
        super().__init__(input_features, output_features, matrix=matrix)
    def _make_transform(self, matrix, **kwargs):
        transform = nflows.transforms.NaiveLinear(self.output_features, **kwargs)
        if matrix is not None:
            assert matrix.shape == (self.output_features, self.output_features)
            with torch.no_grad():
                transform._weight.copy_(matrix)
        return transform





