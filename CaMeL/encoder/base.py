
from torch import nn
class IntractableError(Exception):
    pass
class Encoder(nn.Module):
    def __init__(self, input_features=2, output_features=2):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
    def forward(self, inputs, deterministic=False):
        raise NotImplementedError
    def inverse(self, inputs, deterministic=False):
        raise IntractableError()
class Inverse(Encoder):
    def __init__(self, base_model):
        super().__init__(
            input_features=base_model.output_features, output_features=base_model.input_features
        )
        self.base_model = base_model
    def forward(self, inputs, deterministic=False):
        return self.base_model.inverse(inputs)
    def inverse(self, outputs, deterministic=False):
        return self.base_model(outputs)






