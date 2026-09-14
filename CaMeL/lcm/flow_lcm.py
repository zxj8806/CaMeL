
from CaMeL.lcm.base import BaseLCM
class FlowLCM(BaseLCM):
    def __init__(
        self,
        causal_model,
        encoder,
        intervention_prior=None,
        dim_z=2,
        intervention_set="atomic_or_none",
    ):
        super().__init__(
            causal_model,
            encoder,
            decoder=None,
            intervention_prior=intervention_prior,
            dim_z=dim_z,
            intervention_set=intervention_set,
        )
    def forward(self, x1, x2, interventions=None, **kwargs):
        z1, logdet_x1 = self.encoder(x1)
        z2, logdet_x2 = self.encoder(x2)
        log_prob_z, outputs = self._evaluate_prior_marginalized_over_interventions(
            z1, z2, interventions
        )
        log_prob = log_prob_z + logdet_x1 + logdet_x2
        outputs["log_det_j"] = logdet_x1 + logdet_x2
        return log_prob, outputs
    def log_likelihood(self, x1, x2, interventions=None, **kwargs):
        return self.forward(x1, x2, interventions, **kwargs)[0]
    def encode_to_noise(self, x, deterministic=True):
        z, _ = self.encoder(x, deterministic=deterministic)
        epsilon = self.scm.causal_to_noise(z)
        return epsilon
    def encode_to_causal(self, x, deterministic=True):
        z, _ = self.encoder(x, deterministic=deterministic)
        return z
    def decode_noise(self, epsilon, deterministic=True):
        z = self.scm.noise_to_causal(epsilon)
        x, _ = self.encoder.inverse(z)
        return x
    def decode_causal(self, z, deterministic=True):
        x, _ = self.encoder.inverse(z)
        return x





