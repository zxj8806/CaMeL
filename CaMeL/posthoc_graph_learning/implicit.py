
from functools import lru_cache
import torch
from CaMeL.utils import mask
def dependance(
    transform,
    inputs,
    context,
    component,
    invert=False,
    measure=torch.nn.functional.mse_loss,
    normalize=True,
    **kwargs,
):
    context_shuffled = context.clone()
    batchsize = context.shape[0]
    idx = torch.randperm(batchsize)
    context_shuffled[:, component] = context_shuffled[idx, component]
    function = transform.inverse if invert else transform
    f, _ = function(inputs, context=context, **kwargs)
    f_shuffled, _ = function(inputs, context=context_shuffled, **kwargs)
    if normalize:
        mean, std = torch.mean(f), torch.std(f)
        std = torch.clamp(std, 0.1)
        f = (f - mean) / std
        f_shuffled = (f_shuffled - mean) / std
    difference = measure(f, f_shuffled)
    return difference
def solution_dependance_on_noise(model, i, j, noise):
    transform = model.scm.solution_functions[i]
    inputs = noise[:, i].unsqueeze(1)
    mask_ = torch.ones_like(noise)
    mask_[:, i] = 0
    context = mask(noise, mask_)
    return dependance(transform, inputs, context, j, invert=True)
def find_topological_order(model, noise):
    @lru_cache()
    def solution_dependance_on_noise(i, j):
        transform = model.scm.solution_functions[i]
        inputs = noise[:, i].unsqueeze(1)
        mask_ = torch.ones_like(noise)
        mask_[:, i] = 0
        context = mask(noise, mask_)
        return dependance(transform, inputs, context, j, invert=True)
    topological_order = []
    components = set(range(model.dim_z))
    while components:
        least_dependant_solution = None
        least_dependant_score = float("inf")
        for i in components:
            others = [j for j in components if j != i]
            score = sum(solution_dependance_on_noise(i, j) for j in others)
            if score < least_dependant_score:
                least_dependant_solution = i
                least_dependant_score = score
        topological_order.append(least_dependant_solution)
        components.remove(least_dependant_solution)
    return topological_order
class CausalMechanism(torch.nn.Module):
    def __init__(self, solution_transform, component, ancestor_mechanisms):
        super().__init__()
        self.component = component
        self.solution_transform = solution_transform
        self.ancestor_mechanisms = ancestor_mechanisms
    def forward(self, inputs, context, noise, computed_noise=None):
        solution_context = self._compute_context(inputs, context, noise, computed_noise)
        return self.solution_transform.inverse(inputs, context=solution_context)
    def inverse(self, inputs, context, noise, computed_noise=None):
        solution_context = self._compute_context(inputs, context, noise, computed_noise)
        return self.solution_transform(inputs, context=solution_context)
    def _compute_context(self, inputs, context, noise, computed_noise=None):
        noise = self._randomize_noise(noise)
        if computed_noise is None:
            computed_noise = dict()
        for a, mech in self.ancestor_mechanisms.items():
            if a not in computed_noise:
                this_noise, _ = mech.inverse(
                    context[:, a].unsqueeze(1), context, noise, computed_noise=computed_noise
                )
                computed_noise[a] = this_noise.squeeze()
            noise[:, a] = computed_noise[a]
        return noise
    def _randomize_noise(self, noise):
        noise = noise.clone()
        for k in range(noise.shape[1]):
            noise[:, k] = noise[torch.randperm(noise.shape[0]), k]
        return noise
def construct_causal_mechanisms(model, topological_order):
    causal_mechanisms = {}
    for i in topological_order:
        solution = model.scm.get_masked_solution_function(i)
        causal_mechanisms[i] = CausalMechanism(
            solution,
            component=i,
            ancestor_mechanisms={a: mech for a, mech in causal_mechanisms.items()},
        )
    return causal_mechanisms
def compute_implicit_causal_effects(model, noise):
    model.eval()
    z = model.scm.noise_to_causal(noise)
    causal_effect = torch.zeros((model.dim_z, model.dim_z))
    topological_order = find_topological_order(model, noise)
    mechanisms = construct_causal_mechanisms(model, topological_order)
    for pos, i in enumerate(topological_order):
        for j in topological_order[:pos]:
            causal_effect[j, i] = dependance(
                mechanisms[i], noise[:, i : i + 1], z, j, invert=False, noise=noise
            )
    return causal_effect, topological_order





