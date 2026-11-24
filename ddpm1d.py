"""Vector latent diffusion, following DiffusionSOD/model/ddpm.py's equations.

Unlike the SOD experiment, use epsilon MSE and an 8-layer MLP (paper Sec III-A).
Latents are unconstrained real vectors: no image normalization or [-1,1] clip.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def cosine_beta_schedule(steps, s=0.008):
    x = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
    a = torch.cos(((x / steps) + s) / (1 + s) * math.pi * .5).square()
    a = a / a[0]
    return (1 - a[1:] / a[:-1]).clamp(0, .999).float()


class NoiseMLP(nn.Module):
    def __init__(self, dim=24):
        super().__init__()
        self.dim = dim
        # Widths and time embedding are reconstruction choices, unspecified in paper.
        widths = [dim * 3, 128, 256, 256, 512, 256, 256, 128, dim]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(widths[:-1], widths[1:]))

    def forward(self, z, t, condition):
        freq = torch.exp(-math.log(10000) * torch.arange(self.dim // 2, device=z.device) / (self.dim // 2 - 1))
        phase = t.float()[:, None] * freq[None]
        time = torch.cat([phase.sin(), phase.cos()], dim=-1)
        h = torch.cat([z, condition, time], dim=-1)
        for layer in self.layers[:-1]:
            h = F.leaky_relu(layer(h), .01)
        # Signed, unbounded epsilon output; do not clamp to an image range.
        return self.layers[-1](h)


class GaussianDiffusion1D(nn.Module):
    def __init__(self, dim=24, timesteps=20, sampling_timesteps=10, parameterization='epsilon',
                 objective='epsilon', loss_type='mse'):
        super().__init__()
        assert 1 <= sampling_timesteps <= timesteps
        self.timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps
        assert parameterization in ['epsilon', 'vp', 'x0']
        assert objective in ['epsilon', 'x0', 'v'] and loss_type in ['mse', 'l1']
        self.parameterization = parameterization
        self.objective, self.loss_type = objective, loss_type
        self.model = NoiseMLP(dim)
        self.register_buffer('alpha_bar', (1 - cosine_beta_schedule(timesteps)).cumprod(0))

    def q_sample(self, z0, t, noise):
        a = self.alpha_bar[t, None]
        return a.sqrt() * z0 + (1 - a).sqrt() * noise

    def predict_start_from_noise(self, zt, t, eps):
        a = self.alpha_bar[t, None]
        return (zt - (1 - a).sqrt() * eps) / a.sqrt()

    def model_predictions(self, zt, t, condition):
        raw = self.model(zt, t, condition)
        return self.predictions_from_raw(zt, t, raw)

    def predictions_from_raw(self, zt, t, raw):
        if self.parameterization == 'epsilon':
            return raw, self.predict_start_from_noise(zt, t, raw)
        if self.parameterization == 'x0':
            a = self.alpha_bar[t, None]
            return (zt - a.sqrt() * raw) / (1 - a).sqrt(), raw
        # Explicit stability ablation, not a detail stated in the paper.
        # VP preconditioning: raw models v; the optimized objective stays epsilon MSE.
        # eps = sigma*zt + alpha*v; x0 = alpha*zt - sigma*v.
        # The latter avoids division by alpha (~0.00246 at the final step).
        a = self.alpha_bar[t, None]
        eps = (1 - a).sqrt() * zt + a.sqrt() * raw
        x0 = a.sqrt() * zt - (1 - a).sqrt() * raw
        return eps, x0

    def forward(self, z0, condition, t=None, noise=None):
        if t is None:
            t = torch.randint(self.timesteps, (z0.shape[0],), device=z0.device)
        if noise is None:
            noise = torch.randn_like(z0)
        zt = self.q_sample(z0, t, noise)
        raw = self.model(zt, t, condition)
        eps, z0_hat = self.predictions_from_raw(zt, t, raw)
        if self.objective == 'epsilon':
            prediction, target = eps, noise
        elif self.objective == 'x0':
            prediction, target = z0_hat, z0
        else:
            a = self.alpha_bar[t, None]
            prediction = raw if self.parameterization == 'vp' else a.sqrt() * eps - (1-a).sqrt() * z0_hat
            target = a.sqrt() * noise - (1-a).sqrt() * z0
        loss_fn = F.mse_loss if self.loss_type == 'mse' else F.l1_loss
        return loss_fn(prediction, target), z0_hat

    @torch.no_grad()
    def sample(self, condition, generator=None, initial_noise=None):
        # Same endpoint convention as DiffusionSOD.ddim_sample; eta=0.
        z = (torch.randn(condition.shape, device=condition.device, dtype=condition.dtype,
                         generator=generator) if initial_noise is None else initial_noise.clone())
        times = torch.linspace(-1, self.timesteps - 1, self.sampling_timesteps + 1).int().tolist()[::-1]
        for current, following in zip(times[:-1], times[1:]):
            t = torch.full((z.shape[0],), current, device=z.device, dtype=torch.long)
            eps, z0 = self.model_predictions(z, t, condition)
            if following < 0:
                z = z0
            else:
                a_next = self.alpha_bar[following]
                z = a_next.sqrt() * z0 + (1 - a_next).sqrt() * eps
        return z
