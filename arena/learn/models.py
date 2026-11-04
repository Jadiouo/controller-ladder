"""Four students on one backbone.  All map a normalised flat observation window o (B, Do) to a normalised action chunk
a (B, Ta, n) and share the training interface:  loss(o, a, gen) -> scalar,  sample(o, gen, **kw) -> (B, Ta, n).

  bc        regression  pi(o) -> a                                   (MSE)
  mip       Minimal Iterative Policy, arXiv 2512.01809 sec. 4 (Eq. 4.4/4.5), two-step regression with intermediate
            supervision:  loss = |pi(o, 0, 0) - a|^2 + |pi(o, I, t*) - a|^2,  I = t* a + (1 - t*) z,  z ~ N(0, I), t* = 0.9
            inference:  a0 = pi(o, 0, 0);  a = pi(o, t* a0, t*)    (2 network evaluations, no sampling noise)
  diffusion DDPM training (epsilon prediction, cosine schedule, K = 100), DDIM (eta = 0) or ancestral sampling
  flow      rectified flow: x_t = t a + (1 - t) z, target velocity a - z, Euler sampling with `steps` steps
            (steps = 1 -> one-step flow = z + v(z, 0); it approaches the conditional mean of a)
Denoiser = residual MLP (LayerNorm -> Linear -> Mish -> Linear) over concat[o, x, sin/cos(t)].  The action chunk is
flattened (Ta*n), so chunking needs no special case; the observation horizon is the window inside o.  A 1-D temporal
U-Net (Chi et al. 2023) is NOT implemented: with Ta <= 16 and 2 joints an MLP denoiser is the minimal choice.
Randomness: noise is drawn from the torch Generator passed in (the controller derives it from self.rng).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

KINDS = ("bc", "mip", "diffusion", "flow")
T_STAR = 0.9


def time_embed(t: torch.Tensor, n_freq: int = 8) -> torch.Tensor:
    """t (B,) in [0, 1] -> (B, 2*n_freq)"""
    f = torch.exp(torch.linspace(0, math.log(200.0), n_freq, device=t.device)) * math.pi
    ang = t[:, None] * f[None]
    return torch.cat([ang.sin(), ang.cos()], -1)


class Block(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.ln, self.a, self.b = nn.LayerNorm(w), nn.Linear(w, w), nn.Linear(w, w)

    def forward(self, h):
        return h + self.b(F.mish(self.a(self.ln(h))))


class Backbone(nn.Module):
    def __init__(self, d_in, d_out, width, depth):
        super().__init__()
        self.inp = nn.Linear(d_in, width)
        self.blocks = nn.ModuleList([Block(width) for _ in range(depth)])
        self.ln, self.out = nn.LayerNorm(width), nn.Linear(width, d_out)

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        return self.out(self.ln(h))


class Policy(nn.Module):
    kind = ""
    conditioned = True          # does the network take (x, t)?

    def __init__(self, obs_dim: int, n: int, chunk: int, width: int = 256, depth: int = 4):
        super().__init__()
        self.obs_dim, self.n, self.chunk, self.width, self.depth = obs_dim, n, chunk, width, depth
        self.da = chunk * n
        d_in = obs_dim + (self.da + 16 if self.conditioned else 0)
        self.net = Backbone(d_in, self.da, width, depth)
        for name, size in (("obs_mean", obs_dim), ("obs_std", obs_dim), ("act_mean", n), ("act_std", n)):
            self.register_buffer(name, torch.zeros(size) if "mean" in name else torch.ones(size))

    def config(self) -> dict:
        return dict(kind=self.kind, obs_dim=self.obs_dim, n=self.n, chunk=self.chunk, width=self.width, depth=self.depth)

    def set_norm(self, om, os_, am, as_):
        for name, v in (("obs_mean", om), ("obs_std", os_), ("act_mean", am), ("act_std", as_)):
            getattr(self, name).copy_(torch.as_tensor(v))

    def _f(self, o, x=None, t=None):
        if not self.conditioned:
            return self.net(o)
        return self.net(torch.cat([o, x, time_embed(t)], -1))

    def _noise(self, shape, gen, like):
        return torch.randn(shape, generator=gen, device=like.device, dtype=like.dtype)

    def loss(self, o, a, gen=None):
        raise NotImplementedError

    def sample(self, o, gen=None, **kw):
        raise NotImplementedError

    @torch.inference_mode()
    def act_chunk(self, feat: torch.Tensor, gen=None, **kw) -> torch.Tensor:
        """raw feature rows (B, Do) -> raw action chunk (B, Ta, n) in torque units"""
        o = (feat - self.obs_mean) / self.obs_std
        return self.sample(o, gen, **kw) * self.act_std + self.act_mean


class BC(Policy):
    kind, conditioned = "bc", False

    def loss(self, o, a, gen=None):
        return F.mse_loss(self._f(o).view_as(a), a)

    def sample(self, o, gen=None, **kw):
        return self._f(o).view(-1, self.chunk, self.n)


class MIP(Policy):
    kind = "mip"

    def loss(self, o, a, gen=None):
        B = o.shape[0]
        af = a.reshape(B, -1)
        z = self._noise(af.shape, gen, af)
        i_star = T_STAR * af + (1 - T_STAR) * z
        zero = torch.zeros_like(af)
        t0, t1 = torch.zeros(B, device=o.device), torch.full((B,), T_STAR, device=o.device)
        return F.mse_loss(self._f(o, zero, t0), af) + F.mse_loss(self._f(o, i_star, t1), af)

    def sample(self, o, gen=None, **kw):
        B = o.shape[0]
        zero = torch.zeros(B, self.da, device=o.device, dtype=o.dtype)
        a0 = self._f(o, zero, torch.zeros(B, device=o.device))
        a = self._f(o, T_STAR * a0, torch.full((B,), T_STAR, device=o.device))
        return a.view(B, self.chunk, self.n)


class Flow(Policy):
    kind = "flow"
    default_steps = 10

    def loss(self, o, a, gen=None):
        B = o.shape[0]
        af = a.reshape(B, -1)
        z = self._noise(af.shape, gen, af)
        t = torch.rand(B, generator=gen, device=o.device, dtype=o.dtype)
        x = t[:, None] * af + (1 - t[:, None]) * z
        return F.mse_loss(self._f(o, x, t), af - z)

    def sample(self, o, gen=None, steps: int | None = None, **kw):
        steps = steps or self.default_steps
        B = o.shape[0]
        x = self._noise((B, self.da), gen, o)
        for i in range(steps):
            t = torch.full((B,), i / steps, device=o.device, dtype=o.dtype)
            x = x + self._f(o, x, t) / steps
        return x.view(B, self.chunk, self.n)


class Diffusion(Policy):
    kind = "diffusion"
    K = 100
    default_steps = 10

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        s = 0.008
        m = torch.arange(self.K + 1, dtype=torch.float64) / self.K
        f = torch.cos((m + s) / (1 + s) * math.pi / 2) ** 2
        ab = (f / f[0]).clamp(min=1e-3).float()
        self.register_buffer("abar", ab, persistent=False)        # abar[0] = 1 (clean) ... abar[K] ~ 0 (noise)

    def loss(self, o, a, gen=None):
        B = o.shape[0]
        af = a.reshape(B, -1)
        m = torch.randint(1, self.K + 1, (B,), generator=gen, device=o.device)
        eps = self._noise(af.shape, gen, af)
        ab = self.abar[m][:, None]
        x = ab.sqrt() * af + (1 - ab).sqrt() * eps
        return F.mse_loss(self._f(o, x, m.float() / self.K), eps)

    def sample(self, o, gen=None, steps: int | None = None, eta: float = 0.0, **kw):
        steps = steps or self.default_steps
        B = o.shape[0]
        ms = torch.linspace(self.K, 0, steps + 1).round().long().tolist()
        x = self._noise((B, self.da), gen, o)
        for m, mp in zip(ms[:-1], ms[1:]):
            ab, abp = self.abar[m], self.abar[mp]
            eps = self._f(o, x, torch.full((B,), m / self.K, device=o.device, dtype=o.dtype))
            x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-5, 5)
            sig = eta * ((1 - abp) / (1 - ab) * (1 - ab / abp)).clamp(min=0).sqrt()
            dirx = (1 - abp - sig ** 2).clamp(min=0).sqrt() * eps
            x = abp.sqrt() * x0 + dirx
            if eta > 0 and mp > 0:
                x = x + sig * self._noise(x.shape, gen, x)
        return x.view(B, self.chunk, self.n)


CLASSES = {c.kind: c for c in (BC, MIP, Diffusion, Flow)}


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def build_policy(kind: str, obs_dim: int, n: int, chunk: int, width: int = 256, depth: int = 4,
                 match_to: int | None = None) -> Policy:
    """`match_to`: choose the width of THIS model (searched within +-24 of `width`) so that its parameter count is
    closest to `match_to` (used to give the conditioning-free BC the same budget as the denoisers)."""
    cls = CLASSES[kind]
    if match_to is None:
        return cls(obs_dim, n, chunk, width, depth)
    best = min(range(width - 24, width + 25), key=lambda w: abs(n_params(cls(obs_dim, n, chunk, w, depth)) - match_to))
    return cls(obs_dim, n, chunk, best, depth)
