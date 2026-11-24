"""Network and ODE used by the reversible flow matching model."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def mlp(sizes: list[int], dropout: float = 0.0, norm: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 2):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if norm:
            layers.append(nn.LayerNorm(sizes[i + 1]))
        layers.append(nn.SiLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    return nn.Sequential(*layers)


class GatedResidualBlock(nn.Module):
    def __init__(self, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, 2 * hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.fc1(self.norm(x)).chunk(2, dim=-1)
        h = F.silu(gate) * value
        h = self.fc2(self.drop(h))
        return x + h / math.sqrt(2.0)


class ResidualMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: int,
        depth: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList(
            [GatedResidualBlock(hidden, dropout) for _ in range(max(1, depth))]
        )
        self.out_norm = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.in_proj(x))
        for block in self.blocks:
            h = block(h)
        return self.out(self.out_norm(h))


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = max(2, int(dim))
        half = max(1, self.dim // 2)
        freqs = torch.exp(torch.linspace(0.0, math.log(1000.0), half))
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        freqs = self.freqs.to(device=t.device, dtype=t.dtype)
        angles = math.pi * t * freqs[None, :]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.cat([emb, t], dim=-1)
        return emb[:, : self.dim]


class ConditionEncoder(nn.Module):
    """Encode the padded operating condition with a residual Fourier adapter."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: int,
        depth: int = 2,
        dropout: float = 0.0,
        fourier_features: bool = False,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.fourier_features = bool(fourier_features)
        feature_dim = self.in_dim * (3 if self.fourier_features else 1)
        self.base = (
            nn.Identity() if self.in_dim == self.out_dim else nn.Linear(self.in_dim, self.out_dim)
        )
        self.adapter = ResidualMLP(
            feature_dim, self.out_dim, hidden, depth=max(1, depth), dropout=dropout
        )

    def _features(self, cond: torch.Tensor) -> torch.Tensor:
        if not self.fourier_features:
            return cond
        return torch.cat([cond, torch.sin(math.pi * cond), torch.cos(math.pi * cond)], dim=-1)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.base(cond) + self.adapter(self._features(cond))


class JointAeroFlowDiffusion(nn.Module):
    """Conditional ODE connecting [aerodynamic coefficients, latent] to shape.

    In the released reversible flow matching model, the source contains three normalized
    aerodynamic coefficients and 17 latent components. The destination contains
    20 normalized CST coefficients.
    """

    def __init__(
        self,
        shape_dim: int,
        cond_dim: int,
        target_dim: int = 3,
        hidden: int = 512,
        depth: int = 4,
        dropout: float = 0.02,
        residual: bool = True,
        time_embed_dim: int = 64,
        solver: str = "heun",
        latent_prior_clip: float = 0.0,
        adaptive_coupling: bool = False,
        adaptive_coupling_strength: float = 0.25,
    ):
        super().__init__()
        if int(shape_dim) <= int(target_dim):
            raise ValueError(
                "shape_dim must be larger than target_dim for joint aero flow diffusion"
            )
        self.shape_dim = int(shape_dim)
        self.target_dim = int(target_dim)
        self.latent_dim = self.shape_dim - self.target_dim
        self.cond_dim = int(cond_dim)
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        in_dim = self.shape_dim + self.cond_dim + int(time_embed_dim)
        self.vector_net = (
            ResidualMLP(
                in_dim,
                self.shape_dim,
                hidden,
                depth=max(1, depth),
                dropout=dropout,
            )
            if residual
            else mlp([in_dim] + [hidden] * max(1, depth) + [self.shape_dim], dropout=dropout)
        )
        self.adaptive_coupling = bool(adaptive_coupling)
        self.adaptive_coupling_strength = max(0.0, float(adaptive_coupling_strength))
        self.cond_affine = None
        if self.adaptive_coupling:
            self.cond_affine = mlp(
                [self.cond_dim, max(64, hidden // 2), 2 * self.shape_dim], dropout=0.0
            )
        self.solver = str(solver).strip().lower()
        self.latent_prior_clip = max(0.0, float(latent_prior_clip))

    def velocity(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        time_features = self.time_embed(t)
        velocity = self.vector_net(torch.cat([x_t, cond, time_features], dim=-1))
        if self.cond_affine is not None:
            gain, bias = self.cond_affine(cond).chunk(2, dim=-1)
            strength = self.adaptive_coupling_strength
            velocity = velocity * (1.0 + strength * torch.tanh(gain)) + strength * torch.tanh(bias)
        return velocity

    def integrate(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        steps: int = 16,
        reverse: bool = False,
        solver: str | None = None,
    ) -> torch.Tensor:
        """Integrate source to shape, or shape to source when reverse is true."""
        steps = max(1, int(steps))
        method = str(solver or self.solver).strip().lower()
        direction = -1.0 if reverse else 1.0
        dt = direction / float(steps)
        state = x
        for i in range(steps):
            if reverse:
                t0_value = 1.0 - i / float(steps)
                t1_value = max(0.0, 1.0 - (i + 1) / float(steps))
            else:
                t0_value = i / float(steps)
                t1_value = min(1.0, (i + 1) / float(steps))
            t0 = x.new_full((x.shape[0], 1), t0_value)
            v0 = self.velocity(state, t0, cond)
            if method == "heun":
                # Use an Euler prediction to estimate the slope at the next time.
                euler = state + dt * v0
                t1 = x.new_full((x.shape[0], 1), t1_value)
                v1 = self.velocity(euler, t1, cond)
                state = state + 0.5 * dt * (v0 + v1)
            else:
                state = state + dt * v0
        return state


class BidirectionalAirfoilFlow(nn.Module):
    """Run inference in either direction with the same vector field."""

    def __init__(self, config):
        super().__init__()
        if (
            config["generator_type"] != "joint_flow_diffusion"
            or config.get("joint_forward_residual_weight", 0) != 0
            or not config.get("gated_residual_blocks", True)
        ):
            raise ValueError("Expected the reversible flow matching architecture with 20 CST coefficients")
        hidden = config["hidden"]
        embed_dim = int(config["condition_embed_dim"])
        self.condition_encoder = (
            ConditionEncoder(
                6,
                embed_dim,
                hidden=max(64, hidden // 2),
                depth=config["condition_encoder_depth"],
                dropout=config["dropout"],
                fourier_features=config["condition_fourier_features"],
            )
            if embed_dim > 0
            else nn.Identity()
        )
        self.flow = JointAeroFlowDiffusion(
            20,
            embed_dim if embed_dim > 0 else 6,
            hidden=hidden,
            depth=config["flow_matching_depth"],
            dropout=config["dropout"],
            residual=config["residual_mlp"],
            time_embed_dim=config["flow_matching_time_embed_dim"],
            solver=config["flow_matching_solver"],
            latent_prior_clip=config["latent_prior_clip"],
            adaptive_coupling=config["adaptive_coupling"],
            adaptive_coupling_strength=config["adaptive_coupling_strength"],
        )
        self.steps = max(1, int(config["flow_matching_steps_eval"]))
        self.forward_weight = max(0.0, float(config["joint_forward_flow_weight"]))

    def encode_flow_condition(self, aero: torch.Tensor) -> torch.Tensor:
        # Only normalized Mach and angle of attack condition the vector field.
        condition = torch.zeros_like(aero)
        condition[:, :2] = aero[:, :2]
        return self.condition_encoder(condition)

    def inverse_flow(self, latent: torch.Tensor, aero: torch.Tensor) -> torch.Tensor:
        """Generate normalized CST coefficients from normalized targets and noise."""
        if latent.shape[-1] == self.flow.shape_dim:
            latent = latent[:, -self.flow.latent_dim :]
        if latent.shape[-1] != self.flow.latent_dim:
            raise ValueError(f"Expected {self.flow.latent_dim} latent components")
        clip = self.flow.latent_prior_clip
        if clip > 0:
            latent = latent.clamp(-clip, clip)
        source = torch.cat([aero[:, 2:5], latent], dim=-1)
        return self.flow.integrate(source, self.encode_flow_condition(aero), steps=self.steps)

    def predict_aero(self, shape: torch.Tensor, aero: torch.Tensor) -> torch.Tensor:
        """Recover normalized lift, log drag and moment for a known geometry."""
        source = self.flow.integrate(
            shape, self.encode_flow_condition(aero), steps=self.steps, reverse=True
        )
        return self.forward_weight * source[:, :3]
