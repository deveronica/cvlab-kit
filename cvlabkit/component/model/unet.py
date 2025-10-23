"""U-Net architecture for Rectified Flow with attention and hyper connections."""

from __future__ import annotations

import math
from functools import partial

import einx
import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import cat, nn

from cvlabkit.component.base import Model


# Private helper functions


def _exists(v):
    """Check if value is not None (private helper)."""
    return v is not None


def _default(v, d):
    """Return v if it exists, else d (private helper)."""
    return v if _exists(v) else d


def _cast_tuple(t, length=1):
    """Cast to tuple of given length (private helper)."""
    return t if isinstance(t, tuple) else ((t,) * length)


def _divisible_by(num, den):
    """Check if num is divisible by den (private helper)."""
    return (num % den) == 0


# Layer factory functions


def _create_upsample(dim, dim_out=None):
    """Create upsample layer (private helper)."""
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, _default(dim_out, dim), 3, padding=1),
    )


def _create_downsample(dim, dim_out=None):
    """Create downsample layer (private helper)."""
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, _default(dim_out, dim), 1),
    )


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.zeros(dim, 1, 1))

    def forward(self, x):
        return torch.nn.functional.normalize(x, dim=1) * (self.gamma + 1) * self.scale


# Positional embeddings


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings."""

    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = einx.multiply("i, j -> i j", x, emb)
        emb = cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """Random or learned sinusoidal positional embeddings."""

    def __init__(self, dim, is_random=False):
        """Initialize positional embeddings."""
        super().__init__()
        assert _divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = cat((x, fouriered), dim=-1)
        return fouriered


# Building blocks


class Block(nn.Module):
    """Convolutional block with normalization and activation."""

    def __init__(self, dim, dim_out, dropout=0.0):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift=None):
        """Forward pass."""
        x = x.contiguous()
        x = self.proj(x)
        x = self.norm(x)

        if _exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return self.dropout(x)


class ResnetBlock(nn.Module):
    """ResNet-style block with time embedding."""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, dropout=0.0):
        """Initialize ResNet block."""
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if _exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, dropout=dropout)
        self.block2 = Block(dim_out, dim_out)

    def forward(self, x, time_emb=None):
        """Forward pass."""
        scale_shift = None
        if _exists(self.mlp) and _exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)

        return h


# Attention


class LinearAttention(nn.Module):
    """Linear attention mechanism."""

    def __init__(self, dim, heads=4, dim_head=32, num_mem_kv=4):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), RMSNorm(dim))

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = tuple(
            rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads) for t in qkv
        )

        from einops import repeat

        mk, mv = tuple(repeat(t, "h c n -> b h c n", b=b) for t in self.mem_kv)
        k, v = map(partial(cat, dim=-1), ((mk, k), (mv, v)))

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        from einops import einsum

        context = einsum(k, v, "b h d n, b h e n -> b h d e")

        out = einsum(context, q, "b h d e, b h d n -> b h e n")
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    """Full attention mechanism."""

    def __init__(self, dim, heads=4, dim_head=32, num_mem_kv=4, flash=False):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h (x y) c", h=self.heads), qkv
        )

        from einops import repeat

        mk, mv = map(lambda t: repeat(t, "h n d -> b h n d", b=b), self.mem_kv)
        k, v = map(partial(cat, dim=-2), ((mk, k), (mv, v)))

        q = q * self.scale

        from einops import einsum

        sim = einsum(q, k, "b h i d, b h j d -> b h i j")

        attn = sim.softmax(dim=-1)
        out = einsum(attn, v, "b h i j, b h j d -> b h i d")

        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


# Residual wrapper (simplified, no hyper connections)


class Residual(nn.Module):
    """Residual connection wrapper."""

    def __init__(self, branch, residual_transform=None):
        super().__init__()
        self.branch = branch
        self.residual_transform = residual_transform

    def forward(self, x, *args, **kwargs):
        residual = x
        if self.residual_transform is not None:
            residual = self.residual_transform(residual)
        return self.branch(x, *args, **kwargs) + residual


# Main U-Net model


class Unet(Model):
    """U-Net architecture for Rectified Flow.

    Implements a U-Net with ResNet blocks, attention mechanisms, and optional
    mean-variance output for probabilistic predictions.
    """

    def __init__(self, cfg):
        """Initialize U-Net.

        Args:
            cfg: Configuration object with parameters:
                - dim: Base dimension (required)
                - init_dim: Initial dimension (default: dim)
                - out_dim: Output dimension (default: channels or channels*2)
                - dim_mults: Dimension multipliers (default: (1, 2, 4, 8))
                - channels: Input channels (default: 3)
                - mean_variance_net: Output mean and variance (default: False)
                - learned_sinusoidal_cond: Use learned sinusoidal conditioning (default: False)
                - random_fourier_features: Use random Fourier features (default: False)
                - learned_sinusoidal_dim: Dimension for learned sinusoidal (default: 16)
                - sinusoidal_pos_emb_theta: Theta for sinusoidal embeddings (default: 10000)
                - dropout: Dropout rate (default: 0.0)
                - attn_dim_head: Attention head dimension (default: 32)
                - attn_heads: Number of attention heads (default: 4)
                - full_attn: Use full attention (default: only last layer)
                - flash_attn: Use flash attention (default: False)
                - num_residual_streams: Number of residual streams (default: 2)
                - accept_cond: Accept conditional input (default: False)
                - dim_cond: Conditional dimension (default: None)
        """
        super().__init__()

        # Get configuration
        dim = cfg.dim
        init_dim = cfg.get("init_dim", dim)
        out_dim = cfg.get("out_dim")
        dim_mults = cfg.get("dim_mults", (1, 2, 4, 8))
        channels = cfg.get("channels", 3)
        mean_variance_net = cfg.get("mean_variance_net", False)
        learned_sinusoidal_cond = cfg.get("learned_sinusoidal_cond", False)
        random_fourier_features = cfg.get("random_fourier_features", False)
        learned_sinusoidal_dim = cfg.get("learned_sinusoidal_dim", 16)
        sinusoidal_pos_emb_theta = cfg.get("sinusoidal_pos_emb_theta", 10000)
        dropout = cfg.get("dropout", 0.0)
        attn_dim_head = cfg.get("attn_dim_head", 32)
        attn_heads = cfg.get("attn_heads", 4)
        full_attn = cfg.get("full_attn")
        flash_attn = cfg.get("flash_attn", False)
        accept_cond = cfg.get("accept_cond", False)
        dim_cond = cfg.get("dim_cond")

        # Determine dimensions
        self.channels = channels

        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

        dims = [init_dim, *[dim * m for m in dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))

        # Time embeddings
        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = (
            learned_sinusoidal_cond or random_fourier_features
        )

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                learned_sinusoidal_dim, random_fourier_features
            )
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta=sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Additional cond mlp
        self.cond_mlp = None
        if accept_cond:
            assert _exists(dim_cond), "`dim_cond` must be set on init"
            first_dim = dim if dim_cond == 1 else dim_cond

            self.cond_mlp = nn.Sequential(
                SinusoidalPosEmb(dim, theta=sinusoidal_pos_emb_theta)
                if dim_cond == 1
                else nn.Identity(),
                nn.Linear(first_dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )

        # Attention
        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        full_attn = _cast_tuple(full_attn, num_stages)
        attn_heads = _cast_tuple(attn_heads, num_stages)
        attn_dim_head = _cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        # Prepare blocks
        full_attention_klass = partial(Attention, flash=flash_attn)
        resnet_block = partial(ResnetBlock, time_emb_dim=time_dim, dropout=dropout)

        # Simplified residual streams (no hyper connections for now)
        res_conv = partial(nn.Conv2d, kernel_size=1, bias=False)

        # Layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (
            (dim_in, dim_out),
            layer_full_attn,
            layer_attn_heads,
            layer_attn_dim_head,
        ) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            attn_klass = full_attention_klass if layer_full_attn else LinearAttention

            self.downs.append(
                nn.ModuleList(
                    [
                        Residual(branch=resnet_block(dim_in, dim_in)),
                        Residual(branch=resnet_block(dim_in, dim_in)),
                        Residual(
                            branch=attn_klass(
                                dim_in,
                                dim_head=layer_attn_dim_head,
                                heads=layer_attn_heads,
                            )
                        ),
                        _create_downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = Residual(branch=resnet_block(mid_dim, mid_dim))
        self.mid_attn = Residual(
            branch=full_attention_klass(
                mid_dim, heads=attn_heads[-1], dim_head=attn_dim_head[-1]
            )
        )
        self.mid_block2 = Residual(branch=resnet_block(mid_dim, mid_dim))

        for ind, (
            (dim_in, dim_out),
            layer_full_attn,
            layer_attn_heads,
            layer_attn_dim_head,
        ) in enumerate(
            zip(
                *[
                    list(reversed(x))
                    for x in (in_out, full_attn, attn_heads, attn_dim_head)
                ]
            )
        ):
            is_last = ind == (len(in_out) - 1)

            attn_klass = full_attention_klass if layer_full_attn else LinearAttention

            self.ups.append(
                nn.ModuleList(
                    [
                        Residual(
                            branch=resnet_block(dim_out + dim_in, dim_out),
                            residual_transform=res_conv(dim_out + dim_in, dim_out),
                        ),
                        Residual(
                            branch=resnet_block(dim_out + dim_in, dim_out),
                            residual_transform=res_conv(dim_out + dim_in, dim_out),
                        ),
                        Residual(
                            branch=attn_klass(
                                dim_out,
                                dim_head=layer_attn_dim_head,
                                heads=layer_attn_heads,
                            )
                        ),
                        _create_upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.mean_variance_net = mean_variance_net

        default_out_dim = channels * (1 if not mean_variance_net else 2)
        self.out_dim = _default(out_dim, default_out_dim)

        self.final_res_block = Residual(
            branch=resnet_block(init_dim * 2, init_dim),
            residual_transform=res_conv(init_dim * 2, init_dim),
        )
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)

    @property
    def downsample_factor(self):
        """Calculate downsample factor."""
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, times, cond=None):
        """Forward pass.

        Args:
            x: Input tensor
            times: Time conditioning
            cond: Optional conditional input

        Returns:
            Output tensor or tuple of (mean, variance)
        """
        assert all([_divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), (
            f"your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet"
        )

        x = self.init_conv(x)

        r = x.clone()

        t = self.time_mlp(times)

        # Maybe additional cond
        assert not (_exists(cond) ^ _exists(self.cond_mlp))

        if _exists(cond):
            assert _exists(self.cond_mlp), (
                "`accept_cond` and `dim_cond` must be set on init for `Unet`"
            )
            c = self.cond_mlp(cond)
            t = t + c

        # Hiddens
        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = cat((x, r), dim=1)

        x = self.final_res_block(x, t)

        out = self.final_conv(x)

        if not self.mean_variance_net:
            return out

        mean, log_var = rearrange(
            out, "b (c mean_log_var) h w -> mean_log_var b c h w", mean_log_var=2
        )
        variance = log_var.exp()  # variance needs to be positive
        return torch.stack((mean, variance))
