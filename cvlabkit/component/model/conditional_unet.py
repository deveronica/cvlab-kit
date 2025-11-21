"""U-Net with time + conditional input (from legacy/rectified_flow.py)."""

from __future__ import annotations

import math
from functools import partial

import torch
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from einops.layers.torch import Rearrange
from hyper_connections.hyper_connections_channel_first import (
    Residual,
    get_init_and_expand_reduce_stream_functions,
)
from torch import cat, nn
from torch.nn import Module, ModuleList

from cvlabkit.component.base import Model


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


def divisible_by(num, den):
    return (num % den) == 0


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.zeros(dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * (self.gamma + 1) * self.scale


class SinusoidalPosEmb(Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Block(Module):
    def __init__(self, dim, dim_out, dropout=0.0):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift=None):
        x = x.contiguous()
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return self.dropout(x)


class ResnetBlock(Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, dropout=0.0):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )
        self.block1 = Block(dim, dim_out, dropout=dropout)
        self.block2 = Block(dim_out, dim_out)

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h


class LinearAttention(Module):
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
        mk, mv = tuple(repeat(t, "h c n -> b h c n", b=b) for t in self.mem_kv)
        k, v = map(partial(cat, dim=-1), ((mk, k), (mv, v)))
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        context = einsum(k, v, "b h d n, b h e n -> b h d e")
        out = einsum(context, q, "b h d e, b h d n -> b h e n")
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(Module):
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
        mk, mv = map(lambda t: repeat(t, "h n d -> b h n d", b=b), self.mem_kv)
        k, v = map(partial(cat, dim=-2), ((mk, k), (mv, v)))
        q = q * self.scale
        sim = einsum(q, k, "b h i d, b h j d -> b h i j")
        attn = sim.softmax(dim=-1)
        out = einsum(attn, v, "b h i j, b h j d -> b h i d")
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class ConditionalUnet(Model):
    """U-Net with time + conditional input."""

    def __init__(self, cfg):
        super().__init__()
        dim = cfg.get("dim", 64)
        self.channels = cfg.get("channels", 3)
        dim_mults = cfg.get("dim_mults", (1, 2, 4, 8))
        dropout = cfg.get("dropout", 0.0)
        attn_dim_head = cfg.get("attn_dim_head", 32)
        attn_heads = cfg.get("attn_heads", 4)
        num_residual_streams = cfg.get("num_residual_streams", 2)
        dim_cond = cfg.get("dim_cond", 1)  # Conditional input dimension (1 for scalar)

        init_dim = cfg.get("init_dim", dim)
        self.init_conv = nn.Conv2d(self.channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim * 4
        sinu_pos_emb = SinusoidalPosEmb(dim, theta=10000)
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Conditional MLP (always enabled)
        first_dim = dim if dim_cond == 1 else dim_cond
        self.cond_mlp = nn.Sequential(
            SinusoidalPosEmb(dim, theta=10000) if dim_cond == 1 else nn.Identity(),
            nn.Linear(first_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        full_attn = (*((False,) * (len(dim_mults) - 1)), True)
        num_stages = len(dim_mults)
        full_attn = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        FullAttention = partial(Attention, flash=False)
        resnet_block = partial(ResnetBlock, time_emb_dim=time_dim, dropout=dropout)

        (
            init_hyper_conn,
            self.expand_streams,
            self.reduce_streams,
        ) = get_init_and_expand_reduce_stream_functions(
            num_residual_streams, disable=num_residual_streams == 1
        )
        res_conv = partial(nn.Conv2d, kernel_size=1, bias=False)

        self.downs = ModuleList([])
        num_resolutions = len(in_out)

        for ind, (
            (dim_in, dim_out),
            layer_full_attn,
            layer_attn_heads,
            layer_attn_dim_head,
        ) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)
            attn_klass = FullAttention if layer_full_attn else LinearAttention
            self.downs.append(
                ModuleList(
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
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = init_hyper_conn(
            dim=mid_dim, branch=resnet_block(mid_dim, mid_dim)
        )
        self.mid_attn = init_hyper_conn(
            dim=mid_dim,
            branch=FullAttention(
                mid_dim, heads=attn_heads[-1], dim_head=attn_dim_head[-1]
            ),
        )
        self.mid_block2 = init_hyper_conn(
            dim=mid_dim, branch=resnet_block(mid_dim, mid_dim)
        )

        self.ups = ModuleList([])
        for ind, (
            (dim_in, dim_out),
            layer_full_attn,
            layer_attn_heads,
            layer_attn_dim_head,
        ) in enumerate(
            zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))
        ):
            is_last = ind == (len(in_out) - 1)
            attn_klass = FullAttention if layer_full_attn else LinearAttention
            self.ups.append(
                ModuleList(
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
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.final_res_block = Residual(
            branch=resnet_block(init_dim * 2, init_dim),
            residual_transform=res_conv(init_dim * 2, init_dim),
        )
        self.final_conv = nn.Conv2d(init_dim, self.channels, 1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, times, cond):
        """Forward pass.

        Args:
            x: Input [B, C, H, W]
            times: Time values [B] or [B, 1]
            cond: Conditional input [B] or [B, 1] or [B, D]

        Returns:
            Output [B, C, H, W]
        """
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), (
            f"Input {x.shape[-2:]} must be divisible by {self.downsample_factor}"
        )

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(times)
        c = self.cond_mlp(cond)
        t = t + c  # Combine time and conditional embeddings

        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.expand_streams(x)
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        x = self.reduce_streams(x)

        for block1, block2, attn, upsample in self.ups:
            x = cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        x = cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)
