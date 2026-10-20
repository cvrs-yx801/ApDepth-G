"""Explicit 8/12-channel U-Net inputs and reproducible SD2 initialization."""
from __future__ import annotations

import torch
from torch import nn


def adapt_unet_input(unet: nn.Module, channels: int) -> None:
    """Repeat a base 4ch convolution 2x/3x, scale by 1/2 or 1/3, copy bias.

    Never silently convert one trained ablation architecture into another.
    """
    if channels not in (8, 12):
        raise ValueError("unet_in_channels must be 8 or 12")
    old = unet.conv_in
    if old.in_channels == channels:
        return
    if old.in_channels != 4 or old.groups != 1:
        raise ValueError("Initialize each 8/12ch experiment independently from the same 4ch base U-Net")
    repeats = channels // 4
    new = nn.Conv2d(channels, old.out_channels, old.kernel_size, old.stride, old.padding,
                    old.dilation, old.groups, old.bias is not None, old.padding_mode,
                    device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        new.weight.copy_(old.weight.repeat(1, repeats, 1, 1) / repeats)
        if old.bias is not None:
            new.bias.copy_(old.bias)
    unet.conv_in = new
    unet.register_to_config(in_channels=channels)


def unet_inputs(rgb: torch.Tensor, prior: torch.Tensor | None, depth: torch.Tensor,
                channels: int, conditioning_enabled: bool) -> torch.Tensor:
    if channels == 8:
        if conditioning_enabled:
            raise ValueError("An 8ch U-Net cannot have explicit DA2 conditioning")
        return torch.cat([rgb, depth], dim=1)
    if channels != 12:
        raise ValueError("Expected an 8ch or 12ch U-Net")
    if conditioning_enabled and prior is None:
        raise ValueError("12ch DA2 conditioning requires a prior latent")
    condition = prior if conditioning_enabled else torch.zeros_like(rgb)
    return torch.cat([rgb, condition, depth], dim=1)
