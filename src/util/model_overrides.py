"""Load a main-training U-Net and its matching scheduler for inference."""
from pathlib import Path
import json

from omegaconf import OmegaConf

from src.util.da2_config import load_da2_config


def checkpoint_inference_settings(unet_checkpoint):
    """Read checkpoint metadata, with a fallback for existing training runs."""
    if not unet_checkpoint:
        return {}
    path = Path(unet_checkpoint)
    metadata_path = path / "inference_settings.json"
    if metadata_path.is_file():
        settings = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        # Training layout: <run>/checkpoint/<iteration> and <run>/config.yaml.
        candidates = (path / "config.yaml", path.parent.parent / "config.yaml")
        config_path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if config_path is None:
            raise FileNotFoundError(
                f"No inference_settings.json or run config.yaml found for {path}. "
                "Keep the checkpoint in its training run or copy its saved config.yaml into it."
            )
        cfg = OmegaConf.load(config_path)
        settings = {
            "unet_in_channels": int(cfg.get("unet_in_channels", 12)),
            "da2_conditioning_enabled": cfg.get("da2_conditioning_enabled", True),
            "da2_config": OmegaConf.to_container(cfg.da2, resolve=True) if cfg.get("da2") is not None else None,
            "diffusion_schedule": OmegaConf.to_container(cfg.diffusion_schedule, resolve=True)
            if cfg.get("diffusion_schedule") is not None else None,
        }
    if not isinstance(settings.get("da2_conditioning_enabled"), bool):
        raise ValueError("Checkpoint da2_conditioning_enabled must be a boolean")
    config_path = path / "unet" / "config.json"
    architecture = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    channels = int(architecture.get("in_channels", settings.get("unet_in_channels", 12)))
    if channels not in (8, 12) or settings.get("unet_in_channels", channels) != channels:
        raise ValueError("Checkpoint U-Net config and inference metadata disagree on 8/12ch architecture")
    if channels == 8 and settings["da2_conditioning_enabled"]:
        raise ValueError("8ch checkpoint cannot enable explicit DA2 conditioning")
    settings["unet_in_channels"] = channels
    return settings


def checkpoint_pipeline_kwargs(unet_checkpoint=None, da2_config_path=None, torch_dtype=None, load_components=False):
    settings = checkpoint_inference_settings(unet_checkpoint)
    kwargs = {}
    if settings:
        kwargs["da2_conditioning_enabled"] = settings["da2_conditioning_enabled"]
        kwargs["load_da2"] = settings["da2_conditioning_enabled"]
        # Supply complete components BEFORE constructing the base pipeline.
        # from_pretrained reads these checkpoint-local config files and weights;
        # no file in BASE_CKPT_DIR is copied, edited or replaced.
        if load_components:
            from diffusers import DDIMScheduler, UNet2DConditionModel
            path = Path(unet_checkpoint)
            kwargs["unet"] = UNet2DConditionModel.from_pretrained(
                path / "unet", torch_dtype=torch_dtype, local_files_only=True)
            kwargs["scheduler"] = DDIMScheduler.from_pretrained(path / "scheduler", local_files_only=True)
    if da2_config_path:
        kwargs["da2_config"] = load_da2_config(da2_config_path)
    elif settings.get("da2_config") is not None:
        kwargs["da2_config"] = settings["da2_config"]
    return kwargs


def apply_model_overrides(pipe, unet_checkpoint=None, components_loaded=False):
    from diffusers import DDIMScheduler, UNet2DConditionModel

    if unet_checkpoint:
        path = Path(unet_checkpoint)
        settings = checkpoint_inference_settings(path)
        unet = pipe.unet if components_loaded else UNet2DConditionModel.from_pretrained(
            path / "unet", torch_dtype=pipe.unet.dtype, local_files_only=True)
        if unet.config.in_channels != settings["unet_in_channels"]:
            raise ValueError("Loaded U-Net architecture differs from checkpoint config.json")
        pipe.unet = unet
        scheduler_path = path / "scheduler"
        if not scheduler_path.is_dir():
            raise FileNotFoundError(f"Checkpoint scheduler is missing: {scheduler_path}")
        pipe.scheduler = DDIMScheduler.from_pretrained(scheduler_path, local_files_only=True)
        # The endpoint ablation deliberately uses nonzero terminal SNR and
        # leading spacing. Validate against its own training settings.
        expected = settings.get("diffusion_schedule")
        if expected:
            for key in ("prediction_type", "rescale_betas_zero_snr", "timestep_spacing"):
                if key in expected and getattr(pipe.scheduler.config, key, None) != expected[key]:
                    raise ValueError(f"Checkpoint scheduler disagrees with training config: {key}")
        pipe.da2_conditioning_enabled = settings["da2_conditioning_enabled"]
        pipe.register_to_config(da2_conditioning_enabled=pipe.da2_conditioning_enabled)
    pipe.unet.eval()
    pipe.vae.eval()
    if pipe.da2 is not None:
        pipe.da2.eval()
