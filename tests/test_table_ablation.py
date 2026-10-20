"""Checkpoint/config/architecture regressions; fixtures never download weights."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from src.util.ablation_evaluation import checkpoint_record
from src.util.ablation_tables import TABLES
from src.util.conditioning import adapt_unet_input, unet_inputs
from src.util.model_overrides import checkpoint_pipeline_kwargs


class TableAblationTest(unittest.TestCase):
    def test_paper_row_membership(self) -> None:
        self.assertEqual(len(TABLES[3]), 8)
        self.assertEqual(len(TABLES[4]), 8)
        self.assertEqual(set(TABLES[3]) & set(TABLES[4]), {"00_full"})
        self.assertIn("11_8ch_no_vgc", TABLES[3])
        self.assertIn("12_8ch_vgc", TABLES[3])
        self.assertIn("N1_no_snr_floor", TABLES[4])
        self.assertIn("N2_no_terminal_noise_fade", TABLES[4])

    def test_true_8ch_initialization_and_input_order(self) -> None:
        class TinyUnet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv_in = nn.Conv2d(4, 2, 3, padding=1)
                self.config = SimpleNamespace(in_channels=4)

            def register_to_config(self, **kwargs: int) -> None:
                for key, value in kwargs.items():
                    setattr(self.config, key, value)

        for channels in (8, 12):
            unet = TinyUnet()
            original = unet.conv_in.weight.detach().clone()
            bias = unet.conv_in.bias.detach().clone()
            adapt_unet_input(unet, channels)
            torch.testing.assert_close(unet.conv_in.weight, original.repeat(1, channels // 4, 1, 1) / (channels // 4))
            torch.testing.assert_close(unet.conv_in.bias, bias)
            self.assertEqual(unet.config.in_channels, channels)
        rgb, prior, depth = [torch.full((1, 4, 2, 2), float(value)) for value in (1, 2, 3)]
        self.assertEqual(unet_inputs(rgb, None, depth, 8, False).shape[1], 8)
        torch.testing.assert_close(unet_inputs(rgb, None, depth, 8, False)[:, 4:], depth)
        torch.testing.assert_close(unet_inputs(rgb, prior, depth, 12, False)[:, 4:8], torch.zeros_like(prior))
        with self.assertRaises(ValueError):
            unet_inputs(rgb, prior, depth, 8, True)

    def make_checkpoint(self, root: Path, channels: int = 8) -> Path:
        checkpoint = root / "trained"
        (checkpoint / "unet").mkdir(parents=True)
        (checkpoint / "scheduler").mkdir()
        (checkpoint / "unet/config.json").write_text(json.dumps({"in_channels": channels, "out_channels": 4}))
        schedule = {"prediction_type": "v_prediction", "rescale_betas_zero_snr": True, "timestep_spacing": "trailing"}
        (checkpoint / "scheduler/scheduler_config.json").write_text(json.dumps(schedule))
        (checkpoint / "unet/diffusion_pytorch_model.bin").write_bytes(b"synthetic fingerprint fixture, not actual model weights")
        (checkpoint / "inference_settings.json").write_text(json.dumps({
            "unet_in_channels": channels, "da2_conditioning_enabled": channels == 12, "diffusion_schedule": schedule}))
        return checkpoint

    def test_missing_local_config_fails_before_inference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.make_checkpoint(Path(directory))
            self.assertEqual(checkpoint_record(checkpoint)["in_channels"], 8)
            (checkpoint / "scheduler/scheduler_config.json").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "checkpoint-local"):
                checkpoint_record(checkpoint)

    def test_components_read_own_configs_without_writing_base(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = self.make_checkpoint(root)
            base = root / "base"
            base.mkdir()
            sentinel = base / "config.json"
            sentinel.write_text('{"in_channels":4}')
            before = sentinel.read_bytes()
            unet, scheduler = MagicMock(), MagicMock()
            with patch("diffusers.UNet2DConditionModel.from_pretrained", return_value=unet) as load_unet, \
                    patch("diffusers.DDIMScheduler.from_pretrained", return_value=scheduler) as load_scheduler:
                kwargs = checkpoint_pipeline_kwargs(checkpoint, torch_dtype=torch.float32, load_components=True)
            self.assertIs(kwargs["unet"], unet)
            self.assertIs(kwargs["scheduler"], scheduler)
            self.assertFalse(kwargs["load_da2"])
            self.assertEqual(load_unet.call_args.args[0], checkpoint / "unet")
            self.assertEqual(load_scheduler.call_args.args[0], checkpoint / "scheduler")
            self.assertTrue(load_unet.call_args.kwargs["local_files_only"])
            self.assertEqual(sentinel.read_bytes(), before)

    def test_scheduler_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.make_checkpoint(Path(directory), channels=12)
            path = checkpoint / "scheduler/scheduler_config.json"
            schedule = json.loads(path.read_text())
            schedule["timestep_spacing"] = "leading"
            path.write_text(json.dumps(schedule))
            with self.assertRaisesRegex(ValueError, "timestep_spacing"):
                checkpoint_record(checkpoint)


if __name__ == "__main__":
    unittest.main()
