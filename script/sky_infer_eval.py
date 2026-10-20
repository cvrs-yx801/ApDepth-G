"""Run checkpoint-local full-frame sky inference AND SCOE evaluation."""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_data_dir", default=os.environ.get("BASE_DATA_DIR", "/root/Dataset"))
    parser.add_argument("--mask_run", default=os.environ.get("MASK_RUN", "output/sky_segmentation/mask2former_v1"))
    parser.add_argument("--sky_config", help="Existing generated sky_eval YAML; overrides mask_run's config")
    parser.add_argument("--base_checkpoint", default=os.environ.get("BASE_CHECKPOINT"))
    parser.add_argument("--unet_checkpoint", default=os.environ.get("UNET_CHECKPOINT"))
    parser.add_argument("--output_dir", default=os.environ.get("SKY_OUTPUT_DIR", "output/sky_eval/celeste_full"))
    parser.add_argument("--model_id", default=os.environ.get("MODEL_ID", "celeste_full"))
    parser.add_argument("--datasets", nargs="+", default=["vkitti2", "kitti", "diode"])
    parser.add_argument("--denoise_steps", type=int, default=50)
    parser.add_argument("--ensemble_size", type=int, default=5)
    parser.add_argument("--processing_res", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv)
    if not args.base_checkpoint or not args.unet_checkpoint:
        parser.error("Provide --base_checkpoint and --unet_checkpoint (or BASE_CHECKPOINT/UNET_CHECKPOINT)")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        from src.util.ablation_evaluation import local_path, sky_infer_eval
        config = local_path(args.sky_config) if args.sky_config else local_path(args.mask_run) / "sky_eval.generated.yaml"
        score_run = sky_infer_eval(checkpoint=local_path(args.unet_checkpoint), base=local_path(args.base_checkpoint),
                                  base_data_dir=local_path(args.base_data_dir), sky_config=config,
                                  output=local_path(args.output_dir), model_id=args.model_id, datasets=args.datasets,
                                  steps=args.denoise_steps, ensemble=args.ensemble_size, resolution=args.processing_res,
                                  seed=args.seed, resume=args.resume, dry_run=args.dry_run)
        logging.info("Sky results: %s", score_run)
        return 0
    except (OSError, ValueError, KeyError, TypeError, ImportError, subprocess.CalledProcessError) as exc:
        logging.error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
