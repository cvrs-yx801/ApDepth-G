"""Paper Tables III/IV: train one control, evaluate it, then start the next."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import csv
import json
import logging
import os
import subprocess
import sys

import yaml

from .ablation_evaluation import REPO, checkpoint_record, file_hash, local_path, read_json, run_command, save_json

TABLES = {
    3: ["00_full", "02_no_vgc", "01_no_far_anchor", "09_no_region_anchor", "10_no_interior_smoothness",
        "08_no_da2_prior", "11_8ch_no_vgc", "12_8ch_vgc"],
    4: ["00_full", "03_no_terminal_snr", "N1_no_snr_floor", "N2_no_terminal_noise_fade",
        "04_no_min_snr", "05_no_multi_res_noise", "06_no_offset_noise", "07_no_latent_gradient"],
}
VALID_DATASETS = {"nyu": "config/dataset/data_nyu_test.yaml", "kitti": "config/dataset/data_kitti_eigen_test.yaml"}


def config_dict(path: Path, inherited: bool) -> dict[str, Any]:
    from omegaconf import OmegaConf
    from src.util.config_util import recursive_load_config
    cfg = recursive_load_config(str(path)) if inherited else OmegaConf.load(path)
    data = OmegaConf.to_container(cfg, resolve=True)
    data.pop("base_config", None)
    return data


def final_checkpoint(run: Path) -> Path:
    cfg = config_dict(run / "config.yaml", False)
    return run / "checkpoint" / f"iter_{int(cfg['max_iter']):06d}"


def ensure_training(experiment: str, args: argparse.Namespace, base: Path) -> tuple[Path, Path]:
    source = REPO / "config/ablation" / f"{experiment}.yaml"
    expected = config_dict(source, True)
    if hasattr(args, "training_configs") and expected != args.training_configs[experiment]:
        raise ValueError("Training config changed during the table run; use one fixed experiment matrix")
    run = local_path(args.full_run) if experiment == "00_full" and args.full_run else local_path(args.run_root) / experiment
    checkpoint = run / "checkpoint" / f"iter_{int(expected['max_iter']):06d}"
    if args.dry_run:
        if not args.eval_only and not (experiment == "00_full" and args.full_run):
            run_command([sys.executable, "-B", "train.py", "--config", str(source), "--base_data_dir", str(args.base_data_dir),
                         "--base_ckpt_dir", str(args.base_ckpt_dir), "--output_dir", str(local_path(args.run_root)), "--no_wandb"], True)
        print(f"Require final checkpoint: {checkpoint}; own unet/config.json + scheduler/scheduler_config.json", flush=True)
        return run, checkpoint
    if run.exists():
        saved_path = run / "config.yaml"
        if not saved_path.is_file() or config_dict(saved_path, False) != expected:
            raise ValueError(f"Run config differs from current experiment (including training budget): {run}. Use a matching run/new root.")
        checkpoint = final_checkpoint(run)
        if checkpoint.is_dir():
            checkpoint_record(checkpoint, include_weight_hash=False)
            return run, checkpoint
        if args.eval_only or experiment == "00_full" and args.full_run:
            raise FileNotFoundError(f"Final checkpoint not ready: {checkpoint}")
        latest = run / "checkpoint/latest"
        if not args.resume or not (latest / "trainer.ckpt").is_file():
            raise FileExistsError(f"Incomplete training run: {run}; use --resume with a valid latest/trainer.ckpt")
        command = [sys.executable, "-B", "train.py", "--resume_run", str(latest), "--base_data_dir", str(args.base_data_dir),
                   "--base_ckpt_dir", str(args.base_ckpt_dir), "--no_wandb"]
    else:
        if args.eval_only or experiment == "00_full" and args.full_run:
            raise FileNotFoundError(f"Training run missing: {run}")
        command = [sys.executable, "-B", "train.py", "--config", str(source), "--base_data_dir", str(args.base_data_dir),
                   "--base_ckpt_dir", str(args.base_ckpt_dir), "--output_dir", str(local_path(args.run_root)), "--no_wandb"]
    run_command(command)
    # train.py can return 0 after catching a training exception/time limit;
    # the saved max_iter checkpoint is therefore the real completion gate.
    if not (run / "config.yaml").is_file():
        raise FileNotFoundError(f"Training did not produce a saved run config: {run}")
    checkpoint = final_checkpoint(run)
    if not checkpoint.is_dir():
        raise RuntimeError(f"Training ended/paused without final checkpoint {checkpoint}; evaluation not started")
    checkpoint_record(checkpoint, include_weight_hash=False)
    if config_dict(run / "config.yaml", False) != expected:
        raise ValueError("Saved run configuration changed; do not label this as the requested control")
    return run, checkpoint


def valid_depth_evaluation(experiment: str, checkpoint: Path, base: Path, args: argparse.Namespace) -> None:
    record = None if args.dry_run else checkpoint_record(checkpoint)
    for dataset, config in VALID_DATASETS.items():
        folder = local_path(args.output_dir) / experiment / "valid_depth" / dataset
        marker = folder / "completed.json"
        if marker.exists() and args.resume and not args.dry_run:
            data = read_json(marker)
            if data["checkpoint"] != record:
                raise ValueError(f"Checkpoint changed since evaluation: {folder}")
            if file_hash(folder / "eval/metrics.json") != data["metrics_sha256"]:
                raise ValueError(f"Completed metrics changed: {folder}")
            continue
        if folder.exists() and not args.dry_run:
            raise FileExistsError(f"Existing or incomplete depth evaluation: {folder}; use new --output_dir")
        commands = [
            [sys.executable, "-B", "infer.py", "--checkpoint", str(base), "--unet_checkpoint", str(checkpoint),
             "--dataset_config", config, "--base_data_dir", str(args.base_data_dir), "--output_dir", str(folder / "prediction"),
             "--denoise_steps", str(args.denoise_steps), "--ensemble_size", str(args.ensemble_size),
             "--processing_res", str(args.processing_res), "--seed", str(args.seed)],
            [sys.executable, "-B", "eval.py", "--prediction_dir", str(folder / "prediction"), "--output_dir", str(folder / "eval"),
             "--dataset_config", config, "--base_data_dir", str(args.base_data_dir), "--alignment", "least_square"],
        ]
        for command in commands:
            run_command(command, args.dry_run)
        if not args.dry_run:
            save_json(marker, dict(checkpoint=record,
                                   metrics_sha256=file_hash(folder / "eval/metrics.json")))


def collect_table(table: int, ids: list[str], output: Path) -> None:
    """Only completed model runs are compared; all rows remain visible as pending."""
    from src.sky_eval.aggregate import compare
    models: dict[str, list[dict[str, Any]]] = {}
    reference_data = None
    reference_protocol = None
    for experiment in ids:
        folder = output / experiment / "sky"
        if (folder / "completed.json").is_file():
            for relative, expected in read_json(folder / "completed.json")["files"].items():
                if file_hash(folder / relative) != expected:
                    raise ValueError(f"Sky result changed: {folder / relative}")
            rows = read_json(folder / "scores/scoe/models" / experiment / "per_image.json")
            expected_datasets = {"diode", "kitti", "vkitti2"} if table == 3 else {"diode"}
            if {row["dataset"] for row in rows} != expected_datasets:
                raise ValueError("Completed sky result does not cover every dataset required by this table")
            data = {row["sample_id"]: {key: row[key] for key in ("dataset", "width", "height", "sky_ratio", "rgb_sha256", "mask_sha256")}
                    for row in rows}
            config = read_json(folder / "scores/scoe/config_resolved.json")
            if reference_data is not None and (data != reference_data or config["protocol"] != reference_protocol):
                raise ValueError("Table rows use different RGB/masks/protocols; do not compare different evaluation samples")
            reference_data, reference_protocol = data, config["protocol"]
            models[experiment] = rows
    sky = compare(models, True) if models else []
    lookup = {(r["model_id"], r["dataset"], r["metric"]): r for r in sky}
    rows: list[dict[str, Any]] = []
    for experiment in ids:
        row: dict[str, Any] = {"experiment": experiment, "status": "complete" if experiment in models else "pending"}
        datasets = ("diode", "kitti", "vkitti2") if table == 3 else ("diode",)
        metrics = ("sky_iqr", "sky_p95_p5", "sfoe") if table == 3 else ("sky_iqr", "sfoe")
        for dataset in datasets:
            for metric in metrics:
                item = lookup.get((experiment, dataset, metric), {})
                row[f"{dataset}_{metric}"] = item.get("mean")
                row[f"{dataset}_{metric}_N"] = item.get("N", 0)
        if table == 4:
            for dataset in VALID_DATASETS:
                folder = output / experiment / "valid_depth" / dataset
                if (folder / "completed.json").is_file():
                    saved = read_json(folder / "eval/metrics.json")
                    if file_hash(folder / "eval/metrics.json") != read_json(folder / "completed.json")["metrics_sha256"]:
                        raise ValueError("Valid-depth metrics changed")
                    row[f"{dataset}_AbsRel_x100"] = 100 * saved["metrics"]["abs_relative_difference"]
                    row[f"{dataset}_delta1_percent"] = 100 * saved["metrics"]["delta1_acc"]
                    row[f"{dataset}_N"] = saved["samples"]
                else:
                    row.update({f"{dataset}_AbsRel_x100": None, f"{dataset}_delta1_percent": None, f"{dataset}_N": 0})
                    row["status"] = "pending"
        rows.append(row)
    # These aggregate views are intentionally refreshed after each completed row.
    columns = list(rows[0])
    destination = output / f"table{table}_results.csv"
    temporary = destination.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination)
    lines = [f"# Table {table} results", "", f"Completed sky rows: {len(models)}/{len(ids)}. Pending entries are N/A.", "",
             "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        lines.append("| " + " | ".join("N/A" if row[key] is None else f"{row[key]:.6g}" if isinstance(row[key], float) else str(row[key])
                                      for key in columns) + " |")
    markdown = output / f"table{table}_results.md"
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(table: int, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"Train/evaluate each Table {table} control sequentially")
    parser.add_argument("--run_root", default="output/ablation")
    parser.add_argument("--output_dir", default=f"output/table{table}_eval")
    parser.add_argument("--base_data_dir", default=os.environ.get("BASE_DATA_DIR"))
    parser.add_argument("--base_ckpt_dir", default=os.environ.get("BASE_CKPT_DIR"))
    parser.add_argument("--base_checkpoint", default=os.environ.get("BASE_CHECKPOINT"))
    parser.add_argument("--full_run", help="Reuse matching existing full-model run directory, including saved config.yaml")
    parser.add_argument("--mask_run", default=os.environ.get("MASK_RUN", "output/sky_segmentation/mask2former_v1"))
    parser.add_argument("--sky_config", help="Generated config for the same frozen Mask2Former masks/splits")
    parser.add_argument("--ids", nargs="+", choices=TABLES[table], default=TABLES[table])
    parser.add_argument("--denoise_steps", type=int, default=50)
    parser.add_argument("--ensemble_size", type=int, default=5)
    parser.add_argument("--processing_res", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--resume", action="store_true", help="Resume matching training, skip only verified completed evaluations")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--collect_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv)
    if len(set(args.ids)) != len(args.ids):
        parser.error("Experiment IDs must be unique")
    if Path.cwd().resolve() != REPO:
        parser.error("Run from the repository root")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        output = local_path(args.output_dir)
        if args.collect_only:
            collect_table(table, args.ids, output)
            return 0
        if not args.base_data_dir or not args.base_ckpt_dir:
            parser.error("Set BASE_DATA_DIR and BASE_CKPT_DIR or pass their corresponding flags")
        args.base_data_dir, args.base_ckpt_dir = local_path(args.base_data_dir), local_path(args.base_ckpt_dir)
        full = config_dict(REPO / "config/ablation/00_full.yaml", True)
        args.training_configs = {name: config_dict(REPO / "config/ablation" / f"{name}.yaml", True) for name in args.ids}
        if any(config["max_iter"] != full["max_iter"] for config in args.training_configs.values()):
            raise ValueError("Every table row must use the same full-model training budget")
        training_base = local_path(args.base_ckpt_dir / full["model"]["pretrained_path"])
        base = local_path(args.base_checkpoint) if args.base_checkpoint else training_base
        if not args.eval_only and base != training_base:
            raise ValueError("Training reads BASE_CKPT_DIR/model.pretrained_path; --base_checkpoint must point to that same local base")
        sky_config = local_path(args.sky_config) if args.sky_config else local_path(args.mask_run) / "sky_eval.generated.yaml"
        datasets = ["diode", "kitti", "vkitti2"] if table == 3 else ["diode"]
        if not args.dry_run:
            if not sky_config.is_file() or not base.is_dir():
                raise FileNotFoundError("Local base checkpoint or generated sky config is missing; no downloads are performed")
            if not args.eval_only and int(read_json(base / "unet/config.json")["in_channels"]) != 4:
                raise ValueError("Fresh table controls require the original 4ch base U-Net; do not replace its config with a trained 8/12ch config")
            if table == 4:
                for filename in VALID_DATASETS.values():
                    dataset = yaml.safe_load(local_path(filename).read_text(encoding="utf-8"))
                    if not local_path(args.base_data_dir / dataset["dir"]).exists():
                        raise FileNotFoundError(f"Conventional benchmark data missing before training: {dataset['dir']}")
            template = yaml.safe_load(sky_config.read_text(encoding="utf-8"))
            from src.sky_eval.datasets import index_dataset
            for name in datasets:
                entries = template["datasets"][name]
                samples = index_dataset(name, entries)
                if name == "diode" and not any(s.rgb_path.startswith(("outdoor/", "outdoors/")) for s in samples):
                    raise ValueError("DIODE Outdoor images required before starting expensive training")
            if output.exists() and not args.resume:
                raise FileExistsError("Output exists; use --resume or a new --output_dir")
            output.mkdir(parents=True, exist_ok=True)
            request = dict(table=table, ids=args.ids, base=str(base), sky_config=str(sky_config),
                           run_root=str(local_path(args.run_root)), full_run=str(local_path(args.full_run)) if args.full_run else None,
                           training_configs=args.training_configs,
                           sky_config_sha256=file_hash(sky_config), base_data_dir=str(args.base_data_dir),
                           denoise_steps=args.denoise_steps, ensemble_size=args.ensemble_size,
                           processing_res=args.processing_res, seed=args.seed, full_training_budget=full["max_iter"])
            request["base_config_hashes"] = {str(base / name): file_hash(base / name) for name in
                                           ("model_index.json", "unet/config.json", "scheduler/scheduler_config.json", "vae/config.json", "text_encoder/config.json")
                                           if (base / name).is_file()}
            if table == 4:
                request["valid_depth_split_hashes"] = {}
                for filename in VALID_DATASETS.values():
                    path = local_path(filename)
                    dataset = yaml.safe_load(path.read_text(encoding="utf-8"))
                    split = local_path(dataset["filenames"])
                    request["valid_depth_split_hashes"].update({str(path): file_hash(path), str(split): file_hash(split)})
            if (output / "table_request.json").exists():
                if read_json(output / "table_request.json") != request:
                    raise ValueError("Table request changed; use a new evaluation output directory")
            else:
                save_json(output / "table_request.json", request)
        for experiment in args.ids:
            logging.info("Table %s: %s train -> evaluate", table, experiment)
            _, checkpoint = ensure_training(experiment, args, base)
            if table == 4:
                valid_depth_evaluation(experiment, checkpoint, base, args)
            command = ["bash", "script/eval/0_infer_sky.sh", "--unet_checkpoint", str(checkpoint), "--base_checkpoint", str(base),
                       "--base_data_dir", str(args.base_data_dir), "--sky_config", str(sky_config),
                       "--output_dir", str(output / experiment / "sky"), "--model_id", experiment,
                       "--datasets", *datasets, "--denoise_steps", str(args.denoise_steps),
                       "--ensemble_size", str(args.ensemble_size), "--processing_res", str(args.processing_res), "--seed", str(args.seed)]
            if args.resume:
                command.append("--resume")
            run_command(command, args.dry_run)
            if not args.dry_run:
                collect_table(table, args.ids, output)
        return 0
    except (OSError, ValueError, KeyError, TypeError, ImportError, subprocess.CalledProcessError) as exc:
        logging.error("%s", exc)
        return 2

