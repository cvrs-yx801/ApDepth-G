"""Checkpoint-local, offline sky inference and evaluation orchestration."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
import hashlib
import json
import logging
import os
import re
import shlex
import subprocess
import sys

import yaml

REPO = Path(__file__).resolve().parents[2]
DATASETS = {"vkitti2": "config/dataset/data_vkitti_val.yaml",
            "kitti": "config/dataset/data_kitti_eigen_test.yaml",
            "diode": "config/dataset/data_diode_all.yaml"}


def local_path(value: str | Path) -> Path:
    path = Path(value)
    path = (REPO / path).resolve() if not path.is_absolute() else path.resolve()
    if any(part.casefold() == "apdepth-g-paper" for part in path.parts):
        raise ValueError("Automation writes/reads must stay outside ApDepth-G-paper")
    return path


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def checkpoint_record(checkpoint: Path, include_weight_hash: bool = True) -> dict[str, Any]:
    """Never substitute a base-model config for a missing training artifact."""
    paths = {"unet_config": checkpoint / "unet/config.json",
             "scheduler_config": checkpoint / "scheduler/scheduler_config.json"}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required checkpoint-local configuration is missing: {path}")
    unet, scheduler = read_json(paths["unet_config"]), read_json(paths["scheduler_config"])
    channels = int(unet["in_channels"])
    if channels not in (8, 12) or int(unet.get("out_channels", 4)) != 4:
        raise ValueError(f"Unsupported depth U-Net architecture: {checkpoint}")
    weights = sorted((checkpoint / "unet").glob("*.safetensors")) + sorted((checkpoint / "unet").glob("*.bin"))
    if not weights:
        raise FileNotFoundError(f"Missing trained U-Net weights: {checkpoint / 'unet'}")
    metadata_path = checkpoint / "inference_settings.json"
    if metadata_path.is_file():
        settings = read_json(metadata_path)
        paths["inference_settings"] = metadata_path
    else:
        saved_config = next((p for p in (checkpoint / "config.yaml", checkpoint.parent.parent / "config.yaml") if p.is_file()), None)
        if saved_config is None:
            raise FileNotFoundError(f"Missing inference_settings.json and saved run config: {checkpoint}")
        cfg = yaml.safe_load(saved_config.read_text(encoding="utf-8"))
        settings = {"unet_in_channels": cfg.get("unet_in_channels", 12),
                    "da2_conditioning_enabled": cfg.get("da2_conditioning_enabled", True),
                    "diffusion_schedule": cfg.get("diffusion_schedule", {})}
        paths["saved_run_config"] = saved_config
    conditioning = settings.get("da2_conditioning_enabled")
    if not isinstance(conditioning, bool) or channels == 8 and conditioning:
        raise ValueError("Checkpoint architecture and DA2-conditioning setting disagree")
    if int(settings.get("unet_in_channels", channels)) != channels:
        raise ValueError("inference_settings and unet/config.json disagree")
    for key in ("prediction_type", "rescale_betas_zero_snr", "timestep_spacing"):
        expected = settings.get("diffusion_schedule", {}).get(key)
        if expected is not None and scheduler.get(key) != expected:
            raise ValueError(f"Checkpoint scheduler does not match training: {key}")
    if include_weight_hash:
        paths.update({f"weight:{path.name}": path for path in weights})
    return dict(checkpoint=str(checkpoint), in_channels=channels, da2_conditioning_enabled=conditioning,
                scheduler=scheduler, files={key: {"path": str(path), "sha256": file_hash(path)} for key, path in paths.items()})


def run_command(command: list[str], dry_run: bool = False) -> None:
    print(shlex.join(command), flush=True)
    if not dry_run:
        environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1",
                           PYTHON_BIN=sys.executable)
        subprocess.run(command, cwd=REPO, env=environment, check=True)


def model_metadata(model_id: str, checkpoint: Path, base: Path, predictions: Path,
                   record: dict[str, Any], steps: int, ensemble: int, resolution: int, seed: int) -> dict[str, Any]:
    return dict(root=str(predictions), layout="infer_split", checkpoint=str(checkpoint),
                inference_command=f"script/eval/0_infer_sky.sh --unet_checkpoint {checkpoint} --base_checkpoint {base}",
                git_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                                          text=True, check=False).stdout.strip() or "unavailable",
                seed=seed, resolution=resolution, denoise_steps=steps, ensemble_size=ensemble,
                scheduler=record["files"]["scheduler_config"], inference_config=record,
                native_postprocessing="MarigoldPipeline decode clip [-1,1] then [0,1]; no GT alignment or extra sky clipping",
                representation="relative_depth", distance_direction="larger_is_farther",
                semantics_source="Local MarigoldPipeline.single_infer and scale_shift_depth normalizer: larger=farther",
                amplitude_comparability_verified=True, geometry="full_frame", resampling="reject",
                squeeze_leading_channel=False)


def ensure_text(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Frozen derived file changed: {path}")
    else:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(content)


def outdoor_subset(settings: dict[str, Any], samples: list[Any], folder: Path) -> dict[str, Any]:
    """Reuse existing sky masks while restricting mixed DIODE val to Outdoor."""
    outdoor = [s for s in samples if s.rgb_path.replace("\\", "/").startswith(("outdoor/", "outdoors/"))]
    if not outdoor:
        raise ValueError("The supplied DIODE mask run contains no Outdoor images")
    if len(outdoor) == len(samples):
        return settings
    folder.mkdir(parents=True, exist_ok=True)
    original_split = local_path(settings["data_split"])
    source_rows = {line.split()[0].replace("\\", "/").removeprefix("./"): line
                   for line in original_split.read_text(encoding="utf-8-sig").splitlines()
                   if line.strip() and not line.lstrip().startswith("#")}
    split, manifest, report_path = folder / "files.txt", folder / "manifest.jsonl", folder / "segmentation_report.json"
    ensure_text(split, "\n".join(source_rows[s.rgb_path] for s in outdoor) + "\n")
    ensure_text(manifest, "".join(json.dumps(s.to_dict(), ensure_ascii=False) + "\n" for s in outdoor))
    report = read_json(local_path(settings["segmentation_report"]))
    report.update(total_split=len(outdoor), requested=len(outdoor), succeeded=len(outdoor), failed=0,
                  zero_sky=sum(s.sky_ratio == 0 for s in outdoor),
                  at_least_five_percent_sky=sum(s.sky_ratio >= 0.05 for s in outdoor),
                  split_list=str(split), split_list_sha256=file_hash(split), manifest_sha256=file_hash(manifest),
                  selection_rule="RGB path outdoor/ or outdoors/; subset of existing frozen segmentation, no new masks",
                  source_inputs={str(local_path(settings[key])): file_hash(local_path(settings[key]))
                                 for key in ("manifest", "segmentation_report", "data_split")})
    ensure_text(report_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    derived = deepcopy(settings)
    derived.update(data_split=str(split), manifest=str(manifest), segmentation_report=str(report_path),
                   split_provenance=settings.get("split_provenance", "") + "; deterministic DIODE Outdoor subset")
    logging.info("DIODE: reused existing masks; selected %d Outdoor images from %d", len(outdoor), len(samples))
    return derived


def sky_infer_eval(*, checkpoint: Path, base: Path, base_data_dir: Path, sky_config: Path,
                   output: Path, model_id: str, datasets: list[str], steps: int, ensemble: int,
                   resolution: int, seed: int, resume: bool = False, dry_run: bool = False) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", model_id) or model_id in (".", ".."):
        raise ValueError("Invalid model_id")
    if not datasets or len(datasets) != len(set(datasets)) or set(datasets) - set(DATASETS):
        raise ValueError("Use distinct datasets from diode, kitti, vkitti2")
    if steps < 1 or ensemble < 1 or resolution < 0:
        raise ValueError("Invalid inference settings")
    for dataset in datasets:
        command = [sys.executable, "-B", "infer.py", "--checkpoint", str(base), "--unet_checkpoint", str(checkpoint),
                   "--base_data_dir", str(base_data_dir), "--dataset_config", str(output / "configs" / f"{dataset}.yaml"),
                   "--output_dir", str(output / "predictions" / dataset), "--denoise_steps", str(steps),
                   "--ensemble_size", str(ensemble), "--processing_res", str(resolution), "--seed", str(seed)]
        if dry_run:
            run_command(command, True)
    if dry_run:
        print(f"Then: SCOE audit -> prepare -> evaluate({model_id}) -> report; masks/splits from {sky_config}", flush=True)
        return output / "scores/scoe"
    if not base.is_dir() or not base_data_dir.is_dir():
        raise FileNotFoundError("Provide existing local base checkpoint and BASE_DATA_DIR directories")
    template = yaml.safe_load(sky_config.read_text(encoding="utf-8-sig"))
    if set(datasets) - set(template.get("datasets", {})):
        raise ValueError(f"Generated sky config does not contain all requested datasets: {datasets}")
    record = checkpoint_record(checkpoint)
    request = dict(model_id=model_id, checkpoint=record, base_checkpoint=str(base), base_data_dir=str(base_data_dir),
                   sky_config=str(sky_config), sky_config_sha256=file_hash(sky_config), datasets=datasets,
                   denoise_steps=steps, ensemble_size=ensemble, processing_res=resolution, seed=seed,
                   orchestrator_sha256=file_hash(Path(__file__)))
    request["base_config_hashes"] = {str(base / name): file_hash(base / name) for name in
                                   ("unet/config.json", "scheduler/scheduler_config.json") if (base / name).is_file()}
    request["inference_source_hashes"] = {name: file_hash(REPO / name) for name in
                                        ("infer.py", "marigold/marigold_pipeline.py", "src/util/model_overrides.py", "src/util/conditioning.py")}
    if output.exists():
        if not resume or read_json(output / "request.json") != request:
            raise FileExistsError(f"Output exists or request changed; use a new output directory: {output}")
        if (output / "completed.json").is_file():
            from src.sky_eval.workflow import load_config, verify_lock
            cfg, _, score_run = load_config(output / "sky_eval.yaml", None)
            verify_lock(cfg, score_run)
            from src.sky_eval.workflow import read_manifest
            from src.sky_eval.datasets import inspect_sample
            for sample in read_manifest(score_run / "manifest.jsonl"):
                actual, _ = inspect_sample(sample, cfg["datasets"][sample.dataset])
                if actual != sample:
                    raise ValueError("Dataset changed after the completed sky evaluation")
            for relative, expected in read_json(output / "completed.json")["files"].items():
                if file_hash(local_path(output / relative)) != expected:
                    raise ValueError(f"Completed result changed: {relative}")
            return score_run
    else:
        output.mkdir(parents=True)
        save_json(output / "request.json", request)
    # Validate the original split and generated mask manifest before any GPU work.
    from src.sky_eval.datasets import index_dataset
    from src.sky_eval.workflow import audit, evaluate, load_config, prepare, read_json as read_scoe_json, report
    selected_datasets = {name: deepcopy(template["datasets"][name]) for name in datasets}
    for settings in selected_datasets.values():
        settings["enabled"] = True
    folder = output / "configs"
    folder.mkdir(exist_ok=True)
    for name, settings in selected_datasets.items():
        for key in ("root", "rgb_root", "mask_root", "class_seg_root", "manifest", "segmentation_report", "data_split", "train_list", "test_list"):
            if settings.get(key):
                settings[key] = str(local_path(settings[key]))
        if settings.get("mask_format") != "binary_manifest":
            raise ValueError("Use the Mask2Former-generated binary_manifest configuration")
        samples = index_dataset(name, settings)
        if name == "diode":
            settings = outdoor_subset(settings, samples, folder / "diode_outdoor")
            selected_datasets[name] = settings
            samples = index_dataset(name, settings)
        split = local_path(settings["data_split"])
        source_rows = {line.split()[0].replace("\\", "/"): line for line in split.read_text(encoding="utf-8-sig").splitlines()
                       if line.strip() and not line.lstrip().startswith("#")}
        lines = "\n".join(source_rows[sample.rgb_path] for sample in samples) + "\n"
        filenames = folder / f"{name}_files.txt"
        native = yaml.safe_load(local_path(DATASETS[name]).read_text(encoding="utf-8"))
        native.update(dir=settings["rgb_root"], filenames=str(filenames), disp_name=f"{name}_sky_full_frame")
        if name in ("vkitti2", "kitti"):
            native.update(kitti_bm_crop=False, valid_mask_crop=None)
        destination = folder / f"{name}.yaml"
        for path, content in ((filenames, lines), (destination, yaml.safe_dump(native, sort_keys=False))):
            if path.exists():
                if path.read_text(encoding="utf-8") != content:
                    raise ValueError(f"Existing inference config changed: {path}")
            else:
                with path.open("x", encoding="utf-8") as stream:
                    stream.write(content)
    cfg_path = output / "sky_eval.yaml"
    cfg = dict(protocol=deepcopy(template.get("protocol", {})), datasets=selected_datasets,
               models={model_id: model_metadata(model_id, checkpoint, base, output / "predictions", record, steps, ensemble, resolution, seed)},
               cross_dataset_duplicate_reviewed=template.get("cross_dataset_duplicate_reviewed", False),
               output=dict(root=str(output / "scores"), run_id="scoe"))
    if cfg_path.exists():
        existing = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        # Commit metadata must remain the original request when resuming.
        cfg["models"][model_id]["git_commit"] = existing["models"][model_id]["git_commit"]
        if existing != cfg:
            raise ValueError("Sky evaluation configuration changed; choose a new output directory")
    else:
        with cfg_path.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(cfg, stream, sort_keys=False, allow_unicode=True)
    resolved, protocol, score_run = load_config(cfg_path, None)
    if not score_run.exists():
        audit(resolved, protocol, score_run, None)
    from src.sky_eval.workflow import verify_config
    verify_config(resolved, score_run)
    audit_result = read_scoe_json(score_run / "dataset_audit.json")
    for name, data in audit_result["datasets"].items():
        if data["failures"] or not data["selected"]:
            raise ValueError(f"Sky audit failed: {name}; see {score_run / 'dataset_audit.md'}")
        if name == "vkitti2" and "split_audit" not in data:
            raise ValueError("VKITTI training/test split audit is missing")
    if audit_result["duplicate_rgb"]:
        raise ValueError("Sky audit found duplicate RGB; review before comparison")
    if not (score_run / "protocol_lock.json").exists():
        if (score_run / "sample_pairs").exists():
            raise ValueError("Incomplete prepare stage; keep its diagnostics and use a new output directory")
        prepare(resolved, protocol, score_run)
    from src.sky_eval.workflow import verify_lock
    verify_lock(resolved, score_run)
    for name in datasets:
        marker = output / f"inference_complete_{name}.json"
        prediction = output / "predictions" / name
        if marker.exists():
            for relative, expected in read_json(marker)["files"].items():
                if file_hash(prediction / relative) != expected:
                    raise ValueError(f"Prediction changed: {prediction / relative}")
            continue
        if prediction.exists():
            raise FileExistsError(f"Incomplete prediction directory; use a new output directory: {prediction}")
        command = [sys.executable, "-B", "infer.py", "--checkpoint", str(base), "--unet_checkpoint", str(checkpoint),
                   "--base_data_dir", str(base_data_dir), "--dataset_config", str(folder / f"{name}.yaml"),
                   "--output_dir", str(prediction), "--denoise_steps", str(steps), "--ensemble_size", str(ensemble),
                   "--processing_res", str(resolution), "--seed", str(seed)]
        run_command(command)
        files = sorted(prediction.rglob("*.npy"))
        if not files:
            raise ValueError(f"No raw predictions produced: {name}")
        save_json(marker, {"files": {p.relative_to(prediction).as_posix(): file_hash(p) for p in files}})
    model_folder = score_run / "models" / model_id
    if not (model_folder / "result_lock.json").exists():
        if model_folder.exists():
            raise ValueError("Incomplete metric evaluation; use a new output directory")
        result_code = evaluate(resolved, protocol, score_run, model_id)
    else:
        result_code = int(any(row["status"] != "ok" for row in read_json(model_folder / "per_image.json"))) * 2
    diagnostics = score_run / "diagnostic_report"
    if not diagnostics.exists():
        report(resolved, protocol, score_run, False)
    rows = read_json(model_folder / "per_image.json")
    expected_mask_exclusions = {"insufficient_boundary_pairs", "insufficient_sky_interior"}
    severe = [row for row in rows if row.get("prediction_sha256") is None
              or set(filter(None, row.get("failure_reason", "").split(";"))) - expected_mask_exclusions]
    if result_code and severe:
        raise ValueError(f"Some sky metrics failed; inspect {model_folder / 'per_image.csv'}. No completed table row published.")
    for path, expected in request["base_config_hashes"].items():
        if file_hash(local_path(path)) != expected:
            raise ValueError("BASE_CKPT_DIR configuration changed during evaluation")
    required = [model_folder / name for name in ("per_image.json", "per_image.csv", "metrics.json", "result_lock.json")]
    required += [diagnostics / "comparison.json", diagnostics / "comparison.md"]
    save_json(output / "completed.json", {"mask_exclusions": sum(row["status"] != "ok" for row in rows),
                                          "files": {p.relative_to(output).as_posix(): file_hash(p) for p in required}})
    return score_run
