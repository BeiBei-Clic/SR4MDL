import csv
import gzip
import hashlib
import json
import logging
import os
import signal
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import nd2py as nd2
import numpy as np
import pandas as pd
import torch
import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from nd2py.utils import AttrDict, init_logger, seed_all
from sr4mdl.env import Tokenizer
from sr4mdl.model import MDLformer
from sr4mdl.search import MCTS4MDL
from sr4mdl.utils import R2_score, RMSE_score


DEFAULT_BINARY = [nd2.Mul, nd2.Div, nd2.Add, nd2.Sub]
DEFAULT_UNARY = [
    nd2.Sqrt,
    nd2.Cos,
    nd2.Sin,
    nd2.Pow2,
    nd2.Pow3,
    nd2.Exp,
    nd2.Inv,
    nd2.Neg,
    nd2.Arcsin,
    nd2.Arccos,
    nd2.Cot,
    nd2.Log,
    nd2.Tanh,
]
DEFAULT_LEAF = [nd2.Number(1), nd2.Number(2), nd2.Number(np.pi)]
SINGLE_RESULT_FIELDS = [
    "r2",
    "rmse",
    "runtime",
    "complexity",
    "dataset",
    "rows",
    "device",
    "checkpoint",
    "n_iter",
    "sample_num",
    "expression",
]


def validate_noise_strength(noise_strength):
    if noise_strength < 0:
        raise ValueError(f"noise_strength must be non-negative, got {noise_strength}.")
    return float(noise_strength)


def format_noise_strength(noise_strength):
    return format(float(noise_strength), "g")


def parse_device(device):
    device = str(device).strip()
    if device == "cpu":
        raise ValueError("This pretrained inference path is GPU-only. Please use --device cuda or --device cuda:N.")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no GPU is available in the current environment.")
        return "cuda"
    if device.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"{device} was requested, but no GPU is available in the current environment.")
        index = int(device.split(":", 1)[1])
        count = torch.cuda.device_count()
        if index < 0 or index >= count:
            raise ValueError(f"Requested {device}, but only {count} CUDA device(s) are visible.")
        torch.cuda.set_device(index)
        return f"cuda:{index}"
    raise ValueError(f"Unsupported device: {device}. Use --device cuda or --device cuda:N.")


def configure_signals():
    def handler(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def build_logger(name, run_dir=None):
    if run_dir is not None:
        os.makedirs(run_dir, exist_ok=True)
        init_logger("sr4mdl", name, os.path.join(run_dir, "info.log"))
    else:
        init_logger("sr4mdl", name)
    return logging.getLogger("sr4mdl.search")


def load_metadata(datasets_dir, dataset):
    metadata_path = Path(datasets_dir) / dataset / "metadata.yaml"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    with open(metadata_path, "r") as f:
        metadata = yaml.safe_load(f)
    if metadata.get("task") != "regression":
        raise ValueError(f"Dataset {dataset} is not regression according to {metadata_path}")
    return metadata


def load_dataset(dataset, datasets_dir, max_rows):
    dataset_path = Path(datasets_dir) / dataset / f"{dataset}.tsv.gz"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    metadata = load_metadata(datasets_dir, dataset)
    with gzip.open(dataset_path, "rt", newline="") as f:
        df = pd.read_csv(f, sep="\t")

    if "target" not in df.columns:
        raise ValueError(f"`target` column not found in {dataset_path}")

    df = df.head(max_rows).copy()
    if df.empty:
        raise ValueError(f"Dataset is empty after sampling: {dataset_path}")

    df = df.apply(pd.to_numeric, errors="raise")
    X_df = df.drop(columns=["target"])
    y = df["target"].to_numpy(dtype=np.float64)
    X = {col: X_df[col].to_numpy(dtype=np.float64) for col in X_df.columns}
    return {
        "dataset": dataset,
        "dataset_path": str(dataset_path),
        "metadata": metadata,
        "X": X,
        "y": y,
        "rows": len(df),
        "n_features": len(X_df.columns),
    }


def apply_target_noise(dataset_info, noise_strength, noise_seed):
    noise_strength = validate_noise_strength(noise_strength)
    if noise_strength == 0:
        return dataset_info

    dataset_key = f"{dataset_info['dataset']}::{noise_seed}"
    seed_bytes = hashlib.sha256(dataset_key.encode("utf-8")).digest()[:8]
    rng = np.random.default_rng(int.from_bytes(seed_bytes, byteorder="big", signed=False))

    y = dataset_info["y"]
    sigma = noise_strength * float(np.std(y))
    noisy_y = y + rng.normal(loc=0.0, scale=sigma, size=y.shape)

    updated_info = dict(dataset_info)
    updated_info["y"] = noisy_y.astype(np.float64, copy=False)
    return updated_info


def load_model_bundle(model_path, device, max_input_points, max_var, use_old_model):
    tokenizer = Tokenizer(-100, 100, 4, max_var)
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model_args = AttrDict(
        dropout=0.1,
        d_model=512,
        d_input=64,
        d_output=512,
        n_TE_layers=8,
        max_len=50,
        max_param=5,
        max_var=max_var,
        uniform_sample_number=max_input_points,
        device=device,
        use_SENet=True,
        use_old_model=use_old_model,
    )
    model = MDLformer(model_args, state_dict["xy_token_list"])
    model.load(state_dict["xy_encoder"], state_dict["xy_token_list"], strict=True)
    model.eval()
    return {"tokenizer": tokenizer, "model": model}


def create_estimator(model_bundle, n_iter, max_input_points, keep_vars, normalize_y, normalize_all, remove_abnormal, save_path=None):
    return MCTS4MDL(
        tokenizer=model_bundle["tokenizer"],
        model=model_bundle["model"],
        n_iter=n_iter,
        sample_num=max_input_points,
        keep_vars=keep_vars,
        normalize_y=normalize_y,
        normalize_all=normalize_all,
        remove_abnormal=remove_abnormal,
        binary=DEFAULT_BINARY,
        unary=DEFAULT_UNARY,
        leaf=DEFAULT_LEAF,
        log_per_sec=5,
        save_path=save_path,
    )


def run_single_inference(
    dataset_info,
    model_bundle,
    n_iter,
    max_input_points,
    keep_vars,
    normalize_y,
    normalize_all,
    remove_abnormal,
    noise_strength=0.0,
    noise_seed=0,
    logger=None,
    save_path=None,
):
    dataset_info = apply_target_noise(
        dataset_info=dataset_info,
        noise_strength=noise_strength,
        noise_seed=noise_seed,
    )
    est = create_estimator(
        model_bundle=model_bundle,
        n_iter=n_iter,
        max_input_points=max_input_points,
        keep_vars=keep_vars,
        normalize_y=normalize_y,
        normalize_all=normalize_all,
        remove_abnormal=remove_abnormal,
        save_path=save_path,
    )
    if logger is not None:
        logger.note(f"dataset: {dataset_info['dataset']}")
        logger.note(f"path: {dataset_info['dataset_path']}")
        logger.note(f"rows: {dataset_info['rows']}, features: {dataset_info['n_features']}")
        logger.note(f"noise_strength: {format_noise_strength(noise_strength)}")

    start_time = time.time()
    est.fit(dataset_info["X"], dataset_info["y"], use_tqdm=False)
    seconds = time.time() - start_time

    y_pred = est.predict(dataset_info["X"])
    rmse = RMSE_score(dataset_info["y"], y_pred)
    r2 = R2_score(dataset_info["y"], y_pred)
    expr = str(est.eqtree)
    complexity = len(est.eqtree)

    if logger is not None:
        logger.note(f"Result = {expr}, RMSE = {rmse:.4f}, R2 = {r2:.4f}")

    return {
        "dataset": dataset_info["dataset"],
        "status": "success",
        "n_features": dataset_info["n_features"],
        "refinement_type": "mcts4mdl",
        "r2": r2,
        "rmse": rmse,
        "complexity": complexity,
        "seconds": seconds,
        "error": "",
        "expr": expr,
        "rows": dataset_info["rows"],
        "n_iter": len(est.records),
    }


def append_single_result(csv_path, result):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SINGLE_RESULT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({key: result[key] for key in SINGLE_RESULT_FIELDS})


def build_single_cli():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--sample_num", type=int, default=200)
    parser.add_argument("--n_iter", type=int, default=1000)
    parser.add_argument("--max_var", type=int, default=10)
    parser.add_argument("--load_model", type=str, default="./weights/checkpoint.pth")
    parser.add_argument("--pmlb_root", type=str, default="./pmlb/datasets")
    parser.add_argument("--keep_vars", action="store_true")
    parser.add_argument("--normalize_y", action="store_true")
    parser.add_argument("--normalize_all", action="store_true")
    parser.add_argument("--remove_abnormal", action="store_true")
    parser.add_argument("--use_old_model", action="store_true")
    return parser


def main():
    configure_signals()
    args = build_single_cli().parse_args()
    args.dataset = args.dataset.strip()
    args.name = args.name or f"{args.dataset}_mcts4mdl"
    args.load_model = os.path.abspath(args.load_model)
    args.pmlb_root = os.path.abspath(args.pmlb_root)
    args.device = parse_device(args.device)
    args.run_dir = os.path.join(ROOT_DIR, "experiments", "pmlb", "results", args.name)
    args.csv_path = os.path.join(ROOT_DIR, "experiments", "pmlb", "results", "pmlb_inference.csv")

    logger = build_logger(args.name, args.run_dir)
    logger.info(args)
    seed_all(args.seed)
    logger.note(f"device: {args.device}")

    dataset_info = load_dataset(args.dataset, args.pmlb_root, args.sample_num)
    model_bundle = load_model_bundle(
        model_path=args.load_model,
        device=args.device,
        max_input_points=args.sample_num,
        max_var=args.max_var,
        use_old_model=args.use_old_model,
    )
    result = run_single_inference(
        dataset_info=dataset_info,
        model_bundle=model_bundle,
        n_iter=args.n_iter,
        max_input_points=args.sample_num,
        keep_vars=args.keep_vars,
        normalize_y=args.normalize_y,
        normalize_all=args.normalize_all,
        remove_abnormal=args.remove_abnormal,
        noise_strength=0.0,
        noise_seed=0,
        logger=logger,
        save_path=os.path.join(args.run_dir, "records.json"),
    )

    single_row = {
        "r2": result["r2"],
        "rmse": result["rmse"],
        "runtime": result["seconds"],
        "complexity": result["complexity"],
        "dataset": result["dataset"],
        "rows": result["rows"],
        "device": args.device,
        "checkpoint": args.load_model,
        "n_iter": result["n_iter"],
        "sample_num": args.sample_num,
        "expression": result["expr"],
    }
    append_single_result(args.csv_path, single_row)

    with open(os.path.join(args.run_dir, "result.json"), "w") as f:
        json.dump(
            {
                "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "name": args.name,
                **single_row,
            },
            f,
            indent=4,
        )


if __name__ == "__main__":
    main()
