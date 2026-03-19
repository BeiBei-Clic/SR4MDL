import csv
import gzip
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

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from nd2py.utils import AttrDict, init_logger, seed_all
from sr4mdl.env import Tokenizer
from sr4mdl.model import MDLformer
from sr4mdl.search import MCTS4MDL
from sr4mdl.utils import R2_score, RMSE_score


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
args = parser.parse_args()

args.dataset = args.dataset.strip()
args.name = args.name or f"{args.dataset}_mcts4mdl"
args.load_model = os.path.abspath(args.load_model)
args.pmlb_root = os.path.abspath(args.pmlb_root)
args.run_dir = os.path.join(ROOT_DIR, "experiments", "pmlb", "results", args.name)
args.csv_path = os.path.join(ROOT_DIR, "experiments", "pmlb", "results", "pmlb_inference.csv")

os.makedirs(args.run_dir, exist_ok=True)
init_logger("sr4mdl", args.name, os.path.join(args.run_dir, "info.log"))
logger = logging.getLogger("sr4mdl.search")
logger.info(args)
seed_all(args.seed)


def handler(signum, frame):
    raise KeyboardInterrupt


signal.signal(signal.SIGINT, handler)
signal.signal(signal.SIGTERM, handler)


def read_dataset():
    dataset_path = Path(args.pmlb_root) / args.dataset / f"{args.dataset}.tsv.gz"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    with gzip.open(dataset_path, "rt", newline="") as f:
        df = pd.read_csv(f, sep="\t")

    if "target" not in df.columns:
        raise ValueError(f"`target` column not found in {dataset_path}")

    df = df.head(args.sample_num).copy()
    if df.empty:
        raise ValueError(f"Dataset is empty after sampling: {dataset_path}")

    df = df.apply(pd.to_numeric, errors="raise")
    X_df = df.drop(columns=["target"])
    y = df["target"].to_numpy(dtype=np.float64)
    X = {col: X_df[col].to_numpy(dtype=np.float64) for col in X_df.columns}
    return dataset_path, X, y, len(df)


def build_estimator():
    tokenizer = Tokenizer(-100, 100, 4, args.max_var)
    state_dict = torch.load(args.load_model, map_location=args.device, weights_only=False)
    model_args = AttrDict(
        dropout=0.1,
        d_model=512,
        d_input=64,
        d_output=512,
        n_TE_layers=8,
        max_len=50,
        max_param=5,
        max_var=args.max_var,
        uniform_sample_number=args.sample_num,
        device=args.device,
        use_SENet=True,
        use_old_model=args.use_old_model,
    )
    model = MDLformer(model_args, state_dict["xy_token_list"])
    model.load(state_dict["xy_encoder"], state_dict["xy_token_list"], strict=True)
    model.eval()

    return MCTS4MDL(
        tokenizer=tokenizer,
        model=model,
        n_iter=args.n_iter,
        keep_vars=args.keep_vars,
        normalize_y=args.normalize_y,
        normalize_all=args.normalize_all,
        remove_abnormal=args.remove_abnormal,
        binary=[nd2.Mul, nd2.Div, nd2.Add, nd2.Sub],
        unary=[
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
        ],
        leaf=[nd2.Number(1), nd2.Number(2), nd2.Number(np.pi)],
        log_per_sec=5,
        save_path=os.path.join(args.run_dir, "records.json"),
    )


def append_result(result):
    fieldnames = [
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
    os.makedirs(os.path.dirname(args.csv_path), exist_ok=True)
    write_header = not os.path.exists(args.csv_path) or os.path.getsize(args.csv_path) == 0
    with open(args.csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: result[key] for key in fieldnames})


def main():
    dataset_path, X, y, row_count = read_dataset()
    logger.note(f"dataset: {args.dataset}")
    logger.note(f"path: {dataset_path}")
    logger.note(f"rows: {row_count}, features: {len(X)}")
    logger.note(f"device: {args.device}")

    est = build_estimator()
    start_time = time.time()
    est.fit(X, y, use_tqdm=False)
    runtime = time.time() - start_time

    y_pred = est.predict(X)
    rmse = RMSE_score(y, y_pred)
    r2 = R2_score(y, y_pred)
    expression = str(est.eqtree)
    complexity = len(est.eqtree)
    logger.note(f"Result = {expression}, RMSE = {rmse:.4f}, R2 = {r2:.4f}")

    result = {
        "r2": r2,
        "rmse": rmse,
        "runtime": runtime,
        "complexity": complexity,
        "dataset": args.dataset,
        "rows": row_count,
        "device": args.device,
        "checkpoint": args.load_model,
        "n_iter": len(est.records),
        "sample_num": args.sample_num,
        "expression": expression,
    }
    append_result(result)
    with open(os.path.join(args.run_dir, "result.json"), "w") as f:
        json.dump(
            {
                "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "name": args.name,
                **result,
            },
            f,
            indent=4,
        )


if __name__ == "__main__":
    main()
