import csv
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from nd2py.utils import seed_all

from experiments.pmlb.pmlb_inference import (
    build_logger,
    configure_signals,
    load_dataset,
    load_model_bundle,
    parse_device,
    run_single_inference,
)


BATCH_FIELDS = [
    "dataset",
    "status",
    "n_features",
    "refinement_type",
    "r2",
    "rmse",
    "complexity",
    "seconds",
    "error",
    "expr",
]


def build_cli():
    parser = ArgumentParser()
    parser.add_argument("--datasets_dir", type=str, default="./pmlb/datasets")
    parser.add_argument("--model_path", type=str, default="./weights/checkpoint.pth")
    parser.add_argument("--output_csv", type=str, default="./experiments/pmlb/results/pmlb_batch_results.csv")
    parser.add_argument("--max_rows", type=int, default=200)
    parser.add_argument("--max_input_points", type=int, default=200)
    parser.add_argument("--n_trees_to_refine", type=int, default=1)
    parser.add_argument("--dataset_limit", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_var", type=int, default=10)
    parser.add_argument("--keep_vars", action="store_true")
    parser.add_argument("--normalize_y", action="store_true")
    parser.add_argument("--normalize_all", action="store_true")
    parser.add_argument("--remove_abnormal", action="store_true")
    parser.add_argument("--use_old_model", action="store_true")
    return parser


def list_datasets(datasets_dir):
    datasets = []
    for dataset_dir in sorted(Path(datasets_dir).iterdir()):
        if not dataset_dir.is_dir():
            continue
        metadata_path = dataset_dir / "metadata.yaml"
        data_path = dataset_dir / f"{dataset_dir.name}.tsv.gz"
        if metadata_path.exists() and data_path.exists():
            datasets.append(dataset_dir.name)
    return datasets


def empty_error_row(dataset, message):
    return {
        "dataset": dataset,
        "status": "error",
        "n_features": "",
        "refinement_type": "mcts4mdl",
        "r2": "",
        "rmse": "",
        "complexity": "",
        "seconds": "",
        "error": message,
        "expr": "",
    }


def write_row(writer, file_obj, row):
    writer.writerow({key: row[key] for key in BATCH_FIELDS})
    file_obj.flush()


def main():
    configure_signals()
    args = build_cli().parse_args()
    args.datasets_dir = os.path.abspath(args.datasets_dir)
    args.model_path = os.path.abspath(args.model_path)
    args.output_csv = os.path.abspath(args.output_csv)
    args.device = parse_device(args.device)

    logger = build_logger("pmlb_batch_inference")
    logger.info(args)
    logger.note(f"device: {args.device}")
    seed_all(args.seed)

    datasets = list_datasets(args.datasets_dir)
    if args.dataset_limit is not None:
        datasets = datasets[: args.dataset_limit]

    model_bundle = load_model_bundle(
        model_path=args.model_path,
        device=args.device,
        max_input_points=args.max_input_points,
        max_var=args.max_var,
        use_old_model=args.use_old_model,
    )

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BATCH_FIELDS)
        writer.writeheader()
        f.flush()

        for dataset in datasets:
            logger.note(f"running dataset: {dataset}")
            try:
                dataset_info = load_dataset(dataset, args.datasets_dir, args.max_rows)
                result = run_single_inference(
                    dataset_info=dataset_info,
                    model_bundle=model_bundle,
                    n_iter=args.n_iter,
                    max_input_points=args.max_input_points,
                    keep_vars=args.keep_vars,
                    normalize_y=args.normalize_y,
                    normalize_all=args.normalize_all,
                    remove_abnormal=args.remove_abnormal,
                )
                write_row(writer, f, result)
            except Exception as exc:
                logger.error(f"{dataset}: {exc}")
                write_row(writer, f, empty_error_row(dataset, str(exc)))


if __name__ == "__main__":
    main()
