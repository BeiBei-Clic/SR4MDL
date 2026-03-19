import csv
import math
import os
import re
import sys
from argparse import ArgumentParser
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


SUMMARY_FIELDS = [
    "noise_strength",
    "group",
    "r2_mean",
    "r2_var",
    "r2_valid_count",
    "total_count",
    "recovery_rate",
    "complexity_mean",
    "complexity_var",
    "complexity_count",
    "seconds_mean",
    "seconds_var",
    "seconds_count",
]

GROUPS = ["Feynman", "Strogatz", "Black-box"]
REQUIRED_COLUMNS = {"dataset", "status", "r2", "complexity", "seconds"}
NOISE_PATTERN = re.compile(r"^pmlb_batch_inference_noise_([0-9.]+)\.csv$")


def build_cli():
    parser = ArgumentParser()
    parser.add_argument("--input_csvs", nargs="*", default=None)
    parser.add_argument(
        "--output_csv",
        type=str,
        default=str(ROOT_DIR / "experiments" / "pmlb" / "results" / "pmlb_results_summary.csv"),
    )
    return parser


def format_number(value):
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{value:.6f}"


def parse_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def normalize_r2(value):
    parsed = parse_float(value)
    if parsed is None or parsed < 0:
        return 0.0, False
    return parsed, True


def classify_group(dataset):
    if dataset.startswith("feynman_"):
        return "Feynman"
    if dataset.startswith("strogatz_"):
        return "Strogatz"
    return "Black-box"


def population_variance(values):
    if not values:
        return None
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def default_input_csvs():
    results_dir = ROOT_DIR / "experiments" / "pmlb" / "results"
    inputs = []
    zero_noise = results_dir / "pmlb_batch_inference_noise_0.csv"
    legacy_zero_noise = results_dir / "pmlb_results.csv"
    if zero_noise.exists():
        inputs.append(zero_noise)
    elif legacy_zero_noise.exists():
        inputs.append(legacy_zero_noise)

    for path in sorted(results_dir.glob("pmlb_batch_inference_noise_*.csv")):
        if path.name == "pmlb_batch_inference_noise_0.csv":
            continue
        inputs.append(path)

    if not inputs:
        raise FileNotFoundError("未找到可用的 PMLB 结果 CSV。")
    return [str(path) for path in inputs]


def infer_noise_strength(path, fieldnames, rows):
    match = NOISE_PATTERN.match(path.name)
    if match:
        return float(match.group(1))
    if path.name == "pmlb_results.csv":
        return 0.0
    if "noise_strength" in fieldnames:
        values = {row.get("noise_strength", "").strip() for row in rows if row.get("noise_strength", "").strip()}
        if len(values) == 1:
            parsed = parse_float(values.pop())
            if parsed is not None:
                return parsed
    raise ValueError(f"无法从文件名推断噪声强度: {path}")


def read_rows(csv_path):
    path = Path(csv_path)
    with path.open(newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        if reader.fieldnames is None:
            raise ValueError(f"CSV 缺少表头: {path}")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"CSV 缺少必要列 {missing_text}: {path}")
        rows = list(reader)
    noise_strength = infer_noise_strength(path, reader.fieldnames, rows)
    return noise_strength, rows


def build_empty_group_stats():
    return {
        "total_count": 0,
        "r2_values": [],
        "r2_valid_count": 0,
        "recovery_count": 0,
        "complexity_values": [],
        "seconds_values": [],
    }


def is_ok_status(status):
    return str(status).strip().lower() in {"ok", "success"}


def aggregate_group(rows):
    stats_by_group = {group: build_empty_group_stats() for group in GROUPS}

    for row in rows:
        group = classify_group(row["dataset"])
        group_stats = stats_by_group[group]
        group_stats["total_count"] += 1

        r2_value, is_valid_r2 = normalize_r2(row.get("r2"))
        group_stats["r2_values"].append(r2_value)
        if is_valid_r2:
            group_stats["r2_valid_count"] += 1
        if r2_value > 0.9:
            group_stats["recovery_count"] += 1

        if is_ok_status(row.get("status")):
            complexity_value = parse_float(row.get("complexity"))
            if complexity_value is not None:
                group_stats["complexity_values"].append(complexity_value)
            seconds_value = parse_float(row.get("seconds"))
            if seconds_value is not None:
                group_stats["seconds_values"].append(seconds_value)

    return stats_by_group


def summarize_noise_level(noise_strength, rows):
    stats_by_group = aggregate_group(rows)
    summary_rows = []

    for group in GROUPS:
        stats = stats_by_group[group]
        total_count = stats["total_count"]
        r2_values = stats["r2_values"]
        complexity_values = stats["complexity_values"]
        seconds_values = stats["seconds_values"]
        summary_rows.append(
            {
                "noise_strength": format_number(noise_strength),
                "group": group,
                "r2_mean": format_number(sum(r2_values) / total_count if total_count else None),
                "r2_var": format_number(population_variance(r2_values)),
                "r2_valid_count": str(stats["r2_valid_count"]),
                "total_count": str(total_count),
                "recovery_rate": format_number(stats["recovery_count"] / total_count if total_count else None),
                "complexity_mean": format_number(
                    sum(complexity_values) / len(complexity_values) if complexity_values else None
                ),
                "complexity_var": format_number(population_variance(complexity_values)),
                "complexity_count": str(len(complexity_values)),
                "seconds_mean": format_number(sum(seconds_values) / len(seconds_values) if seconds_values else None),
                "seconds_var": format_number(population_variance(seconds_values)),
                "seconds_count": str(len(seconds_values)),
            }
        )
    return summary_rows


def write_summary(output_csv, summary_rows):
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)


def print_table(summary_rows):
    widths = {field: len(field) for field in SUMMARY_FIELDS}
    for row in summary_rows:
        for field in SUMMARY_FIELDS:
            widths[field] = max(widths[field], len(row[field]))

    def render_row(row):
        return " | ".join(row[field].ljust(widths[field]) for field in SUMMARY_FIELDS)

    separator = "-+-".join("-" * widths[field] for field in SUMMARY_FIELDS)
    header = {field: field for field in SUMMARY_FIELDS}
    print(render_row(header))
    print(separator)
    for row in summary_rows:
        print(render_row(row))


def main():
    args = build_cli().parse_args()
    input_csvs = args.input_csvs or default_input_csvs()

    summary_rows = []
    seen_noise_strengths = set()
    for csv_path in input_csvs:
        noise_strength, rows = read_rows(csv_path)
        if noise_strength in seen_noise_strengths:
            raise ValueError(f"噪声强度重复: {noise_strength}")
        seen_noise_strengths.add(noise_strength)
        summary_rows.extend(summarize_noise_level(noise_strength, rows))

    summary_rows.sort(key=lambda row: (float(row["noise_strength"]), GROUPS.index(row["group"])))
    write_summary(args.output_csv, summary_rows)
    print_table(summary_rows)
    print(f"\nsummary csv saved to: {os.path.abspath(args.output_csv)}")


if __name__ == "__main__":
    main()
