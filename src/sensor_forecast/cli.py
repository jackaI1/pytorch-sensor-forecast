"""Small command line interface; network access is isolated to fetch."""

import argparse
import json
from dataclasses import fields
from pathlib import Path

from .benchmark import Config, run_benchmark, save_results
from .data import fetch_weather, load_csv, validate_hourly


def main():
    parser = argparse.ArgumentParser(description="Compare hourly forecasts without future data")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="Download and checksum public UCI temperature data")
    fetch.add_argument("--output", type=Path, default=Path("data/weather.csv"))
    validate = sub.add_parser("validate", help="Reject gaps, duplicates and missing values")
    validate.add_argument("data", type=Path)
    run = sub.add_parser(
        "run", help="Fit all three models and write predictions and error intervals"
    )
    run.add_argument("--data", type=Path, default=Path("data/weather.csv"))
    run.add_argument("--output", type=Path, default=Path("outputs/benchmark"))
    for field in fields(Config):
        if field.name == "horizons":
            run.add_argument("--horizons", type=int, nargs="+", default=field.default)
        else:
            run.add_argument("--" + field.name.replace("_", "-"), type=int, default=field.default)
    args = parser.parse_args()
    try:
        if args.command == "fetch":
            print(json.dumps(fetch_weather(args.output), indent=2))
        elif args.command == "validate":
            print(json.dumps(validate_hourly(load_csv(args.data)), indent=2))
        else:
            options = {field.name: getattr(args, field.name) for field in fields(Config)}
            options["horizons"] = tuple(options["horizons"])
            forecasts, metrics, report = run_benchmark(load_csv(args.data), Config(**options))
            report["input_path"] = str(args.data)
            save_results(args.output, forecasts, metrics, report)
            print(metrics.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
            print(f"Saved predictions, metrics, fold audit and summary to {args.output}")
    except (ValueError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
