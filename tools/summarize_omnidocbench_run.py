#!/usr/bin/env python3
"""Summarize OmniDocBench run artifacts with throughput and scores."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def get(data: dict[str, Any], path: list[str], default: Any = None) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def fmt_float(value: Any, digits: int = 3, scale: float = 1.0) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value) * scale:.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def fmt_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "N/A"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rest = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {rest:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {rest:.0f}s"


def read_inference_log(log_path: Path) -> dict[str, Any]:
    seconds: list[float] = []
    statuses: dict[str, int] = {}
    chars: list[int] = []
    if not log_path.is_file():
        return {"seconds_sum": None, "seconds_median": None, "statuses": statuses, "chars_median": None}

    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            status = str(row.get("status", "unknown"))
            statuses[status] = statuses.get(status, 0) + 1
            if status in {"success", "failed"} and isinstance(row.get("seconds"), (int, float)):
                seconds.append(float(row["seconds"]))
            if status == "success" and isinstance(row.get("chars"), int):
                chars.append(int(row["chars"]))

    return {
        "seconds_sum": sum(seconds) if seconds else None,
        "seconds_median": statistics.median(seconds) if seconds else None,
        "statuses": statuses,
        "chars_median": statistics.median(chars) if chars else None,
    }


def first_metric_prefix(eval_dir: Path, engine: str) -> str | None:
    preferred = eval_dir / f"{engine}_quick_match_metric_result.json"
    if preferred.is_file():
        return f"{engine}_quick_match"
    matches = sorted(eval_dir.glob("*_metric_result.json"))
    if not matches:
        return None
    return matches[0].name.removesuffix("_metric_result.json")


def summarize_engine(run_root: Path, engine: str) -> dict[str, Any]:
    pred_dir = run_root / "predictions" / engine
    eval_dir = run_root / "eval" / engine
    summary = load_json(pred_dir / "inference_summary.json")
    log_stats = read_inference_log(pred_dir / "inference_log.jsonl")

    successes = int(summary.get("successes") or log_stats["statuses"].get("success", 0) or 0)
    failures = int(summary.get("failures") or log_stats["statuses"].get("failed", 0) or 0)
    total = int(summary.get("total") or successes + failures + log_stats["statuses"].get("skipped", 0) or 0)
    empty_md = 0
    if pred_dir.is_dir():
        empty_md = sum(1 for path in pred_dir.glob("*.md") if path.stat().st_size == 0)

    page_seconds_sum = summary.get("page_seconds_sum")
    if page_seconds_sum is None:
        page_seconds_sum = log_stats["seconds_sum"]
    page_seconds_sum = float(page_seconds_sum) if page_seconds_sum is not None else None
    wall_seconds = summary.get("wall_seconds")
    wall_seconds = float(wall_seconds) if wall_seconds is not None else None
    processed_pages = successes + failures
    warm_ppm = None if not page_seconds_sum else processed_pages * 60.0 / page_seconds_sum
    wall_ppm = None if not wall_seconds else successes * 60.0 / wall_seconds

    prefix = first_metric_prefix(eval_dir, engine)
    metric = load_json(eval_dir / f"{prefix}_metric_result.json") if prefix else {}
    run_summary = load_json(eval_dir / f"{prefix}_run_summary.json") if prefix else {}
    stage = load_json(eval_dir / f"{prefix}_stage_execution.json") if prefix else {}
    eval_timing = load_json(eval_dir / "eval_timing.json")
    eval_seconds = eval_timing.get("eval_seconds")
    eval_seconds = float(eval_seconds) if eval_seconds is not None else None

    return {
        "engine": engine,
        "pages": total,
        "successes": successes,
        "failures": failures,
        "empty_md": empty_md,
        "warm_inference_seconds": page_seconds_sum,
        "warm_pages_per_minute": warm_ppm,
        "wall_seconds": wall_seconds,
        "wall_pages_per_minute": wall_ppm,
        "median_page_seconds": log_stats["seconds_median"],
        "median_output_chars": log_stats["chars_median"],
        "eval_seconds": eval_seconds,
        "overall": get(run_summary, ["notebook_metric_summary", "overall_notebook"]),
        "text_edit": get(metric, ["text_block", "all", "Edit_dist", "ALL_page_avg"]),
        "formula_cdm": get(metric, ["display_formula", "page", "CDM", "ALL"]),
        "formula_edit": get(metric, ["display_formula", "all", "Edit_dist", "ALL_page_avg"]),
        "table_teds": get(metric, ["table", "page", "TEDS", "ALL"]),
        "table_edit": get(metric, ["table", "all", "Edit_dist", "ALL_page_avg"]),
        "reading_edit": get(metric, ["reading_order", "all", "Edit_dist", "ALL_page_avg"]),
        "cdm_samples": get(stage, ["metrics", "display_formula", "CDM", "sample_count"]),
    }


def discover_engines(run_root: Path) -> list[str]:
    pred_root = run_root / "predictions"
    if not pred_root.is_dir():
        return []
    return sorted(path.name for path in pred_root.iterdir() if path.is_dir())


def print_markdown(rows: list[dict[str, Any]]) -> None:
    print("| Engine | Pages | Success | Empty | Warm Infer Time | Warm pages/min | Eval Time | Overall | Text ED ↓ | Formula CDM ↑ | Formula ED ↓ | Table TEDS ↑ | Table ED ↓ | Reading ED ↓ |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            "| {engine} | {pages} | {successes}/{pages} | {empty_md} | {warm_time} | {warm_ppm} | {eval_time} | {overall} | {text_edit} | {formula_cdm} | {formula_edit} | {table_teds} | {table_edit} | {reading_edit} |".format(
                engine=row["engine"],
                pages=row["pages"],
                successes=row["successes"],
                empty_md=row["empty_md"],
                warm_time=fmt_seconds(row["warm_inference_seconds"]),
                warm_ppm=fmt_float(row["warm_pages_per_minute"], 2),
                eval_time=fmt_seconds(row["eval_seconds"]),
                overall=fmt_float(row["overall"], 3),
                text_edit=fmt_float(row["text_edit"], 6),
                formula_cdm=fmt_float(row["formula_cdm"], 3, 100.0),
                formula_edit=fmt_float(row["formula_edit"], 6),
                table_teds=fmt_float(row["table_teds"], 3, 100.0),
                table_edit=fmt_float(row["table_edit"], 6),
                reading_edit=fmt_float(row["reading_edit"], 6),
            )
        )
    print()
    print("Notes:")
    print("- Warm Infer Time/pages-min is summed from per-page inference log entries and excludes model download/startup for older runs.")
    print("- Overall follows OmniDocBench: ((1 - Text ED) * 100 + Table TEDS + Formula CDM) / 3.")
    print("- Formula ED and Table ED are diagnostic normalized edit-distance metrics; they are not part of the official Overall formula.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize an OmniDocBench RUN_ROOT.")
    parser.add_argument("run_root", type=Path, help="Run root, e.g. runs/omnidocbench_100_sparse")
    parser.add_argument("--engines", nargs="*", help="Engines to include. Defaults to all prediction folders.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of Markdown")
    args = parser.parse_args()

    engines = args.engines or discover_engines(args.run_root)
    if not engines:
        raise SystemExit(f"No engines found under {args.run_root / 'predictions'}")

    rows = [summarize_engine(args.run_root, engine) for engine in engines]
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        print_markdown(rows)


if __name__ == "__main__":
    main()
