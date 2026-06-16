#!/usr/bin/env python3
"""Build a deterministic, diverse OmniDocBench page subset.

The selector balances the primary page attribute (``data_source`` by default)
and then greedily maximizes coverage over language, layout, element categories,
and rich document features such as tables, formulas, figures, notes, and code.
It never edits the source dataset or source images.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PAGE_ATTRIBUTE_KEYS = ("data_source", "language", "layout")
BLOCK_FEATURES = {
    "table": {"table"},
    "display_formula": {"equation_isolated"},
    "figure": {"figure", "figure_caption", "figure_footnote"},
    "code": {"code_txt", "code_txt_caption"},
    "reference": {"reference"},
    "abandon": {"abandon"},
    "header_footer": {"header", "footer", "page_number", "page_footnote"},
}
SPAN_FEATURES = {
    "inline_formula": {"equation_inline"},
    "footnote_mark": {"footnote_mark"},
}


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(f"Expected a list of samples in {path}")
    return data


def page_basename(sample: dict[str, Any]) -> str:
    image_path = sample.get("page_info", {}).get("image_path", "")
    basename = os.path.basename(image_path)
    if not basename:
        raise ValueError("sample is missing page_info.image_path")
    return basename


def deterministic_noise(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def safe_attribute(sample: dict[str, Any], key: str) -> str:
    return str(sample.get("page_info", {}).get("page_attribute", {}).get(key, "unknown"))


def iter_non_ignored_blocks(sample: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in sample.get("layout_dets", []) if not item.get("ignore", False)]


def span_category_counts(blocks: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for block in blocks:
        for span in block.get("line_with_spans", []) or []:
            if span.get("ignore", False):
                continue
            category = span.get("category_type")
            if category:
                counts[str(category)] += 1
    return counts


def sample_profile(index: int, sample: dict[str, Any]) -> dict[str, Any]:
    blocks = iter_non_ignored_blocks(sample)
    block_counts = Counter(str(item.get("category_type", "unknown")) for item in blocks)
    span_counts = span_category_counts(blocks)
    attrs = {key: safe_attribute(sample, key) for key in PAGE_ATTRIBUTE_KEYS}

    tokens: set[str] = set()
    for key, value in attrs.items():
        tokens.add(f"page:{key}:{value}")
    for category, count in block_counts.items():
        if count:
            tokens.add(f"block:{category}")
    for category, count in span_counts.items():
        if count:
            tokens.add(f"span:{category}")

    features: dict[str, int] = {}
    for name, categories in BLOCK_FEATURES.items():
        features[name] = int(any(block_counts.get(category, 0) > 0 for category in categories))
        if features[name]:
            tokens.add(f"feature:{name}")
    for name, categories in SPAN_FEATURES.items():
        features[name] = int(any(span_counts.get(category, 0) > 0 for category in categories))
        if features[name]:
            tokens.add(f"feature:{name}")

    element_count = sum(block_counts.values()) + sum(span_counts.values())
    rich_feature_count = sum(features.values())
    richness = math.log1p(element_count) + 0.75 * rich_feature_count + 0.2 * len(block_counts)

    return {
        "index": index,
        "basename": page_basename(sample),
        "attrs": attrs,
        "block_counts": block_counts,
        "span_counts": span_counts,
        "features": features,
        "tokens": tokens,
        "richness": richness,
        "element_count": element_count,
    }


def allocate_quotas(profiles: list[dict[str, Any]], sample_count: int, balance_key: str) -> dict[str, int]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for profile in profiles:
        groups[profile["attrs"].get(balance_key, "unknown")].append(profile)
    if not groups:
        return {}

    keys = sorted(groups)
    base = sample_count // len(keys)
    remainder = sample_count % len(keys)
    quotas = {key: min(base, len(groups[key])) for key in keys}

    capacity = {key: len(groups[key]) - quotas[key] for key in keys}
    for key in sorted(keys, key=lambda item: (-capacity[item], item)):
        if remainder <= 0:
            break
        if capacity[key] <= 0:
            continue
        quotas[key] += 1
        remainder -= 1

    while sum(quotas.values()) < sample_count:
        candidates = [key for key in keys if quotas[key] < len(groups[key])]
        if not candidates:
            break
        key = min(candidates, key=lambda item: (quotas[item] / max(len(groups[item]), 1), item))
        quotas[key] += 1

    return quotas


def build_token_weights(profiles: list[dict[str, Any]]) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for profile in profiles:
        counts.update(profile["tokens"])
    total = max(len(profiles), 1)
    return {token: math.log((total + 1) / (count + 1)) + 1.0 for token, count in counts.items()}


def profile_score(
    profile: dict[str, Any],
    selected_counts: Counter[str],
    token_weights: dict[str, float],
    max_richness: float,
) -> float:
    coverage = 0.0
    repeat_bonus = 0.0
    for token in profile["tokens"]:
        token_count = selected_counts[token]
        weight = token_weights.get(token, 1.0)
        if token_count == 0:
            coverage += weight
        elif token_count < 3:
            repeat_bonus += weight / (4.0 + token_count)

    richness = profile["richness"] / max(max_richness, 1.0)
    jitter = deterministic_noise(profile["basename"]) * 1e-6
    return coverage + repeat_bonus + 0.35 * richness + jitter


def select_profiles(
    profiles: list[dict[str, Any]],
    sample_count: int,
    balance_key: str,
) -> list[dict[str, Any]]:
    if sample_count < 1:
        raise SystemExit("--sample-count must be >= 1")
    if len(profiles) < sample_count:
        raise SystemExit(f"Requested {sample_count} samples, but only {len(profiles)} are available")

    quotas = allocate_quotas(profiles, sample_count, balance_key)
    token_weights = build_token_weights(profiles)
    max_richness = max((profile["richness"] for profile in profiles), default=1.0)
    selected: list[dict[str, Any]] = []
    selected_indices: set[int] = set()
    selected_counts: Counter[str] = Counter()

    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for profile in profiles:
        by_group[profile["attrs"].get(balance_key, "unknown")].append(profile)

    for group_key in sorted(quotas):
        for _ in range(quotas[group_key]):
            candidates = [p for p in by_group[group_key] if p["index"] not in selected_indices]
            if not candidates:
                break
            best = max(candidates, key=lambda p: profile_score(p, selected_counts, token_weights, max_richness))
            selected.append(best)
            selected_indices.add(best["index"])
            selected_counts.update(best["tokens"])

    while len(selected) < sample_count:
        candidates = [p for p in profiles if p["index"] not in selected_indices]
        best = max(candidates, key=lambda p: profile_score(p, selected_counts, token_weights, max_richness))
        selected.append(best)
        selected_indices.add(best["index"])
        selected_counts.update(best["tokens"])

    return selected


def resolve_image_path(sample: dict[str, Any], images_root: Path) -> Path | None:
    image_path = str(sample.get("page_info", {}).get("image_path", ""))
    basename = os.path.basename(image_path)
    raw_path = Path(image_path)
    candidates = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    candidates.extend(
        [
            images_root / image_path,
            images_root / basename,
            images_root.parent / image_path,
            images_root.parent / basename,
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def write_image(src: Path, dst: Path, copy_mode: str, overwrite: bool) -> None:
    if copy_mode == "none":
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()
    if copy_mode == "copy":
        shutil.copy2(src, dst)
    elif copy_mode == "symlink":
        os.symlink(src, dst)
    else:
        raise ValueError(f"Unknown copy mode: {copy_mode}")


def counter_to_plain(counter: Counter[str]) -> dict[str, int]:
    return {key: int(value) for key, value in sorted(counter.items())}


def build_report(selected_profiles: list[dict[str, Any]]) -> dict[str, Any]:
    page_attrs = {key: Counter() for key in PAGE_ATTRIBUTE_KEYS}
    features: Counter[str] = Counter()
    block_categories: Counter[str] = Counter()
    span_categories: Counter[str] = Counter()
    for profile in selected_profiles:
        for key in PAGE_ATTRIBUTE_KEYS:
            page_attrs[key][profile["attrs"].get(key, "unknown")] += 1
        for name, enabled in profile["features"].items():
            if enabled:
                features[name] += 1
        block_categories.update(profile["block_counts"])
        span_categories.update(profile["span_counts"])

    return {
        "sample_count": len(selected_profiles),
        "page_attributes": {key: counter_to_plain(value) for key, value in page_attrs.items()},
        "features": counter_to_plain(features),
        "block_categories": counter_to_plain(block_categories),
        "span_categories": counter_to_plain(span_categories),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a diverse OmniDocBench subset.")
    parser.add_argument("--gt-json", required=True, type=Path, help="Full OmniDocBench JSON file")
    parser.add_argument("--images-root", required=True, type=Path, help="Directory containing page images")
    parser.add_argument("--output-root", required=True, type=Path, help="Directory where subset files are written")
    parser.add_argument("--sample-count", type=int, default=100, help="Number of pages to select")
    parser.add_argument("--balance-key", default="data_source", help="Page attribute to balance first")
    parser.add_argument(
        "--copy-mode",
        choices=("symlink", "copy", "none"),
        default="symlink",
        help="How to materialize selected images under output-root/images",
    )
    parser.add_argument("--allow-missing-images", action="store_true", help="Write subset even if images are missing")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite generated files if they already exist")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_json(args.gt_json)
    profiles = [sample_profile(index, sample) for index, sample in enumerate(samples)]

    duplicate_basenames = [name for name, count in Counter(p["basename"] for p in profiles).items() if count > 1]
    if duplicate_basenames:
        raise SystemExit("Duplicate image basenames are not supported: " + ", ".join(sorted(duplicate_basenames)[:10]))

    selected_profiles = select_profiles(profiles, args.sample_count, args.balance_key)
    selected_profiles_by_index = {profile["index"]: profile for profile in selected_profiles}
    selected_indices = set(selected_profiles_by_index)

    args.output_root.mkdir(parents=True, exist_ok=True)
    images_out = args.output_root / "images"
    subset_json = args.output_root / f"OmniDocBench_{args.sample_count}_diverse.json"
    manifest_csv = args.output_root / "manifest.csv"
    filelist_txt = args.output_root / "filelist.txt"
    report_json = args.output_root / "selection_report.json"

    generated_files = [subset_json, manifest_csv, filelist_txt, report_json]
    existing = [path for path in generated_files if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit("Generated files already exist. Re-run with --overwrite: " + ", ".join(str(p) for p in existing))

    subset_samples: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    missing_images: list[str] = []

    for index, sample in enumerate(samples):
        if index not in selected_indices:
            continue
        profile = selected_profiles_by_index[index]
        src_image = resolve_image_path(sample, args.images_root)
        basename = profile["basename"]
        dst_image = images_out / basename
        if src_image is None:
            missing_images.append(basename)
            if not args.allow_missing_images:
                continue
        else:
            write_image(src_image, dst_image, args.copy_mode, args.overwrite)

        sample_copy = json.loads(json.dumps(sample, ensure_ascii=False))
        sample_copy["page_info"]["image_path"] = f"images/{basename}"
        subset_samples.append(sample_copy)
        manifest_rows.append(
            {
                "original_index": index,
                "basename": basename,
                "source_image": "" if src_image is None else str(src_image),
                "subset_image": "" if args.copy_mode == "none" else str(dst_image),
                "data_source": profile["attrs"].get("data_source", "unknown"),
                "language": profile["attrs"].get("language", "unknown"),
                "layout": profile["attrs"].get("layout", "unknown"),
                "element_count": profile["element_count"],
                "features": ";".join(name for name, enabled in sorted(profile["features"].items()) if enabled),
                "block_categories": json.dumps(counter_to_plain(profile["block_counts"]), sort_keys=True),
                "span_categories": json.dumps(counter_to_plain(profile["span_counts"]), sort_keys=True),
            }
        )

    if missing_images and not args.allow_missing_images:
        raise SystemExit(
            f"Missing {len(missing_images)} selected images under {args.images_root}. "
            f"Examples: {', '.join(missing_images[:10])}"
        )
    if len(subset_samples) != args.sample_count:
        raise SystemExit(f"Prepared {len(subset_samples)} samples, expected {args.sample_count}")

    with subset_json.open("w", encoding="utf-8") as f:
        json.dump(subset_samples, f, ensure_ascii=False, indent=2)
    with manifest_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    with filelist_txt.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(row["basename"] + "\n")

    report = build_report([selected_profiles_by_index[index] for index in sorted(selected_indices)])
    report["gt_json"] = str(args.gt_json)
    report["images_root"] = str(args.images_root)
    report["copy_mode"] = args.copy_mode
    report["missing_images"] = missing_images
    with report_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"subset_json={subset_json}")
    print(f"images_dir={images_out}")
    print(f"manifest={manifest_csv}")
    print(f"report={report_json}")


if __name__ == "__main__":
    main()
