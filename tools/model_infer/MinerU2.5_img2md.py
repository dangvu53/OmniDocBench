#!/usr/bin/env python3
"""Convert OmniDocBench page images to Markdown with MinerU2.5/MinerU2.5-Pro."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from PIL import Image
from tqdm import tqdm


DEFAULT_MODEL_ID = "opendatalab/MinerU2.5-Pro-2605-1.2B"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def read_filelist(path: Path) -> list[str]:
    names: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                names.append(os.path.basename(stripped))
    return names


def read_manifest(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "basename" not in (reader.fieldnames or []):
            raise SystemExit(f"{path} must contain a 'basename' column")
        return [row["basename"] for row in reader if row.get("basename")]


def collect_images(input_dir: Path, manifest: Path | None, filelist: Path | None, limit: int | None) -> list[Path]:
    if manifest and filelist:
        raise SystemExit("Use only one of --manifest or --filelist")
    if manifest:
        names = read_manifest(manifest)
    elif filelist:
        names = read_filelist(filelist)
    else:
        names = sorted(path.name for path in input_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)

    images = [input_dir / name for name in names]
    missing = [str(path) for path in images if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing {len(missing)} images. Examples: {', '.join(missing[:10])}")
    if limit is not None:
        images = images[:limit]
    if not images:
        raise SystemExit(f"No images found in {input_dir}")
    return images


def build_vllm_client(args: argparse.Namespace) -> object:
    try:
        from mineru_vl_utils import MinerUClient
        from vllm import LLM
    except ImportError as exc:
        raise SystemExit('Install MinerU vLLM dependencies first: pip install -U "mineru-vl-utils[vllm]"') from exc

    llm_kwargs: dict[str, object] = {
        "model": args.model_id,
        "tensor_parallel_size": args.tensor_parallel_size,
    }
    if args.gpu_memory_utilization is not None:
        llm_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len

    try:
        from mineru_vl_utils import MinerULogitsProcessor

        llm_kwargs["logits_processors"] = [MinerULogitsProcessor]
    except ImportError:
        pass

    try:
        llm = LLM(**llm_kwargs)
    except TypeError:
        if "logits_processors" not in llm_kwargs:
            raise
        llm_kwargs.pop("logits_processors")
        llm = LLM(**llm_kwargs)
    return MinerUClient(backend="vllm-engine", vllm_llm=llm, image_analysis=args.image_analysis)


def build_transformers_client(args: argparse.Namespace) -> object:
    try:
        from mineru_vl_utils import MinerUClient
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    except ImportError as exc:
        raise SystemExit(
            'Install MinerU transformers dependencies first: pip install -U "mineru-vl-utils[transformers]"'
        ) from exc

    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(args.model_id, dtype="auto", device_map="auto")
    except TypeError:
        model = Qwen2VLForConditionalGeneration.from_pretrained(args.model_id, torch_dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(args.model_id, use_fast=True)
    return MinerUClient(
        backend="transformers",
        model=model,
        processor=processor,
        image_analysis=args.image_analysis,
    )


def build_client(args: argparse.Namespace) -> object:
    if args.backend == "vllm-engine":
        return build_vllm_client(args)
    if args.backend == "transformers":
        return build_transformers_client(args)
    raise SystemExit(f"Unsupported backend: {args.backend}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MinerU2.5/MinerU2.5-Pro over page images.")
    parser.add_argument("--model-id", "--model_path", default=DEFAULT_MODEL_ID)
    parser.add_argument("--backend", choices=("vllm-engine", "transformers"), default="vllm-engine")
    parser.add_argument("--input-dir", "--input_path", required=True, type=Path)
    parser.add_argument("--output-dir", "--save_dir", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, help="Optional OmniDocBench subset manifest.csv")
    parser.add_argument("--filelist", type=Path, help="Optional text file of image basenames")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-json", action="store_true", help="Write raw MinerU JSON beside Markdown outputs")
    parser.add_argument("--image-analysis", action="store_true", help="Enable image/chart analysis")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--max-model-len", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_dir = args.output_dir / "mineru_json"
    if args.save_json:
        json_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(args.input_dir, args.manifest, args.filelist, args.limit)

    from mineru_vl_utils.post_process import json2md

    client = build_client(args)
    for image_path in tqdm(images, desc="MinerU2.5"):
        md_path = args.output_dir / f"{image_path.stem}.md"
        if md_path.exists() and md_path.stat().st_size > 0 and not args.overwrite:
            continue

        image = Image.open(image_path).convert("RGB")
        content_list = client.two_step_extract(image)
        md_path.write_text(json2md(content_list).strip() + "\n", encoding="utf-8")

        if args.save_json:
            json_path = json_dir / f"{image_path.stem}.json"
            json_path.write_text(json.dumps(content_list, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
