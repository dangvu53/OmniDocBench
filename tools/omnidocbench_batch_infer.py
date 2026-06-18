#!/usr/bin/env python3
"""Run page-image-to-Markdown inference for OmniDocBench-compatible outputs."""

from __future__ import annotations

import argparse
import csv
import base64
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_MODELS = {
    "lighton_bbox": "lightonai/LightOnOCR-2-1B-bbox",
    "chandra": "datalab-to/chandra-ocr-2",
    "mineru25": "opendatalab/MinerU2.5-2509-1.2B",
    "mineru25pro": "opendatalab/MinerU2.5-Pro-2605-1.2B",
    "kdl_frontier_nano": "KDLAI/KDL-Frontier-Parser-nano",
}
DEFAULT_KDL_PROMPT = (
    "Parse this document page into clean Markdown. Preserve reading order, headings, paragraphs, tables, "
    "lists, equations, and visible captions. Return only Markdown without commentary."
)


class EngineError(RuntimeError):
    """Raised when an inference engine cannot be initialized or run."""


def read_filelist(path: Path) -> list[str]:
    names = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                names.append(os.path.basename(stripped))
    return names


def read_manifest_basenames(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "basename" not in (reader.fieldnames or []):
            raise SystemExit(f"{path} must contain a 'basename' column")
        return [row["basename"] for row in reader if row.get("basename")]


def collect_images(input_dir: Path, filelist: Path | None, manifest: Path | None, limit: int | None) -> list[Path]:
    if filelist and manifest:
        raise SystemExit("Use only one of --filelist or --manifest")

    if manifest:
        basenames = read_manifest_basenames(manifest)
    elif filelist:
        basenames = read_filelist(filelist)
    else:
        basenames = sorted(path.name for path in input_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)

    images = [input_dir / name for name in basenames]
    missing = [str(path) for path in images if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing {len(missing)} images. Examples: {', '.join(missing[:10])}")

    if limit is not None:
        images = images[:limit]
    if not images:
        raise SystemExit(f"No images found in {input_dir}")
    return images


def clean_markdown(text: str) -> str:
    stripped = text.strip()
    for fence in ("```markdown", "```md", "```"):
        if stripped.startswith(fence) and stripped.endswith("```"):
            stripped = stripped[len(fence) : -3].strip()
            break
    return stripped


def output_path_for(image_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{image_path.stem}.md"


def harvest_markdown(search_root: Path, preferred_stem: str) -> str:
    md_files = sorted(search_root.rglob("*.md"))
    if not md_files:
        raise EngineError(f"No Markdown output found under {search_root}")

    preferred = [path for path in md_files if path.stem == preferred_stem]
    chosen = preferred[0] if preferred else max(md_files, key=lambda path: path.stat().st_size)
    return chosen.read_text(encoding="utf-8")


def build_marker_converter(_: argparse.Namespace) -> Callable[[Path], str]:
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered
    except ImportError as exc:
        raise EngineError("Marker is not installed. Install with: pip install marker-pdf") from exc

    converter = PdfConverter(artifact_dict=create_model_dict())

    def convert(image_path: Path) -> str:
        rendered = converter(str(image_path))
        text, _, _ = text_from_rendered(rendered)
        return clean_markdown(text)

    return convert


def build_docling_converter(_: argparse.Namespace) -> Callable[[Path], str]:
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:
        raise EngineError("Docling is not installed. Install with: pip install docling") from exc

    converter = DocumentConverter()

    def convert(image_path: Path) -> str:
        result = converter.convert(str(image_path))
        return clean_markdown(result.document.export_to_markdown())

    return convert


def build_paddleocr_vl15_converter(args: argparse.Namespace) -> Callable[[Path], str]:
    try:
        from paddleocr import PaddleOCRVL
    except ImportError as exc:
        if shutil.which("paddleocr"):
            return build_paddleocr_cli_converter(args)
        raise EngineError(
            'PaddleOCR-VL is not installed. Install with: python -m pip install -U "paddleocr[doc-parser]"'
        ) from exc

    kwargs: dict[str, object] = {"pipeline_version": args.paddle_pipeline_version}
    if args.device != "auto":
        kwargs["device"] = args.device
    if args.paddle_engine:
        kwargs["engine"] = args.paddle_engine
    pipeline = PaddleOCRVL(**kwargs)

    def convert(image_path: Path) -> str:
        with tempfile.TemporaryDirectory(prefix=f"paddle_{image_path.stem}_") as tmp:
            tmp_path = Path(tmp)
            output = pipeline.predict(
                str(image_path),
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
            )
            for res in output:
                res.save_to_markdown(save_path=str(tmp_path), pretty=False)
            return clean_markdown(harvest_markdown(tmp_path, image_path.stem))

    return convert


def build_paddleocr_cli_converter(args: argparse.Namespace) -> Callable[[Path], str]:
    def convert(image_path: Path) -> str:
        with tempfile.TemporaryDirectory(prefix=f"paddle_cli_{image_path.stem}_") as tmp:
            cmd = [
                "paddleocr",
                "doc_parser",
                "-i",
                str(image_path),
                "--save_path",
                tmp,
                "--pipeline_version",
                args.paddle_pipeline_version,
            ]
            if args.paddle_engine:
                cmd.extend(["--engine", args.paddle_engine])
            subprocess.run(cmd, check=True, text=True, capture_output=True)
            return clean_markdown(harvest_markdown(Path(tmp), image_path.stem))

    return convert


def torch_dtype_for_device(torch_module: object, device: str) -> object:
    if device == "cuda":
        return getattr(torch_module, "bfloat16")
    return getattr(torch_module, "float32")


def build_lighton_converter(args: argparse.Namespace) -> Callable[[Path], str]:
    model_id = args.model_id or DEFAULT_MODELS["lighton_bbox"]
    try:
        import torch
        from PIL import Image
        from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor
    except ImportError as exc:
        raise EngineError(
            "LightOnOCR requires transformers from source plus torch and Pillow. "
            "Install with: pip install git+https://github.com/huggingface/transformers pillow"
        ) from exc

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    dtype = torch_dtype_for_device(torch, device)
    model = LightOnOcrForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype).to(device)
    processor = LightOnOcrProcessor.from_pretrained(model_id)
    model.eval()

    def convert(image_path: Path) -> str:
        image = Image.open(image_path).convert("RGB")
        conversation = [{"role": "user", "content": [{"type": "image", "image": image}]}]
        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(device=device, dtype=dtype) if value.is_floating_point() else value.to(device)
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
        return clean_markdown(processor.decode(generated_ids, skip_special_tokens=True))

    return convert


def from_pretrained_image_text_model(model_cls: object, model_id: str, torch_module: object) -> object:
    try:
        return model_cls.from_pretrained(model_id, dtype=getattr(torch_module, "bfloat16"), device_map="auto")
    except TypeError:
        return model_cls.from_pretrained(model_id, torch_dtype=getattr(torch_module, "bfloat16"), device_map="auto")


def build_chandra_converter(args: argparse.Namespace) -> Callable[[Path], str]:
    model_id = args.model_id or DEFAULT_MODELS["chandra"]
    try:
        import torch
        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor

        from chandra.model.hf import generate_hf
        from chandra.model.schema import BatchInputItem
        from chandra.output import parse_markdown
    except ImportError as exc:
        raise EngineError("Chandra HF inference is not installed. Install with: pip install 'chandra-ocr[hf]'") from exc

    model = from_pretrained_image_text_model(AutoModelForImageTextToText, model_id, torch)
    model.eval()
    model.processor = AutoProcessor.from_pretrained(model_id)
    model.processor.tokenizer.padding_side = "left"

    def convert(image_path: Path) -> str:
        batch = [
            BatchInputItem(
                image=Image.open(image_path).convert("RGB"),
                prompt_type=args.chandra_prompt_type,
            )
        ]
        with torch.inference_mode():
            try:
                result = generate_hf(batch, model, max_output_tokens=args.max_new_tokens)[0]
            except TypeError:
                result = generate_hf(batch, model)[0]
        return clean_markdown(parse_markdown(result.raw))

    return convert


def build_mineru_vllm_client(args: argparse.Namespace, model_id: str) -> object:
    try:
        from mineru_vl_utils import MinerUClient
        from vllm import LLM
    except ImportError as exc:
        raise EngineError(
            'MinerU vLLM inference is not installed. Install with: pip install -U "mineru-vl-utils[vllm]"'
        ) from exc

    llm_kwargs: dict[str, object] = {
        "model": model_id,
        "tensor_parallel_size": args.mineru_tensor_parallel_size,
    }
    if args.mineru_gpu_memory_utilization is not None:
        llm_kwargs["gpu_memory_utilization"] = args.mineru_gpu_memory_utilization
    if args.mineru_max_model_len is not None:
        llm_kwargs["max_model_len"] = args.mineru_max_model_len

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
    return MinerUClient(
        backend="vllm-engine",
        vllm_llm=llm,
        image_analysis=args.mineru_image_analysis,
    )


def build_mineru_transformers_client(args: argparse.Namespace, model_id: str) -> object:
    try:
        from mineru_vl_utils import MinerUClient
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    except ImportError as exc:
        raise EngineError(
            'MinerU transformers inference is not installed. Install with: pip install -U "mineru-vl-utils[transformers]"'
        ) from exc

    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, dtype="auto", device_map="auto")
    except TypeError:
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    return MinerUClient(
        backend="transformers",
        model=model,
        processor=processor,
        image_analysis=args.mineru_image_analysis,
    )


def build_mineru_converter(args: argparse.Namespace) -> Callable[[Path], str]:
    model_id = args.model_id or DEFAULT_MODELS[args.engine]
    try:
        from PIL import Image
        from mineru_vl_utils.post_process import json2md
    except ImportError as exc:
        raise EngineError(
            "MinerU utilities are not installed. Install with the vLLM or transformers extra for your backend."
        ) from exc

    if args.mineru_backend == "vllm-engine":
        client = build_mineru_vllm_client(args, model_id)
    elif args.mineru_backend == "transformers":
        client = build_mineru_transformers_client(args, model_id)
    else:
        raise EngineError(f"Unsupported MinerU backend: {args.mineru_backend}")

    def convert(image_path: Path) -> str:
        image = Image.open(image_path).convert("RGB")
        content_list = client.two_step_extract(image)
        return clean_markdown(json2md(content_list))

    return convert


def build_kdl_transformers_converter(args: argparse.Namespace, model_id: str) -> Callable[[Path], str]:
    try:
        import torch
        from PIL import Image
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    except ImportError as exc:
        raise EngineError(
            "KDL transformers inference requires torch, Pillow, and transformers with Qwen2-VL support."
        ) from exc

    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
    except TypeError:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model.eval()

    def convert(image_path: Path) -> str:
        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": args.kdl_prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt")
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
        return clean_markdown(processor.decode(generated_ids, skip_special_tokens=True))

    return convert


def image_to_data_url(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def build_kdl_openai_converter(args: argparse.Namespace, model_id: str) -> Callable[[Path], str]:
    try:
        import requests
    except ImportError as exc:
        raise EngineError("KDL OpenAI-compatible inference requires requests. Install with: pip install requests") from exc

    if not args.kdl_api_base:
        raise EngineError("--kdl-api-base is required when --kdl-backend openai")
    api_base = args.kdl_api_base.rstrip("/")
    if api_base.endswith("/v1"):
        url = f"{api_base}/chat/completions"
    else:
        url = f"{api_base}/v1/chat/completions"

    headers = {"Content-Type": "application/json"}
    if args.kdl_api_key:
        headers["Authorization"] = f"Bearer {args.kdl_api_key}"

    def convert(image_path: Path) -> str:
        payload = {
            "model": args.kdl_served_model_name or model_id,
            "temperature": 0,
            "max_tokens": args.max_new_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": args.kdl_prompt},
                        {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                    ],
                }
            ],
        }
        response = requests.post(url, headers=headers, json=payload, timeout=args.kdl_timeout)
        response.raise_for_status()
        data = response.json()
        return clean_markdown(data["choices"][0]["message"]["content"])

    return convert


def build_kdl_converter(args: argparse.Namespace) -> Callable[[Path], str]:
    model_id = args.model_id or DEFAULT_MODELS["kdl_frontier_nano"]
    if args.kdl_backend == "transformers":
        return build_kdl_transformers_converter(args, model_id)
    if args.kdl_backend == "openai":
        return build_kdl_openai_converter(args, model_id)
    raise EngineError(f"Unsupported KDL backend: {args.kdl_backend}")


def build_converter(args: argparse.Namespace) -> Callable[[Path], str]:
    if args.engine == "marker":
        return build_marker_converter(args)
    if args.engine == "docling":
        return build_docling_converter(args)
    if args.engine == "paddleocr_vl15":
        return build_paddleocr_vl15_converter(args)
    if args.engine == "lighton_bbox":
        return build_lighton_converter(args)
    if args.engine == "chandra":
        return build_chandra_converter(args)
    if args.engine in {"mineru25", "mineru25pro"}:
        return build_mineru_converter(args)
    if args.engine == "kdl_frontier_nano":
        return build_kdl_converter(args)
    raise SystemExit(f"Unsupported engine: {args.engine}")


def write_log(log_path: Path, record: dict[str, object]) -> None:
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a document parser over OmniDocBench page images.")
    parser.add_argument(
        "--engine",
        required=True,
        choices=(
            "marker",
            "docling",
            "paddleocr_vl15",
            "lighton_bbox",
            "chandra",
            "mineru25",
            "mineru25pro",
            "kdl_frontier_nano",
        ),
    )
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory containing selected page images")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for OmniDocBench .md predictions")
    parser.add_argument("--filelist", type=Path, help="Optional text file of image basenames to process")
    parser.add_argument("--manifest", type=Path, help="Optional subset manifest.csv with a basename column")
    parser.add_argument("--limit", type=int, help="Process only the first N images")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing markdown files")
    parser.add_argument("--write-empty-on-error", action="store_true", help="Write an empty .md if a page fails")
    parser.add_argument("--min-success-rate", type=float, default=0.95, help="Fail if success rate is below this value")
    parser.add_argument("--device", default="auto", help="auto, cuda, cpu, gpu:0, etc., depending on engine")
    parser.add_argument("--model-id", help="Override model ID for LightOn or Chandra")
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--paddle-pipeline-version", default="v1.5")
    parser.add_argument("--paddle-engine", default="", help="Optional PaddleOCR engine, e.g. paddle or transformers")
    parser.add_argument("--chandra-prompt-type", default="ocr_layout")
    parser.add_argument("--mineru-backend", choices=("vllm-engine", "transformers"), default="vllm-engine")
    parser.add_argument("--mineru-image-analysis", action="store_true", help="Enable MinerU image/chart analysis")
    parser.add_argument("--mineru-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--mineru-gpu-memory-utilization", type=float)
    parser.add_argument("--mineru-max-model-len", type=int)
    parser.add_argument("--kdl-backend", choices=("transformers", "openai"), default="transformers")
    parser.add_argument("--kdl-prompt", default=DEFAULT_KDL_PROMPT)
    parser.add_argument("--kdl-api-base", help="OpenAI-compatible server base URL, e.g. http://127.0.0.1:8000")
    parser.add_argument("--kdl-api-key")
    parser.add_argument("--kdl-served-model-name", default="kdl-frontier-parser-nano")
    parser.add_argument("--kdl-timeout", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "inference_log.jsonl"
    images = collect_images(args.input_dir, args.filelist, args.manifest, args.limit)

    run_started = time.time()
    converter = build_converter(args)
    converter_init_seconds = time.time() - run_started
    total = len(images)
    successes = 0
    failures = 0
    skipped = 0
    page_seconds_sum = 0.0

    for idx, image_path in enumerate(images, start=1):
        target_path = output_path_for(image_path, args.output_dir)
        if target_path.exists() and target_path.stat().st_size > 0 and not args.overwrite:
            successes += 1
            skipped += 1
            write_log(log_path, {"image": image_path.name, "status": "skipped", "output": str(target_path)})
            print(f"[{idx}/{total}] skip {image_path.name}")
            continue

        started = time.time()
        print(f"[{idx}/{total}] {args.engine} {image_path.name}", flush=True)
        try:
            markdown = converter(image_path)
            target_path.write_text(markdown, encoding="utf-8")
            page_seconds = time.time() - started
            page_seconds_sum += page_seconds
            successes += 1
            write_log(
                log_path,
                {
                    "image": image_path.name,
                    "status": "success",
                    "output": str(target_path),
                    "seconds": round(page_seconds, 3),
                    "chars": len(markdown),
                },
            )
        except Exception as exc:
            page_seconds = time.time() - started
            page_seconds_sum += page_seconds
            failures += 1
            if args.write_empty_on_error:
                target_path.write_text("", encoding="utf-8")
            write_log(
                log_path,
                {
                    "image": image_path.name,
                    "status": "failed",
                    "output": str(target_path) if target_path.exists() else "",
                    "seconds": round(page_seconds, 3),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            print(f"ERROR {image_path.name}: {type(exc).__name__}: {exc}", flush=True)

    wall_seconds = time.time() - run_started
    processed = successes + failures - skipped
    success_rate = successes / total
    summary = {
        "engine": args.engine,
        "total": total,
        "successes": successes,
        "failures": failures,
        "skipped": skipped,
        "success_rate": success_rate,
        "converter_init_seconds": round(converter_init_seconds, 3),
        "page_seconds_sum": round(page_seconds_sum, 3),
        "wall_seconds": round(wall_seconds, 3),
        "pages_per_minute": None if wall_seconds <= 0 else round((successes / wall_seconds) * 60.0, 6),
        "processed_pages_per_minute": None if page_seconds_sum <= 0 else round((processed / page_seconds_sum) * 60.0, 6),
        "output_dir": str(args.output_dir),
        "log": str(log_path),
    }
    (args.output_dir / "inference_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)

    if success_rate < args.min_success_rate:
        raise SystemExit(
            f"{args.engine} success rate {success_rate:.3f} is below --min-success-rate {args.min_success_rate:.3f}"
        )


if __name__ == "__main__":
    main()
