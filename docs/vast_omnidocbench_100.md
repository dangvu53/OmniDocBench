# Vast AI 100-Page OmniDocBench Runbook

This run path evaluates a deterministic 100-page OmniDocBench subset across:

- Marker
- Docling
- PaddleOCR-VL-1.5
- LightOnOCR-2-1B-bbox
- Chandra

It runs one engine at a time so a single 32 GB GPU is not shared by multiple loaded models.

## Inputs

You need the full OmniDocBench JSON and page images on the Vast instance:

```bash
export GT_JSON=/workspace/OmniDocBench/OmniDocBench.json
export IMAGES_ROOT=/workspace/OmniDocBench/images
```

The subset builder writes derived files only under `RUN_ROOT`; it does not edit raw data.

## Recommended Environments

The model packages are easier to keep stable in separate virtual environments. These commands are for a CUDA-capable Vast container where the NVIDIA driver supports CUDA 12.x wheels.

```bash
cd /workspace/OmniDocBench

python3 -m venv ~/venvs/omni-eval
source ~/venvs/omni-eval/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
deactivate

python3 -m venv ~/venvs/marker-docling
source ~/venvs/marker-docling/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install marker-pdf docling
deactivate

python3 -m venv ~/venvs/paddle-vl
source ~/venvs/paddle-vl/bin/activate
python -m pip install --upgrade pip
python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
python -m pip install -U "paddleocr[doc-parser]"
deactivate

python3 -m venv ~/venvs/lighton
source ~/venvs/lighton/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install git+https://github.com/huggingface/transformers pillow accelerate pypdfium2
deactivate

python3 -m venv ~/venvs/chandra
source ~/venvs/chandra/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install "chandra-ocr[hf]"
deactivate
```

If your Paddle install fails, use the current PaddleOCR install matrix for your CUDA/runtime combination. The runner uses the full PaddleOCR-VL pipeline with `--paddle-pipeline-version v1.5`, not the standalone VLM component.

## Run

```bash
cd /workspace/OmniDocBench

GT_JSON=/workspace/OmniDocBench/OmniDocBench.json \
IMAGES_ROOT=/workspace/OmniDocBench/images \
EVAL_PYTHON=~/venvs/omni-eval/bin/python \
PY_MARKER=~/venvs/marker-docling/bin/python \
PY_DOCLING=~/venvs/marker-docling/bin/python \
PY_PADDLE=~/venvs/paddle-vl/bin/python \
PY_LIGHTON=~/venvs/lighton/bin/python \
PY_CHANDRA=~/venvs/chandra/bin/python \
EVAL_WORKERS=4 \
bash tools/run_vast_omnidocbench_100.sh
```

Outputs land in:

```text
runs/omnidocbench_100_<timestamp>/
  subset/
  predictions/<engine>/
  eval/<engine>/
  configs/
  checkpoints/
```

## One Model at a Time

Use a fixed `RUN_ROOT` and set `ENGINE` to benchmark exactly one model. The subset is prepared once and reused by later model runs:

```bash
export RUN_ROOT=/workspace/OmniDocBench/runs/omnidocbench_100_manual

ENGINE=marker bash tools/run_vast_omnidocbench_100.sh
ENGINE=docling bash tools/run_vast_omnidocbench_100.sh
ENGINE=paddleocr_vl15 bash tools/run_vast_omnidocbench_100.sh
ENGINE=lighton_bbox bash tools/run_vast_omnidocbench_100.sh
ENGINE=chandra bash tools/run_vast_omnidocbench_100.sh
```

Each engine gets its own prediction and eval folders, so a failure in one does not affect the others.

## Checkpointing

The runner writes checkpoint files after each successful stage:

```text
checkpoints/subset.done
checkpoints/<engine>.inference.done
checkpoints/<engine>.eval.done
```

Re-run the same command with the same `RUN_ROOT` to resume. Existing non-empty `.md` files are skipped; missing or empty files are retried. If a process dies halfway through a model, no success checkpoint is written for that stage.

Useful controls:

```bash
# Prepare/reuse the 100-page subset without running any model
PREPARE_ONLY=1 bash tools/run_vast_omnidocbench_100.sh

# Re-run only evaluation for an engine using existing predictions
ENGINE=marker RUN_INFERENCE=0 FORCE_EVAL=1 bash tools/run_vast_omnidocbench_100.sh

# Force regeneration of predictions for one model
ENGINE=marker FORCE_INFERENCE=1 FORCE_EVAL=1 bash tools/run_vast_omnidocbench_100.sh

# In a multi-engine run, keep going after a model fails
CONTINUE_ON_FAILURE=1 bash tools/run_vast_omnidocbench_100.sh

# Disable checkpoint skipping entirely
SKIP_COMPLETED=0 bash tools/run_vast_omnidocbench_100.sh
```

## CDM Formula Metric

`NO_CDM=auto` is the default. It enables CDM only when `pdflatex`, `kpsewhich`, `gs`, and ImageMagick are visible; otherwise it evaluates text, table, and reading order without CDM to avoid runtime crashes.

For full OmniDocBench-style overall scoring, install TeX/CJK, Ghostscript, and ImageMagick, then run with:

```bash
NO_CDM=0 bash tools/run_vast_omnidocbench_100.sh
```

For a faster smoke pass:

```bash
NO_CDM=1 SAMPLE_COUNT=10 ENGINES="marker docling" bash tools/run_vast_omnidocbench_100.sh
```

## Model IDs

Defaults:

```bash
LIGHTON_MODEL_ID=lightonai/LightOnOCR-2-1B-bbox
CHANDRA_MODEL_ID=datalab-to/chandra-ocr-2
```

Datalab's current public Chandra OCR 2 checkpoint is `datalab-to/chandra-ocr-2`. If you have a private or alternate Chandra-9B checkpoint, point the runner at it:

```bash
CHANDRA_MODEL_ID=your-org/your-chandra-9b-checkpoint bash tools/run_vast_omnidocbench_100.sh
```

Set `HF_TOKEN` in the shell before running if a model is gated or private.

## Recovery

The runner skips already generated non-empty `.md` files and completed checkpoints. Re-running the same `RUN_ROOT` resumes inference and evaluation:

```bash
RUN_ROOT=/workspace/OmniDocBench/runs/omnidocbench_100_20260616_170000 \
bash tools/run_vast_omnidocbench_100.sh
```

If one engine fails but you want the rest to continue:

```bash
CONTINUE_ON_FAILURE=1 bash tools/run_vast_omnidocbench_100.sh
```
