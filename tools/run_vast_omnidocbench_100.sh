#!/usr/bin/env bash
set -euo pipefail

# Vast AI runner for a 100-page diverse OmniDocBench evaluation.
#
# Required environment variables:
#   GT_JSON=/abs/path/to/OmniDocBench.json
#   IMAGES_ROOT=/abs/path/to/images
#
# Common optional variables:
#   RUN_ROOT=$PWD/runs/omnidocbench_100_YYYYmmdd_HHMMSS
#   ENGINES="marker docling paddleocr_vl15 lighton_bbox chandra"
#   ENGINE=marker      # shortcut for running exactly one model
#   SAMPLE_COUNT=100
#   EVAL_WORKERS=4
#   NO_CDM=auto       # auto, 0, or 1
#   SKIP_COMPLETED=1
#   FORCE_INFERENCE=0
#   FORCE_EVAL=0
#   PREPARE_ONLY=0
#   CONTINUE_ON_FAILURE=0

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GT_JSON="${GT_JSON:-}"
IMAGES_ROOT="${IMAGES_ROOT:-}"
SAMPLE_COUNT="${SAMPLE_COUNT:-100}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/runs/omnidocbench_100_$RUN_ID}"
ENGINES="${ENGINES:-marker docling paddleocr_vl15 lighton_bbox chandra}"
ENGINE="${ENGINE:-}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
NO_CDM="${NO_CDM:-auto}"
COPY_MODE="${COPY_MODE:-symlink}"
PREPARE_SUBSET="${PREPARE_SUBSET:-auto}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"
RUN_INFERENCE="${RUN_INFERENCE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
FORCE_INFERENCE="${FORCE_INFERENCE:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-0}"
WRITE_EMPTY_ON_ERROR="${WRITE_EMPTY_ON_ERROR:-1}"
MIN_SUCCESS_RATE="${MIN_SUCCESS_RATE:-0.90}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"

EVAL_PYTHON="${EVAL_PYTHON:-python3}"
PY_MARKER="${PY_MARKER:-python3}"
PY_DOCLING="${PY_DOCLING:-python3}"
PY_PADDLE="${PY_PADDLE:-python3}"
PY_LIGHTON="${PY_LIGHTON:-python3}"
PY_CHANDRA="${PY_CHANDRA:-python3}"

MARKER_TORCH_DEVICE="${MARKER_TORCH_DEVICE:-cuda}"
PADDLE_DEVICE="${PADDLE_DEVICE:-gpu:0}"
PADDLE_ENGINE="${PADDLE_ENGINE:-paddle}"
PADDLE_PIPELINE_VERSION="${PADDLE_PIPELINE_VERSION:-v1.5}"
LIGHTON_MODEL_ID="${LIGHTON_MODEL_ID:-lightonai/LightOnOCR-2-1B-bbox}"
LIGHTON_DEVICE="${LIGHTON_DEVICE:-auto}"
CHANDRA_MODEL_ID="${CHANDRA_MODEL_ID:-datalab-to/chandra-ocr-2}"
CHANDRA_DEVICE="${CHANDRA_DEVICE:-auto}"
CHANDRA_PROMPT_TYPE="${CHANDRA_PROMPT_TYPE:-ocr_layout}"

if [[ -n "$ENGINE" ]]; then
  ENGINES="$ENGINE"
fi

if [[ -z "$GT_JSON" || -z "$IMAGES_ROOT" ]]; then
  echo "Set GT_JSON and IMAGES_ROOT before running." >&2
  echo "Example: GT_JSON=/workspace/OmniDocBench/OmniDocBench.json IMAGES_ROOT=/workspace/OmniDocBench/images bash tools/run_vast_omnidocbench_100.sh" >&2
  exit 2
fi

if [[ ! -f "$GT_JSON" ]]; then
  echo "GT_JSON not found: $GT_JSON" >&2
  exit 2
fi
if [[ ! -d "$IMAGES_ROOT" ]]; then
  echo "IMAGES_ROOT not found: $IMAGES_ROOT" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT"
SUBSET_ROOT="$RUN_ROOT/subset"
SUBSET_JSON="$SUBSET_ROOT/OmniDocBench_${SAMPLE_COUNT}_diverse.json"
SUBSET_IMAGES="$SUBSET_ROOT/images"
MANIFEST="$SUBSET_ROOT/manifest.csv"
PRED_ROOT="$RUN_ROOT/predictions"
EVAL_ROOT="$RUN_ROOT/eval"
CONFIG_ROOT="$RUN_ROOT/configs"
CHECKPOINT_ROOT="$RUN_ROOT/checkpoints"
mkdir -p "$PRED_ROOT" "$EVAL_ROOT" "$CONFIG_ROOT" "$CHECKPOINT_ROOT"

echo "RUN_ROOT=$RUN_ROOT"
echo "GT_JSON=$GT_JSON"
echo "IMAGES_ROOT=$IMAGES_ROOT"
echo "ENGINES=$ENGINES"
echo "CHECKPOINT_ROOT=$CHECKPOINT_ROOT"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

epoch_seconds() {
  date -u +"%s"
}

subset_ready() {
  [[ -f "$SUBSET_JSON" && -f "$MANIFEST" && -d "$SUBSET_IMAGES" ]]
}

prepare_subset() {
  python3 tools/prepare_omnidocbench_subset.py \
    --gt-json "$GT_JSON" \
    --images-root "$IMAGES_ROOT" \
    --output-root "$SUBSET_ROOT" \
    --sample-count "$SAMPLE_COUNT" \
    --copy-mode "$COPY_MODE" \
    --overwrite
  {
    echo "timestamp=$(timestamp)"
    echo "subset_json=$SUBSET_JSON"
    echo "manifest=$MANIFEST"
    echo "images=$SUBSET_IMAGES"
  } > "$CHECKPOINT_ROOT/subset.done"
}

case "$PREPARE_SUBSET" in
  1|true|yes)
    prepare_subset
    ;;
  0|false|no)
    if ! subset_ready; then
      echo "PREPARE_SUBSET=0 but subset files are missing under $SUBSET_ROOT" >&2
      exit 2
    fi
    ;;
  auto)
    if subset_ready && [[ "$SKIP_COMPLETED" == "1" ]]; then
      echo "========== SUBSET skip: already prepared =========="
    else
      prepare_subset
    fi
    ;;
  *)
    echo "Unknown PREPARE_SUBSET value: $PREPARE_SUBSET" >&2
    exit 2
    ;;
esac

if [[ "$PREPARE_ONLY" == "1" ]]; then
  echo "DONE subset: $SUBSET_ROOT"
  exit 0
fi

resolve_no_cdm() {
  if [[ "$NO_CDM" == "0" || "$NO_CDM" == "1" ]]; then
    echo "$NO_CDM"
    return
  fi
  if command -v pdflatex >/dev/null 2>&1 && command -v kpsewhich >/dev/null 2>&1 && command -v gs >/dev/null 2>&1 && command -v magick >/dev/null 2>&1; then
    echo "0"
  else
    echo "1"
  fi
}

RESOLVED_NO_CDM="$(resolve_no_cdm)"
if [[ "$NO_CDM" == "auto" && "$RESOLVED_NO_CDM" == "1" ]]; then
  echo "NO_CDM=auto selected --no-cdm because TeX/ImageMagick/Ghostscript tools were not all found."
fi

run_engine() {
  local engine="$1"
  local engine_py="python3"
  local pred_dir="$PRED_ROOT/$engine"
  local eval_dir="$EVAL_ROOT/$engine"
  local config_path="$CONFIG_ROOT/${engine}_end2end.yaml"
  local prefix="${engine}_quick_match"
  local infer_done="$CHECKPOINT_ROOT/${engine}.inference.done"
  local eval_done="$CHECKPOINT_ROOT/${engine}.eval.done"
  mkdir -p "$pred_dir" "$eval_dir"

  local args=(
    tools/omnidocbench_batch_infer.py
    --engine "$engine"
    --input-dir "$SUBSET_IMAGES"
    --manifest "$MANIFEST"
    --output-dir "$pred_dir"
    --min-success-rate "$MIN_SUCCESS_RATE"
    --max-new-tokens "$MAX_NEW_TOKENS"
  )
  if [[ "$WRITE_EMPTY_ON_ERROR" == "1" ]]; then
    args+=(--write-empty-on-error)
  fi
  if [[ "$FORCE_INFERENCE" == "1" ]]; then
    args+=(--overwrite)
  fi

  case "$engine" in
    marker)
      engine_py="$PY_MARKER"
      ;;
    docling)
      engine_py="$PY_DOCLING"
      ;;
    paddleocr_vl15)
      engine_py="$PY_PADDLE"
      args+=(--device "$PADDLE_DEVICE" --paddle-engine "$PADDLE_ENGINE" --paddle-pipeline-version "$PADDLE_PIPELINE_VERSION")
      ;;
    lighton_bbox)
      engine_py="$PY_LIGHTON"
      args+=(--model-id "$LIGHTON_MODEL_ID" --device "$LIGHTON_DEVICE")
      ;;
    chandra)
      engine_py="$PY_CHANDRA"
      args+=(--model-id "$CHANDRA_MODEL_ID" --device "$CHANDRA_DEVICE" --chandra-prompt-type "$CHANDRA_PROMPT_TYPE")
      ;;
    *)
      echo "Unknown engine: $engine" >&2
      return 2
      ;;
  esac

  if [[ "$RUN_INFERENCE" == "1" ]]; then
    if [[ "$SKIP_COMPLETED" == "1" && "$FORCE_INFERENCE" != "1" && -f "$infer_done" ]]; then
      echo "========== INFER $engine skip: checkpoint exists =========="
    else
      echo "========== INFER $engine =========="
      if [[ "$engine" == "marker" ]]; then
        env TORCH_DEVICE="$MARKER_TORCH_DEVICE" "$engine_py" "${args[@]}" || return $?
      else
        "$engine_py" "${args[@]}" || return $?
      fi
    fi
  fi

  local md_count
  md_count="$(find "$pred_dir" -maxdepth 1 -name '*.md' | wc -l | tr -d ' ')"
  if [[ "$md_count" != "$SAMPLE_COUNT" ]]; then
    echo "$engine produced $md_count markdown files, expected $SAMPLE_COUNT" >&2
    return 3
  fi
  {
    echo "timestamp=$(timestamp)"
    echo "engine=$engine"
    echo "pred_dir=$pred_dir"
    echo "md_count=$md_count"
  } > "$infer_done"

  if [[ "$RUN_EVAL" == "1" ]]; then
    if [[ "$SKIP_COMPLETED" == "1" && "$FORCE_EVAL" != "1" && -f "$eval_done" ]]; then
      echo "========== EVAL $engine skip: checkpoint exists =========="
    else
      echo "========== EVAL $engine =========="
      local cdm_flag=()
      if [[ "$RESOLVED_NO_CDM" == "1" ]]; then
        cdm_flag=(--no-cdm)
      fi
      local eval_started
      local eval_finished
      local eval_seconds
      eval_started="$(epoch_seconds)"
      "$EVAL_PYTHON" skills/scripts/generate_end2end_config.py \
        --gt "$SUBSET_JSON" \
        --pred "$pred_dir" \
        --out "$config_path" \
        --workers "$EVAL_WORKERS" \
        "${cdm_flag[@]}" || return $?
      "$EVAL_PYTHON" pdf_validation.py --config "$config_path" 2>&1 | tee "$eval_dir/eval.log" || return $?
      "$EVAL_PYTHON" skills/scripts/parse_results.py "$REPO_ROOT/result" --prefix "$prefix" --pred "$pred_dir" | tee "$eval_dir/report.md" || return $?
      find "$REPO_ROOT/result" -maxdepth 1 -type f -name "${prefix}*" -exec cp {} "$eval_dir/" \; || return $?
      eval_finished="$(epoch_seconds)"
      eval_seconds="$((eval_finished - eval_started))"
      {
        echo "{"
        echo "  \"engine\": \"$engine\","
        echo "  \"eval_started\": $eval_started,"
        echo "  \"eval_finished\": $eval_finished,"
        echo "  \"eval_seconds\": $eval_seconds,"
        echo "  \"no_cdm\": $RESOLVED_NO_CDM,"
        echo "  \"workers\": $EVAL_WORKERS"
        echo "}"
      } > "$eval_dir/eval_timing.json"
      {
        echo "timestamp=$(timestamp)"
        echo "engine=$engine"
        echo "eval_dir=$eval_dir"
        echo "config=$config_path"
        echo "no_cdm=$RESOLVED_NO_CDM"
        echo "eval_seconds=$eval_seconds"
      } > "$eval_done"
    fi
  fi
}

FAILED=()
for engine in $ENGINES; do
  if ! run_engine "$engine"; then
    FAILED+=("$engine")
    if [[ "$CONTINUE_ON_FAILURE" != "1" ]]; then
      echo "Stopping after $engine failure. Set CONTINUE_ON_FAILURE=1 to keep going." >&2
      exit 1
    fi
  fi
done

if [[ "${#FAILED[@]}" -gt 0 ]]; then
  echo "Completed with failed engines: ${FAILED[*]}" >&2
  exit 1
fi

echo "DONE: $RUN_ROOT"
