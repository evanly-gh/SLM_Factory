#!/usr/bin/env bash
# run_model_sweep.sh - the whole benchmark, unattended.
#
# For each model:  fetch + convert + quantize  ->  push to the phone  ->  benchmark every engine x backend x quant
#                  ->  save results  ->  ranked summary
#   SmolChat (llama.cpp GGUF): Q4_K_M, Q8_0, BF16    MNN: 4-bit (lm_head 8-bit), 8-bit, 16-bit (fp16)
#   backends: CPU and OpenCL on both engines            = 12 configurations per model
# Every configuration uses the per-question protocol: for each of the 10 built-in questions, gate (battery
# temp + CPU clocks), force-stop the app, evict the model from the page cache, then 3 back-to-back runs
# (run 1 cold, last run reported), with Perfetto energy recording (valid only when unplugged).
#
# Usage:
#   ./run_model_sweep.sh                                        # the nine default models
#   ./run_model_sweep.sh --models "HuggingFaceTB/SmolLM2-135M"  # any Hugging Face model ids
#   ./run_model_sweep.sh --plan                                 # show what would run, then exit
#   options: --gate-rise 2  --rest-between-configs 120
# Resumable: a configuration whose result file exists is skipped, and so are finished conversions. A
# configuration that fails twice is logged and skipped so the rest continues.
# Results: sweep_results/<model>/{smolchat,mnn}_<quant>_<backend>.json + sweep.log + SUMMARY.txt,
#          sweep_results/ALL_SUMMARY.txt, sweep_results/sweep.log
# Afterwards, choose configurations by constraint:
#   python3 ../mnn-benchmark-harness/compare_engines.py --where "decode_tps>=30" --where "cold_start_ms<=500"

HERE="$(cd "$(dirname "$0")" && pwd)"
SMOL="$(cd "$HERE/.." && pwd)"   # repo root: Model-Conversion/, SmolChat-Android/ (vendored llama.cpp), Benchmark-Harness/
PY="$SMOL/.venv/bin/python"                                   # repo .venv (requirements.txt)
[[ -x "$PY" ]] || PY="$SMOL/.venv/Scripts/python.exe"         # Windows (Git Bash)
MNN_DEVICE=/data/local/tmp/mnn_models
APK_DIR="$SMOL/SmolChat-Android/app/build/outputs/apk"
RESULTS="$HERE/sweep_results"
# Small to large, so a problem with a big model does not block the small ones.
MODELS="HuggingFaceTB/SmolLM2-135M HuggingFaceTB/SmolLM2-360M google/gemma-3-270m-it \
Qwen/Qwen3-0.6B Qwen/Qwen3.5-0.8B Qwen/Qwen3-1.7B Qwen/Qwen3.5-2B \
Qwen/Qwen3-4B-Instruct-2507 Qwen/Qwen3.5-4B"
GATE_RISE=2   # wait only while the battery is > 2 C above its temperature when the configuration started
REST_BETWEEN=120
PLAN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models) MODELS="$2"; shift ;;
    --gate-rise) GATE_RISE="$2"; shift ;;
    --rest-between-configs) REST_BETWEEN="$2"; shift ;;
    --plan) PLAN=1 ;;
    -h|--help) sed -n 2,24p "$0"; exit 0 ;;
    *) echo "unknown argument: $1 (see --help)"; exit 1 ;;
  esac
  shift
done

mkdir -p "$RESULTS"
LOG="$RESULTS/sweep.log"
LOG_M="$LOG"
FAILED=""
# gate-timeout: if the phone will not cool within the limit in 5 min, run anyway (flagged in the result).
# The limit is relative (baseline + rise): with the screen on and USB plugged in this phone idles at ~28 C
# (room temperature and charging), so an absolute 28 C limit made 3 of 10 questions wait the full timeout.
COMMON=(--runs-per-question 3 --gate-temp-rise "$GATE_RISE" --gate-timeout 300 --rest-seconds 10 --timeout 300 --energy)

log() { echo "=== $(date '+%H:%M:%S') $*" | tee -a "$LOG"; }
prefix_of() { basename "$1" | tr 'A-Z_ ' 'a-z--'; }   # HuggingFaceTB/SmolLM2-135M -> smollm2-135m

need_phone() {
  local i
  for i in 1 2 3 4 5 6 7 8 9 10; do
    [[ "$(adb get-state 2>/dev/null)" == "device" ]] && return 0
    log "phone not connected - waiting (attempt $i/10)"; sleep 60
  done
  log "ABORT: phone not connected"; exit 1
}

install_smolchat() {  # cpu | opencl
  adb shell am force-stop io.shubham0204.smollmandroid
  adb install -r "$APK_DIR/$1/debug/app-$1-debug.apk" | tail -1
}

convert_model() {  # hf-id prefix
  local hf="$1" m="$2"
  local gg="$SMOL/Model-Conversion/output-$m" mn="$HERE/../Model-Conversion/mnn-output-$m"
  if [[ -f "$gg/$m-q4_k_m.gguf" && -f "$gg/$m-q8_0.gguf" && -f "$gg/$m-bf16.gguf" ]]; then
    log "GGUF files for $m already exist - skipping conversion"
  else
    log "converting $hf -> GGUF (BF16, Q4_K_M, Q8_0)"
    (cd "$SMOL/Model-Conversion" && "$PY" convert_to_gguf.py --model "$hf" --quant ALL --output "output-$m") >>"$LOG_M" 2>&1
    [[ -f "$gg/$m-q4_k_m.gguf" && -f "$gg/$m-q8_0.gguf" && -f "$gg/$m-bf16.gguf" ]] || { log "FAILED: GGUF conversion of $hf"; return 1; }
  fi
  if [[ -f "$mn/$m-mnn-q4/llm.mnn.weight" && -f "$mn/$m-mnn-q8/llm.mnn.weight" && -f "$mn/$m-mnn-q16/llm.mnn.weight" ]]; then
    log "MNN exports for $m already exist - skipping conversion"
  else
    log "converting $hf -> MNN (4/8/16-bit)"
    (cd "$HERE/../Model-Conversion" && "$PY" convert_to_mnn.py --model "$hf" --output "mnn-output-$m" --quant ALL) >>"$LOG_M" 2>&1
    [[ -f "$mn/$m-mnn-q4/llm.mnn.weight" && -f "$mn/$m-mnn-q8/llm.mnn.weight" && -f "$mn/$m-mnn-q16/llm.mnn.weight" ]] || { log "FAILED: MNN conversion of $hf"; return 1; }
  fi
}

push_mnn() {  # prefix
  local m="$1" b
  adb shell mkdir -p "$MNN_DEVICE"
  for b in 4 8 16; do
    adb push "$HERE/../Model-Conversion/mnn-output-$m/$m-mnn-q$b" "$MNN_DEVICE/" >>"$LOG_M" 2>&1 || return 1
  done
  log "pushed MNN models for $m to $MNN_DEVICE"
}

smol_cmd() {  # gguf n_gpu_layers out
  (cd "$SMOL/Benchmark-Harness" && "$PY" -u run_autobench.py --model "$1" --n-gpu-layers "$2" --no-eos-suppress "${COMMON[@]}" --output "$3")
}

mnn_cmd() {  # model-path backend out
  (cd "$HERE/../mnn-benchmark-harness" && "$PY" -u run_mnn_autobench.py --model-path "$1" --backend-type "$2" "${COMMON[@]}" --output "$3")
}

run_config() {  # label out-json command args...
  local label="$1" out="$2" attempt
  shift 2
  if [[ -f "$out" ]]; then log "skip: $label (result exists)"; return; fi
  if [[ $PLAN == 1 ]]; then log "would run: $label"; return; fi
  need_phone
  log "START $label"
  for attempt in 1 2; do
    "$@" >>"$LOG_M" 2>&1
    if [[ -f "$out" ]]; then log "DONE  $label"; sleep "$REST_BETWEEN"; return; fi
    log "attempt $attempt failed: $label"; sleep 60
  done
  log "FAILED $label (no result file after 2 attempts)"
  FAILED="$FAILED"$'\n'"  $label"
}

log "sweep start | models: $MODELS | gate: battery <= start+${GATE_RISE}C | plan=$PLAN"
[[ $PLAN == 1 ]] || need_phone

for hf in $MODELS; do
  m="$(prefix_of "$hf")"
  OUT="$RESULTS/$m"; mkdir -p "$OUT"; LOG_M="$OUT/sweep.log"
  GG="$SMOL/Model-Conversion/output-$m"
  log "MODEL $hf  ($m)"
  if [[ $PLAN == 0 ]]; then
    convert_model "$hf" "$m" || { FAILED="$FAILED"$'\n'"  $m: conversion"; continue; }
    push_mnn "$m" || { log "FAILED: pushing MNN models for $m"; FAILED="$FAILED"$'\n'"  $m: push"; continue; }
    install_smolchat cpu
  fi
  for q in "q4_k_m q4" "q8_0 q8" "bf16 bf16"; do
    set -- $q
    run_config "$m SmolChat $2 cpu" "$OUT/smolchat_${2}_cpu.json" smol_cmd "$GG/$m-$1.gguf" 0 "$OUT/smolchat_${2}_cpu.json"
  done
  [[ $PLAN == 0 ]] && install_smolchat opencl
  for q in "q4_k_m q4" "q8_0 q8" "bf16 bf16"; do
    set -- $q
    run_config "$m SmolChat $2 opencl" "$OUT/smolchat_${2}_opencl.json" smol_cmd "$GG/$m-$1.gguf" 99 "$OUT/smolchat_${2}_opencl.json"
  done
  for backend in cpu opencl; do
    for b in 4 8 16; do
      run_config "$m MNN q$b $backend" "$OUT/mnn_q${b}_${backend}.json" mnn_cmd "$MNN_DEVICE/$m-mnn-q$b" "$backend" "$OUT/mnn_q${b}_${backend}.json"
    done
  done
  if [[ $PLAN == 0 ]]; then
    install_smolchat cpu   # leave the CPU build installed
    "$PY" "$HERE/../mnn-benchmark-harness/compare_engines.py" --results-dir "$OUT" --list > "$OUT/SUMMARY.txt" 2>&1
    log "MODEL $m finished - summary: $OUT/SUMMARY.txt"
  fi
done

if [[ $PLAN == 0 ]]; then
  "$PY" "$HERE/../mnn-benchmark-harness/compare_engines.py" --results-dir "$RESULTS" --list > "$RESULTS/ALL_SUMMARY.txt" 2>&1
fi
log "ALL DONE | failures:${FAILED:- none}"
