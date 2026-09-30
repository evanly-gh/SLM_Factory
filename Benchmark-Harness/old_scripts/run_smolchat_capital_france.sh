#!/usr/bin/env bash
# SmolChat (llama.cpp) CPU vs OpenCL on the four Table 5 models (Q4_K_M), one prompt:
# "What is the capital of France?" (one_question.txt). Per model/backend: 1 discarded warm-up
# + 3 recorded trials (--trials 3). Output length = app default DEFAULT_MAX_TOKENS (256).
# Results: smolchat_capital_france_results/<model>_<backend>.json (+ run.log)
set -uo pipefail

# Archived in old_scripts/; paths resolve from the Benchmark-Harness dir one level up.
HERE="$(cd "$(dirname "$0")/.." && pwd)"
APK_DIR="$HERE/../SmolChat-Android/app/build/outputs/apk"
MODELS_DIR="$HERE/../gguf_models_verified"
OUT="$HERE/new_mnn_paper_table_5_results/smolchat_capital_france_results"
MODELS="qwen2.5-1.5b-q4_k_m llama-3.2-1b-q4_k_m llama-3.2-3b-q4_k_m qwen2.5-7b-q4_k_m"
mkdir -p "$OUT"
cd "$HERE"

for backend in cpu opencl; do
  if [[ $backend == cpu ]]; then ngl=0; else ngl=99; fi
  echo "=== installing $backend APK"
  adb shell am force-stop io.shubham0204.smollmandroid
  adb install -r "$APK_DIR/$backend/debug/app-$backend-debug.apk" | tail -1
  for m in $MODELS; do
    echo "=== $m [$backend] $(date '+%H:%M:%S')"
    python3 run_autobench.py --model "$MODELS_DIR/$m.gguf" --questions one_question.txt \
      --trials 3 --timeout 600 --n-gpu-layers $ngl --output "$OUT/${m}_${backend}.json"
  done
done

echo "=== restoring cpu APK"
adb shell am force-stop io.shubham0204.smollmandroid
adb install -r "$APK_DIR/cpu/debug/app-cpu-debug.apk" | tail -1
echo "=== done $(date '+%H:%M:%S')"
