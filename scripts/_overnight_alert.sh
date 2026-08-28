#!/bin/bash
# Emit a sentinel when a monitored run needs attention. Paired with scripts/monitor_runs.py, which
# does the real analysis; this only decides WHEN to look.
set -uo pipefail
cd /mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory || exit 1
seen_evals=""
for i in $(seq 1 4000); do
  for j in "$@"; do
    f=$(ls logs/slurm/*"$j".out 2>/dev/null | head -1); [ -s "$f" ] || continue
    if grep -qE "RUN FAILED|CudaWorkerError|OutOfMemoryError|GGML_ASSERT|exit=-6" "$f"; then
      echo "ALERT_FAIL $j"
    fi
    if ! squeue -h -j "$j" >/dev/null 2>&1 || [ -z "$(squeue -h -j "$j" -o %T 2>/dev/null)" ]; then
      echo "ALERT_ENDED $j"
    fi
    n=$(grep -c "Score: " "$f" 2>/dev/null || echo 0)
    key="$j:$n"
    case " $seen_evals " in *" $key "*) ;; *)
      [ "$n" -gt 0 ] && echo "ALERT_NEW_EVAL $j (eval #$n)"
      seen_evals="$seen_evals $key" ;;
    esac
  done
  [ $((i % 40)) -eq 0 ] && echo "tick $((i/2))min :: $(squeue -u evanly -h -o '%i=%T' | tr '\n' ' ')"
  sleep 30
done
