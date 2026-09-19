#!/bin/bash
# run_pipeline.sh — Runs both engines end to end, as ONE command, while
# keeping Stage 1 (MNN) and Stage 2 (GGUF) as two genuinely separate
# top-level processes.
#
# INDEPENDENT FULL RUNS, NOT MNN-FIRST/GGUF-FALLBACK ANYMORE: both stages now
# run the ENTIRE question set against every model+quant in --fit-report,
# completely independently - no data passed between them at all. Stage 2 is
# invoked with --full-run (run_fallback_agent_gguf.py's new mode), so it
# reads --fit-report/--questions directly instead of Stage 1's
# needs_gguf_fallback.json - there's no handoff left to depend on, so both
# stages run unconditionally regardless of what the other one found.
# (run_fallback_agent_gguf.py's old fallback-file mode - reading
# needs_gguf_fallback.json, only covering what Stage 1 flagged - is still
# fully intact in that script; this wrapper just no longer invokes it that
# way. Run it directly, without --full-run, if you need that mode again.)
#
# WHY TWO SEPARATE PROCESSES (still applies, unchanged): an earlier
# single-process design ran GGUF/SmolChat as a subprocess NESTED inside a
# long-running Python process, and that nesting had a confirmed, unresolved
# hang that survived 7 separate fix attempts across fundamentally different
# mechanisms (stdin handling, an MNN force-stop, uncaptured output with
# polling, session isolation, os.system() invocation, an ADB server restart)
# - all failed identically. The one thing that reliably worked every time
# was a fresh, non-nested, top-level process invocation. Stage 1 fully
# exits - this script waits for it, not backgrounds it - before Stage 2
# starts, so Stage 2's run_autobench.py subprocess is never nested inside
# anything but this plain shell script. Stage 2 no longer depends on Stage
# 1's output file, but this script still runs them sequentially rather than
# in parallel, to keep that guarantee simple and unchanged.
#
# The SAME arguments are passed through to both stages (each stage's own
# argparse uses parse_known_args(), so it just ignores flags meant for the
# other stage). Both now take the same --fit-report/--questions, so one
# shared argument list works for both, plus the added --full-run flag for
# Stage 2.
#
# Usage:
#   ./run_pipeline.sh --fit-report model_fit_report_q4q8.json --questions eval_questions_phase3.json
#   ./run_pipeline.sh --fit-report ... --questions ... --timeout 300 --no-think

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "======================================================================"
echo "STAGE 1: MNN evaluation (run_fallback_agent_mnn.py) - full question set"
echo "======================================================================"
# "|| STAGE1_EXIT=$?" instead of a bare call + separate "STAGE1_EXIT=$?":
# under `set -e`, a bare nonzero exit would kill this script immediately at
# that line, before ever reaching the exit-code check below - this idiom
# captures the code without triggering errexit.
STAGE1_EXIT=0
python3 "$SCRIPT_DIR/run_fallback_agent_mnn.py" "$@" || STAGE1_EXIT=$?

if [ "$STAGE1_EXIT" -ne 0 ]; then
    echo ""
    echo "[run_pipeline.sh] Stage 1 exited with code $STAGE1_EXIT." >&2
fi

echo ""
echo "======================================================================"
echo "STAGE 2: GGUF evaluation (run_fallback_agent_gguf.py --full-run) - full question set, independent of Stage 1"
echo "======================================================================"
# Stage 2 now runs unconditionally, regardless of Stage 1's exit code -
# these two stages are fully independent, so a Stage 1 failure no longer
# blocks Stage 2 (there's no needs_gguf_fallback.json handoff left for
# Stage 1 to have failed to produce).
STAGE2_EXIT=0
python3 "$SCRIPT_DIR/run_fallback_agent_gguf.py" --full-run "$@" || STAGE2_EXIT=$?

echo ""
echo "======================================================================"
echo "PIPELINE DONE (Stage 1 exit=$STAGE1_EXIT, Stage 2 exit=$STAGE2_EXIT)"
echo "======================================================================"

if [ "$STAGE1_EXIT" -ne 0 ]; then
    exit "$STAGE1_EXIT"
fi
exit "$STAGE2_EXIT"
