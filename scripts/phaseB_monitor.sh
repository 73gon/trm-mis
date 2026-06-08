#!/bin/bash
# Background monitor for Phase B jobs (2506 pw20, 2507 deep).
# Writes one snapshot every $INTERVAL seconds to logs/phaseB_monitor.status.
# Applies it=50k go/no-go gate automatically:
#   PASS = model_lift > +0.05  (bce gate dropped: scale-dependent on pos_weight)
#   On FAIL after first eval at it >= 50000: scancel + sentinel file.
# Stops when both jobs are no longer in squeue.

set -u
cd "$(dirname "$0")/.."

STATUS=logs/phaseB_monitor.status
DECISIONS=logs/phaseB_monitor.decisions
INTERVAL="${INTERVAL:-300}"   # 5 min

DECODER_ER=0.5949    # back-filled (job 2505)

JOBS=(
  "2506:sweep1_pe_er_N25k_400k_pw20"
  "2531:sweep1_pe_er_N25k_400k_deep"
)

# Track which jobs already had the gate applied.
declare -A GATED
for spec in "${JOBS[@]}"; do
  jid="${spec%%:*}"
  GATED[$jid]=0
done

echo "[$(date -Iseconds)] monitor starting (interval=${INTERVAL}s)" >| "$STATUS"

while true; do
  alive=0

  {
    echo "===================================================================="
    echo "[$(date -Iseconds)] snapshot"
    echo "--- squeue ---"
    squeue -u "$USER" -o "%.10i %.30j %.10T %.20R %.8M" 2>&1
    echo

    for spec in "${JOBS[@]}"; do
      jid="${spec%%:*}"
      jname="${spec##*:}"
      log="logs/${jname}_${jid}.log"
      err="logs/${jname}_${jid}.err"

      state=$(squeue -h -j "$jid" -o "%T" 2>/dev/null || true)
      [[ -n "$state" ]] && alive=1

      echo "--- ${jname} (job ${jid}) state=${state:-DONE} ---"

      if [[ -f "$log" ]]; then
        # Last training tick.
        last_train=$(grep -E "^\[it " "$log" | tail -1)
        echo "train: ${last_train:-<none>}"

        # Last eval block (the [pe] eval line, has 'lift=').
        last_eval=$(grep -E "^\[pe\] eval@" "$log" 2>/dev/null | tail -1)
        if [[ -z "$last_eval" ]]; then
          last_eval=$(grep -E "model_lift|lift=|test pp_AR=" "$log" 2>/dev/null | tail -1)
        fi
        echo "eval:  ${last_eval:-<none>}"

        # Decoder baseline line printed once at startup.
        dec=$(grep -E "decoder-only baseline" "$log" 2>/dev/null | tail -1)
        echo "decoder: ${dec:-<none>}"

        # Gate logic: parse latest eval iter and metrics.
        if [[ ${GATED[$jid]} -eq 0 ]]; then
          # Find any eval line with iter parseable.
          gate_line=$(grep -oE "it [0-9]+ .* bce=[0-9.]+ .* lift=[+-][0-9.]+" "$log" 2>/dev/null | tail -1)
          if [[ -n "$gate_line" ]]; then
            it=$(echo "$gate_line" | grep -oE "it [0-9]+" | grep -oE "[0-9]+")
            bce=$(echo "$gate_line" | grep -oE "bce=[0-9.]+" | head -1 | cut -d= -f2)
            lift=$(echo "$gate_line" | grep -oE "lift=[+-][0-9.]+" | head -1 | cut -d= -f2)
            echo "gate-parse: it=$it bce=$bce lift=$lift"
            if [[ -n "$it" && "$it" -ge 50000 ]]; then
              pass_lift=$(awk -v l="$lift" 'BEGIN{print (l+0 > 0.05) ? 1 : 0}')
              if [[ "$pass_lift" = "1" ]]; then
                msg="[$(date -Iseconds)] PASS job=$jid it=$it bce=$bce lift=$lift -> continue"
                echo "$msg" >> "$DECISIONS"
                echo "DECISION: $msg"
                GATED[$jid]=1
              else
                msg="[$(date -Iseconds)] FAIL job=$jid it=$it bce=$bce lift=$lift -> scancel"
                echo "$msg" >> "$DECISIONS"
                echo "DECISION: $msg"
                scancel "$jid" 2>&1 || true
                GATED[$jid]=1
              fi
            fi
          fi
        fi
      else
        echo "log file missing: $log"
      fi

      if [[ -s "$err" ]]; then
        echo "stderr (last 3):"
        tail -3 "$err"
      fi
      echo
    done

    echo "alive=$alive"
  } >> "$STATUS" 2>&1

  if [[ "$alive" -eq 0 ]]; then
    echo "[$(date -Iseconds)] all Phase B jobs finished, exiting monitor" >> "$STATUS"
    break
  fi

  sleep "$INTERVAL"
done
