#!/usr/bin/env bash
# Resumable local MiniMax-H3 Ref2VA queue for every compact plan.
set -u

project=/root/Pipline
venv="$project/.venv/bin/apimart-h3-sequential"
jobs=/root/autodl-tmp/Pipline_inputs/compact_plans_compiled_jobs.json
template=/root/autodl-tmp/Pipline_inputs/minimax_h3_ref2va_api.json
root=/root/autodl-tmp/Pipline_runs/local_compact_plans
input_dir=/root/autodl-tmp/ComfyUI/input
output_dir=/root/autodl-tmp/ComfyUI/output
summary="$root/batch_status.tsv"
start_task_id=${APIMART_H3_START_TASK_ID:-1}
local_server=${APIMART_H3_LOCAL_SERVER:-http://127.0.0.1:6006}
input_dir=${APIMART_H3_LOCAL_INPUT_DIR:-$input_dir}
output_dir=${APIMART_H3_LOCAL_OUTPUT_DIR:-$output_dir}
lock_file=${APIMART_H3_BATCH_LOCK:-$root/.batch.lock}
task_parity=${APIMART_H3_TASK_PARITY:-all}
task_bucket=${APIMART_H3_TASK_BUCKET:-}
mkdir -p "$root"
# Only one queue owner is allowed.  A second runner can otherwise submit work
# concurrently and a cleanup/interrupt from either process can cancel the
# other's ComfyUI sampling job.
exec 9>"$lock_file"
if ! flock -n 9; then
  printf 'another local compact-plan runner already owns %s\n' "$lock_file" >&2
  exit 2
fi
printf 'started=%s\n' "$(date -Is)" >> "$root/batch.log"

wait_for_comfy() {
  local attempt
  for attempt in $(seq 1 60); do
    if curl --noproxy '*' -fsS --max-time 5 "$local_server/system_stats" >/dev/null 2>&1; then
      return 0
    fi
    printf 'comfyui_unavailable_attempt_%s\t%s\n' "$attempt" "$(date -Is)" >> "$root/batch_status.tsv"
    sleep 10
  done
  printf 'comfyui_unavailable_after_60_checks\n' >> "$root/batch.log"
  return 1
}

image_edit_retries_exhausted() {
  python3 - "$1" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for path in root.rglob("*image_edit_state*.json"):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        continue
    if value.get("status") in {"failed_retry_exhausted", "other_retry_exhausted"}:
        raise SystemExit(0)
raise SystemExit(1)
PY
}

for task_id in $(python3 - "$jobs" <<'PY'
import json, sys
data=json.load(open(sys.argv[1], encoding='utf-8'))
for task in sorted(data['tasks'], key=lambda item: int(str(item['task_id']))):
    print(task['task_id'])
PY
); do
  if [ -n "$task_bucket" ]; then
    case "$task_bucket" in
      0|1|2)
        if [ $((task_id % 3)) -ne "$task_bucket" ]; then
          continue
        fi
        ;;
      *)
        printf 'invalid APIMART_H3_TASK_BUCKET=%s (expected 0, 1, or 2)\n' "$task_bucket" >&2
        exit 2
        ;;
    esac
  else
    case "$task_parity" in
      odd)
        if [ $((task_id % 2)) -ne 1 ]; then
          continue
        fi
        ;;
      even)
        if [ $((task_id % 2)) -ne 0 ]; then
          continue
        fi
        ;;
      all)
        ;;
      *)
        printf 'invalid APIMART_H3_TASK_PARITY=%s (expected all, odd, or even)\n' "$task_parity" >&2
        exit 2
        ;;
    esac
  fi
  if [ "$task_id" -lt "$start_task_id" ]; then
    printf '%s	skipped_before_start_task_%s\t%s\n' "$task_id" "$start_task_id" "$(date -Is)" >> "$summary"
    continue
  fi
  task_dir="$root/task_$task_id"
  manifest="$task_dir/sequence_manifest.json"
  if [ "$task_id" = "1" ] && screen -ls 2>/dev/null | grep -q '\.pipline-local-task1[[:space:]]'; then
    printf '%s\twaiting_for_smoke\t%s\n' "$task_id" "$(date -Is)" >> "$summary"
    while screen -ls 2>/dev/null | grep -q '\.pipline-local-task1[[:space:]]'; do
      sleep 30
    done
  fi
  if [ -f "$manifest" ] && python3 - "$manifest" <<'PY'
import json, sys
try:
    value=json.load(open(sys.argv[1], encoding='utf-8'))
    ok=value.get('status') in {'success', 'degraded'} and isinstance(value.get('output'), str)
    ok=ok and __import__('pathlib').Path(value['output']).is_file()
except Exception:
    ok=False
raise SystemExit(0 if ok else 1)
PY
  then
    printf '%s\tskipped_success\t%s\n' "$task_id" "$(date -Is)" >> "$summary"
    continue
  fi
  mkdir -p "$task_dir/media"
  attempt=0
  while :; do
    attempt=$((attempt + 1))
    printf '%s\tstarted_attempt_%s\t%s\n' "$task_id" "$attempt" "$(date -Is)" >> "$summary"
    if ! wait_for_comfy; then
      printf '%s\tfailed_comfyui_unavailable_next_task\t%s\n' "$task_id" "$(date -Is)" >> "$summary"
      break
    fi
    "$venv" \
      --h3-backend local \
      --local-server "$local_server" \
      --local-workflow-template "$template" \
      --local-input-dir "$input_dir" \
      --local-output-dir "$output_dir" \
      --local-timeout 21600 \
      --local-poll-seconds 15 \
      --compiled-jobs "$jobs" \
      --task-id "$task_id" \
      --out-dir "$task_dir" \
      --media-dir "$task_dir/media" \
      --dashscope-env /root/.dashscope.env \
      --grsai-env /root/.grsai.env \
      --failure-recovery targeted \
      >> "$root/task_${task_id}.log" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ]; then
      printf '%s\tcompleted\t%s\n' "$task_id" "$(date -Is)" >> "$summary"
      break
    fi
    if image_edit_retries_exhausted "$task_dir"; then
      printf '%s\tfailed_final_image_edit_retries_exhausted\t%s\n' "$task_id" "$(date -Is)" >> "$summary"
      printf 'task=%s image-edit retry budget exhausted; continuing with the next task\n' "$task_id" >> "$root/batch.log"
      break
    fi
    printf '%s\tfailed_rc_%s_retrying\t%s\n' "$task_id" "$rc" "$(date -Is)" >> "$summary"
    if [ "$attempt" -ge 10 ]; then
      printf '%s\tfailed_final_after_%s_attempts\t%s\n' "$task_id" "$attempt" "$(date -Is)" >> "$summary"
      printf 'task=%s halted after %s failed attempts; continuing with the next task\n' "$task_id" "$attempt" >> "$root/batch.log"
      break
    fi
    sleep 30
  done
done
printf 'finished=%s\n' "$(date -Is)" >> "$root/batch.log"
