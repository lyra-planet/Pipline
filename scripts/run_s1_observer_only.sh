#!/usr/bin/env bash
# Run only S1 for every compact-plan task, then stop after Observer.
set -u

project=/root/Pipline
venv="$project/.venv/bin/apimart-h3-sequential"
jobs=/root/autodl-tmp/Pipline_inputs/compact_plans_compiled_jobs.json
template=/root/autodl-tmp/Pipline_inputs/minimax_h3_ref2va_api.json
root=/root/autodl-tmp/Pipline_runs/s1_observer_only_20260905
previous_root=/root/autodl-tmp/Pipline_runs/local_compact_plans
summary="$root/batch_status.tsv"
local_server=${APIMART_H3_LOCAL_SERVER:-http://127.0.0.1:6006}
input_dir=${APIMART_H3_LOCAL_INPUT_DIR:-/root/autodl-tmp/ComfyUI/input}
output_dir=${APIMART_H3_LOCAL_OUTPUT_DIR:-/root/autodl-tmp/ComfyUI/output}
lock_file=${APIMART_H3_BATCH_LOCK:-$root/.batch.lock}
task_bucket=${APIMART_H3_TASK_BUCKET:-}

mkdir -p "$root"
exec 9>"$lock_file"
if ! flock -n 9; then
  printf 'another S1 observer-only runner already owns %s\n' "$lock_file" >&2
  exit 2
fi

printf 'started=%s\n' "$(date -Is)" >> "$root/batch.log"

if [ ! -x "$venv" ]; then
  printf 'missing runner: %s\n' "$venv" >&2
  exit 2
fi

for task_id in $(python3 - "$jobs" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
for task in sorted(data["tasks"], key=lambda item: int(str(item["task_id"]))):
    print(task["task_id"])
PY
); do
  if [ -n "$task_bucket" ]; then
    case "$task_bucket" in
      0|1|2|3|4)
        if [ $((task_id % 5)) -ne "$task_bucket" ]; then
          continue
        fi
        ;;
      *)
        printf 'invalid APIMART_H3_TASK_BUCKET=%s (expected 0, 1, 2, 3, or 4)\n' "$task_bucket" >&2
        exit 2
        ;;
    esac
  fi

  new_s1="$root/task_${task_id}/stages/S1/output.mp4"
  old_s1="$previous_root/task_${task_id}/stages/S1/output.mp4"
  if [ -s "$new_s1" ]; then
    printf '%s\tskipped_existing_s1_new_run\t%s\n' "$task_id" "$(date -Is)" >> "$summary"
    continue
  fi
  if [ -s "$old_s1" ]; then
    mkdir -p "$root/task_${task_id}"
    printf '%s\tskipped_existing_s1_previous_run\t%s\n' "$task_id" "$(date -Is)" >> "$summary"
    printf '{"task_id":"%s","status":"skipped_existing_s1","source":"%s"}\n' "$task_id" "$old_s1" > "$root/task_${task_id}/s1_run_state.json"
    continue
  fi

  task_dir="$root/task_${task_id}"
  mkdir -p "$task_dir/media"
  printf '%s\tstarted_s1\t%s\n' "$task_id" "$(date -Is)" >> "$summary"
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
    --last-stage S1 \
    --failure-recovery disabled \
    >> "$root/task_${task_id}.log" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then
    printf '%s\tcompleted_s1_observer\t%s\n' "$task_id" "$(date -Is)" >> "$summary"
  else
    printf '%s\tstopped_after_s1_observer_error_rc_%s\t%s\n' "$task_id" "$rc" "$(date -Is)" >> "$summary"
  fi
done

printf 'finished=%s\n' "$(date -Is)" >> "$root/batch.log"
