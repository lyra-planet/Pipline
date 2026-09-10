#!/usr/bin/env bash
# Rerun selected Stage 1 tasks on five local H3 workers, Observer only.
set -u
PROJECT=/root/Pipline
RUNNER="$PROJECT/.venv/bin/apimart-h3-sequential"
JOBS=${S1_RERUN_JOBS:-/root/autodl-tmp/Pipline_inputs/s1_rerun_compiled_jobs_20260906.json}
TEMPLATE=${S1_RERUN_TEMPLATE:-/root/autodl-tmp/Pipline_inputs/temp/minimax_h3_ref2va_api.json}
ROOT=${S1_RERUN_ROOT:-/root/autodl-tmp/Pipline_runs/s1_rerun_20260906}
INPUT_DIR=${S1_RERUN_INPUT_DIR:-/root/autodl-tmp/ComfyUI/input}
COMFY_ROOT=/root/ComfyUI
COMFY_PYTHON=/root/miniconda3/bin/python
SUMMARY="$ROOT/batch_status.tsv"
LOG="$ROOT/batch.log"
LOCK="$ROOT/.batch.lock"

mkdir -p "$ROOT" "$INPUT_DIR"
exec 9>"$LOCK"
if ! flock -n 9; then
  printf 'another rerun already owns %s\n' "$LOCK" >&2
  exit 2
fi
log() { printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }
status() {
  (
    flock -x 8
    printf '%s\t%s\t%s\n' "$1" "$2" "$(date -Is)" >> "$SUMMARY"
  ) 8>"$ROOT/.status.lock"
}

if [ ! -x "$RUNNER" ] || [ ! -f "$JOBS" ] || [ ! -f "$TEMPLATE" ]; then
  log "unrecoverable: runner, jobs, or workflow template is missing"
  sync
  /usr/bin/shutdown -h now >> "$LOG" 2>&1 || true
  exit 2
fi

printf 'task_id\tstatus\ttimestamp\n' > "$SUMMARY"
task_count=$(python3 - "$JOBS" <<'PY'
import json, sys
print(len(json.load(open(sys.argv[1], encoding="utf-8")).get("tasks", [])))
PY
)
log "starting selected S1 observer-only rerun; task_count=${task_count}"

start_comfy() {
  local gpu="$1" port="$2" output="$3" user_dir="$4" service_log="$5"
  mkdir -p "$output" "$user_dir"
  screen -S "h3-rerun-gpu${gpu}" -X quit >/dev/null 2>&1 || true
  screen -dmS "h3-rerun-gpu${gpu}" bash -lc \
    "cd '$COMFY_ROOT'; \
     export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES='$gpu' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; \
     while true; do \
       printf '%s starting ComfyUI gpu%s port %s\\n' \"\$(date -u +%FT%TZ)\" '$gpu' '$port' >> '$service_log'; \
       '$COMFY_PYTHON' -u main.py --listen 127.0.0.1 --port '$port' --disable-auto-launch --preview-method none --log-stdout --disable-manager --disable-api-nodes --disable-triton-backend --output-directory '$output' --input-directory '$INPUT_DIR' --user-directory '$user_dir' >> '$service_log' 2>&1; \
       rc=\$?; printf '%s ComfyUI gpu%s exited rc=%s; restarting in 5s\\n' \"\$(date -u +%FT%TZ)\" '$gpu' \"\$rc\" >> '$service_log'; sleep 5; \
     done"
}

for spec in \
  "0 8191 /root/autodl-tmp/ComfyUI/output_gpu0 /root/ComfyUI/user_gpu0 /root/autodl-tmp/ComfyUI/comfy_gpu0_s1_rerun.log" \
  "1 8192 /root/autodl-tmp/ComfyUI/output_gpu1 /root/ComfyUI/user_gpu1 /root/autodl-tmp/ComfyUI/comfy_gpu1_s1_rerun.log" \
  "2 8193 /root/autodl-tmp/ComfyUI/output_gpu2 /root/ComfyUI/user_gpu2 /root/autodl-tmp/ComfyUI/comfy_gpu2_s1_rerun.log" \
  "3 8194 /root/autodl-tmp/ComfyUI/output_gpu3 /root/ComfyUI/user_gpu3 /root/autodl-tmp/ComfyUI/comfy_gpu3_s1_rerun.log" \
  "4 8195 /root/autodl-tmp/ComfyUI/output_gpu4 /root/ComfyUI/user_gpu4 /root/autodl-tmp/ComfyUI/comfy_gpu4_s1_rerun.log"; do
  read -r gpu port output user_dir service_log <<< "$spec"
  start_comfy "$gpu" "$port" "$output" "$user_dir" "$service_log"
done

healthy=0
for attempt in $(seq 1 60); do
  healthy=0
  for port in 8191 8192 8193 8194 8195; do
    if curl --noproxy '*' -fsS --max-time 3 "http://127.0.0.1:${port}/system_stats" >/dev/null 2>&1; then
      healthy=$((healthy + 1))
    fi
  done
  log "ComfyUI health check ${attempt}/60: ${healthy}/5 ready"
  [ "$healthy" -eq 5 ] && break
  sleep 10
done
if [ "$healthy" -ne 5 ]; then
  log "unrecoverable: not all five ComfyUI services became healthy"
  sync
  /usr/bin/shutdown -h now >> "$LOG" 2>&1 || true
  exit 3
fi

run_worker() {
  local gpu="$1" port="$2" bucket="$3"
  log "worker gpu${gpu} started on port ${port}, bucket=${bucket}"
  while read -r task_id; do
    [ -z "$task_id" ] && continue
    task_dir="$ROOT/task_${task_id}"
    if [ -s "$task_dir/stages/S1/output.mp4" ] && [ -f "$task_dir/sequence_manifest.json" ] && \
       python3 - "$task_dir/sequence_manifest.json" <<'PY'
import json,sys
try:
    d=json.load(open(sys.argv[1],encoding='utf-8'))
    stage=next((s for s in d.get('stages',[]) if isinstance(s,dict) and s.get('stage')=='S1'),{})
    raise SystemExit(0 if stage.get('status') == 'success' else 1)
except Exception:
    raise SystemExit(1)
PY
    then
      status "$task_id" "skipped_existing_success"
      continue
    fi
    status "$task_id" "started_gpu${gpu}"
    "$RUNNER" \
      --h3-backend local \
      --local-server "http://127.0.0.1:${port}" \
      --local-workflow-template "$TEMPLATE" \
      --local-input-dir "$INPUT_DIR" \
      --local-output-dir "/root/autodl-tmp/ComfyUI/output_gpu${gpu}" \
      --local-timeout 21600 \
      --local-poll-seconds 15 \
      --compiled-jobs "$JOBS" \
      --task-id "$task_id" \
      --out-dir "$task_dir" \
      --media-dir "$task_dir/media" \
      --dashscope-env /root/.dashscope.env \
      --grsai-env /root/.grsai.env \
      --last-stage S1 \
      --failure-recovery disabled \
      >> "$ROOT/task_${task_id}.log" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ]; then
      status "$task_id" "completed_s1_observer"
    else
      status "$task_id" "failed_rc_${rc}_no_repair"
    fi
  done < <(python3 - "$JOBS" "$bucket" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding='utf-8'))
b=int(sys.argv[2])
for i,t in enumerate(d.get('tasks',[])):
    if i % 5 == b:
        print(t['task_id'])
PY
  )
  log "worker gpu${gpu} finished"
}

run_worker 0 8191 0 & pids0=$!
run_worker 1 8192 1 & pids1=$!
run_worker 2 8193 2 & pids2=$!
run_worker 3 8194 3 & pids3=$!
run_worker 4 8195 4 & pids4=$!
wait "$pids0" "$pids1" "$pids2" "$pids3" "$pids4"

completed=$(awk -F '\t' '$2=="completed_s1_observer"{n++} END{print n+0}' "$SUMMARY")
failed=$(awk -F '\t' '$2 ~ /^failed_/{n++} END{print n+0}' "$SUMMARY")
skipped=$(awk -F '\t' '$2=="skipped_existing_success"{n++} END{print n+0}' "$SUMMARY")
log "batch finished: completed=${completed}, skipped=${skipped}, failed_no_repair=${failed}"
sync
log "executing automatic shutdown"
/usr/bin/shutdown -h now >> "$LOG" 2>&1 || log "shutdown command returned non-zero; machine may require shutdown authorization"
exit 0
