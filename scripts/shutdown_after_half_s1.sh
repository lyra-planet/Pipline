#!/usr/bin/env bash
# Power off after half of the compact-plan tasks reach a terminal S1 state.
set -u

jobs=/root/autodl-tmp/Pipline_inputs/compact_plans_compiled_jobs.json
run_root=/root/autodl-tmp/Pipline_runs/s1_observer_only_20260905
status_file="$run_root/batch_status.tsv"
log="$run_root/half_shutdown_monitor.log"
marker="$run_root/.half_shutdown_triggered"

mkdir -p "$run_root"

read -r total target < <(python3 - "$jobs" <<'PY'
import json, math, sys
tasks = json.load(open(sys.argv[1], encoding="utf-8"))["tasks"]
total = len({str(item["task_id"]) for item in tasks})
print(total, math.ceil(total / 2))
PY
)

printf 'monitor_started=%s total_tasks=%s target_terminal_tasks=%s\n' "$(date -Is)" "$total" "$target" >> "$log"

while [ ! -e "$marker" ]; do
  read -r terminal < <(python3 - "$jobs" "$status_file" <<'PY'
import json, sys
from pathlib import Path

jobs = json.load(open(sys.argv[1], encoding="utf-8"))["tasks"]
task_ids = {str(item["task_id"]) for item in jobs}
latest = {}
try:
    lines = Path(sys.argv[2]).read_text(encoding="utf-8").splitlines()
except OSError:
    lines = []
for line in lines:
    fields = line.split("\t")
    if len(fields) >= 2 and fields[0] in task_ids:
        latest[fields[0]] = fields[1]
terminal_states = {
    "skipped_existing_s1_new_run",
    "skipped_existing_s1_previous_run",
    "completed_s1_observer",
    "stopped_after_s1_observer_error_rc_1",
}
print(sum(state in terminal_states for state in latest.values()))
PY
  )
  timestamp=$(date -Is)
  printf 'checked=%s terminal_tasks=%s target=%s\n' "$timestamp" "$terminal" "$target" >> "$log"
  if [ "$terminal" -ge "$target" ]; then
    printf 'threshold_reached=%s terminal_tasks=%s target=%s; syncing and shutting down\n' "$timestamp" "$terminal" "$target" >> "$log"
    printf '%s\n' "$timestamp" > "$marker"
    sync
    /usr/bin/shutdown -h now >> "$log" 2>&1
    rc=$?
    printf 'shutdown_return_code=%s time=%s\n' "$rc" "$(date -Is)" >> "$log"
    exit "$rc"
  fi
  sleep 30
done
