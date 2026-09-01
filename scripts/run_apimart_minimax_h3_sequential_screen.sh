#!/usr/bin/env bash
# Serve only this sequence's temporary media and run its resume-safe APIMart worker.
set -Eeuo pipefail

# Use the maintained multi-step pipeline copy.  The parent aurora_msr_control
# directory contains an older runner that lacks --h3-model and the uniform S1
# bridge contract.
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project=$(cd -- "$script_dir/.." && pwd)
runner="$project/scripts/run_apimart_minimax_h3_sequential.py"
proxy_wrapper=${APIMART_PROXY_WRAPPER:-}
cloudflared=${CLOUDFLARED_BIN:-cloudflared}
run_dir=${APIMART_H3_RUN_DIR:?APIMART_H3_RUN_DIR is required}
compiled_jobs=${APIMART_H3_COMPILED_JOBS:?APIMART_H3_COMPILED_JOBS is required}
task_id=${APIMART_H3_TASK_ID:?APIMART_H3_TASK_ID is required}
http_port=${APIMART_H3_MEDIA_PORT:-18683}
http_screen=${APIMART_H3_HTTP_SCREEN:-apimart_h3_media_http_${task_id}_20260818}
tunnel_screen=${APIMART_H3_TUNNEL_SCREEN:-apimart_h3_media_tunnel_${task_id}_20260818}
allow_resubmit=${APIMART_H3_ALLOW_RESUBMIT:-0}
last_stage=${APIMART_H3_LAST_STAGE:-}
initial_reference=${APIMART_H3_INITIAL_REFERENCE:-0}
global_style_reference_count=${APIMART_H3_GLOBAL_STYLE_REFERENCE_COUNT:-1}
failure_recovery=${APIMART_H3_FAILURE_RECOVERY:-targeted}
allow_unverified_output=${APIMART_H3_ALLOW_UNVERIFIED_OUTPUT:-0}
dashscope_env=${APIMART_H3_DASHSCOPE_ENV:?APIMART_H3_DASHSCOPE_ENV is required}
dashscope_model=${APIMART_H3_DASHSCOPE_MODEL:-qwen-vl-plus}
apimart_env=${APIMART_H3_API_ENV:?APIMART_H3_API_ENV is required}
grsai_env=${APIMART_H3_GRSAI_ENV:?APIMART_H3_GRSAI_ENV is required}
h3_model=${APIMART_H3_MODEL:-MiniMax-H3}
prepared_initial=${APIMART_H3_PREPARED_INITIAL_VIDEO:-}
probe_proxy=${APIMART_H3_PROBE_PROXY:-http://127.0.0.1:17890}

screen_exists() {
  screen -ls 2>/dev/null | grep -q "\\.${1}[[:space:]]"
}

stop_media() {
  screen_exists "$tunnel_screen" && screen -S "$tunnel_screen" -X quit || true
  screen_exists "$http_screen" && screen -S "$http_screen" -X quit || true
}

on_exit() {
  local rc=$?
  trap - EXIT HUP INT TERM
  set +e
  printf 'event=media_cleanup exit_code=%s time=%s\n' "$rc" "$(date -Is)"
  stop_media
  exit "$rc"
}

[[ -f "$runner" ]] || { echo "missing sequential runner: $runner" >&2; exit 1; }
if [[ -n "$proxy_wrapper" ]]; then
  [[ -x "$proxy_wrapper" ]] || { echo "missing proxy wrapper: $proxy_wrapper" >&2; exit 1; }
fi
command -v "$cloudflared" >/dev/null 2>&1 || { echo "cloudflared is not executable: $cloudflared" >&2; exit 1; }
[[ -f "$compiled_jobs" ]] || { echo "missing compiled jobs: $compiled_jobs" >&2; exit 1; }
[[ -f "$dashscope_env" ]] || { echo "missing DashScope environment file: $dashscope_env" >&2; exit 1; }
if [[ -n "$prepared_initial" ]]; then
  [[ -f "$prepared_initial" ]] || { echo "missing prepared initial video: $prepared_initial" >&2; exit 1; }
  [[ -f "${prepared_initial%.mp4}.geometry.json" ]] || { echo "missing prepared geometry sidecar: ${prepared_initial%.mp4}.geometry.json" >&2; exit 1; }
fi
screen_exists "$http_screen" && { echo "media HTTP screen already exists: $http_screen" >&2; exit 1; }
screen_exists "$tunnel_screen" && { echo "media tunnel screen already exists: $tunnel_screen" >&2; exit 1; }

media_dir="$run_dir/public_media"
mkdir -p "$run_dir/logs" "$media_dir"
if [[ -n "$prepared_initial" ]]; then
  cp "$prepared_initial" "$media_dir/task_${task_id}_initial.mp4"
  cp "${prepared_initial%.mp4}.geometry.json" "$media_dir/task_${task_id}_initial.geometry.json"
fi
exec > >(tee -a "$run_dir/logs/supervisor.log") 2>&1
trap on_exit EXIT HUP INT TERM

# A Quick Tunnel hostname expires when its process exits. Keep prior logs for
# diagnosis but never let a restart parse an already-closed hostname.
if [[ -f "$run_dir/logs/cloudflared.log" ]]; then
  mv "$run_dir/logs/cloudflared.log" "$run_dir/logs/cloudflared.$(date -u +%Y%m%dT%H%M%SZ).log"
fi

screen -dmS "$http_screen" bash -lc "cd '$media_dir' && exec python3 -m http.server '$http_port' --bind 127.0.0.1 >> '$run_dir/logs/media_http.log' 2>&1"
for attempt in $(seq 1 10); do
  if curl -fsS --max-time 2 -o /dev/null "http://127.0.0.1:${http_port}/"; then
    break
  fi
  sleep 1
  [[ "$attempt" != 10 ]] || { echo "media HTTP server did not start" >&2; exit 1; }
done

tunnel_command="'$cloudflared' tunnel --protocol http2 --url 'http://127.0.0.1:${http_port}'"
if [[ -n "$proxy_wrapper" ]]; then
  tunnel_command="'$proxy_wrapper' $tunnel_command"
fi
screen -dmS "$tunnel_screen" bash -lc "exec $tunnel_command >> '$run_dir/logs/cloudflared.log' 2>&1"
tunnel_url=
for attempt in $(seq 1 45); do
  tunnel_url=$(rg -o 'https://[-a-z0-9]+\.trycloudflare\.com' "$run_dir/logs/cloudflared.log" 2>/dev/null | tail -n 1 || true)
  [[ -n "$tunnel_url" ]] && break
  screen_exists "$tunnel_screen" || { tail -n 80 "$run_dir/logs/cloudflared.log" >&2; exit 1; }
  sleep 2
done
[[ -n "$tunnel_url" ]] || { echo "Quick Tunnel URL was not created" >&2; exit 1; }

# Verify that the public media path is reachable through the same proxy route
# used to create the tunnel before spending an H3 request. Prefer the latest
# stage media; the normalized initial source is always available as fallback.
probe_file="$media_dir/task_${task_id}_S2.mp4"
[[ -f "$probe_file" ]] || probe_file="$media_dir/task_${task_id}_initial.mp4"
probe_name=$(basename "$probe_file")
probe_url="$tunnel_url/$(python3 -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1]))' "$probe_name")"
probe_host=${tunnel_url#https://}
probe_ip=
probe_ok=0
for attempt in $(seq 1 45); do
  # A newly-created trycloudflare hostname can briefly return NXDOMAIN.
  # Under pipefail, getent's transient non-zero status would abort the
  # supervisor before the next retry, so treat it as an empty resolution.
  probe_ip=$(getent ahostsv4 "$probe_host" 2>/dev/null | awk 'NR == 1 {print $1}' || true)
  if curl -x "$probe_proxy" -fsS --max-time 45 -o /dev/null "$probe_url"; then
    probe_ok=1
    break
  fi
  if [[ -n "$probe_ip" ]] && curl --noproxy '*' --resolve "${probe_host}:443:${probe_ip}" -fsS --max-time 45 -o /dev/null "$probe_url"; then
    probe_ok=1
    break
  fi
  sleep 2
done
[[ "$probe_ok" = 1 ]] || { echo "public media probe failed for $probe_name" >&2; exit 1; }

echo "event=online_sequence_start task_id=$task_id tunnel_ready=true time=$(date -Is)"
runner_extra=()
if [[ "$allow_resubmit" = 1 ]]; then
  runner_extra+=(--allow-resubmit)
fi
if [[ -n "$last_stage" ]]; then
  runner_extra+=(--last-stage "$last_stage")
fi
if [[ "$initial_reference" = 1 ]]; then
  runner_extra+=(--initial-reference)
fi
if [[ -n "$prepared_initial" ]]; then
  runner_extra+=(--prepared-initial-video "$prepared_initial")
fi
runner_extra+=(--global-style-reference-count "$global_style_reference_count")
runner_extra+=(--failure-recovery "$failure_recovery")
if [[ "$allow_unverified_output" = 1 ]]; then
  runner_extra+=(--allow-unverified-output)
fi
runner_command=(python3 "$runner")
if [[ -n "$proxy_wrapper" ]]; then
  runner_command=("$proxy_wrapper" "${runner_command[@]}")
fi
"${runner_command[@]}" \
  --compiled-jobs "$compiled_jobs" \
  --task-id "$task_id" \
  --out-dir "$run_dir" \
  --media-dir "$media_dir" \
  --media-public-base-url "$tunnel_url" \
  --dashscope-env "$dashscope_env" \
  --dashscope-model "$dashscope_model" \
  --apimart-env "$apimart_env" \
  --grsai-env "$grsai_env" \
  --h3-model "$h3_model" \
  --duration 4 \
  --resolution 768P \
  --aspect-ratio 16:9 \
  --request-timeout 120 \
  --poll-seconds 7 \
  --total-timeout 900 \
  "${runner_extra[@]}"
