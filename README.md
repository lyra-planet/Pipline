# APIMart MiniMax-H3 Sequential Pipeline

This repository contains the portable execution layer for multi-stage
MiniMax-H3 video editing.  It is deliberately separated from Aurora training
code, model weights, benchmark data, and historical run artifacts.

## What is included

- `scripts/apimart_h3_pipeline/`: the implementation split by responsibility;
- `scripts/run_apimart_minimax_h3_sequential.py`: backwards-compatible CLI;
- `scripts/run_apimart_minimax_h3_sequential_screen.sh`: optional HTTP/tunnel
  supervisor with no machine-specific project paths;
- `scripts/run_apimart_minimax_h3.py` and `scripts/vetra_failure_repair.py`:
  provider client and closed-set failure-repair policy;
- `tests/`: media, bridge, retry, and modular-import regression tests;
- `docs/`: the Chinese protocol and control-plane documentation.

The implementation is intentionally provider-boundary aware.  Media
normalization, reference policy, Qwen-VL observation, GRSAI image editing,
bridge construction, artifact archival, and stage orchestration are separate
modules.  A failed stage output is archived and observed but can never become
the input of its own retry.  Global style edits start with a frame-0 style
master and use frame `0/53/106` only for a failed retry; the middle and end
anchors receive the frame-0 master as their style reference.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
```

`ffmpeg` and `ffprobe` must be available on `PATH`.  Provider credentials are
never stored in this repository.  Pass `--apimart-env`, `--grsai-env`, and
`--dashscope-env`, or set `APIMART_ENV_FILE`, `GRSAI_ENV_FILE`, and
`DASHSCOPE_ENV_FILE`; defaults are `~/.apimart.env`, `~/.grsai.env`, and
`~/.dashscope.env`.

## Run

```bash
python scripts/run_apimart_minimax_h3_sequential.py \
  --compiled-jobs /data/compiled_jobs.json --task-id 139 \
  --out-dir /data/runs/task139 \
  --media-dir /data/runs/task139/public_media \
  --media-public-base-url https://media.example.invalid \
  --apimart-env /secure/apimart.env \
  --grsai-env /secure/grsai.env \
  --dashscope-env /secure/dashscope.env
```

The same command is available as `apimart-h3-sequential` after installation.
Use `--dry-run` to validate the compiled task and reference policies without
calling a provider.

## Test

```bash
python -m pytest -q
```

The tests use fake provider responses and do not spend API credits.
