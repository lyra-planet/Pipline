# APIMart MiniMax-H3 Sequential Pipeline

This repository contains the portable execution layer for multi-stage
MiniMax-H3 video editing.  It supports both the hosted APIMart API and a local
MiniMax-H3 Ref2VA workflow served by ComfyUI.  It is deliberately separated
from Aurora training code, model weights, benchmark data, and historical run
artifacts.

## What is included

- `src/apimart_h3_pipeline/`: the canonical implementation split by responsibility;
- `scripts/run_apimart_minimax_h3_sequential.py`: backwards-compatible CLI;
- `scripts/run_apimart_minimax_h3_sequential_screen.sh`: optional HTTP/tunnel
  supervisor with no machine-specific project paths;
- `scripts/run_apimart_minimax_h3.py` and `scripts/vetra_failure_repair.py`:
  thin compatibility commands for the packaged provider client and repair policy;
- `tests/`: media, bridge, retry, and modular-import regression tests;
- `docs/`: the Chinese protocol and control-plane documentation.

The implementation is intentionally provider-boundary aware.  The source
package uses two levels of responsibility: `core/`, `providers/`, `bridge/`,
`execution/`, `media/`, and `resources/`.  Media normalization, reference
policy, Qwen-VL observation, GRSAI image editing, bridge construction, artifact
archival, and stage orchestration are separate modules.  A failed stage output
is archived and observed but can never become the input of its own retry.
Global style edits start with a frame-0 style master and use frame `0/53/106`
only for a failed retry; the middle and end anchors receive the frame-0 master
as their style reference.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
```

`ffmpeg` and `ffprobe` must be available on `PATH`.  Provider credentials are
never stored in this repository.  The online backend accepts
`--apimart-env`, `--grsai-env`, and `--dashscope-env`, or the corresponding
`APIMART_ENV_FILE`, `GRSAI_ENV_FILE`, and `DASHSCOPE_ENV_FILE` variables; the
defaults are `~/.apimart.env`, `~/.grsai.env`, and `~/.dashscope.env`.

The local backend does not need an APIMart credential.  Start a compatible
ComfyUI server, make its input/output directories visible to this process, and
provide an API-format workflow template:

```bash
python scripts/run_apimart_minimax_h3_sequential.py \
  --h3-backend local \
  --local-server http://127.0.0.1:8188 \
  --local-workflow-template /models/workflows/minimax_h3_api.json \
  --local-input-dir /models/comfyui/input \
  --local-output-dir /models/comfyui/output \
  --compiled-jobs /data/compiled_jobs.json --task-id 139 \
  --out-dir /data/runs/task139-local --media-dir /data/runs/task139-local/public_media \
  --dashscope-env /secure/dashscope.env --grsai-env /secure/grsai.env
```

The template must contain one node whose `class_type` includes
`ReferenceToVideo`, one `LoadVideo` node, and one `SaveVideo` node.  Reference
images are attached by cloning `LoadImage` nodes when the stage policy needs
three anchors.  The server must use the same input/output directories passed
to the runner; no machine-specific absolute path is embedded in the package.

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
