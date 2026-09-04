# APIMart MiniMax-H3 Pipeline

This package is the implementation behind
`scripts/run_apimart_minimax_h3_sequential.py`.  The historical script is a
thin compatibility entry point; new integrations should import this package
or use the `apimart-h3-sequential` console command.

H3 generation has two interchangeable provider boundaries: `providers/apimart.py`
submits hosted MiniMax-H3 requests, while `providers/local.py` submits an
API-format workflow to a local ComfyUI server.  The stage state machine,
reference bridge, Observer gate, and repair policy are shared by both modes.

The source layout is deliberately shallow: `core/`, `providers/`, `bridge/`,
`execution/`, `media/`, and `resources/` are the only responsibility
subpackages.  This keeps ownership visible without turning every helper into
another directory.

## Module boundaries

| Module | Responsibility |
| --- | --- |
| `core/` | protocol constants, reference policy, and closed-set repair policy |
| `providers/` | APIMart, DashScope, and GRSAI transport boundaries |
| `bridge/` | reference planning, uploads, prompt contracts, and bridge execution |
| `execution/` | CLI, stage lifecycle, immutable parent handling, and artifact records |
| `media/` | `ffprobe`, letterbox/crop normalization, frame extraction, and sidecars |
| `resources/` | package-resource prompt loading and reviewable prompt templates |
| root compatibility modules | stable imports such as `apimart_h3_pipeline.vision` |

For local H3, pass `--h3-backend local`, `--local-server`, and
`--local-workflow-template`, plus the ComfyUI `--local-input-dir` and
`--local-output-dir` when those directories are not the defaults under the
run directory.  The workflow graph is adapted by node type, so the package
does not depend on the node IDs from one particular ComfyUI export.

The package deliberately keeps the control-plane boundary explicit.  A stage
is executed against one immutable parent video.  A failed output is observed
and archived, but is never fed into its own retry.  The first generation is
`attempt_1`; the bounded repair generation is `attempt_2`.  Once that retry is
submitted, its media is validated and propagated to the next stage without a
second Observer call.  The stage and sequence manifest mark this as
`semantic_failure_propagated`/`degraded`, so downstream processing can continue
without presenting the retry as semantically confirmed.  A normal static/global
edit starts with the first parent frame as the style master.  Only a failed
global style attempt is escalated to frame `0/53/106` anchors, with the
first-frame master passed as the style reference for the middle and end image
edits.
Pure action or pose changes are the other explicit path: they use only
`<Video 1>` and a temporal-onset contract, so a target pose is not baked into
the first frame as a static picture.  Action edits that also change or expose a
static object, text, appearance, or composition remain image-conditioned.

## Install on another machine

The folder is self-contained apart from the provider clients, `ffmpeg`/
`ffprobe`, and model/API credentials.  From the folder root:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
ffmpeg -version
```

Set credentials through the process environment or three local files.  The
runner accepts `--apimart-env`, `--grsai-env`, and `--dashscope-env`; their
defaults are `~/.apimart.env`, `~/.grsai.env`, and `~/.dashscope.env`, and can
also be supplied with `APIMART_ENV_FILE`, `GRSAI_ENV_FILE`, and
`DASHSCOPE_ENV_FILE`.  No credential file is part of the repository.

Run directly with the old-compatible path:

```bash
python scripts/run_apimart_minimax_h3_sequential.py \
  --compiled-jobs /data/jobs.json --task-id 139 \
  --out-dir /data/runs/task139 --media-dir /data/runs/task139/public_media \
  --media-public-base-url https://media.example.invalid \
  --apimart-env /secure/apimart.env \
  --grsai-env /secure/grsai.env \
  --dashscope-env /secure/dashscope.env
```

For the screen supervisor, set `APIMART_H3_COMPILED_JOBS`,
`APIMART_H3_RUN_DIR`, `APIMART_H3_TASK_ID`, the three `*_ENV` variables, and
optionally `APIMART_PROXY_WRAPPER`/`CLOUDFLARED_BIN`.  The supervisor resolves
its project path relative to the script, so the folder can be moved without
editing the shell file.
