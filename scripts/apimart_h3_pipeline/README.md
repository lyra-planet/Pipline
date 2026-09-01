# APIMart MiniMax-H3 Pipeline

This package is the implementation behind
`scripts/run_apimart_minimax_h3_sequential.py`.  The historical script is a
thin compatibility entry point; new integrations should import this package
or use the `apimart-h3-sequential` console command.

## Module boundaries

| Module | Responsibility |
| --- | --- |
| `constants.py` | H3 media contract, frame roles, and schema identifiers |
| `media.py` | `ffprobe`, letterbox/crop normalization, frame extraction, sidecars |
| `policy.py` | static/video-only reference policy and prompt/tag validation |
| `repair.py` | public boundary for the closed-set VETRA failure repair policy |
| `vision.py` | DashScope Qwen-VL refinement and five-frame success gate |
| `image_editor.py` | GRSAI asynchronous image-edit client |
| `bridge.py` | parent frames, image references, uploads, and H3 prompt assembly |
| `artifacts.py` | request resume, attempt archival, and manifest records |
| `runner.py` | stage lifecycle, immutable parent handling, retry routing, CLI |

The package deliberately keeps the control-plane boundary explicit.  A stage
is executed against one immutable parent video.  A failed output is observed
and archived, but is never fed into its own retry.  A normal static/global edit
starts with the first parent frame as the style master.  Only a failed global
style attempt is escalated to frame `0/53/106` anchors, with the first-frame
master passed as the style reference for the middle and end image edits.

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
