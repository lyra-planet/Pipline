# 当前 APIMart MiniMax-H3 多阶段推理流程

> **实现状态：已接入 VETRA 失败诊断与定向修复。** 默认使用 `--failure-recovery targeted`；普通执行包括全局风格在内先使用一张 primary reference，只有失败 retry 才按诊断决定定向修复或三锚点。`fixed-three-anchor` 是兼容实验基线，`disabled` 用于不自动修复的对照。当前已完成 schema、策略、bridge、恢复和 mocked state-machine 验证；真实 Task 139 的线上付费 provider 验证记录在 `runs/apimart_task139_S1_targeted_from_attempt1_20260901/`。

本文档以当前实现的
~~~text
scripts/run_apimart_minimax_h3_sequential.py
~~~
为准，记录在线和本地 MiniMax-H3 多阶段视频编辑的实际行为。配套的单阶段 API 客户端是
~~~text
scripts/run_apimart_minimax_h3.py
~~~

## 1. 流程定位

该 runner 是“已编译计划 + 可选在线/本地 H3 生成 + Qwen-VL 观察 + 可选图片参考 + 成功门 + 断点恢复”的执行器。

它会读取已经编译好的 compiled-jobs，按 S1、S2、... 固定顺序执行，每个 stage 使用上一个 stage 的视频作为输入。它不会重新规划 MSR、不会自动重排风格 stage、不会运行 Aurora 控制器。H3 可以通过 `--h3-backend online` 使用 APIMart，也可以通过 `--h3-backend local` 使用本地 ComfyUI workflow；风格阶段是否在最后，取决于输入计划本身。

## 2. 默认契约

~~~text
H3 画布             1344 x 768
H3 中间视频帧数     107
H3 中间视频帧率     24 fps
Qwen 上下文帧       0, 26, 53, 80, 106
Qwen 默认模型       qwen-vl-plus
普通静态编辑默认参考图数量  1
全局风格首次执行参考图数量  1
全局风格失败重试参考图数量  3
默认 H3 分辨率       768P
默认宽高比           16:9
默认时长             4 秒（允许 4 到 15 秒）
默认轮询间隔         7 秒
单任务总超时         900 秒
~~~

核心常量见 `src/apimart_h3_pipeline/core/constants.py`，CLI 定义见
`src/apimart_h3_pipeline/execution/cli.py`；`scripts/` 下的文件只是兼容入口。

## 3. 总体数据流

~~~text
compiled-jobs
    |
    v
初始视频归一化为 1344x768 / 107 帧 / 24fps
    |
    v
当前 parent video
    |
    +--> 抽取 0/26/53/80/106 五帧
    |
    +--> 本地 reference_policy 判断参考图策略
    |       |
    |       +--> 无参考图：Qwen 观察，H3 使用原始 prompt + <Video 1>
    |       |
    |       +--> 一张图：固定使用上下文首帧 0
    |       |       -> nano-banana-2 编辑首帧 style master
    |       |       -> Qwen 生成带 Picture/Video 标签的 H3 prompt
    |       |
    |       +--> 全局风格/普通静态编辑：默认编辑一张 primary anchor
    |       +--> 显式三图实验：首帧 style master + 中帧/末帧 temporal anchors
    |
    v
APIMart 或 CTMOAI MiniMax-H3 生成
    |
    v
Qwen-VL 五帧成功门 + failure_type 诊断
    |
    +--> success=true：完成当前 stage，更新 parent
    |
    +--> success=null：默认 observation_pending，停止传播
    |
    +--> success=false：按 failure_type 定向修复并重试同一个 stage
            |
            +--> edit_missing / identity_drift / previous_stage_lost / composition_weak
            |       保持原参考数量，强化对应约束
            +--> motion_weak：video_only，强化原 motion 语义
            +--> style_inconsistency：start/primary/end 三锚点
            +--> 不可修复：semantic_failure，停止传播
            +--> 已发起 retry：只做媒体合法性检查，不再调用 Observer；
                    输出标记 semantic_failure_propagated，物化后传给下一 stage
    |
    v
归一化 stage 输出，作为下一轮 parent video
    |
    v
全部 stage 完成后裁掉 letterbox，生成 output.mp4
~~~

## 4. 计划输入

每个任务至少包含：

~~~json
{
  "task_id": "139",
  "source_video": "/path/to/source.mp4",
  "sequential_nominal_plan": [
    {
      "stage_id": "S1",
      "audited_content_only_prompt": "Apply the first atomic edit."
    }
  ]
}
~~~

加载时会检查 task_id、source_video、连续的 S1/S2/... 编号和每个 stage 的非空 audited_content_only_prompt。实际只取这个 atomic prompt；完整结构化计划、其他 stage、依赖关系、CoVEBench 字段和 planner 原始回复不会发送给 Qwen 或 H3。

## 5. 初始视频和几何归一化

传入 prepared-initial-video 时，文件必须已经是 1344x768、107 帧、24fps、带音轨，并带有匹配的 .geometry.json。否则脚本从源视频计算可逆的 letterbox geometry：

~~~python
scale = min(1344 / source_width, 768 / source_height)
content_width  = floor_even(source_width * scale)
content_height = floor_even(source_height * scale)
offset_x = (1344 - content_width) // 2
offset_y = (768 - content_height) // 2
~~~

随后执行 Lanczos 等比例缩放、黑边补齐到 1344x768、按完整源时长均匀重采样到 107 帧、必要时复制末帧、设为 24fps，并编码为 H.264/yuv420p/CRF18。初始视频使用静音 stereo AAC 48kHz 音轨，原视频音频不会保留。

实现函数 `materialize_initial_video()` 位于
`src/apimart_h3_pipeline/media/video.py`。

## 6. 每个 stage 的 parent video

~~~text
S1 = 初始归一化视频
S2 = S1 归一化后的输出
S3 = S2 归一化后的输出
...
~~~

每个 stage 都从当前 parent video 抽取 0、26、53、80、106 五帧。这五帧用于 Qwen 内容观察、编写 H3 prompt 和检查输出；参考图主帧固定取首帧 0。

同一个 stage 的 retry 使用进入该 stage 时锁定的 `stage_parent_video` 和 `stage_parent_url`。首轮失败的 H3 输出会被 Observer 读取并写入 `attempts/attempt_1/`，绝不会成为自己的 retry 输入。唯一 retry（`attempt_2`）仍使用原始 parent；retry 生成后不再调用 Observer，而是验证媒体、物化为该 stage 的标准输出并作为下一 stage parent。每次 attempt 还记录 `h3_input_video` 与 `h3_input_video_url`，因此可以直接审计 retry 是否仍指向原始 parent。对于 CTMOAI 恢复的持久化请求，如果其视频 URL 与锁定 parent 不一致，runner 会中止而不是继续提交。

## 7. 参考图策略

reference_policy() 是本地正则启发式，不是 Qwen 分类器。它会先清理空白，并在 to keep、while preserving、keep、preserving 等保持性短语处分割，只用前半段作分类。

优先级如下：

~~~text
全局风格 -> 纯动作/姿态 -> 静态视觉 -> 时序/音频 -> 普通对象编辑 -> 默认使用参考图
~~~

全局风格关键词包括 style、appearance、look、watercolor、oil painting、anime、vintage、sepia、monochrome 等。静态视觉关键词包括 recolor、lighting、rain、wet、reflective、background、face、hair、material、add、remove、replace、reposition 等。

明确的纯动作/姿态变化（例如 action、pose、jump、run、walk、sit、stand、flap 等）也不生成静态图片参考，而是只把原始视频交给 H3，并附加动作的自然起始和时间展开约束。若动作同时涉及对象替换、报纸/白板文字、持物/阅读等静态内容，则回到内容参考图路径。camera、pan、push-in、pull-out、zoom、dolly、tilt、orbit、tracking、motion、movement、speed、sway、blur、temporal、fps、audio、sound、music、voice、wind 等通常判为不需要参考图。

如果一个 prompt 同时包含静态视觉词和 motion/camera 词，静态视觉分支先匹配，因此可能仍会使用参考图。这是当前实现的实际行为，需要在编译计划阶段避免含义混杂的 stage prompt。

全局风格首次执行固定使用一张 primary reference。`global-style-reference-count` 只控制普通静态、普通对象和模糊默认分支；即使把它设为 3，也不会把全局风格提前变成三锚点。当该 stage 的输出被 Observer 判定失败时，runner 已经知道 `is_global_style=true`，会直接创建三锚点 repair record 并在同一 stage 重试；这才是全局风格的时序锁定路径。

## 8. Qwen-VL 选帧和图片编辑

需要参考图时，Qwen 仍接收原始 atomic prompt 和 parent video 的五帧，用于内容观察和审计；参考帧不再由 Qwen 自由选择，普通路径固定使用上下文首帧 `frame 0` 作为 primary style master。只有进入三锚点 retry/显式三图实验时，才额外编辑中间帧 `frame 53` 和末帧 `frame 106`；这两次图片编辑都把首帧生成的 primary style master 作为 `style_reference`。Qwen 返回观察和固定帧契约的元数据。

S1 只要需要图片编辑，使用一张已经按 H3 画布处理的带黑边参考帧，调用 GRSAI `nano-banana-2` 做一次 atomic image edit；不在图片编辑阶段裁剪、回填或再次扩展画布。三锚点 S1 的首帧、中间帧和末帧分别使用各自的带黑边 H3 参考帧编辑，中间帧和末帧额外接收首帧作为风格参考。S1 之外的 stage 使用 `gpt-image-2`，仍保持每个普通 stage 一轮图片编辑。

图片编辑服务是 GRSAI；模型按 stage 选择：

~~~text
POST <GRSAI_BASE_URL>/v1/api/generate
model       = nano-banana-2  # S1
model       = gpt-image-2    # S1 之外的 stage
replyType   = async
aspectRatio = closest supported ratio of the actual edit input
~~~

图片编辑 prompt 始终由代码从原始 atomic prompt 构造，只增加一个不改变语义的保持约束：

~~~text
image_edit_prompt = image_edit_prompt(next_raw_prompt)
# 结果示例：<原始 requirement> Preserve all other elements exactly as they are.
~~~

因此 Qwen 不会把观察到的对象、材质、颜色或场景细节自行添加到图片编辑指令；图片模型只执行原始编辑，并明确保持其他元素不变。该约束通过 `resources/prompts/image_edit_preservation.txt` 外置，恢复时会按请求字段匹配，不会重复提交已经完成的图片任务。图片任务提交后每 5 秒查询，下载 PNG，并保存独立的 image_edit_state 文件。如果 GRSAI 明确返回 `excessive system load`，provider 会把状态标为 `waiting_for_capacity`，等待 60 秒后用同一输入重新提交，直到成功；其他失败仍立即停止。

## 9. Qwen 生成最终 H3 prompt

普通一图路径（包括全局风格首次执行）下，Qwen 会看到五个 parent-video 帧、实际生成的 primary reference 和原始 atomic prompt，然后生成最终 H3 prompt。进入三锚点 retry 后采用本地 temporal-anchor contract，避免再付费调用一次自由式 H3 prompt compose；repair 路径只允许闭集 repair clause。

一张图时，prompt 必须包含：

~~~text
<Picture 1>
<Video 1>
~~~

Picture 1 负责外观和静态变化；Video 1 负责原视频动作、运动和时间推进。Qwen 不得添加原始 prompt 中不存在的编辑，不得泄露结构化 wrapper，并且 prompt 不得超过 1200 字符。

三张图时，固定语义为（帧索引来自 `QWEN_CONTEXT_FRAME_INDICES` 的首项、中间项和末项）：

~~~text
<Picture 1> = edited start anchor / first-frame style master, source frame 0
<Picture 2> = edited primary anchor (middle temporal anchor), source frame 53
<Picture 3> = edited end anchor, source frame 106
<Video 1>    = 原视频的动作、运动和时间结构
~~~

其中 Picture 1 由 parent 的第一帧直接编辑得到，同时是后续两张图的共同风格母版；Picture 2 和 Picture 3 分别从 parent 的中间帧和末帧编辑得到，并显式接收 Picture 1 作为 `style_reference`。H3 prompt 必须包含全部标签、source frame 编号、anchor role，并要求 start -> primary -> end 平滑过渡。非法 JSON 或校验失败时，会把上一次输出和错误追加回上下文，最多修复 3 次。

三锚点 retry 还必须要求 frame lock：start、primary、end 三个时间位置分别锁定对应的编辑后参考外观，所有中间帧保持该编辑风格，不能回退到源视频的原始外观。这个要求是 H3 的输入语言契约，不宣称像素级逐帧复制。

无参考图时，Qwen 仍会观察五帧，但最终传给 H3 的 prompt 被固定为：

~~~text
Apply only this edit to <Video 1>: <原始 atomic prompt>
~~~

第一次 H3 推理前，只有 atomic requirement 明确表达镜头运动时才会注入简短 camera 澄清：左右移动按请求方向做平滑水平 pan 并保持同一主体焦点，push-in/pull-out 按实体相机前后运动，zoom 按镜头 framing，其他已命名运动保持其原类型。每种澄清句存放在 `resources/prompts/h3_camera_motion_clauses.json`，公共约束存放在 `h3_camera_motion_contract.txt`。Qwen 生成 H3 prompt 时必须先写当前操作，再补充必要的 `<Picture>/<Video>` 关系；不添加对象、动作、风格或新的 requirement，也不让图片模型参与 camera motion。

## 10. H3 生成后端

线上模式通过 `providers/apimart.py` 子进程传入：

~~~text
--prompt <最终 H3 prompt>
--model MiniMax-H3
--duration <4-15>
--resolution 768P
--aspect-ratio 16:9
--video-url <当前 parent video URL>
--image-url <参考图 URL，可重复>
~~~

APIMart 使用 POST /v1/videos/generations，字段是 video_urls、image_urls、duration。CTMOAI 使用 POST /v1/videos，字段是 reference_videos、images、seconds；16:9 的 768P 明确设置为 1376x768。

CTMOAI 会把初始视频、每轮 stage 视频和参考图片上传到稳定 media store；APIMart 通常使用 media-public-base-url 暴露的视频和图片 URL。H3 POST 只提交一次，相同请求按 task state 恢复，状态 GET 按轮询间隔重试，视频下载支持断点续传。

本地模式通过 `providers/local.py` 向 ComfyUI 的 `/prompt` 提交 API-format
workflow，并轮询 `/history/{prompt_id}`。runner 将当前 parent video 和 bridge
生成的参考图复制到 `--local-input-dir`，再从 SaveVideo history 输出复制到
当前 stage 的 `output.mp4`。local workflow 必须包含一个 H3
`ReferenceToVideo` 节点、一个 `LoadVideo` 节点和一个 `SaveVideo` 节点；节点按
`class_type` 发现，模板可以来自不同的 ComfyUI 导出。

## 11. 五帧成功门、失败诊断和定向修复

H3 输出完成后，Qwen 用输出的 0、26、53、80、106 五帧检查 atomic edit：

~~~json
{
  "success": false,
  "failure_type": "identity_drift",
  "observation": "The person's identity is not preserved across output frames.",
  "observer_evidence": "face and clothing changed",
  "confidence": 0.92
}
~~~

静态编辑缺失、部分成功、不一致或不明确时返回 `success=false` 和闭集 `failure_type`。camera、motion、temporal、audio 无法仅凭静态帧判断时返回 `success=true` 和 `not_frame_judgeable`；这不证明运动质量，只表示静态 gate 不触发错误修复。Qwen 网络失败记录为 `success=null`、`observer_unavailable`，默认不触发额外付费重试。

定向修复按 failure type 选择下一次执行策略。全局风格首次执行只有一张图；一旦该 stage 失败，runner 直接把它导向三锚点 retry。普通静态编辑只有 `style_inconsistency`，或显式 `fixed-three-anchor` 实验基线，才使用三锚点：

~~~python
policy["is_global_style"] and first_attempt_used_one_anchor
或 diagnosis["failure_type"] == "style_inconsistency"
或 --failure-recovery fixed-three-anchor
~~~

每次付费重试前先把输出、state、bridge 和 observation 归档到 `attempts/attempt_N/`，并优先复用归档的首帧 primary style master。需要三锚点时动作随后：

1. 使用 parent 原始上下文首帧生成或恢复 Picture 1 style master；
2. 用 parent 原始上下文中间帧编辑 Picture 2 primary/middle anchor，并把 Picture 1 作为 `style_reference`；
3. 用 parent 原始上下文末帧编辑 Picture 3 end anchor，并把 Picture 1 作为 `style_reference`；
4. 组成首帧/中帧/末帧三锚点；
5. 重新生成 H3 prompt，提交同一 stage；
6. retry 输出只做媒体合法性检查，不再调用 Observer；物化后作为下一 stage parent。

定向或三锚点 retry 只允许执行一次；该预算是 runner 的固定协议，不提供 CLI 覆盖。首轮失败归档为 `attempt_1`，retry 为 `attempt_2`。retry 输出不再进行第二次 Observer 评估，而是记录为 `semantic_failure_propagated`，物化为当前 stage 的标准输出并作为下一 stage parent；sequence 最终状态为 `degraded`。不可修复、Observer 不可用或媒体无效且没有可传播 retry 时仍停止。

## 12. Stage 输出归一化和最终输出

线上输出必须是 107 帧、24fps、高度 768、宽度 1344 或 1376。进入下一 stage 前执行 materialize_stage_video()：

1. 若宽度是 1376，先居中裁到 1344；
2. 按原始 geometry 裁掉 letterbox 外部区域；
3. 将内容放回 1344x768 黑边画布；
4. 保持 107 帧和 24fps；
5. 保留已有音频，缺失时补静音音频；
6. 写入 stage_input 类型的 geometry sidecar。

全部 stage 完成后，materialize_final_video() 只裁剪内容区域：

~~~text
crop=<content_width>:<content_height>:<offset_x>:<offset_y>
~~~

最终文件为 <out-dir>/output.mp4，保持 107 帧、24fps、H.264、yuv420p 和源视频宽高比。最终尺寸是 1344x768 画布拟合后的内容尺寸，不一定等于源视频像素尺寸。

## 13. 断点恢复产物

~~~text
<out-dir>/sequence_manifest.json
<out-dir>/stages/Sn/apimart_task_state.json
<out-dir>/stages/Sn/bridge_for_next/bridge.json
<out-dir>/stages/Sn/bridge_for_next/*qwen_reference_plan.json
<out-dir>/stages/Sn/bridge_for_next/*three_anchor_plan.json（全局风格或三锚点 repair）
<out-dir>/stages/Sn/bridge_for_next/*image_edit_state*.json
<out-dir>/stages/Sn/observation/observation.json
<out-dir>/stages/Sn/attempts/attempt_1/attempt.json
~~~

复用前会比较 stage_id、previous_video、raw prompt、reference policy、Qwen model、failure observation、H3 prompt、参考图 URL 和 H3 参数。只有完全一致时才复用；仅有旧 output.mp4 不足以跳过新请求。

如果进程在某次失败并归档、但下一次请求尚未提交时退出，重启会识别最新的 `attempt_N/` 和 repair record，恢复对应的一图或三锚点路径；不会因目录存在就无条件升级三锚点。

## 14. 服务、环境和运行模板

~~~text
APIMART_API_KEY / CTMOAI_API_KEY：视频生成
DASHSCOPE_API_KEY：Qwen-VL
GRSAI_API_KEY：nano-banana-2

~/.apimart.env（也可通过 `APIMART_ENV_FILE` 指定）
~/.grsai.env（也可通过 `GRSAI_ENV_FILE` 指定）
~/.dashscope.env（也可通过 `DASHSCOPE_ENV_FILE` 指定）
~~~

Qwen-VL 和 GRSAI 使用 direct opener；APIMart H3 客户端继承进程中的 HTTP/HTTPS 代理。当前 refiner 拒绝 qwen-max 和 qwen3-max，默认使用 qwen-vl-plus。

~~~bash
python scripts/run_apimart_minimax_h3_sequential.py --compiled-jobs /path/to/compiled_jobs.json --task-id 139 --out-dir /path/to/run/task_139 --media-dir /path/to/media/task_139 --media-public-base-url http://<public-media-host>:<port> --apimart-env /secure/apimart.env --grsai-env /secure/grsai.env --dashscope-env /secure/dashscope.env --duration 4 --resolution 768P --aspect-ratio 16:9
~~~

`--last-stage S2` 可截断到指定 stage；`dry-run` 只写计划和策略，不访问在线服务；`initial-reference` 只是历史兼容参数，不会让 S1 走特殊流程。失败恢复相关参数只有 `--failure-recovery targeted|fixed-three-anchor|disabled` 和兼容旧行为的 `--allow-unverified-output`；每个 stage 的 retry 预算固定为一次。

## 15. 当前实现的关键边界

1. stage 顺序完全来自输入计划，runner 不自动把风格编辑移到最后。
2. 是否需要参考图由本地 reference_policy() 判断，不是 Qwen 判断。
3. Qwen 观察五帧，但不再自由选择主帧；图片编辑固定使用首帧 0，且 prompt 始终是原始 atomic prompt。S1 图片编辑使用一张带黑边 H3 画布帧做一次 edit。
4. 无参考图时 Qwen 仍观察五帧，但 H3 使用原始 prompt 加 Video 绑定。
5. 全局风格首次使用一张 primary reference，失败后直接升级三锚点；普通编辑默认按 failure type 定向修复，只有 `style_inconsistency` 或显式基线才在失败后升级三锚点。
6. 下一轮输入是上一轮归一化后的输出，不是永远重新使用源视频。
7. 中间视频固定为 1344x768、107 帧、24fps，最终输出才移除 letterbox。
8. Qwen、GRSAI 和 H3 的状态均持久化，并按已保存请求字段精确匹配恢复；项目不新增指纹或 ID。

## 16. 已接入的失败诊断与定向修复

当前实现对 `success=false` 的静态失败先归一化为闭集 `failure_type`，再按动作表改写当前 H3 prompt：

```text
五帧观察 -> failure_type -> 当前 stage 定向修复 -> 同一 stage 重试 -> 再观察
```

例如，编辑缺失只强化当前 edit 的可见性，人物身份漂移只增加 identity preservation，前序编辑消失只增加已确认父 stage 的保持约束，风格时序不一致才使用三锚点，camera push-in 太弱则只强化原有 motion wording。该组件不修改 `sequential_nominal_plan`、DAG、CLT 或 CEG，每个 stage 固定只允许一次 retry。

旧的 `runs/apimart_task139_S1_targeted_from_attempt1_20260901/` 实验把 `attempt_1/output.mp4` 复制成了 `public_media/task_139_initial.mp4`，两者 SHA256 都是 `dcc99c...`，因此它不满足“retry 只能使用干净原始视频”的输入约束，不能作为该约束的有效证据。按同样的 `attempt_1` 失败结果、但改用干净原始 normalized video 重新恢复的验证目录是 `runs/apimart_task139_S1_target_runner_original_parent_20260901/`：retry 的 initial SHA256 为 `749290...`，失败输出 SHA256 为 `dcc99c...`；H3 请求使用三张 temporal anchors 和该 initial URL，Observer 在 0/26/53/80/106 五帧确认油画风格（`success=true`, `confidence=0.95`），S1 一次 retry 成功且没有执行第二次 retry。这说明 parent 不可变约束在真实线上流程中生效，同时三锚点是否成功仍受 provider 采样影响。

详细的输入/输出 JSON、代码接入位置、闭集失败类型、恢复记录和 fixed-three-anchor 对照实验见：

`docs/FAILURE_DIAGNOSIS_AND_TARGETED_REPAIR_CN.md`
