# S1 Observer-only 批次审计与 Fallback 优化建议

审计对象：`/root/autodl-tmp/Pipline_runs/s1_observer_only_20260905`

审计日期：2026-09-07

本报告只读检查了该目录中的任务目录、S1 输出、bridge、状态文件、Observer JSON、sequence manifest 和 task 日志。报告中的“失败”指该批次的 Observer 或运行状态没有通过，不等同于视频文件一定损坏。

## 1. 总体统计

| 项目 | 数量 |
|---|---:|
| 数字 task 目录 | 496 |
| 有 `stages/S1/output.mp4` 的 task | 496 |
| 有 `observation/observation.json` 的 task | 482 |
| Observer 成功 | 304 |
| Observer 语义失败 | 178 |
| 有 `local_task_state.json` 的 task | 491 |
| 有 `bridge.json` 的 task | 491 |
| 有完整图片编辑状态的成功记录 | 451 |

因此，本批次没有出现 S1 输出视频整体缺失的问题；主要问题是输出视频虽然生成，但没有通过 Observer 的语义成功门。

任务级 manifest 状态为：

- `success`：300 个。
- `semantic_failure`：178 个。
- `skipped_camera_motion`：5 个。
- `task_322` 的 manifest 没有 `stages`，属于任务在阶段记录完成前中断或提前结束。

## 2. Observer 语义失败

178 个失败按 `failure_type` 分布如下：

| failure_type | 数量 | 解释 |
|---|---:|---|
| `edit_missing` | 156 | 请求的修改没有出现、只出现部分，或跨帧不稳定 |
| `identity_drift` | 14 | 主体身份、外观或主体内容发生漂移 |
| `motion_weak` | 7 | 请求的动作、速度、运动方向或位置变化不明显 |
| `composition_weak` | 1 | 构图、布局或主体位置没有达到要求 |

`edit_missing` 占语义失败的 87.6%，是最需要优化的路径。

按原子 prompt 的操作类型粗分，失败情况如下：

| 操作类型 | `edit_missing` | 其他失败 | 典型问题 |
|---|---:|---:|---|
| 动作、速度、行为变化 | 52 | 6 | 动作没有持续发生，或只改变了单帧 |
| 替换人物、物体、服装、文字 | 39 | 7 | 原对象仍存在，替换不完整，或身份漂移 |
| 移动人物或物体位置 | 33 | 6 | 画面位置没有改变，或被误做成相机移动 |
| 删除元素 | 21 | 0 | 要删除的对象仍可见，遮挡区域补全失败 |
| 添加对象或 VFX | 5 | 2 | 新对象没有出现，或主体身份被带偏 |
| 尺寸放大、缩小 | 4 | 0 | 尺寸变化不足，比例和位置不稳定 |
| 其他 | 2 | 0 | 目标变化与保留内容的边界不清楚 |

Observer 失败任务的置信度并不是普遍偏低；大量失败的置信度是 `0.95` 或 `1.0`。因此不能只用降低置信度阈值来解决，应该修复 H3 prompt、参考图使用和重试验证路径。

## 3. 基础设施和传输错误

从 task 日志中发现 23 个 task 有明确的底层异常。下面的数量按 task 去重；这些 task 可能同时有 Observer 失败。

| 错误 | 数量 | task |
|---|---:|---|
| GRSAI 图片任务 `status=failed` | 7 | 23、41、57、62、99、100、289 |
| GRSAI 任务没有返回图片 URL | 12 | 37、80、86、120、125、162、184、186、191、204、260、288 |
| GRSAI 网络连接被重置 | 1 | 268 |
| 本地 ComfyUI 任务从 queue/history 消失 | 3 | 19、20、33 |

这些错误不能全部交给语义 fallback：

- `status=failed` 应进入图片编辑任务重试或失败后跳过分支。
- 没有 URL 应保存完整响应状态和 task id，并允许按原 task id 恢复查询；不能只把它当成普通空结果。
- 网络错误应使用传输重试和退避，不应消耗语义修复次数。
- ComfyUI queue/history 消失时应检查服务存活、队列状态和输出目录，再决定是否重新提交。

## 4. Observer 和状态缺失

缺少 `observation/observation.json` 的 task：

`5、9、10、11、13、15、18、21、31、239、322、451、562、579`

其中：

- `31、239` 的日志记录了 `camera_motion_stage_disabled`，属于被相机阶段跳过后没有执行 Observer 的任务。
- `322` 有输出视频，但 `sequence_manifest.json` 的 `stages` 为空，属于阶段记录不完整。
- `451、562、579` 有 S1 输出，但没有完整的状态和 bridge 元数据，应支持从现有输出重建状态，或明确标记为不可恢复。
- 其余缺失 Observer 的 task 需要从日志和输出文件判断是复制已有结果、提前退出还是 Observer 调用未完成。

Fallback 应把“Observer 不可用/状态缺失”和“Observer 语义失败”分开。前者不能自动归类成 `edit_missing`，否则会对一个没有被检查的输出错误地做语义重试。

## 5. 当前代码行为与本批次的直接原因

本批次使用的脚本是 [`scripts/run_s1_observer_only.sh`](../scripts/run_s1_observer_only.sh)，每个任务传入：

```text
--last-stage S1
--failure-recovery disabled
```

所以这次运行的设计是“生成 S1、执行 Observer、记录结果后停止”。Observer 返回失败时，runner 会记录 `semantic_failure` 并退出当前 task，不会执行修复。这解释了为什么 178 个失败没有产生 fallback 重试结果。

当前 [`execution/runner.py`](../src/apimart_h3_pipeline/execution/runner.py) 和 [`core/repair_policy.py`](../src/apimart_h3_pipeline/core/repair_policy.py) 还有以下限制：

1. `STAGE_RETRY_LIMIT = 1`，每个阶段最多一次语义修复。这个固定值与“最多重试 10 次”的运行要求不一致。
2. 语义重试成功提交 H3 后，重试输出被标记为 `observer_skipped_retry`，不会再次调用 Observer；未验证的视频可能直接成为后续 stage 的输入。
3. `edit_missing`、`identity_drift`、`motion_weak` 和 `composition_weak` 已有动作映射，但主要依靠通用修复子句，未针对动作、替换、删除、添加、位置、速度、文字和尺寸做细分。
4. `motion_weak` 使用 `video_only` 参考策略，这适合纯相机或速度操作，但不适合所有主体位置和主体动作变化。
5. 图片编辑传输失败与 H3 语义失败共享任务级错误路径，日志中难以区分“没有得到可用参考图”和“参考图正确但视频编辑没有完成”。

## 6. Fallback 优化优先级

建议按以下顺序修改：

### 6.1 先分离传输恢复和语义恢复

建立独立的错误类别和状态字段：

- `image_edit_failed`
- `image_edit_no_url`
- `image_edit_timeout`
- `image_edit_network_error`
- `local_queue_lost`
- `observer_unavailable`
- `semantic_failure`

传输重试不消耗语义修复次数。每次重试必须保存 provider、task id、HTTP 状态、响应 detail、重试次数和最后一次轮询时间。

### 6.2 对 `edit_missing` 按操作类型修复

不要只追加一段通用 preservation 文本，应让修复 prompt 明确：

- 动作：动作从何时开始、持续到哪些帧、主体哪些外观必须不变。
- 替换：只替换指定对象，原对象不能残留，遮挡区域要自然补全。
- 删除：对象完全不可见，背景补全，其他对象和相机保持原样。
- 添加：新增对象的位置、大小、动作和持续时间，禁止改变原主体。
- 位置：区分画面内重新布局和相机运动，保持物理动作连续。
- 速度：只改变指定运动速度，保留轨迹、主体和场景。
- 文字：固定文字内容、位置、字体可读性和跨帧一致性。
- 尺寸：指定目标比例和参照物，避免只做轻微缩放。

### 6.3 对 `identity_drift` 加强主体锁定

修复时必须重新声明主体的身份、衣着、体态、动作、位置和时序，并明确只有指定属性允许改变。人物替换任务要区分“替换人物身份”和“保留原人物动作/位置”，避免模型把整个场景重新生成。

### 6.4 对 `motion_weak` 区分四种运动

将运动修复拆成：相机运动、主体运动、物体运动、速度/时间变化。每种类型需要不同的 prompt 约束和 Observer 检查条件，不能全部使用同一条 camera motion 规则。

### 6.5 重试后重新 Observer

语义重试输出应再次抽取五个时间点，并重新执行 Observer。只有 `success=true` 才能进入后续 stage；`observer_skipped_retry` 只能作为中间状态，不能作为已验证成功。

### 6.6 状态恢复和幂等

对已有输出但缺少状态文件的任务，应从输出视频、workflow、manifest 和日志重建状态。对已有图片编辑 task id 的任务，应先恢复轮询，避免重复提交造成多个孤立任务。对 ComfyUI queue 消失的任务，应在重新提交前检查输出目录，防止重复生成和覆盖。

### 6.7 将重试次数改成配置项

语义重试、图片编辑重试、网络重试和本地队列恢复应分别配置上限。建议至少提供：

```text
semantic_retry_limit
image_edit_retry_limit
network_retry_limit
local_queue_retry_limit
```

达到上限后记录结构化失败原因并继续下一个 task；不要因为单个 task 的异常终止整个批次。

## 7. 结论

本批次最主要的问题不是视频没有生成，而是 178 个已生成的视频没有通过 Observer，其中 156 个属于目标编辑没有完成或不稳定。另有 23 个 task 存在 GRSAI 或 ComfyUI 基础设施错误，14 个 task 缺少 Observer 或状态记录。由于运行脚本显式关闭了 failure recovery，这些失败没有进入 fallback 属于当前批次的预期行为；后续完整推理应启用修复模式，并按本报告的传输恢复、语义修复、重试后复核和状态恢复四条路径分别处理。

## 8. 已实施的 Qwen-VL Fallback 改造

fallback 的修复 prompt 已改为由 Qwen-VL 根据实际上下文重新编写。修复调用会同时提供：

- 原始 atomic prompt。
- 第一次提交给 MinMax-H3 的失败 prompt。
- Observer 的 `failure_type` 和可读错误证据。
- 五个 parent-video 时间采样帧。
- 当前策略生成的 `<Picture N>` 参考图及其 frame role。

Qwen-VL 只返回新的纯文本 H3 prompt，并被要求保留原始编辑目标，不得从错误证据或参考图中发明额外编辑。参考图不是强制项：对于参考图会损害主体或背景保持的任务，修复 prompt 可以依赖 `<Video 1>`；对于需要跨时间保持外观的任务，可以使用首帧、中间帧和末帧作为可选时序锚点。修复结果在 bridge 中标记为 `qwen_vl_failure_repair`，便于后续审计。

本改造只改变语义 fallback 的 prompt 编排，不改变 GRSAI、网络或 ComfyUI 基础设施错误的处理，也不会在 Observer-only 批次中自动触发 H3 生成。

fallback 的参考图拓扑不是由原始任务是否有图、`failure_type` 或某个编辑类型固定决定的。Qwen-VL 会在看到原始 prompt、失败前 H3 prompt、Observer 证据、五个输入帧和已有 Picture 后，返回 `video_only`、`one_anchor` 或 `three_anchor`。如果选择三锚点，模型必须提供三个不同的 image-edit prompt；bridge 只负责把模型给出的时序提示映射到首帧、选定的内部帧和末帧，并把它们作为三张独立参考图传给 H3。

### 修复后的 prompt 输出位置

每次 fallback bridge 完成后，优化后的最终 H3 prompt 会同时保存到：

- `stages/<stage>/bridge_for_next/bridge.json` 的 `h3_prompt` 字段。
- 同一个 `bridge.json` 的 `optimized_h3_prompt_path` 字段指向的纯文本文件。
- `fallback_result.json` 的 `h3_prompt` 字段（bridge-only 验证结果）。

10 个样例的可读 prompt 汇总在：

`/root/autodl-tmp/Pipline_runs/fallback_qwen_10_bridge_only_20260907_final/optimized_prompts/`

例如 task 26 的修复 prompt 是：

`/root/autodl-tmp/Pipline_runs/fallback_qwen_10_bridge_only_20260907_final/optimized_prompts/task_26_S1_optimized_h3_prompt.txt`

这个文件的来源标记为 `h3_prompt_source=qwen_vl_failure_repair`。它是 Qwen-VL 根据原始 prompt、失败前 prompt、Observer 报错、输入帧和参考图重新生成的最终 H3 prompt，不是原始 Qwen prompt 的简单复制。`qwen_failure_repair_plan.json` 单独保存参考图拓扑和每张图片的 image-edit prompt；它不是最终 H3 prompt。
