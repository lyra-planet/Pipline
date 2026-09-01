# VETRA 失败诊断与定向修复设计

> 本文把“Observer 未确认后统一三锚点重试”实现为“先诊断失败类型，再只修复当前 stage 的失败部分”。文档以当前实现
> `scripts/run_apimart_minimax_h3_sequential.py` 为代码基线，说明 VETRA 执行层如何在不修改 MSR 控制平面的前提下运行。
>
> **状态：已接入目标 runner，尚未完成线上付费端到端验证。** 本文的 schema、策略、bridge 和 mocked state machine 已实现并测试；真实 Task 139 的 provider 请求、轮询、上传和费用仍需受控运行确认。

## 1. 结论先行

当前执行器的默认恢复路径是：

```text
MiniMax-H3 输出
    -> Qwen-VL 五帧 success gate + failure_type
    -> success=false
    -> 按 failure_type 选择当前 stage 的定向 repair action
    -> 全局风格首次是一张图；失败时直接使用三锚点
    -> 普通编辑只在 style_inconsistency 或 fixed-three-anchor 基线时使用三锚点
    -> 同一个 stage 固定只重试 1 次（无 CLI 覆盖）
```

已接入的 VETRA 执行层是：

```text
MiniMax-H3 输出
    -> Qwen-VL 五帧观察 + failure diagnosis
    -> failure_type
    -> 只修复该类型对应的 prompt/reference 部分
    -> 同一个 stage 重试
    -> 再观察；仍失败则停止传播并记录 semantic_failure
```

这不是重新规划器，也不是对失败结果进行自由发挥。它只在已经冻结的 `sequential_nominal_plan` 内工作：

- 当前 stage ID 不变；
- 当前 `audited_content_only_prompt` 是不可变的语义锚点；
- 只能增加针对失败的连续性或可见性约束，不能增加新的对象、动作、风格、镜头或音频 requirement；
- 不重排 stage，不修改 DAG、CLT、CEG 或编译计划；
- 每个 stage 固定只进行 1 次定向修复，不提供额外重试配置；
- 修复失败不能把未确认的视频静默当作成功父状态传给下一 stage。

### 1.1 研究想法的迁移边界

这个想法与 Adaptive Task Reformulation 和 EditRefiner 的共同点是：模型先观察执行结果，再根据失败证据改写当前任务表达，而不是盲目重复相同请求。迁移到本系统时不需要训练新的模型，原因是当前执行器已经具备三个可复用的原语：

1. Qwen-VL 可以看到 parent/output 的五帧并给出局部视觉证据；
2. `compose_h3_prompt()` 可以在保持原子语义的前提下生成 H3 prompt；
3. `bridge_for_stage()` 已经能在同一个 stage 内复用父视频、参考图和请求状态。

因此 VETRA 的实现不是自由的“再规划”，而是一个受限的错误恢复器：

```text
冻结的 atomic requirement
        +
五帧失败证据
        +
闭集 repair action / allow-list 短语
        -> 当前 stage 的新执行参数
```

与研究论文中的通用 reformulation 相比，本系统需要更强的工程约束：

| 维度 | 通用任务重写 | VETRA 定向修复 |
| --- | --- | --- |
| 重写范围 | 可以重述整个任务 | 只能重写当前 atomic requirement 的执行表达 |
| 任务拓扑 | 可能重新规划步骤 | `sequential_nominal_plan`、DAG、CLT/CEG 全部冻结 |
| 反馈 | 任意 evaluator 反馈 | 只使用当前 stage 的五帧 observer evidence |
| 动作空间 | 自由生成下一任务 | `failure_type -> repair_action -> reference_policy` 闭集映射 |
| 失败后状态 | 可选择新状态 | semantic failure 不得更新 parent，不得静默传播 |
| 预算 | 固定协议 | 每 stage 恰好最多 1 次 retry |

这也解释了为什么定向修复比固定三锚点更适合本 pipeline：除全局风格这种天然需要跨时间外观约束的 stage 外，三锚点只解决“跨时间外观约束不足”；编辑缺失、身份漂移、前序编辑消失、运动过弱和构图变化过小分别属于语义可见性、保持约束、历史状态保持、时序强度和空间表达问题。把所有普通失败都变成三张图片会增加 GRSAI/H3 成本，却不一定修复真正的因果缺陷。

## 2. 当前代码基线

### 2.1 当前状态机的真实行为

目标 runner 的主循环位于 `scripts/run_apimart_minimax_h3_sequential.py` 的 `main()`，约第 1837 行开始。每个 stage 的实际顺序是：

1. `load_task()` 只读取编译计划中的 `task_id`、源视频和连续的 `S1...Sn` 原子 prompt；
2. `reference_policy()` 用本地正则决定 `video_only/one_anchor/three_anchor` 初始策略；
3. `bridge_for_stage()` 抽取五帧，调用 Qwen-VL 观察内容，固定使用 parent 首帧作为主风格参考，调用 GRSAI 编辑静态参考图，并生成带 `<Video 1>`/`<Picture N>` 的 H3 prompt；
4. `invoke_h3_client()` 通过 `run_apimart_minimax_h3.py` 提交或恢复同一 H3 请求；
5. `is_aligned_video()` 做媒体有效性检查；
6. `observe_stage_output()` 抽取 `0,26,53,80,106` 五帧并调用 Qwen-VL success gate，返回闭集 `failure_type`；
7. `FailureDiagnosisAndRepair` 根据失败类型生成一个 stage-local repair record，校验当前 requirement token、reference policy、重试预算和控制平面 guard；
8. `bridge_for_stage()` 对定向 repair 使用确定性 prompt，复用已归档的首帧 primary style master；全局风格的一图首次尝试失败时直接构造首帧/中帧/末帧三锚点，普通编辑只有 `style_inconsistency` 或显式 `fixed-three-anchor` 模式才构造三锚点；
9. `stage_outcome()` 是严格传播闸门：语义成功才更新 parent；最终语义失败写入 manifest 并停止，Observer 不可用默认写入 `observation_pending` 并停止。

`--allow-unverified-output` 仅为旧实验保留：Observer 返回 `success=null` 时允许媒体继续流转，但 manifest 仍标记 `observation_pending`，不会伪造语义成功。

### 2.2 代码位置和现有字段

| 责任 | 当前代码 | 当前输出/输入 | 定向修复接入点 |
| --- | --- | --- | --- |
| 计划读取 | `load_task()`，约 1286 行 | `stage_id`、`prompt` | 只读，不允许 repair 修改 |
| 参考策略 | `reference_policy()`，约 409 行 | `needs_reference_image`、`reference_image_count` | diagnosis 只可选择下一次策略 |
| 五帧抽取 | `select_keyframe()`、`observe_stage_output()`，约 957/1012 行 | `0,26,53,80,106` | 诊断的证据来源 |
| 输出观察 | `DashScopeVisionRefiner.observe()`，约 957 行 | `success`、`failure_type`、`observer_evidence`、`confidence` | 结构化失败诊断 |
| H3 prompt | `compose_h3_prompt()`，约 727 行 | `h3_prompt`、`frame_observation` | 接收受限 repair context |
| 图片桥 | `bridge_for_stage()`，约 1425 行 | `bridge.json`、图片 URL、reference roles | 应用 `reference_policy` |
| 三锚点规划 | `three_anchor_reference_plan()` | `middle/end_image_edit_prompt`、`style_reference_frame_index` | 本地确定性元数据；首帧是共享 style master，repair 只在 `style_inconsistency` 等类型触发 |
| stage 状态机 | `main()`，约 1928 行 | `attempts`、`post_edit_observation` | 诊断、修复、传播闸门 |
| 在线 H3 请求 | `scripts/run_apimart_minimax_h3.py`（实现位于 `src/apimart_h3_pipeline/providers/apimart.py`） | request/poll/download state | 修复后按变化后的请求字段提交新 task |
| 现有严格尝试模式 | `scripts/run_preplanned_h3_full_pipeline.py`，约 333-470 行 | attempt manifest、`semantic_failure` | 可借鉴停止传播和 stage attempt 结构 |

### 2.3 当前 observer 与诊断边界

当前 `DashScopeVisionRefiner.observe()` 的 system prompt 要求返回：

```json
{
  "success": false,
  "failure_type": "identity_drift",
  "observation": "The person's identity is not preserved across output frames.",
  "observer_evidence": "face and clothing changed",
  "confidence": 0.92
}
```

它现在使用闭集 `failure_type` 区分编辑缺失、身份漂移、前序编辑消失、风格不一致、运动过弱和构图过弱。`observe_stage_output()` 统一校验并持久化 `success/failure_type/observer_evidence/confidence`。`FailureDiagnosisAndRepair` 再按置信度和已确认父 stage 选择 repair action。

仍然存在一个明确边界：静态五帧不能可靠证明 camera、motion、temporal 或 audio 的速度/幅度。Observer 对这类不可由静帧判断的请求返回 `success=true` + `failure_type=not_frame_judgeable`；这表示“不触发静态失败修复”，不是运动质量已经被证明。需要运动/音频质量时必须另做视频级评估。

### 2.4 已拆开的执行层耦合

实现中已经拆开以下耦合，避免 `failure_type` 被旧 fallback 覆盖：

- 全局视觉风格由 `reference_policy()` 首次设为一张 primary reference；失败后的 repair record 才把当前 stage 切换为三锚点。普通失败本身不会强制三锚点。
- bridge cache 同时比较 `failure_observation` 和完整 `repair_context`，不同 failure type/action/retry 不会误复用。
- video-only repair 使用确定性模板，不为被丢弃的 paraphrase 支付额外 Qwen compose 调用。
- 图片编辑始终使用原始 `raw_prompt`；`repaired_h3_prompt` 只用于 H3，不会改变 GRSAI 的语义范围。
- `attempt_N`、显式 `retry_index=1` 和归档 bridge 支持一次定向修复；第一次 retry 失败后停止传播，并可恢复首帧 primary anchor。

这些约束仍属于执行层；repair 组件不修改 compiled plan 或 MSR 控制器。

## 3. 已实现的 VETRA 执行层协议

### 3.1 组件边界

执行层组件 `FailureDiagnosisAndRepair` 位于
`src/apimart_h3_pipeline/core/repair_policy.py`，由目标 runner 直接导入；
`scripts/vetra_failure_repair.py` 仅为兼容入口。它不拥有 API 提交权限，也不读完整控制计划，职责包括：

1. 校验 observer diagnosis；
2. 从有限枚举中选择 repair action；
3. 生成受约束的 repaired H3 prompt；
4. 选择 `video_only`、`one_anchor` 或 `three_anchor` reference policy；
5. 输出可持久化、可恢复的 repair record。

H3、GRSAI、媒体上传和断点恢复仍由现有组件负责。这样可以让“诊断/修复决策”和“付费请求执行”保持清晰分层。

### 3.2 Diagnosis 输入与输出

诊断器的输入只包含当前 stage 所需的最小上下文：

```json
{
  "stage_id": "S2",
  "current_requirement": "Change the selfie into two people reading the newspaper together.",
  "failed_prompt": "Apply only this edit to <Video 1>: Change the selfie into two people reading the newspaper together.",
  "observer_evidence": "The original selfie remains visible; the second person and shared newspaper reading are not established.",
  "previous_requirements": [
    {"stage_id": "S1", "prompt": "Render the scene as an oil painting.", "status": "confirmed"}
  ],
  "attempt": 1
}
```

建议的结构化输出：

```json
{
  "kind": "qwen_vl_failure_diagnosis_v1",
  "success": false,
  "failure_type": "edit_missing",
  "observer_evidence": "The requested relation is absent across the five frames.",
  "confidence": 0.91,
  "repairable": true,
  "affected_scope": "current_stage_only",
  "preserved_stage_ids": ["S1"],
  "evidence_frames": [0, 26, 53, 80, 106]
}
```

字段约束：

- `success=true` 时 `failure_type` 必须为 `none`，不得触发 repair；
- `success=null`、`observer_unavailable` 或 `not_frame_judgeable` 不是 H3 语义失败，不能据此付费重试；
- `failure_type` 必须属于下表枚举，未知值一律降级为 `unclassified`，不自动生成激进 prompt；
- `observer_evidence` 只描述观察到的失败，不得成为新 requirement；
- `affected_scope` 必须是 `current_stage_only`；任何其它值直接拒绝；
- `confidence` 小于配置阈值时默认不做定向修复，记录 `diagnosis_uncertain`。

### 3.3 Repair 输入与输出

修复器的输入对应用户提出的接口，并补充 stage/retry 元数据：

```json
{
  "stage_id": "S2",
  "current_requirement": "Change the selfie into two people reading the newspaper together.",
  "failed_prompt": "Apply only this edit to <Video 1>: Change the selfie into two people reading the newspaper together.",
  "observer_evidence": "The original selfie remains visible; the second person and shared newspaper reading are absent.",
  "failure_type": "edit_missing",
  "previous_requirements": [
    {"stage_id": "S1", "prompt": "Render the scene as an oil painting.", "status": "confirmed"}
  ],
  "retry_index": 1,
  "max_retries": 1,
  "original_reference_policy": "one_anchor"
}
```

其中 `max_retries` 只是归档审计字段，代码固定写为 `1`，不接受 CLI 或调用方覆盖。

修复器输出：

```json
{
  "kind": "vetra_failure_repair_v1",
  "stage_id": "S2",
  "repair_action": "strengthen_edit",
  "repaired_h3_prompt": "Apply only this edit to <Video 1>: Make the requested transition to two people reading the newspaper together clearly visible while preserving the confirmed oil-painting appearance and all unrelated scene details.",
  "reference_policy": "one_anchor",
  "reference_image_count": 1,
  "retry_index": 1,
  "allowed_semantic_source": "current_requirement_only",
  "preservation_source": ["S1"],
  "guard": {
    "same_stage": true,
    "topology_changed": false,
    "new_requirement_added": false,
    "clt_written": false,
    "ceg_written": false
  }
}
```

`repaired_h3_prompt` 不是新的 atomic requirement。它必须通过以下确定性检查后才能交给 H3：

1. 包含当前 requirement 的核心动作/对象语义；
2. 没有引入当前 requirement 中不存在的新目标编辑；
3. 只使用允许的 preservation、visibility、temporal consistency 修复短语；
4. 对 `video_only` 阶段保留 `<Video 1>` 且不出现 `<Picture N>`；
5. 对图片阶段的 `<Picture N>` 数量和 temporal role 与 `reference_policy` 一致；
6. `stage_id`、父视频和原始 prompt 与失败尝试一致。

## 4. 失败类型与定向动作

下表是第一版的闭集动作集合。每个失败只能选择一个主类型；若 observer 发现多个问题，选择阻止当前 requirement 成功的主因，并在 evidence 中保留其它观察。

| `failure_type` | 诊断信号 | `repair_action` | 下一次 reference policy | 允许的修复方向 |
| --- | --- | --- | --- | --- |
| `edit_missing` | 当前编辑在五帧中没有出现或只有极弱局部出现 | `strengthen_edit` | 保持原策略 | 强化当前动作/对象/关系的可见性，不增加目标 |
| `identity_drift` | 人物脸、身份、服装、人数或角色发生漂移 | `strengthen_identity_preservation` | `one_anchor`；若同时风格跨时序不一致才 `three_anchor` | 加入人物身份和未编辑属性保持约束 |
| `previous_stage_lost` | 已确认的前序编辑在当前输出中消失 | `strengthen_previous_stage_preservation` | 保持原策略 | 只引用已确认 stage 的 preservation summary |
| `style_inconsistency` | 开头、中间、结尾风格不一致或风格 master 只在局部出现 | `use_three_anchor` | `three_anchor` | 使用 `0/53/106` 时间锚点；首帧 style master 供中帧、末帧编辑继承，并使用统一 style contract |
| `motion_weak` | camera push-in、pan、zoom 等存在但幅度/速度不可见 | `strengthen_motion` | `video_only` | 强调原 prompt 已声明的 motion type、方向、幅度和速度，不发明新运动 |
| `composition_weak` | 要求的构图/位置变化过小或不明确 | `strengthen_composition` | 原策略 | 强调原 prompt 中已有的空间关系和目标位置 |
| `unclassified` | observer 不能可靠归类 | `no_automatic_repair` | 不变 | 记录人工/离线复核，不自动扩大语义 |
| `observer_unavailable` | Qwen transport、JSON 或服务错误 | `no_semantic_retry` | 不变 | 这是观察失败，不是 H3 失败 |
| `not_frame_judgeable` | camera/motion/audio 无法从静态帧确认 | `no_semantic_retry` | 不变 | 保持现有 video-only 规则，另行做视频级评估 |
| `media_invalid` | ffprobe、帧数、FPS、下载或编码失败 | `execution_retry` | 不适用 | 走 API/下载恢复，不调用语义 repair |

### 4.1 动作短语的安全边界

修复短语只能改变“当前要求如何被执行和保持”，不能改变“要做什么”。建议使用受限模板，而不是让 Qwen 自由重写：

```text
strengthen_edit:
  Make the requested edit clearly visible across the sequence. Apply only the current edit.

strengthen_identity_preservation:
  Preserve the identity, count, role, face, body, and unedited appearance of existing people.

strengthen_previous_stage_preservation:
  Preserve all edits already confirmed before this stage, then apply only the current edit.

strengthen_motion:
  Make the requested motion visibly clear with its stated type, direction, amplitude, and speed;
  do not introduce a different camera or object motion.

strengthen_composition:
  Make the requested spatial/compositional change clearly visible at its stated target position;
  preserve all unrequested layout.
```

其中“amplitude”和“speed”只能强调 raw prompt 已经给出的程度。若 raw prompt 说 `slight push-in`，不能把它改成 `large dramatic zoom`；最多要求该 `slight push-in` 在视频中可见且连续。

`strengthen_previous_stage_preservation` 的 `previous_requirements` 只能来自已经确认的父 stage，不能把尚未执行的未来 stage 或 benchmark target 拼进 H3 prompt。

### 4.2 全局风格与失败后的三锚点边界

全局视觉风格首次执行固定使用 parent 首帧生成的一张 primary style master；`global_style_reference_count` 不会把它提前变成三锚点。全局风格一图失败后才使用首帧/中帧/末帧三锚点，这是当前 stage 的时序锁定 repair。普通静态编辑仍默认使用一张首帧 primary reference，失败后的三锚点只在 `style_inconsistency` 或显式 `fixed-three-anchor` 实验基线中触发：

- `edit_missing`：先重写当前 H3 prompt，通常复用一张内容对齐主参考图；
- `identity_drift`：先加强 identity preservation，只有跨时间外观也漂移时才升级三锚点；
- `previous_stage_lost`：先加入已确认父 stage 的 preservation summary，不立即生成三张图；
- `motion_weak`：不生成图片，保持 `video_only`，加强原有 motion wording；
- `composition_weak`：保留原 reference policy，强调目标位置/关系；
- `style_inconsistency`：直接使用上下文首帧/中间帧/上下文末帧三锚点；中间帧和末帧的 GRSAI 编辑都传入首帧 primary style master。

三锚点计划由本地 `three_anchor_reference_plan()` 生成，不再额外调用 Qwen fallback planner。首帧图片编辑使用原始 atomic requirement 生成 style master；中帧和末帧也从各自 parent 帧使用原始 atomic requirement 编辑，但都把首帧产物作为共同 `style_reference`；H3 prompt 则使用统一的 temporal-anchor contract。这样可以区分“全局风格失败后的时序锁定”和“普通编辑因风格时序不一致而升级锚点”。

该 temporal-anchor contract 明确要求 frame lock：首帧 style master、中间 anchor、末帧 anchor 在各自 source frame 对应的输出时间位置锁定编辑后外观，所有中间帧保持同一编辑风格，禁止回退到源视频外观。它是 H3 的强语言约束，不等同于像素级逐帧复制保证。

## 5. 当前 runner 的实际接入

### 5.1 Observer 扩展

`DashScopeVisionRefiner.observe()` 现在直接返回结构化诊断，并保留旧 observation 字段兼容：

```python
def diagnose_failure(
    self,
    frames: Sequence[Path],
    current_requirement: str,
    failed_prompt: str,
    previous_requirements: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ...
```

当前 Qwen diagnosis 的 system contract：

```text
Return JSON only with exactly:
failure_type, observer_evidence, confidence, repairable.

Classify only the current atomic requirement. Do not propose a new requirement.
Use one closed-set failure_type. If the requirement is camera, motion, temporal,
or audio and cannot be judged from still frames, return not_frame_judgeable.
```

`observe_stage_output()` 把 success gate 和 failure diagnosis 合并成一次 Qwen 请求，返回：

```json
{
  "success": false,
  "failure_type": "identity_drift",
  "observation": "...",
  "observer_evidence": "...",
  "confidence": 0.88
}
```

`kind` 版本字段和 deterministic schema validator 会保留；旧的 `observation.json` 缺少 `failure_type` 时按兼容规则归一化，不重复付费。

### 5.2 主循环的实际控制流

`main()` 当前的控制流等价于以下逻辑：

```python
attempt = 1
parent_for_stage = parent
stage_parent_url = parent_url
policy = reference_policy(stage["prompt"], args.global_style_reference_count)

while True:
    image_urls, h3_prompt, bridge = bridge_for_stage(
        ..., parent_for_stage, stage["prompt"], policy_override=policy,
        repair_context=repair_context,
    )
    invoke_h3_client(args, stage, h3_prompt, stage_parent_url, image_urls, stage_dir)
    require_aligned_output(output)

    observation = observe_stage_output(...)
    if observation["success"] is True:
        break
    if observation["success"] is None:
        # None 表示 observer 不可用，不能伪造 semantic success，也不触发图片重试。
        mark_observation_pending_and_stop(...)

    diagnosis = failure_diagnosis(...)
    repair = failure_repair(..., diagnosis, retry_index=attempt)
    persist_attempt_before_next_paid_call(..., diagnosis, repair)
    if not repair["guard"]["same_stage"] or attempt >= 1:
        mark_semantic_failure_and_stop(stage_dir, observation, diagnosis, repair)
        raise SemanticStageFailure(stage["stage_id"])

    attempt += 1
    policy = policy_from_repair(repair)
    repair_context = repair
```

实现中 `success=None` 默认标记为 `observation_pending` 并停止；只有显式传入 `--allow-unverified-output` 才允许媒体继续流转，manifest 仍保留 pending 状态。

### 5.3 `bridge_for_stage()` 的 repair 接口

现有 `bridge_for_stage()` 已有 `failure_observation` 参数，并增加只读 `repair_context`：

```python
repair_context: Mapping[str, Any] | None = None
```

它只允许影响：

- 最终 H3 prompt 的 preservation/visibility wording；
- 是否从一张图切换为三锚点；
- 当前图片编辑是否需要沿用已生成的 primary reference；
- bridge 元数据中的 diagnosis/repair record。

它不允许影响：

- `next_prompt` 的原始语义；
- `stage_label` 或 stage 顺序；
- `previous_video` 父状态；
- 未确认的未来 requirement；
- H3 duration、canvas、帧数、FPS 等媒体协议。

当 repair action 不是 `use_three_anchor` 时，不应因为存在 `failure_observation` 就自动把普通 stage 的 `reference_count` 改成 3。全局风格是一条明确例外：它首次仍按配置使用一张图，但一图尝试失败后，`is_global_style` 会选择 `global_style_three_anchor` repair。`global_style_reference_count` 和 `active_reference_count` 仍与 repair action 分开，避免“传入诊断文本”隐式触发统一 fallback。

### 5.4 H3 prompt 生成策略

普通路径的 `compose_h3_prompt()` 负责 `<Video 1>`/`<Picture N>` 合同和标签校验；全局风格失败和三锚点 repair 使用 `temporal_anchor_h3_prompt()` 的确定性时序合同，再附加一个经过 validator 的短 repair clause：

```text
Raw atomic requirement: <immutable current prompt>
Repair clause: <one allow-listed clause>
Preservation evidence: <confirmed parent-stage summary only>
```

最终 prompt 必须满足：

- 当前 raw atomic requirement 的每一个动作/对象/关系仍然出现；
- repair clause 不引入新的目标编辑；
- video-only 阶段固定为 `Apply only this edit to <Video 1>: ...`，不附加图片标签；
- 图片阶段继续由 `validate_h3_reference_tags()` 检查图片数量、source frame 和 anchor role；
- 重试 prompt、repair action、输入父视频和参考图 URL 写入 bridge，作为 H3 请求恢复时的精确字段。

## 6. 状态、恢复和传播闸门

### 6.1 当前每次尝试保存的记录

`stage_dir/attempts/attempt_N/attempt.json` 当前保存：

```json
{
  "attempt": 2,
  "stage": "S2",
  "status": "semantic_failure",
  "raw_prompt": "...",
  "failed_h3_prompt": "...",
  "post_edit_observation": {
    "success": false,
    "failure_type": "previous_stage_lost",
    "observer_evidence": "...",
    "confidence": 0.87
  },
  "repair": {
    "repair_action": "strengthen_previous_stage_preservation",
    "reference_policy": "one_anchor",
    "repaired_h3_prompt": "...",
    "retry_index": 1
  },
  "parent_video": "...",
  "reference_images": ["..."],
  "h3_request_state": "...",
  "cost": {
    "qwen_calls": 1,
    "image_calls": 0,
    "h3_calls": 1
  }
}
```

`bridge.json` 同时保存图片编辑、prompt、诊断和 repair context：

```json
{
  "diagnosis": {...},
  "repair": {...},
  "repair_schema_version": "vetra_failure_repair_v1"
}
```

实现的写入顺序是：

1. 持久化 diagnosis；
2. 持久化 repair 和新的 prompt；
3. 持久化图片编辑状态/bridge；
4. 最后提交 H3 POST。

这样进程在付费请求前退出时可以恢复同一修复，不会重新调用图片模型或重复提交相同任务。

### 6.2 当前断点恢复条件

复用一个 repair bridge 必须同时匹配：

- `task_id`、`stage_id`；
- 原始 `raw_prompt`；
- `previous_video` 路径或内容 hash；
- `failure_type`、`observer_evidence` hash 和 `retry_index`；
- `repair_action`、`repaired_h3_prompt`；
- `reference_policy`、参考图数量和参考图 URL；
- H3 duration、resolution、aspect ratio、model；
- APIMart/CTMOAI provider。

只匹配 `output.mp4` 或只匹配“success=false”是不够的。修复 prompt 或参考图变化时，保存的请求字段必须不同；CTMOAI 的稳定 `/sd-media/` URL 可以复用，临时 tunnel URL 仍要遵守现有风险控制。

### 6.3 当前严格的 stage 传播规则

当前 VETRA 模式把媒体成功和语义成功分成两个状态：

```text
media_valid && semantic_confirmed  -> stage_success -> 允许 parent 更新
media_valid && semantic_failed     -> semantic_failure -> 停在当前 stage
media_invalid                      -> execution_failure -> 恢复/重试媒体请求
observer_unavailable               -> observation_pending -> 不自动宣称成功
```

三锚点或定向 prompt 的第二次尝试仍然 `success=false` 时：

- 保留输出作为诊断证据；
- 不写入 `parent`；
- 不进入下一个 stage；
- 在 task manifest 中记录 `semantic_failure`；
- 由上层决定人工复核、回到最近依赖祖先，或终止任务。

这一点可以复用 `scripts/run_preplanned_h3_full_pipeline.py` 中 `attempt_manifest`、`semantic_failure` 和失败后停止任务的结构，但不能直接复用它的固定三锚点策略。

## 7. 与控制平面的不变量

定向修复之所以不会破坏“控制平面先冻结、执行平面只读消费”，依赖以下硬约束：

| 不变量 | 校验方式 |
| --- | --- |
| 当前 stage 不变 | repair record 的 `stage_id` 等于失败 attempt 的 stage |
| 原子 requirement 不变 | `raw_prompt_sha256` 在所有 attempts 中一致 |
| 不增加新 requirement | repaired prompt 只能由 raw prompt + allow-list repair clause 组成 |
| 不重排拓扑 | `sequence_manifest` 的 stage 顺序和数量不变 |
| 不回写 CLT/CEG | repair 组件无控制层写权限；guard 字段必须为 false |
| 只保留已确认父 stage | `previous_requirements` 来自 manifest 中 confirmed stages |
| retry 输入不漂移 | 同一 stage 的所有 attempt 固定使用 `stage_parent_video`/`stage_parent_url`；失败输出只归档，不回写 parent |
| 重试有上限 | `retry_index == 1`；第二次失败后停止传播 |
| 语义失败不传播 | `semantic_failure` 不更新 `parent` |
| media/API 失败与语义失败分开 | 状态使用 `execution_failure`/`semantic_failure`/`observation_pending` |

对于“不能添加新 requirement”的解释应写入测试和代码注释：`strengthen_identity_preservation`、`strengthen_previous_stage_preservation` 属于对已有内容的保持约束；它们不能引入新对象、新人数、新动作或新的视觉风格。

## 8. 评价方案：固定三锚点 vs 定向修复

### 8.1 两个实验臂

**Arm A：Fixed Three-Anchor Recovery**

- 保持当前 runner 的行为；
- 所有满足 `success=false + one reference` 的静态 stage 都升级为首帧/中帧/末帧 `0/53/106`，且中帧、末帧编辑继承首帧 style master；
- 记录第二次输出是否通过媒体和语义检查。

**Arm B：Failure-Type Targeted Repair**

- observer 返回闭集 `failure_type`；
- 按动作表只修复失败部分；
- 只有 `style_inconsistency` 等类型才使用三锚点；
- 每个 stage 默认最多一次定向修复；
- 第二次仍失败时停止传播。

两个实验臂必须固定：源视频、compiled plan、stage 顺序、H3 model、GRSAI model、Qwen model、duration、seed（若 provider 支持）、媒体规格和超时。使用相同任务集合，最好以 task-stage 为单位配对比较。

### 8.2 指标定义

| 指标 | 定义 |
| --- | --- |
| 当前编辑成功率 | 每个 stage 最终通过语义 gate 的比例；同时报告 first-attempt 和 after-repair 两个版本 |
| 历史编辑保持率 | 当前 stage 成功后，已确认父 stage 的 requirement 在五帧/视频级复核中仍保持的比例 |
| 三锚点触发率 | 触发 `reference_policy=three_anchor` 的失败 stage 数 / 语义失败 stage 数 |
| H3 重试次数 | 每个 stage 的 H3 POST 次数，报告均值、P95 和按 failure type 分布 |
| 总成本 | Qwen diagnosis/refine + GRSAI image edit + H3 generation + 上传/下载的实际费用或调用计数 |
| 每种 failure type 成功率 | `failure_type` 分桶后，repair 前后成功率及置信区间 |
| 错误修复率 | repair 后引入新的身份/构图/风格问题的比例 |
| 传播阻断率 | semantic failure 被正确停在当前 stage 的比例 |
| 恢复重复率 | 进程重启后重复支付的 H3/图片调用次数 |

其中“当前编辑成功率”不能只看 `output.mp4` 存在；必须同时满足媒体有效和语义确认。对于 `not_frame_judgeable`，要单独报告视频级 motion/audio 评估，不能将静态 gate 的 `success=true` 当成运动质量证明。

### 8.3 推荐日志行

每个 stage 完成时在 stdout 和 manifest 中输出同一组字段：

```json
{
  "event": "stage_repair_summary",
  "task_id": "139",
  "stage": "S2",
  "first_attempt_success": false,
  "failure_type": "edit_missing",
  "repair_action": "strengthen_edit",
  "reference_image_count_before": 1,
  "reference_image_count_after": 1,
  "h3_attempts": 2,
  "final_success": true,
  "previous_stage_preserved": true,
  "qwen_calls": 2,
  "image_calls": 1,
  "h3_calls": 2
}
```

实际金额若 provider 返回 usage/price，应另存原始 provider 字段和计算后的 `cost_total`，不能只凭轮询次数推算价格。

## 9. 分阶段落地状态与剩余工作

### Phase 0：离线 schema 和 fixture（已完成）

- 为 `failure_type`、repair action 和 guard 编写纯函数 validator；
- 用现有 Task 139 观察记录构造 `edit_missing`、`identity_drift`、`previous_stage_lost`、`style_inconsistency`、`motion_weak`、`composition_weak` fixtures；
- 验证不改变 raw prompt、stage 顺序和 reference tag。

### Phase 1：Observer diagnosis（已完成并纳入执行记录）

- Observer 在同一次五帧请求中返回 `failure_type`、evidence 和 confidence；
- diagnosis 通过 deterministic validator 后写入 attempt/manifest；
- 不额外调用一个独立的 Qwen fallback planner，也不让自然语言 evidence 直接改变图片数量。

### Phase 2：Prompt-only targeted repair（已完成）

- 启用 `strengthen_edit`、`strengthen_identity_preservation`、`strengthen_previous_stage_preservation`、`strengthen_motion`、`strengthen_composition`；
- 默认保持原 reference policy；
- 只允许一次重试；
- 对每个 repair prompt 做 requirement guard，并持久化完整请求字段以支持恢复。

### Phase 3：选择性三锚点（已完成，保留固定基线）

- 仅对 `style_inconsistency` 或明确的跨时间 appearance drift 使用三锚点；
- 比较三锚点触发率、成本和风格一致性；
- 保留 fixed fallback 作为可切换实验臂。

### Phase 4：严格语义传播闸门（已完成，待线上验证）

- 任何最终 `success=false` 都停止当前任务，不更新 parent；
- `observer_unavailable` 进入 `observation_pending`，由离线视频级复核决定；
- 上层仍可选择回到最近的依赖祖先，但该回溯属于控制层显式决策，不能由 repair 组件偷偷改拓扑。

本轮尚未完成的是线上付费端到端验证：需要使用真实 Task 139，在受控媒体服务和 provider 配额下确认请求字段恢复、上传 URL 和实际费用。

## 10. 风险和未决问题

1. **Observer 误分类**：Qwen 可能把 `edit_missing` 和 `composition_weak` 混淆。应使用闭集枚举、置信度阈值和 shadow 评估，不要让自由文本直接决定图片数量。
2. **修复 prompt 语义膨胀**：实现使用短模板和 deterministic builder，仍需在线样本持续检查是否引入目标竞争。
3. **身份保持短语的边界**：identity preservation 是保持约束，不是新 requirement；但不能借此添加“增加第二个人”等原 prompt 没有的内容。
4. **运动不可由静态帧确认**：`motion_weak` 需要视频级 temporal observer 或光流/轨迹证据。五帧静态 observer 只能判断“是否明显发生了构图差异”，不能证明真实 push-in 速度。
5. **前序编辑保持率**：只把最近一条父 stage prompt 塞进 H3 可能仍然过长；应在 manifest 中保存短的 preservation summary，并单独评估历史编辑保持率。
6. **API 成本和恢复**：每一个不同的 repaired prompt 都会形成不同的已保存请求字段。必须在 POST 前持久化 repair state，CTMOAI 优先使用稳定 media URL。
7. **当前三锚点输出的语义门**：实现已在第二次失败时停止传播并写入 `semantic_failure`；线上 provider 仍需确认不会产生旁路状态。
8. **计划审计边界**：repair 组件不能回写 compiler 产物，也不能把 observer 结果当作新的 MSR requirement；需要在代码和 manifest 中明确只读边界。

## 11. 代码索引与实现状态

### 当前已有能力

- `DashScopeVisionRefiner.observe()`：五帧 success gate + 闭集 failure diagnosis；
- `observe_stage_output()`：抽帧、持久化观察、observer transport failure 记录；
- `reference_policy()`：静态/时序/全局风格参考图分类；
- `compose_h3_prompt()`：普通图片/视频标签合同和失败观察文本；
- `temporal_anchor_h3_prompt()`：全局风格与三锚点 repair 的统一时序 prompt 合同；
- `three_anchor_reference_plan()`：不调用模型的首帧 style master + 中帧/末帧 temporal anchor 计划元数据；
- `bridge_for_stage()`：图片编辑、上传、bridge 持久化和恢复；
- `main()`：按 failure type 定向修复、可选 fixed-three-anchor 基线、attempt 归档和严格传播闸门；
- `vetra_failure_repair.py`：闭集 diagnosis、repair action/reference policy 映射、prompt/record validator 和重试预算；
- `run_preplanned_h3_full_pipeline.py`：可参考的 stage attempt 和 semantic failure 停止结构。

### 尚需验证或扩展

- 使用真实 Task 139 做一次受控线上付费运行，验证 provider 请求字段恢复和费用；
- 补充视频级 motion/audio observer，并把它与静态五帧 gate 分开统计；
- 编写固定三锚点与定向修复的配对实验脚本和汇总工具；
- 在真实多 stage 任务上统计当前编辑成功率、历史编辑保持率、三锚点触发率、重试次数、成本及按 failure type 的成功率。

因此，当前默认代码路径是 **failure-type targeted repair**；`fixed-three-anchor` 仅作为可切换实验基线，始终保持控制平面冻结。
