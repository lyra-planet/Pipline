# Observer 时间覆盖更新

## 背景

旧版 post-edit Observer 只把 `0/26/53/80/106` 五个检查点交给 Qwen-VL。这个协议同时承担了参考图/父视频上下文和语义成功门，容易漏掉短暂的接触动作、impact VFX、动作起止和目标进入/离开画面的情况。`task_37`、`task_46`、`task_53` 的人工复核显示，固定五帧失败结论不能直接当作整段视频的语义真值。

CoVEBench 的主观 checklist 评测把视频作为一个完整时序输入，并使用等间隔帧覆盖整段视频；它还把 instruction compliance 和 semantic preservation 分开，不能用单个静态帧替代动作过程判断。本 Pipeline 的 Qwen-VL 接口目前以图片输入，因此采用等间隔检查点实现同一原则。

## 新协议

- H3 输入、参考图选择和 prompt 上下文仍使用原来的五帧 `0/26/53/80/106`，不改变任何 anchor frame 映射。
- post-edit Observer 单独使用十个等间隔检查点：`0/12/24/35/47/59/71/82/94/106`。
- 有 source video 时，Observer 对每个同索引的 source/output 帧做对照；没有 source 时只判断 output 的可见事实。
- persistent edit（替换、移除、添加）检查目标在其可见区间是否持续正确，是否后续重新出现或漂移。
- action、speed、timing、event-triggered VFX 按时间进展判断，不再要求效果出现在每一帧；短暂效果要在触发事件附近核验。
- camera edit 只在原子 prompt 明确要求时检查 camera operation；不再硬编码“frame 0 必须不变”。
- 仅靠静态检查点无法可靠判断的时序问题返回 `success=true`、`failure_type=not_frame_judgeable`，而不是触发语义 fallback。

记录版本由 `qwen_vl_uniform_temporal_success_gate_v2` 标识。旧的五帧 `observation.json` 不会被静默改写；需要复核时使用 Observer 脚本的 `--force` 重新生成。
