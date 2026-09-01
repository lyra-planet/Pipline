# Pipeline 模块与 Prompt 资源布局

当前顺序视频编辑 pipeline 按职责拆成几个小边界。历史入口
`scripts/run_apimart_minimax_h3_sequential.py` 只负责兼容旧的导入和命令行
调用；实际执行代码位于 `apimart_h3_pipeline` 包中。

## 模块职责

- `runner.py` 负责命令行参数、阶段生命周期、父视频选择、一次重试和最终产物。
- `bridge_execution.py` 负责一个 stage 的参考图/H3 bridge 执行与 `bridge.json` 持久化。
- `bridge_helpers.py` 负责任务读取、上传、首帧主参考复用、三锚点计划和确定性 repair prompt。
- `vision_client.py` 只负责 DashScope 请求、图片编码和多模态消息中的帧/图片附件。
- `vision_refiner.py` 负责 Qwen-VL 的参考规划、H3 prompt 组成和五帧 observer；`vision.py` 是稳定的兼容导出层。
- `image_editor.py` 负责 GRSAI 图片编辑；`media.py` 负责 ffprobe、帧抽取和视频几何。
- `policy.py` 和 `vetra_failure_repair.py` 负责闭集策略、标签校验和 stage-local 不变量。
- `artifacts.py` 负责 attempt 归档、observer 记录和 manifest 更新。

这样拆分后，纯策略和 prompt contract 可以在没有网络/模型的情况下单元测试，
provider transport 也不会混入 runner 的阶段状态机。

## Prompt 资源

Qwen system prompt、user prompt、图片标签、三锚点合同和 repair clause 位于
`scripts/apimart_h3_pipeline/prompts/`。它们不是 Python 字符串常量，而是随包发布
的 UTF-8 资源。`prompt_catalog.py` 使用 `importlib.resources.files()` 读取资源，
因此从任意当前工作目录启动都有效；`pyproject.toml` 通过
`[tool.setuptools.package-data]` 明确声明 `*.txt` 和 `*.json`。

模板使用 `${name}` 占位符。`render_prompt()` 对缺少参数、非法模板和路径穿越
直接报错；调用方必须显式提供所需值。`repair_clauses.json` 是 repair action 的
闭集，`vetra_failure_repair.py` 只接受其中的 action 和对应原文。

Prompt 改动会自然进入已有的 `bridge.json`、attempt 和 observer 记录，便于审阅
和复现实验；没有额外的缓存标识或运行时配置项。

## 运行边界

控制平面在执行前冻结 stage 拓扑。执行平面只读消费当前 raw atomic requirement，
失败时最多对同一 stage 重试一次。重试只能使用原始 parent video；失败输出只归档
并交给 observer，不会成为自己的 retry 输入。全局风格首次执行仍是一张首帧参考图，
只有该尝试失败后才切换到首帧/中帧/末帧三锚点，并要求 H3 对三个锚点进行帧锁定。
