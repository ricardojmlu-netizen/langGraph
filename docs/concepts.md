# 设计与学习笔记

## 1. State 不是全局可变字典

节点返回局部更新，由 LangGraph 合并。普通字段覆盖；`findings` 使用 `operator.add` 累加并发研究结果；消息使用 `add_messages` 按消息 ID 合并。多个并行节点同时覆盖同一普通字段会冲突。汇总前排序，避免把并发完成顺序误当稳定顺序。

重复提交同一输入可能重复追加 findings，所以 CLI 为每次 run 建立独立 thread。真正的多轮对话应设计 turn_id、按轮次组织结果，或显式重置 reducer 字段；不能简单反复 invoke 相同初始字典。

## 2. 边、条件边与 Command

固定边表达始终执行的步骤；条件边读取节点输出后选择去向；`Send` 把不同输入发给同一个 worker，实现动态 map-reduce。所有 research 分支同一 super-step 完成后进入 synthesize。

`Command` 同时返回状态更新和下一节点。review 没有额外固定出边，否则固定边仍可能执行，导致非预期双路径。

## 3. 子图与工具循环

父图关注 question/findings/report；子图关注 topic/messages/model_turns。research 负责映射输入输出，通过节点内调用继承执行配置与 checkpoint 上下文。`subgraphs=True` 显示嵌套执行命名空间。

模型只提出 tool_calls，ToolNode 执行 Python 工具并生成对应 tool_call_id 的 ToolMessage。每个工具结果都回到 model，形成真正循环。模型轮次上限与图 recursion_limit 分别限制模型调用和整体 super-step；达到模型上限返回明确终止消息，不假装完整研究成功。

## 4. Checkpointer 和 Store

Checkpointer 以 thread_id 保存一个执行的状态与待执行节点，支持故障恢复和历史检查。Store 以 namespace/key 保存跨 thread 的业务记忆。一个系统不自动替代另一个。

示例用 SQLite 持久化 checkpoint，用 InMemoryStore 展示跨 thread 用户偏好。多用户隔离依赖调用方可信的 user_id；生产应从认证上下文注入，不能信任任意客户端自报身份。

## 5. 中断、恢复和副作用

interrupt 抛出框架控制信号；不要用宽泛 except 吞掉它。恢复通过 Command(resume=...)，会重新进入节点；interrupt 返回恢复值。不要在 interrupt 之前做不可重复的外部写入。

Runtime context 不属于 checkpoint 状态，所以 CLI 恢复时重新提供审核配置。演示只有一个人工中断；多中断应按 interrupt ID 提供映射并保持调用顺序稳定。

历史 fork 创建新 checkpoint。若后续节点有模型或工具调用，它们可能重新执行并产生费用或副作用。测试验证了修改分支不覆盖原 checkpoint 的 report。

## 6. 重试、缓存和异步

RetryPolicy 仅对选定异常重试。实验第一次主动抛 ConnectionError，第二次成功，随后相同输入从缓存返回。真实模型客户端另有网络重试；不要同时无限叠加两层重试。

缓存用于结果只依赖输入、允许在 TTL 内复用的节点。不要把包含用户身份、权限或外部状态但未纳入 cache key 的结果共享缓存。实际 research 节点没有开启缓存。

异步例子使用 InMemorySaver 的异步接口；同步 SqliteSaver 不应直接用于 ainvoke，可改用 AsyncSqliteSaver。示例子图节点是同步函数，LangGraph 在异步图运行时通过执行器调度；若做高吞吐模型调用，进一步改为 async 节点 + await model.ainvoke。

## 7. Streaming 与可观测性

updates 只输出节点更新；values 输出累计状态；custom 输出业务进度；messages 输出模型消息分片及 metadata；debug 提供更详细的执行事件。包含子图时，事件额外包含 namespace。

流事件不是数据库记录，最终状态从 checkpoint 获取。消息和 debug 流可能包含原始问题、工具输出和内部状态。需要对外服务时应明确选择展示字段。LangSmith 是可选外部追踪服务，图执行不依赖它。

## 8. Functional API

`@entrypoint` 使普通 Python 控制流成为可持久化入口；`@task` 为计算或副作用划定可记录的执行边界。先创建多个 task future，再取 result，可并行调度。本实验和 Graph API 分开，便于比较两种风格；并非所有逻辑都需要图状表达。
