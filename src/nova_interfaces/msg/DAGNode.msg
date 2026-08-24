# 任务图节点:每个节点 = 一次 executor 工具调用
string id            # 唯一节点 id,供 depends_on 引用
string type          # 执行类型,当前统一 "tool",预留 verify/decide
string tool_name     # 调用的 MCP 工具名
string params_json   # 工具参数(JSON 字符串),值可为 "$ref:<node_id>" 引用前序节点结果
string goal          # 节点目标描述(自然语言,便于校验/可解释)
string[] depends_on  # 依赖的节点 id 列表
