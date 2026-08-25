# 一个 MCP 工具的能力描述(由 executor 通过心跳发布)
string name                  # 工具名,LLM 调用时使用
string description           # 工具说明(注入 LLM)
string params_schema_json    # 参数 JSON Schema(openai function parameters 格式)
string action_server_name    # 执行该工具的 MCPExecute action server 名
string obs_bindings          # JSON,如 {"cameras": ["agentview"], "state": ["robot0_eef_pos"]};空串表示纯工具调用、不需要转发
