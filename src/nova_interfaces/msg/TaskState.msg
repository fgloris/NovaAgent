# agent 消息(全局话题 /nova/agentos/agent_msg,按 kind 区分消息类型)
string task_id      # 任务唯一标识
string status       # working / done / failed
bool done           # 任务是否已结束(完成或失败)
string kind         # status | text | tool_call | tool_result —— 消息类型
string message      # 消息内容(规划文本/工具调用与参数/工具结果/总结/错误)
