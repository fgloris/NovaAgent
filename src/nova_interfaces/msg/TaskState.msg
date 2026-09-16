# agent 消息(全局话题 /nova/agentos/agent_msg,按 kind 区分消息类型)
string task_id      # 任务唯一标识
string session_id   # 所属 session(用于多客户端按会话过滤)
string status       # working / done / failed
bool done           # 任务是否已结束(完成或失败)
string kind         # status | text | tool_call | tool_feedback | tool_result | session_renamed
string message      # 消息内容(规划文本/工具调用与参数/工具结果/总结/错误/新会话名)
