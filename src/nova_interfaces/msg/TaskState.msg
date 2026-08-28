# agent 任务状态(每任务一个 topic:/nova/agentos/task_state/t_{task_id})
string task_id      # 任务唯一标识
string status       # working / done / failed
bool done           # 任务是否已结束(完成或失败)
string message      # agent 最后一条消息(纯文本回复/完成总结/错误信息)
