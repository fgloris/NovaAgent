# 格式化后的任务 DAG 图
string task_id       # 任务唯一标识
string instruction   # 原始用户指令
string description   # LLM 对整体任务的概括
DAGNode[] nodes      # 全部任务节点
