# 任务执行状态(agentos 周期/状态变化时发布)
string task_id        # 任务唯一标识
string instruction    # 原始指令
string status         # planning / executing / done / failed
string plan_json      # TaskGraph 序列化
string current_step   # 当前执行的节点 id
string detail         # 附加信息(错误信息等)
