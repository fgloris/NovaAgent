# executor 节点周期发布,供 executor_manager 自动发现与剔除(热插拔)
string executor_name     # executor 节点名
ToolDescriptor[] tools   # 该 executor 支持的全部工具
