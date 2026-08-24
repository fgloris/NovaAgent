# nova_common

NovaAgent 的共享基础设施,节点无关。

## 内容

- `config/llm.yaml`:全项目 LLM 配置的唯一权威来源。`providers` 按列表顺序作为调用优先级,调用失败自动回退到下一个 provider。支持 `kind: openai`(OpenAI 兼容接口)和 `kind: anthropic`。
- `llm_config.py`:统一配置加载,查找顺序:`NOVA_LLM_CONFIG` 环境变量 → 包内 `share/nova_common/config/llm.yaml`。
- `llm_client.py`:统一 LLM 客户端,内部使用 OpenAI 兼容消息格式(含 tool/function calling),自动切换 openai / anthropic 后端并按 providers 顺序 failover。

## 使用

```python
from nova_common.llm_client import LLMClient

client = LLMClient()  # 也可传入自定义 config dict
result = client.chat(
    [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
    tools=[{"type": "function", "function": {...}}],
)
# result.content: 文本回复
# result.tool_calls: [{"id", "type", "function": {"name", "arguments"}}]
```

## 配置优先级

1. 环境变量 `NOVA_LLM_CONFIG=/path/to/llm.yaml`
2. 构建安装后的 `share/nova_common/config/llm.yaml`
3. 源码目录内的 `config/llm.yaml`
