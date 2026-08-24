# 统一 LLM 客户端:内部使用 OpenAI 兼容消息格式(含 function/tool calling)。
# 按 providers 顺序尝试调用,失败自动回退下一个 provider。
import json
import os
from dataclasses import dataclass, field
from typing import Any

import requests

import nova_common.llm_config as llm_config

DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_TOKENS = 8192


@dataclass
class ChatResult:
    content: str = ""
    tool_calls: list = field(default_factory=list)
    raw: dict = field(default_factory=dict)


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, config: dict | None = None):
        self.config = config or llm_config.load()
        self.providers = self.config.get("providers", [])

    def chat(
        self,
        messages: list[dict],
        tools: list | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        errors = []
        for provider in self.providers:
            try:
                return self._chat_one(provider, messages, tools, temperature, max_tokens)
            except Exception as exc:
                errors.append(f"{provider.get('name')}: {exc}")
        raise LLMError("所有 LLM provider 均失败: " + " | ".join(errors))

    def _chat_one(self, provider, messages, tools, temperature, max_tokens) -> ChatResult:
        if provider.get("kind") == "anthropic":
            return self._chat_anthropic(provider, messages, tools, temperature, max_tokens)
        return self._chat_openai(provider, messages, tools, temperature, max_tokens)

    def _chat_openai(self, provider, messages, tools, temperature, max_tokens) -> ChatResult:
        url = provider["base_url"].rstrip("/") + "/chat/completions"
        body = {
            "model": provider["model"],
            "messages": messages,
            "temperature": self._value(temperature, provider, "temperature", 0.1),
            "max_tokens": self._value(max_tokens, provider, "max_tokens", DEFAULT_MAX_TOKENS),
        }
        if tools:
            body["tools"] = tools
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {self._api_key(provider)}"},
            json=body,
            timeout=provider.get("timeout_sec", DEFAULT_TIMEOUT),
        )
        if resp.status_code != 200:
            raise LLMError(f"openai {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        message = data["choices"][0]["message"]
        return ChatResult(message.get("content") or "", message.get("tool_calls") or [], data)

    def _chat_anthropic(self, provider, messages, tools, temperature, max_tokens) -> ChatResult:
        url = provider["base_url"].rstrip("/") + "/v1/messages"
        system, an_messages = self._to_anthropic_messages(messages)
        body = {
            "model": provider["model"],
            "max_tokens": self._value(max_tokens, provider, "max_tokens", DEFAULT_MAX_TOKENS),
            "temperature": self._value(temperature, provider, "temperature", 0.1),
            "messages": an_messages,
        }
        if system:
            body["system"] = system
        if tools:
            body["tools"] = [
                {
                    "name": t["function"]["name"],
                    "description": t["function"].get("description", ""),
                    "input_schema": t["function"]["parameters"],
                }
                for t in tools
            ]
        resp = requests.post(
            url,
            headers={"x-api-key": self._api_key(provider), "anthropic-version": "2023-06-01"},
            json=body,
            timeout=provider.get("timeout_sec", DEFAULT_TIMEOUT),
        )
        if resp.status_code != 200:
            raise LLMError(f"anthropic {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        content = ""
        tool_calls = []
        for block in data.get("content", []):
            if block["type"] == "text":
                content += block.get("text", "")
            elif block["type"] == "tool_use":
                tool_calls.append(
                    {
                        "id": block["id"],
                        "type": "function",
                        "function": {"name": block["name"], "arguments": json.dumps(block["input"])},
                    }
                )
        return ChatResult(content, tool_calls, data)

    # 把 OpenAI 风格消息转成 Anthropic 格式
    @staticmethod
    def _to_anthropic_messages(messages: list[dict]) -> tuple[str, list[dict]]:
        system_parts = []
        out = []
        for m in messages:
            role = m["role"]
            if role == "system":
                system_parts.append(m.get("content", ""))
            elif role == "assistant":
                content = []
                if m.get("content"):
                    content.append({"type": "text", "text": m["content"]})
                for tc in m.get("tool_calls", []):
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["function"]["name"],
                            "input": json.loads(tc["function"]["arguments"]),
                        }
                    )
                out.append({"role": "assistant", "content": content})
            elif role == "tool":
                # tool 结果就近挂到上一条 assistant 的 tool_use 之后
                if out and out[-1]["role"] == "assistant":
                    out[-1]["content"].append(
                        {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m.get("content", "")}
                    )
                else:
                    out.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m.get("content", "")}]})
            else:
                out.append({"role": "user", "content": m.get("content", "")})
        return "\n\n".join(system_parts), out

    @staticmethod
    def _value(value, provider, key, default):
        return value if value is not None else provider.get(key, default)

    @staticmethod
    def _api_key(provider) -> str:
        env = provider.get("api_key_env")
        if env:
            key = os.environ.get(env)
            if key:
                return key
        raise LLMError(f"provider {provider.get('name')} 缺少 API key(环境变量 {env})")
