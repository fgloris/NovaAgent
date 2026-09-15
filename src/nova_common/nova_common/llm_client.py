"""统一 LLM 客户端:内部使用 OpenAI 兼容消息格式(含 function/tool calling)。

按 providers 顺序尝试调用,失败自动回退下一个 provider。
"""
import json
import os
import time
import base64
from dataclasses import dataclass, field
from typing import Any

import requests

import nova_common.llm_config as llm_config

DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_TOKENS = 8192


@dataclass
class ChatResult:
    """一次对话返回结果:正文、工具调用、思考内容与原始响应。"""

    content: str = ""
    tool_calls: list = field(default_factory=list)
    reasoning_content: str = ""  # 思考模式(如 deepseek-v4-pro)必须原样带回,否则 400
    raw: dict = field(default_factory=dict)


class LLMError(RuntimeError):
    """LLM 调用或配置相关错误。"""


class LLMClient:
    # vision=True/False 时只保留带(不带)vision: true 标记的 provider;
    # vision=None 保持原行为(全部 provider 按序尝试)。
    def __init__(self, config: dict | None = None, vision: bool | None = None, api_logger=None):
        self.config = config or llm_config.load()
        providers = self.config.get("providers", [])
        if vision is not None:
            providers = [p for p in providers if p.get("vision", False) == vision]
        self.providers = providers
        self.api_logger = api_logger

    def chat(
        self,
        messages: list[dict],
        tools: list | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        task_id: str = "",
        session_id: str = "",
    ) -> ChatResult:
        """按序尝试各 provider,任一成功即返回;全部失败抛 LLMError。"""
        errors = []
        for fallback_no, provider in enumerate(self.providers):
            try:
                return self._chat_one(
                    provider, messages, tools, temperature, max_tokens,
                    task_id=task_id, session_id=session_id, fallback_no=fallback_no,
                )
            except Exception as exc:
                errors.append(f"{provider.get('name')}: {exc}")
        raise LLMError("所有 LLM provider 均失败: " + " | ".join(errors))

    def _chat_one(self, provider, messages, tools, temperature, max_tokens, task_id="", session_id="", fallback_no=0) -> ChatResult:
        """调用单个 provider:生成请求元数据后,按 kind 分派到 OpenAI/Anthropic 实现。"""
        request_id = f"req_{os.urandom(8).hex()}"
        metadata = {
            "request_id": request_id,
            "task_id": task_id,
            "session_id": session_id,
            "provider": provider.get("name", ""),
            "model": provider.get("model", ""),
            "requested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "fallback_no": fallback_no,
        }
        if provider.get("kind") == "anthropic":
            return self._chat_anthropic(provider, messages, tools, temperature, max_tokens, metadata)
        return self._chat_openai(provider, messages, tools, temperature, max_tokens, metadata)

    def ping(self, max_tokens: int = 8) -> list[dict]:
        """逐个 provider 发一条最小 chat 测连接延迟;返回 [{name, ok, latency_ms} 或 {name, ok, error}]。"""
        results = []
        for provider in self.providers:
            t0 = time.time()
            try:
                self._chat_one(provider, [{"role": "user", "content": "ping"}], None, 0.0, max_tokens)
                results.append(
                    {
                        "name": provider.get("name"),
                        "ok": True,
                        "latency_ms": round((time.time() - t0) * 1000),
                    }
                )
            except Exception as exc:
                results.append({"name": provider.get("name"), "ok": False, "error": str(exc)[:200]})
        return results

    def _chat_openai(self, provider, messages, tools, temperature, max_tokens, metadata=None) -> ChatResult:
        """调用 OpenAI 兼容的 /chat/completions 接口,并把请求/响应写入 api_logger。"""
        url = provider["base_url"].rstrip("/") + "/chat/completions"
        body = {
            "model": provider["model"],
            "messages": messages,
            "temperature": self._value(temperature, provider, "temperature", 0.1),
            "max_tokens": self._value(max_tokens, provider, "max_tokens", DEFAULT_MAX_TOKENS),
        }
        if tools:
            body["tools"] = tools
        request_id = (metadata or {}).get("request_id", "req_unknown")
        started = time.perf_counter()
        response_logged = False
        if self.api_logger:
            self.api_logger.request(request_id, metadata or {}, messages, tools, body)
        try:
            # 在已记录的请求生命周期内解析凭据,这样即便配置出错也能产生配对的响应事件
            api_key = self._api_key(provider)
            resp = requests.post(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                json=body,
                timeout=provider.get("timeout_sec", DEFAULT_TIMEOUT),
            )
            elapsed = round((time.perf_counter() - started) * 1000)
            if resp.status_code != 200:
                if self.api_logger:
                    self.api_logger.response(request_id, {**(metadata or {}), "duration_ms": elapsed}, error=resp.text[:1000], http_status=resp.status_code)
                    response_logged = True
                raise LLMError(f"openai {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
            if self.api_logger:
                self.api_logger.response(request_id, {**(metadata or {}), "duration_ms": elapsed}, response=data, http_status=resp.status_code)
                response_logged = True
        except Exception as exc:
            if self.api_logger and not response_logged:
                self.api_logger.response(request_id, {**(metadata or {}), "duration_ms": round((time.perf_counter() - started) * 1000)}, error=str(exc))
            raise
        message = data["choices"][0]["message"]
        return ChatResult(
            message.get("content") or "",
            message.get("tool_calls") or [],
            message.get("reasoning_content") or "",
            data,
        )

    def _chat_anthropic(self, provider, messages, tools, temperature, max_tokens, metadata=None) -> ChatResult:
        """调用 Anthropic /v1/messages 接口,并把 OpenAI 风格消息/工具转换为 Anthropic 格式。"""
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
        request_id = (metadata or {}).get("request_id", "req_unknown")
        started = time.perf_counter()
        response_logged = False
        if self.api_logger:
            self.api_logger.request(request_id, metadata or {}, messages, tools, body)
        try:
            api_key = self._api_key(provider)
            resp = requests.post(
                url,
                headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
                json=body,
                timeout=provider.get("timeout_sec", DEFAULT_TIMEOUT),
            )
            elapsed = round((time.perf_counter() - started) * 1000)
            if resp.status_code != 200:
                if self.api_logger:
                    self.api_logger.response(request_id, {**(metadata or {}), "duration_ms": elapsed}, error=resp.text[:1000], http_status=resp.status_code)
                    response_logged = True
                raise LLMError(f"anthropic {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
            if self.api_logger:
                self.api_logger.response(request_id, {**(metadata or {}), "duration_ms": elapsed}, response=data, http_status=resp.status_code)
                response_logged = True
        except Exception as exc:
            if self.api_logger and not response_logged:
                self.api_logger.response(request_id, {**(metadata or {}), "duration_ms": round((time.perf_counter() - started) * 1000)}, error=str(exc))
            raise
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
        return ChatResult(content, tool_calls, "", data)

    @staticmethod
    def _to_anthropic_messages(messages: list[dict]) -> tuple[str, list[dict]]:
        """把 OpenAI 风格消息转成 Anthropic 格式,返回 (system 文本, messages 列表)。"""
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
                out.append({"role": "user", "content": LLMClient._to_anthropic_content(m.get("content", ""))})
        return "\n\n".join(system_parts), out

    @staticmethod
    def _to_anthropic_content(content: Any) -> Any:
        """把 OpenAI 风格的多模态 content 列表转成 Anthropic 的 text/image 块。"""
        if not isinstance(content, list):
            return content
        out = []
        for part in content:
            if not isinstance(part, dict):
                out.append({"type": "text", "text": str(part)})
                continue
            if part.get("type") == "text":
                out.append({"type": "text", "text": str(part.get("text", ""))})
            elif part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                media_type, data = LLMClient._parse_data_url(url)
                out.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": data,
                        },
                    }
                )
        return out

    @staticmethod
    def _parse_data_url(url: str) -> tuple[str, str]:
        """拆解 data:image/*;base64 形式的 URL,返回 (media_type, base64 数据)。"""
        prefix = "data:"
        if not url.startswith(prefix) or ";base64," not in url:
            raise LLMError("Anthropic vision 只支持 data:image/*;base64 图像 URL")
        header, data = url[len(prefix):].split(";base64,", 1)
        # 轻量校验,尽早发现截断或非 base64 数据。
        base64.b64decode(data, validate=True)
        return header or "image/jpeg", data

    @staticmethod
    def _value(value, provider, key, default):
        """显式传入的参数优先,否则取 provider 配置,最后回退默认值。"""
        return value if value is not None else provider.get(key, default)

    @staticmethod
    def _api_key(provider) -> str:
        """从 api_key_env 指定的环境变量读取 API key,缺失时抛 LLMError。"""
        env = provider.get("api_key_env")
        if env:
            key = os.environ.get(env)
            if key:
                return key
        raise LLMError(f"provider {provider.get('name')} 缺少 API key(环境变量 {env})")
