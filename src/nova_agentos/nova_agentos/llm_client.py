"""轻量 LLM 客户端:用标准库 urllib 支持 OpenAI 与 Anthropic 协议,零依赖。

providers 按配置顺序逐个尝试,失败则回退下一个。
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from typing import Any


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, config: dict[str, Any], logger=None) -> None:
        providers = config.get("providers") or []
        if not providers:
            raise LLMError("llm.yaml: providers is empty")
        self.providers = []
        for provider in providers:
            self.providers.append(
                {
                    "name": provider.get("name", "unnamed"),
                    "kind": provider.get("kind", "openai"),
                    "base_url": provider.get("base_url", "").rstrip("/"),
                    "model": provider.get("model", ""),
                    "api_key_env": provider.get("api_key_env", ""),
                    "max_tokens": int(provider.get("max_tokens", 2048)),
                    "temperature": float(provider.get("temperature", 0.2)),
                    "timeout_sec": float(provider.get("timeout_sec", 60.0)),
                }
            )
        self.logger = logger

    def _api_key(self, provider: dict) -> str:
        key = os.environ.get(provider["api_key_env"], "")
        if not key:
            raise LLMError(
                f"provider '{provider['name']}': env var '{provider['api_key_env']}' not set"
            )
        return key

    def _post(self, url: str, headers: dict, body: bytes, timeout: float) -> dict:
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _chat_openai(self, provider: dict, system: str, messages: list[dict]) -> str:
        url = f"{provider['base_url']}/chat/completions"
        body = {
            "model": provider["model"],
            "messages": [{"role": "system", "content": system}, *messages],
            "temperature": provider["temperature"],
            "max_tokens": provider["max_tokens"],
        }
        data = self._post(
            url,
            {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key(provider)}",
            },
            json.dumps(body).encode("utf-8"),
            provider["timeout_sec"],
        )
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"unexpected openai response: {json.dumps(data)[:500]}")

    def _chat_anthropic(self, provider: dict, system: str, messages: list[dict]) -> str:
        url = f"{provider['base_url'] or 'https://api.anthropic.com'}/v1/messages"
        body = {
            "model": provider["model"],
            "system": system,
            "messages": messages,
            "max_tokens": provider["max_tokens"],
        }
        data = self._post(
            url,
            {
                "Content-Type": "application/json",
                "x-api-key": self._api_key(provider),
                "anthropic-version": "2023-06-01",
            },
            json.dumps(body).encode("utf-8"),
            provider["timeout_sec"],
        )
        try:
            return "".join(block.get("text", "") for block in data["content"])
        except (KeyError, TypeError):
            raise LLMError(f"unexpected anthropic response: {json.dumps(data)[:500]}")

    def chat(self, system: str, messages: list[dict]) -> str:
        """按顺序尝试每个 provider,返回第一个成功结果。"""
        errors = []
        for provider in self.providers:
            try:
                if provider["kind"] == "anthropic":
                    text = self._chat_anthropic(provider, system, messages)
                else:
                    text = self._chat_openai(provider, system, messages)
                if self.logger is not None:
                    self.logger.info(f"LLM provider '{provider['name']}' ok")
                return text
            except Exception as exc:
                errors.append(f"{provider['name']}: {exc}")
                if self.logger is not None:
                    self.logger.warning(f"LLM provider '{provider['name']}' failed: {exc}")
        raise LLMError("all providers failed: " + "; ".join(errors))
