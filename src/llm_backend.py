#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 15:22:43 2026

@author: fabriziocoiai
"""
# llm_backend.py
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Protocol

import numpy as np
from openai import APIError, OpenAI, RateLimitError


class LLMBackend(Protocol):
    def chat(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        ...

    def embed(self, texts: List[str]) -> np.ndarray:
        ...


class OpenAIBackend:
    def __init__(
        self,
        chat_model: str = "gpt-5.2",
        embed_model: str = "text-embedding-3-small",
        api_key_env: str = "OPENAI_API_KEY",
        temperature: float = 0.2,
        max_output_tokens: int = 800,
    ):
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"{api_key_env} is not set. Export your OpenAI API key first.")

        self.client = OpenAI(api_key=api_key)
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    @staticmethod
    def _to_responses_input(messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for m in messages:
            role = (m.get("role") or "user").lower()
            if role not in {"system", "user", "assistant", "developer"}:
                role = "user"

            items.append(
                {
                    "role": role,
                    "content": [
                        {
                            "type": "input_text",
                            "text": m.get("content") or "",
                        }
                    ],
                }
            )
        return items

    def chat(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        payload = self._to_responses_input(messages)

        for attempt in range(5):
            try:
                resp = self.client.responses.create(
                    model=self.chat_model,
                    input=payload,
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens,
                )
                text = getattr(resp, "output_text", None)
                if text is None:
                    text = str(resp)
                return {"content": text.strip()}
            except RateLimitError:
                time.sleep(1.5 * (attempt + 1))
                continue
            except APIError as e:
                if getattr(e, "status_code", None) == 429:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise

        raise RuntimeError("OpenAI chat failed after retries due to rate limits/quota.")

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        try:
            resp = self.client.embeddings.create(
                model=self.embed_model,
                input=texts,
            )
        except APIError as e:
            raise RuntimeError(
                f"Embedding failed for model '{self.embed_model}'. "
                f"Original error: {e}"
            ) from e

        vectors = [item.embedding for item in resp.data]
        return np.array(vectors, dtype=np.float32)


ChatGPTBackend = OpenAIBackend
