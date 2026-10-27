"""Explicit, operator-triggered assistant boundary.

Supports the default local Ollama endpoint and an OpenAI-compatible hosted
provider (Pollinations). This module is never called by capture, inference,
tracking, state, or alerts. The console invokes it only from an explicit user
request and persists the request/response through the PLOMID repository
boundary.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class OllamaConfig:
    enabled: bool = True
    provider: Literal["ollama", "pollinations"] = "ollama"
    url: str = "http://127.0.0.1:11434"
    model: str = "mistral"
    timeout_s: float = 30.0


class OllamaUnavailable(RuntimeError):
    """Configured assistant provider is unavailable or rejected the request."""


OUT_OF_SCOPE_MESSAGE = (
    "This question is outside the PLOMID Cabin assistant scope. "
    "Ask about the current cabin session, seats, detections, layout, "
    "inference, health, events, evidence, PLOMID persistence, or this MVP."
)

_APP_SCOPE_TERMS = frozenset(
    {
        "plomid",
        "odc",
        "cabin",
        "seat",
        "seats",
        "occupant",
        "occupancy",
        "person",
        "people",
        "adult",
        "child",
        "object",
        "belt",
        "buckle",
        "pose",
        "dms",
        "driver",
        "attention",
        "drowsy",
        "drowsiness",
        "phone",
        "steering",
        "rhd",
        "lhd",
        "layout",
        "row",
        "vehicle",
        "camera",
        "video",
        "frame",
        "detection",
        "detector",
        "inference",
        "model",
        "tracking",
        "track",
        "event",
        "alert",
        "evidence",
        "confidence",
        "bbox",
        "vector",
        "database",
        "db",
        "sql",
        "persistence",
        "schema",
        "health",
        "status",
        "error",
        "replay",
        "live",
        "session",
        "calibration",
        "calibrated",
        "export",
        "report",
        "mistral",
        "ollama",
        "start",
        "stop",
        "pause",
        "resume",
        "unknown",
        "unsupported",
        "current",
        "state",
        "explain",
        "working",
        "works",
        "issue",
        "problem",
        "down",
        "up",
        "available",
        "support",
        "function",
        "feature",
        "application",
        "app",
        "product",
        "purpose",
        "architecture",
        "pipeline",
        "algorithm",
        "algorithms",
        "module",
        "modules",
        "documentation",
        "requirement",
        "requirements",
        "customer",
        "pilot",
        "setup",
        "install",
        "installation",
        "configuration",
        "config",
        "source",
        "stream",
        "processing",
        "pending",
        "implemented",
        "implementation",
        "mvp",
        "ui",
        "ux",
        "screen",
    }
)


def is_in_scope(prompt: str) -> bool:
    """Return whether a question contains an explicit PLOMID product topic."""
    terms = set(re.findall(r"[a-z0-9_]+", prompt.lower()))
    return bool(terms & _APP_SCOPE_TERMS)


def generate(prompt: str, context: dict[str, object], cfg: OllamaConfig) -> str:
    """Ask the configured assistant once, on demand, with structured cabin context."""
    if not cfg.enabled:
        raise OllamaUnavailable("assistant is disabled")
    if not is_in_scope(prompt):
        return OUT_OF_SCOPE_MESSAGE
    system = (
        "You are the PLOMID Cabin engineering assistant. Use only the supplied "
        "structured cabin evidence. Answer the user's exact question directly "
        "in natural language, normally in 2-5 concise sentences; use bullets "
        "only when they make a comparison or procedure clearer. Do not repeat "
        "the full cabin summary, layout, or every seat unless asked. If a seat is named, "
        "focus only on that seat. Use exact state and reason-code values. Never "
        "infer a total person count from occupied seats, infer age from adult or "
        "child candidates, turn unknown into positive/negative evidence, or "
        "claim belt fastening, airbag control, safety qualification, medical "
        "diagnosis, or production readiness. If the evidence does not answer the "
        "question, say: 'Insufficient evidence in the current frame.' Do not "
        "describe the phrase 'provided cabin evidence' or add generic caveats. "
        "For questions about child presence or male/female counts, use the "
        "assistant_facts object exactly. Report child candidates separately from "
        "child-unknown seats. Report apparent male/female labels as experimental "
        "appearance evidence, never as verified gender. Do not call occupied-seat "
        "count a certified onboard-person count. "
        "If the user asks about a topic unrelated to PLOMID Cabin, occupant "
        "detection, this application, its models, its UI, or its database, reply "
        "only: 'This question is outside the PLOMID Cabin assistant scope.'"
    )
    evidence = json.dumps(context, default=str)
    headers = {"Content-Type": "application/json"}
    if cfg.provider == "pollinations":
        body = {
            "model": cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": f"User question:\n{prompt}\n\nCabin evidence:\n{evidence}",
                },
            ],
            "temperature": 0.0,
            "max_tokens": 500,
            "stream": False,
        }
        headers["User-Agent"] = "PLOMID-ODC/0.1"
        endpoint = cfg.url.rstrip("/")
    else:
        body = {
            "model": cfg.model,
            "stream": False,
            "system": system,
            "prompt": f"User question:\n{prompt}\n\nCabin evidence:\n{evidence}",
            "options": {
                "temperature": 0.0,
                "top_p": 0.8,
                "repeat_penalty": 1.15,
                "num_predict": 140,
            },
        }
        endpoint = cfg.url.rstrip("/") + "/api/generate"
    request = Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=cfg.timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        if cfg.provider == "pollinations":
            raise OllamaUnavailable(
                f"Pollinations request failed (HTTP {exc.code}); try again shortly"
            ) from exc
        raise OllamaUnavailable(f"Ollama HTTP {exc.code}; check model '{cfg.model}'") from exc
    except (URLError, TimeoutError, OSError) as exc:
        if cfg.provider == "pollinations":
            raise OllamaUnavailable(
                f"Pollinations is not reachable at {cfg.url}; try again shortly"
            ) from exc
        raise OllamaUnavailable(
            "Ollama is not reachable at " + cfg.url + "; start Ollama and pull model " + cfg.model
        ) from exc
    if cfg.provider == "pollinations":
        text = None
        if isinstance(payload, dict):
            choices = payload.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                message = choices[0].get("message")
                if isinstance(message, dict):
                    text = message.get("content")
        if not isinstance(text, str) or not text.strip():
            raise OllamaUnavailable("Pollinations returned no answer text")
        return text.strip()
    text = payload.get("response") if isinstance(payload, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise OllamaUnavailable("Ollama returned no response text")
    return text.strip()
