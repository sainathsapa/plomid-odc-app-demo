"""Tests for the explicit, operator-triggered assistant boundary."""

from __future__ import annotations

import json

import pytest

from odc.ollama import OUT_OF_SCOPE_MESSAGE, OllamaConfig, OllamaUnavailable, generate, is_in_scope


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        body = {"response": "Seat 3 is unknown because evidence is insufficient."}
        return json.dumps(body).encode()


def test_generate_uses_local_endpoint_and_returns_text(monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["timeout"] = timeout
        seen["body"] = json.loads(request.data.decode())
        return _Response()

    monkeypatch.setattr("odc.ollama.urlopen", fake_urlopen)
    result = generate("Why is Seat 3 unknown?", {"seat_id": "seat_3"}, OllamaConfig())
    assert "insufficient" in result
    assert seen["url"] == "http://127.0.0.1:11434/api/generate"
    assert seen["body"]["model"] == "mistral"
    assert seen["body"]["stream"] is False
    assert seen["body"]["options"]["num_predict"] == 140
    assert "natural language" in seen["body"]["system"]


def test_generate_pollinations_uses_openai_chat_endpoint(monkeypatch):
    seen = {}

    class PollinationsResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            body = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Seat 3 has insufficient evidence.",
                        }
                    }
                ]
            }
            return json.dumps(body).encode()

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["timeout"] = timeout
        seen["body"] = json.loads(request.data.decode())
        seen["authorization"] = request.get_header("Authorization")
        return PollinationsResponse()

    monkeypatch.setattr("odc.ollama.urlopen", fake_urlopen)
    cfg = OllamaConfig(
        provider="pollinations",
        url="https://text.pollinations.ai/openai",
        model="openai-fast",
        timeout_s=90,
    )
    result = generate("Why is Seat 3 unknown?", {"seat_id": "seat_3"}, cfg)
    assert "insufficient" in result
    assert seen["url"] == "https://text.pollinations.ai/openai"
    assert seen["timeout"] == 90
    assert seen["authorization"] is None
    assert seen["body"]["model"] == "openai-fast"
    assert seen["body"]["stream"] is False
    assert seen["body"]["max_tokens"] == 500
    assert seen["body"]["messages"][0]["role"] == "system"
    assert "natural language" in seen["body"]["messages"][0]["content"]
    assert "Cabin evidence" in seen["body"]["messages"][1]["content"]
    assert seen["body"]["messages"][1]["role"] == "user"


def test_generate_disabled_fails_closed():
    with pytest.raises(OllamaUnavailable, match="disabled"):
        generate("hello", {}, OllamaConfig(enabled=False))


def test_unrelated_question_is_rejected_before_model_call(monkeypatch):
    def fail_urlopen(*_args, **_kwargs):
        raise AssertionError("Ollama must not receive unrelated questions")

    monkeypatch.setattr("odc.ollama.urlopen", fail_urlopen)
    assert generate("What is the capital of France?", {}, OllamaConfig()) == OUT_OF_SCOPE_MESSAGE


def test_application_questions_are_in_scope():
    assert is_in_scope("What is the purpose of this application and how does the pipeline work?")
    assert is_in_scope("Why is the database down?")
