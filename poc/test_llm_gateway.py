from llm.gateway import LLMConfig, LLMGateway


class _FakeResponse:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "model": "test/model",
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"total_tokens": 1},
        }


class _FakeSession:
    def __init__(self):
        self.calls = []

    def post(self, url, *, headers=None, json=None, timeout=None):
        self.calls.append({
            "url": url,
            "headers": headers,
            "json": json,
            "timeout": timeout,
        })
        return _FakeResponse()


def test_llm_gateway_reuses_thread_local_session_for_chat_calls():
    gateway = LLMGateway(LLMConfig(
        api_key="test",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        max_retries=1,
    ))
    fake_session = _FakeSession()
    gateway._session_local.session = fake_session

    first = gateway.chat("one")
    second = gateway.chat("two")

    assert first.content == "ok"
    assert second.content == "ok"
    assert len(fake_session.calls) == 2
    assert fake_session.calls[0]["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert fake_session.calls[1]["json"]["messages"][0]["content"] == "two"


def test_llm_gateway_builds_openrouter_provider_routing_from_env(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_OPENROUTER_PROVIDER_SORT", "throughput")
    monkeypatch.setenv("DRAFTPROOF_OPENROUTER_PROVIDER_ORDER", "together,fireworks,deepinfra")
    monkeypatch.setenv("DRAFTPROOF_OPENROUTER_ALLOW_FALLBACKS", "true")
    gateway = LLMGateway(LLMConfig(
        api_key="test",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        max_retries=1,
    ))
    fake_session = _FakeSession()
    gateway._session_local.session = fake_session

    gateway.chat("provider route")

    provider = fake_session.calls[0]["json"]["provider"]
    assert provider == {
        "order": ["together", "fireworks", "deepinfra"],
        "allow_fallbacks": True,
        "sort": "throughput",
    }
