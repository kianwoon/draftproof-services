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


class _FakeNullContentResponse:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "model": "qwen/qwen3-next-80b-a3b-thinking-2509",
            "provider": "Alibaba",
            "choices": [{
                "finish_reason": "stop",
                "native_finish_reason": "stop",
                "message": {"role": "assistant", "content": None},
            }],
            "usage": {
                "total_tokens": 10,
                "completion_tokens_details": {"reasoning_tokens": 8},
            },
        }


class _FakeSequenceSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, *, headers=None, json=None, timeout=None):
        self.calls.append({
            "url": url,
            "headers": headers,
            "json": json,
            "timeout": timeout,
        })
        return self.responses.pop(0)


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


def test_llm_gateway_retries_null_content_response():
    gateway = LLMGateway(LLMConfig(
        api_key="test",
        base_url="https://openrouter.ai/api/v1",
        model="qwen/qwen3-next-80b-a3b-thinking",
        max_retries=2,
    ))
    fake_session = _FakeSequenceSession([_FakeNullContentResponse(), _FakeResponse()])
    gateway._session_local.session = fake_session

    response = gateway.chat("return json")

    assert response.content == "ok"
    assert len(fake_session.calls) == 2


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


def test_llm_gateway_bounds_mandatory_qwen_reasoning_when_effort_none(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_OPENROUTER_REASONING_EFFORT", "none")
    gateway = LLMGateway(LLMConfig(
        api_key="test",
        base_url="https://openrouter.ai/api/v1",
        model="qwen/qwen3-next-80b-a3b-thinking",
        max_retries=1,
    ))
    fake_session = _FakeSession()
    gateway._session_local.session = fake_session

    gateway.chat("return json")

    payload = fake_session.calls[0]["json"]
    assert payload["reasoning"] == {"max_tokens": 256, "exclude": True}
    assert payload["include_reasoning"] is False


def test_llm_gateway_preserves_reasoning_disable_for_non_reasoning_model(monkeypatch):
    monkeypatch.setenv("DRAFTPROOF_OPENROUTER_REASONING_EFFORT", "none")
    gateway = LLMGateway(LLMConfig(
        api_key="test",
        base_url="https://openrouter.ai/api/v1",
        model="z-ai/glm-5.1",
        max_retries=1,
    ))
    fake_session = _FakeSession()
    gateway._session_local.session = fake_session

    gateway.chat("return json")

    payload = fake_session.calls[0]["json"]
    assert payload["reasoning"] == {"effort": "none", "exclude": True}
    assert payload["include_reasoning"] is False


def test_llm_gateway_checks_cancellation_before_request():
    class Canceled(BaseException):
        pass

    def cancel():
        raise Canceled()

    gateway = LLMGateway(LLMConfig(
        api_key="test",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        max_retries=1,
        cancellation_check=cancel,
    ))
    fake_session = _FakeSession()
    gateway._session_local.session = fake_session

    try:
        gateway.chat("cancel me")
    except Canceled:
        pass
    else:
        raise AssertionError("Expected cancellation to stop the LLM request")

    assert fake_session.calls == []
