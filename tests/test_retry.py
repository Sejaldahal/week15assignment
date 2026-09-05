import pytest

from backend.llm.base import LLMError
from backend.utils.retry import retry_async


@pytest.mark.asyncio
async def test_retries_transient_errors_then_succeeds():
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise LLMError("transient", retryable=True)
        return "ok"

    result = await retry_async(flaky, max_retries=3, base_delay=0.01, retryable_exc=LLMError)
    assert result == "ok"
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_does_not_retry_permanent_errors():
    calls = {"n": 0}

    async def always_fails():
        calls["n"] += 1
        raise LLMError("permanent", retryable=False)

    with pytest.raises(LLMError):
        await retry_async(always_fails, max_retries=3, base_delay=0.01, retryable_exc=LLMError)
    assert calls["n"] == 1
