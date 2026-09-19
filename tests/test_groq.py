import json

from backend.llm.groq import GroqProvider


def test_messages_convert_to_openai_tool_format():
    msgs = [
        {"role": "user", "text": "q"},
        {"role": "model", "call": {"name": "calculator", "args": {"expression": "1+1"}}, "raw": {"id": "call_1"}},
        {"role": "tool", "name": "calculator", "result": {"result": 2}},
    ]
    out = GroqProvider._to_messages("sys", msgs)
    assert out[0] == {"role": "system", "content": "sys"}
    assert out[2]["tool_calls"][0]["id"] == "call_1"
    assert json.loads(out[2]["tool_calls"][0]["function"]["arguments"]) == {"expression": "1+1"}
    assert out[3] == {"role": "tool", "tool_call_id": "call_1", "content": '{"result": 2}'}
