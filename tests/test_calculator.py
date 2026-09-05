from backend.tools.calculator import execute


def test_basic_arithmetic():
    result = execute({"expression": "25 * 4 + 10"})
    assert result["result"] == 110


def test_division_by_zero():
    result = execute({"expression": "1 / 0"})
    assert "error" in result


def test_rejects_non_arithmetic():
    result = execute({"expression": "__import__('os').system('ls')"})
    assert "error" in result


def test_parentheses_and_precedence():
    result = execute({"expression": "(2 + 3) * 4"})
    assert result["result"] == 20
