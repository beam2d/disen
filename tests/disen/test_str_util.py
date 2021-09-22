import disen


def test_parse_optional() -> None:
    parse = disen.parse_optional(float)
    assert parse("1.5") == 1.5
    assert parse("None") is None
