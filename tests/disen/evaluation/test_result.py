import tempfile

import disen


def test_save_load() -> None:
    result = disen.evaluation.Result.new()

    result.history += [
        {"a": 1.0, "b": 2.0},
        {"a": 2.0, "b": 3.0, "c": 4.0},
    ]
    result.add_metric("x", 10.0)
    result.add_parameterized_metric("p", -2.0, "y", -5.0)

    with tempfile.NamedTemporaryFile() as f:
        result.save(f.name)
        loaded = disen.evaluation.Result.load(f.name)
        assert result == loaded
