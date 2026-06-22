from pdebug.pdag.io import DataCatalog, MemoryDataSet
from pdebug.pdag.pipeline import node, pipeline
from pdebug.pdag.runner import SequentialRunner

import pytest


def _dataset(data):
    return MemoryDataSet(data, copy_mode="assign")


def test_sequential_runner_returns_free_outputs():
    def add_one(value):
        return value + 1

    graph = pipeline([node(add_one, inputs="value", outputs="out")])
    catalog = DataCatalog({"value": _dataset(2)})

    assert SequentialRunner().run(graph, catalog) == {"out": 3}


def test_runner_reports_unsatisfied_inputs():
    graph = pipeline(
        [node(lambda value: value, inputs="missing", outputs="out")]
    )

    with pytest.raises(ValueError, match="Pipeline input\\(s\\).*missing"):
        SequentialRunner().run(graph, DataCatalog())


def test_runner_warns_before_saving_none_output(capsys):
    graph = pipeline([node(lambda: None, inputs=None, outputs="out")])

    with pytest.raises(Exception, match="Saving 'None'"):
        SequentialRunner().run(graph, DataCatalog())

    assert "please provide output" in capsys.readouterr().out


def test_runner_suggests_resume_and_reraises(monkeypatch):
    runner = SequentialRunner()
    calls = []

    def fail():
        raise RuntimeError("boom")

    monkeypatch.setattr(
        runner,
        "_suggest_resume_scenario",
        lambda pipeline, done_nodes: calls.append(set(done_nodes)),
    )
    graph = pipeline([node(fail, inputs=None, outputs="out")])

    with pytest.raises(RuntimeError, match="boom"):
        runner.run(graph, DataCatalog())

    assert calls == [set()]


def test_runner_releases_intermediate_datasets():
    def add_one(value):
        return value + 1

    def double(value):
        return value * 2

    graph = pipeline(
        [
            node(add_one, inputs="input", outputs="intermediate"),
            node(double, inputs="intermediate", outputs="output"),
        ]
    )
    runner = SequentialRunner()
    catalog = DataCatalog(
        {
            "input": _dataset(1),
            "intermediate": MemoryDataSet(copy_mode="assign"),
            "output": MemoryDataSet(copy_mode="assign"),
        }
    )

    runner.run(graph, catalog)

    assert catalog.load("input") == 1
    assert catalog.load("output") == 4
    with pytest.raises(Exception, match="has not been saved"):
        catalog.load("intermediate")
