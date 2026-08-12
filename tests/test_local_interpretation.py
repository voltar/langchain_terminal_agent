"""Tests for local result interpretation (no cloud LLM on result rows)."""

from unittest.mock import MagicMock

from data_agent.models.outputs import QueryResult
from data_agent.nodes.response import ResponseNode
from data_agent.nodes.visualization import VisualizationNode
from data_agent.utils.local_interpretation import (
    extract_tabular_data,
    local_interpret_results,
    local_visualize_results,
)


class TestExtractTabularData:
    def test_query_result_model(self) -> None:
        result = QueryResult(
            columns=["name", "amount"],
            rows=[["Alice", 10], ["Bob", 20]],
            row_count=2,
        )
        cols, rows, count = extract_tabular_data(result)
        assert cols == ["name", "amount"]
        assert rows == [["Alice", 10], ["Bob", 20]]
        assert count == 2

    def test_dict_result(self) -> None:
        cols, rows, count = extract_tabular_data(
            {"columns": ["x"], "rows": [[1], [2], [3]], "row_count": 3}
        )
        assert cols == ["x"]
        assert count == 3
        assert len(rows) == 3

    def test_none(self) -> None:
        assert extract_tabular_data(None) == ([], [], 0)


class TestLocalInterpretResults:
    def test_empty(self) -> None:
        text = local_interpret_results("how many?", None)
        assert "No rows" in text
        assert "Local evaluation" in text

    def test_tabular_sample(self) -> None:
        result = QueryResult(
            columns=["region", "sales"],
            rows=[["EMEA", 100], ["APAC", 200], ["AMER", 150]],
            row_count=3,
        )
        text = local_interpret_results(
            "sales by region",
            result,
            sql="SELECT region, sales FROM t",
        )
        assert "3" in text
        assert "EMEA" in text
        assert "SELECT region" in text
        assert "Local evaluation" in text

    def test_raw_sql_payload(self) -> None:
        result = QueryResult(
            columns=["result"],
            rows=[["[('a', 1), ('b', 2)]"]],
            row_count=1,
            metadata={"raw": True},
        )
        text = local_interpret_results("list items", result)
        assert "Result" in text
        assert "('a', 1)" in text


class TestLocalVisualizeResults:
    def test_bar_chart_from_category_and_value(self) -> None:
        result = QueryResult(
            columns=["region", "total"],
            rows=[["A", 10], ["B", 20], ["C", 15]],
            row_count=3,
        )
        img, code, err = local_visualize_results(result, question="totals")
        assert err is None
        assert img is not None
        assert len(img) > 100
        assert code is not None

    def test_no_data(self) -> None:
        img, _code, err = local_visualize_results(
            QueryResult(columns=["x"], rows=[], row_count=0)
        )
        assert img is None
        assert err == "No data to visualize"


class TestResponseNodeLocal:
    def test_local_path_does_not_call_llm(self) -> None:
        llm = MagicMock()
        structured = MagicMock()
        llm.with_structured_output.return_value = structured

        from data_agent.config import DataAgentConfig

        node = ResponseNode(
            llm,
            DataAgentConfig(name="test"),
            local_interpretation=True,
        )
        state = {
            "question": "How many orders?",
            "generated_sql": "SELECT COUNT(*) AS c FROM orders",
            "result": QueryResult(
                columns=["c"],
                rows=[[42]],
                row_count=1,
            ),
        }
        out = node.generate_response(state)
        structured.invoke.assert_not_called()
        assert "Local evaluation" in out["final_response"]
        assert "42" in out["final_response"]

    def test_cloud_path_calls_llm(self) -> None:
        llm = MagicMock()
        structured = MagicMock()
        structured.invoke.return_value = MagicMock(response="There are 42 orders.")
        llm.with_structured_output.return_value = structured

        from data_agent.config import DataAgentConfig
        from data_agent.models.outputs import ResponseGeneratorOutput

        structured.invoke.return_value = ResponseGeneratorOutput(
            response="There are 42 orders.", confidence=1.0
        )

        node = ResponseNode(
            llm,
            DataAgentConfig(name="test"),
            local_interpretation=False,
        )
        state = {
            "question": "How many orders?",
            "generated_sql": "SELECT 1",
            "result": QueryResult(columns=["c"], rows=[[42]], row_count=1),
        }
        out = node.generate_response(state)
        structured.invoke.assert_called_once()
        assert out["final_response"] == "There are 42 orders."


class TestVisualizationNodeLocal:
    def test_local_path_does_not_call_llm(self) -> None:
        llm = MagicMock()
        executor = MagicMock()
        node = VisualizationNode(llm, executor, local_interpretation=True)
        state = {
            "question": "chart sales",
            "result": QueryResult(
                columns=["region", "sales"],
                rows=[["N", 1], ["S", 2]],
                row_count=2,
            ),
        }
        # run async method synchronously
        import asyncio

        out = asyncio.run(node.generate_visualization(state))
        llm.ainvoke.assert_not_called()
        executor.execute.assert_not_called()
        assert out.get("visualization_image")
        assert out.get("visualization_error") is None
