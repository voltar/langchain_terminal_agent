"""Local result interpretation without sending row data to cloud LLMs.

Mirrors DATA-ARC's LOCAL_INTERPRETATION mode: query results stay in-process.
Only schema/metadata and generated SQL are used with remote models elsewhere.
"""

from __future__ import annotations

import base64
import io
import logging
import re
from typing import Any

from data_agent.models.outputs import QueryResult

logger = logging.getLogger(__name__)

_SAMPLE_ROWS = 5
_MAX_COLS_PER_ROW = 12


def extract_tabular_data(
    result: QueryResult | dict[str, Any] | Any | None,
) -> tuple[list[str], list[list[Any]], int]:
    """Normalize a query result into columns, rows, and row_count.

    Args:
        result: QueryResult model, dict, or other result payload.

    Returns:
        Tuple of (columns, rows, row_count).
    """
    if result is None:
        return [], [], 0

    if isinstance(result, QueryResult):
        columns = list(result.columns or [])
        rows = [list(row) for row in (result.rows or [])]
        row_count = result.row_count if result.row_count else len(rows)
        return columns, rows, row_count

    if isinstance(result, dict):
        columns = list(result.get("columns") or [])
        raw_rows = result.get("rows") or []
        rows = [list(row) if not isinstance(row, list) else row for row in raw_rows]
        row_count = int(result.get("row_count") or len(rows))
        return columns, rows, row_count

    columns = list(getattr(result, "columns", None) or [])
    raw_rows = getattr(result, "rows", None) or []
    rows = [list(row) if not isinstance(row, list) else row for row in raw_rows]
    row_count = int(getattr(result, "row_count", None) or len(rows))
    return columns, rows, row_count


def local_interpret_results(
    question: str,
    result: QueryResult | dict[str, Any] | Any | None,
    *,
    sql: str = "",
    sample_rows: int = _SAMPLE_ROWS,
) -> str:
    """Build a natural-language summary of query results without an LLM.

    Args:
        question: Original user question (shown for context only).
        result: Query execution payload.
        sql: Optional SQL that was executed (shown for transparency).
        sample_rows: Number of sample rows to include in the summary.

    Returns:
        Markdown-formatted local interpretation text.
    """
    columns, rows, row_count = extract_tabular_data(result)

    if not rows and row_count == 0:
        return (
            "**Local evaluation** (no cloud AI on results)\n\n"
            "No rows were returned for this query."
        )

    # SQLDatabase.run often returns a single string cell under column "result"
    if (
        len(columns) == 1
        and columns[0] == "result"
        and len(rows) == 1
        and len(rows[0]) == 1
    ):
        return _interpret_raw_payload(question, sql, rows[0][0])

    lines = [
        "**Local evaluation** (no cloud AI on results)",
        "",
        f"Found **{row_count:,}** row(s) for: _{question.strip()}_",
    ]
    if sql:
        lines.extend(["", f"Executed SQL:\n```sql\n{sql.strip()}\n```"])

    lines.extend(["", "### Sample results", ""])

    display_rows = rows[:sample_rows]
    if columns:
        header = " | ".join(str(c) for c in columns[:_MAX_COLS_PER_ROW])
        sep = " | ".join("---" for _ in columns[:_MAX_COLS_PER_ROW])
        lines.append(f"| {header} |")
        lines.append(f"| {sep} |")
        for row in display_rows:
            cells = [
                _format_cell(row[i] if i < len(row) else "")
                for i in range(min(len(columns), _MAX_COLS_PER_ROW))
            ]
            lines.append(f"| {' | '.join(cells)} |")
    else:
        for i, row in enumerate(display_rows, 1):
            lines.append(f"**{i}.** {_format_cell(row)}")

    if row_count > sample_rows:
        lines.append("")
        lines.append(f"... and **{row_count - sample_rows:,}** more row(s).")

    lines.extend(
        [
            "",
            "_Tip: set `LOCAL_INTERPRETATION=false` for cloud LLM narrative analysis "
            "(sends result rows to the model)._",
        ]
    )
    return "\n".join(lines)


def local_visualize_results(
    result: QueryResult | dict[str, Any] | Any | None,
    *,
    question: str = "",
    max_categories: int = 25,
) -> tuple[str | None, str | None, str | None]:
    """Create a simple chart locally without sending data to an LLM.

    Heuristic:
    - 2+ columns with a numeric series → bar chart (first non-numeric label col
      vs first numeric value col), or line if labels look sequential.
    - Single numeric column → histogram of values.
    - Otherwise → skip with an error message (no cloud fallback).

    Args:
        result: Query execution payload.
        question: Optional title context.
        max_categories: Cap categories plotted to keep charts readable.

    Returns:
        Tuple of (base64_png, code_or_description, error). Exactly one of
        image or error is set on success/failure.
    """
    columns, rows, _row_count = extract_tabular_data(result)
    if not rows:
        return None, None, "No data to visualize"

    # Unwrap raw SQLDatabase string payloads — cannot chart reliably
    if (
        len(columns) == 1
        and columns[0] == "result"
        and len(rows) == 1
        and isinstance(rows[0][0], str)
    ):
        return (
            None,
            None,
            "Local visualization needs structured columns/rows; "
            "raw SQL string results are not charted.",
        )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None, None, "matplotlib is required for local visualization"

    label_idx, value_idx = _pick_chart_columns(columns, rows)
    title = (question.strip()[:80] if question else "Query results") or "Query results"

    try:
        fig, ax = plt.subplots(figsize=(8, 4.5))

        if label_idx is not None and value_idx is not None:
            labels = [str(row[label_idx]) for row in rows[:max_categories]]
            values = [_to_float(row[value_idx]) for row in rows[:max_categories]]
            values = [v if v is not None else 0.0 for v in values]
            use_line = _looks_sequential(labels)
            if use_line:
                ax.plot(range(len(labels)), values, marker="o")
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha="right")
                code_desc = "local line chart"
            else:
                ax.bar(range(len(labels)), values)
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha="right")
                code_desc = "local bar chart"
            ax.set_xlabel(str(columns[label_idx]))
            ax.set_ylabel(str(columns[value_idx]))
        elif value_idx is not None:
            values = [
                v
                for row in rows
                if (v := _to_float(row[value_idx] if value_idx < len(row) else None))
                is not None
            ]
            if not values:
                plt.close(fig)
                return None, None, "No numeric values found for local chart"
            ax.hist(values, bins=min(20, max(5, len(set(values)))), color="steelblue")
            ax.set_xlabel(str(columns[value_idx]) if columns else "value")
            ax.set_ylabel("count")
            code_desc = "local histogram"
        else:
            plt.close(fig)
            return (
                None,
                None,
                "Local visualization could not infer a numeric series from the result",
            )

        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        img_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
        return img_b64, code_desc, None
    except Exception as e:
        logger.exception("Local visualization failed")
        return None, None, f"Local visualization failed: {e}"


def _interpret_raw_payload(question: str, sql: str, payload: Any) -> str:
    """Summarize a raw single-cell SQLDatabase.run payload."""
    text = str(payload).strip() if payload is not None else ""
    lines = [
        "**Local evaluation** (no cloud AI on results)",
        "",
        f"Query for: _{question.strip()}_",
    ]
    if sql:
        lines.extend(["", f"Executed SQL:\n```sql\n{sql.strip()}\n```"])
    lines.extend(["", "### Result", "", "```", text[:4000], "```"])
    if len(text) > 4000:
        lines.append("")
        lines.append(f"... truncated ({len(text):,} characters total).")
    lines.extend(
        [
            "",
            "_Tip: set `LOCAL_INTERPRETATION=false` for cloud LLM narrative analysis "
            "(sends result rows to the model)._",
        ]
    )
    return "\n".join(lines)


def _format_cell(value: Any) -> str:
    """Format a cell value for markdown tables."""
    if value is None:
        return ""
    text = str(value).replace("|", "\\|").replace("\n", " ")
    return text if len(text) <= 80 else text[:77] + "..."


def _to_float(value: Any) -> float | None:
    """Best-effort numeric conversion."""
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _pick_chart_columns(
    columns: list[str], rows: list[list[Any]]
) -> tuple[int | None, int | None]:
    """Pick label and value column indexes for a simple chart."""
    if not rows:
        return None, None

    width = max((len(r) for r in rows), default=0)
    if width == 0:
        return None, None

    numeric_idxs: list[int] = []
    non_numeric_idxs: list[int] = []
    for i in range(width):
        sample = [row[i] for row in rows[:20] if i < len(row)]
        nums = [_to_float(v) for v in sample]
        if sample and sum(1 for n in nums if n is not None) >= max(1, len(sample) // 2):
            numeric_idxs.append(i)
        else:
            non_numeric_idxs.append(i)

    if not numeric_idxs:
        return None, None

    value_idx = numeric_idxs[0]
    label_idx = non_numeric_idxs[0] if non_numeric_idxs else None
    if label_idx is None and len(numeric_idxs) >= 2:
        label_idx = numeric_idxs[0]
        value_idx = numeric_idxs[1]
    return label_idx, value_idx


def _looks_sequential(labels: list[str]) -> bool:
    """Heuristic: treat labels as a sequence (dates/years/indexes) → line chart."""
    if len(labels) < 3:
        return False
    dateish = sum(1 for lab in labels if re.search(r"\d{4}-\d{2}", lab) or lab.isdigit())
    return dateish >= max(2, len(labels) // 2)
