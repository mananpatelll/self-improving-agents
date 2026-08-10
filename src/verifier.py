"""Decide whether a predicted SQL query is correct.

Correctness here means "returns the same data as the gold query", 

This is deliberately code and not a language model: it is the ground truth the
whole improvement loop depends on, so it has to be cheap and deterministic.
"""

import re
from collections import Counter
from dataclasses import dataclass
from src.sql_executor import ExecResult, run_sql

# Floats are rounded before comparison so that an AVG computed two different
# ways does not count as a mismatch on the last decimal place.
_FLOAT_PRECISION = 6

# ORDER BY anywhere in the gold query means the caller asked for a specific
# row order, so we compare rows as a sequence instead of as a bag.
_ORDER_BY_PATTERN = re.compile(r"\border\s+by\b", re.IGNORECASE)


@dataclass(frozen=True)
class Verdict:
    """Result of checking one predicted query.

    The two ExecResults are kept so the teacher can look at what actually
    happened.
    """

    correct: bool
    reason: str
    predicted: ExecResult
    gold: ExecResult


def _normalize_value(value: object) -> object:
    """Make one cell comparable across queries that produce it differently."""
    if value is None:
        return None

    # COUNT(*) may come back as int in one query and float in another.
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return round(float(value), _FLOAT_PRECISION)

    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")

    if isinstance(value, str):
        return value.strip()

    return value


def _normalize_rows(rows: tuple[tuple, ...]) -> list[tuple]:
    """Normalize every cell in every row."""
    return [tuple(_normalize_value(cell) for cell in row) for row in rows]


def gold_requires_order(gold_sql: str) -> bool:
    """True if the gold query asked for rows in a particular order."""
    return bool(_ORDER_BY_PATTERN.search(gold_sql))


def results_match(
    predicted_rows: tuple[tuple, ...],
    gold_rows: tuple[tuple, ...],
    ordered: bool,
) -> bool:
    """Compare two result sets.

    When order does not matter we compare as bags, not sets, so a query that
    drops or duplicates rows is still caught.
    """
    predicted = _normalize_rows(predicted_rows)
    gold = _normalize_rows(gold_rows)

    if ordered:
        return predicted == gold

    return Counter(predicted) == Counter(gold)


def verify(
    db_id: str,
    predicted_sql: str,
    gold_sql: str,
) -> Verdict:
    """Run both queries and decide whether the prediction is correct.

    The reason field is written for the teacher to read, so it names the kind
    of failure rather than just saying no.
    """
    gold = run_sql(db_id, gold_sql)
    if not gold.ok:
        # The gold query comes from the dataset, so this means a broken example
        # rather than a worker mistake. Surfaced so it is not counted as a
        # failure the teacher should try to learn from.
        return Verdict(False, f"gold query failed: {gold.error}", ExecResult(), gold)

    predicted = run_sql(db_id, predicted_sql)
    if not predicted.ok:
        return Verdict(False, predicted.error or "query failed", predicted, gold)

    if results_match(predicted.rows, gold.rows, gold_requires_order(gold_sql)):
        return Verdict(True, "", predicted, gold)

    # Narrow the mismatch so the teacher gets a specific starting point.
    reason = _describe_mismatch(predicted.rows, gold.rows)
    return Verdict(False, reason, predicted, gold)


def _describe_mismatch(
    predicted_rows: tuple[tuple, ...],
    gold_rows: tuple[tuple, ...],
) -> str:
    """Say how two result sets differ, in one line."""
    predicted_columns = len(predicted_rows[0]) if predicted_rows else 0
    gold_columns = len(gold_rows[0]) if gold_rows else 0

    if predicted_rows and gold_rows and predicted_columns != gold_columns:
        return f"returned {predicted_columns} columns, expected {gold_columns}"

    if len(predicted_rows) != len(gold_rows):
        return f"returned {len(predicted_rows)} rows, expected {len(gold_rows)}"

    # Same shape, different contents, so ordering or values are wrong.
    if Counter(_normalize_rows(predicted_rows)) == Counter(_normalize_rows(gold_rows)):
        return "correct rows but in the wrong order"

    return "returned the right number of rows but the wrong values"


if __name__ == "__main__":
    checks = [
        # Correct, written differently from the gold query.
        ("concert_singer",
         "SELECT count(*) FROM singer",
         "SELECT count(singer_id) FROM singer"),
        # Wrong column.
        ("concert_singer",
         "SELECT country FROM singer",
         "SELECT name FROM singer"),
        # Invalid SQL.
        ("concert_singer",
         "SELECT * FROM nope",
         "SELECT name FROM singer"),
        # Right rows, wrong order, where the gold query asked for an order.
        ("concert_singer",
         "SELECT name FROM singer ORDER BY age ASC",
         "SELECT name FROM singer ORDER BY age DESC"),
    ]

    for db_id, predicted_sql, gold_sql in checks:
        verdict = verify(db_id, predicted_sql, gold_sql)
        status = "PASS" if verdict.correct else "FAIL"
        print(f"{status}  {predicted_sql[:44]:46s} {verdict.reason}")
