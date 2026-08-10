"""Run SQL against a Spider database, and expose it to the worker as tools.

Two consumers:
  - the verifier, which needs structured rows so it can compare result sets
  - the worker, which needs LangChain tools it can call while writing a query

SQL here is model-generated, so every query is treated as possibly invalid,
slow, or destructive.
"""

from dataclasses import dataclass
from pathlib import Path

from langchain_community.utilities import SQLDatabase
from langchain_core.tools import BaseTool, tool
from sqlalchemy import Engine, create_engine

DATABASE_DIR = Path("spider_data/database")

# A query slower than this is a runaway join, not a real answer.
DEFAULT_TIMEOUT_SECONDS = 5.0

# No correct Spider answer is this large. Anything bigger means a broken join.
MAX_ROWS = 10_000

# How often SQLite checks whether we want to stop, in VM instructions.
# Small enough to react quickly, large enough that the check costs nothing.
_PROGRESS_CHECK_INTERVAL = 1_000

# Engines are reused across queries so we open each database file only once.
_engines: dict[str, Engine] = {}


@dataclass(frozen=True)
class ExecResult:
    """Outcome of running one query.

    Either rows is set or error is, never both. Errors are returned rather than
    raised because a failed query is a normal event here, and the error text is
    what the teacher needs in order to diagnose the failure.
    """

    rows: tuple[tuple, ...] | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True if the query ran successfully."""
        return self.error is None


def database_path(db_id: str, database_dir: Path = DATABASE_DIR) -> Path:
    """Return the .sqlite file for a database id."""
    return Path(database_dir) / db_id / f"{db_id}.sqlite"


def get_engine(db_id: str, database_dir: Path = DATABASE_DIR) -> Engine:
    """Return a cached read-only engine for a database.

    Model-generated SQL can contain DROP or UPDATE, and these database files are
    committed to the repo, so mode=ro makes writes impossible rather than just
    discouraged.
    """
    if db_id not in _engines:
        path = database_path(db_id, database_dir)
        if not path.exists():
            raise FileNotFoundError(f"database not found: {db_id}")

        _engines[db_id] = create_engine(
            f"sqlite:///file:{path.as_posix()}?mode=ro&uri=true"
        )

    return _engines[db_id]


def get_database(db_id: str, database_dir: Path = DATABASE_DIR) -> SQLDatabase:
    """Return a LangChain SQLDatabase, used for schema inspection."""
    return SQLDatabase(get_engine(db_id, database_dir))


def _raw_connection(connection) -> object:
    """Get the underlying sqlite3 connection from a SQLAlchemy connection.

    Needed because the query timeout is a sqlite3 feature with no SQLAlchemy
    equivalent. The attribute moved between SQLAlchemy versions.
    """
    proxy = connection.connection
    return getattr(proxy, "driver_connection", proxy)


def run_sql(
    db_id: str,
    sql: str,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_rows: int = MAX_ROWS,
    database_dir: Path = DATABASE_DIR,
) -> ExecResult:
    """Run one query and return its rows, or an explanation of what went wrong."""
    import time

    try:
        engine = get_engine(db_id, database_dir)
    except FileNotFoundError as exc:
        return ExecResult(error=str(exc))

    try:
        with engine.connect() as connection:
            raw = _raw_connection(connection)

            # Some Spider databases hold text that is not valid UTF-8. Without
            # this, reading those rows raises instead of returning data.
            raw.text_factory = lambda value: value.decode("utf-8", errors="replace")

            # Stop the query once the deadline passes. SQLite calls this handler
            # every _PROGRESS_CHECK_INTERVAL instructions and aborts on a
            # non-zero return, which is what lets us interrupt a running query.
            deadline = time.monotonic() + timeout_seconds
            raw.set_progress_handler(
                lambda: 1 if time.monotonic() > deadline else 0,
                _PROGRESS_CHECK_INTERVAL,
            )

            try:
                # exec_driver_sql, not text(), because text() treats a colon as
                # a bind parameter and Spider queries contain times like '12:30'.
                result = connection.exec_driver_sql(sql)

                # Fetch one row past the cap so we can tell "exactly at the cap"
                # from "over the cap" instead of silently truncating.
                rows = result.fetchmany(max_rows + 1)
            finally:
                raw.set_progress_handler(None, 0)

        if len(rows) > max_rows:
            return ExecResult(error=f"query returned more than {max_rows} rows")

        return ExecResult(rows=tuple(tuple(row) for row in rows))

    except Exception as exc:
        # An aborted query surfaces as an "interrupted" operational error, so
        # separate a real timeout from ordinary bad SQL.
        message = str(exc)
        if "interrupt" in message.lower():
            return ExecResult(error=f"query timed out after {timeout_seconds}s")
        return ExecResult(error=f"sql error: {message}")


def format_rows(rows: tuple[tuple, ...], preview: int = 20) -> str:
    """Render rows as text for the worker to read."""
    if not rows:
        return "(no rows)"

    shown = "\n".join(str(row) for row in rows[:preview])
    if len(rows) > preview:
        shown += f"\n... {len(rows) - preview} more rows"
    return shown


def make_sql_tools(db_id: str, database_dir: Path = DATABASE_DIR) -> list[BaseTool]:
    """Build the worker's tools, bound to one database.

    Returns a fresh list per episode, which is also where the teacher's added
    skills get appended.
    """

    @tool
    def list_tables() -> str:
        """List the tables in the database."""
        return ", ".join(get_database(db_id, database_dir).get_usable_table_names())

    @tool
    def describe_tables(table_names: str) -> str:
        """Show columns, types, and sample rows for tables.

        Args:
            table_names: comma-separated table names, e.g. "singer, concert"
        """
        names = [name.strip() for name in table_names.split(",") if name.strip()]
        try:
            return get_database(db_id, database_dir).get_table_info(names)
        except ValueError as exc:
            return f"error: {exc}"

    @tool
    def run_query(sql: str) -> str:
        """Run a read-only SQL query and return the rows it produces."""
        result = run_sql(db_id, sql, database_dir=database_dir)
        return format_rows(result.rows) if result.ok else f"error: {result.error}"

    return [list_tables, describe_tables, run_query]


if __name__ == "__main__":
    checks = [
        ("concert_singer", "SELECT name FROM singer LIMIT 3"),
        ("concert_singer", "SELECT * FROM no_such_table"),
        ("concert_singer", "DELETE FROM singer"),
        ("nonexistent_db", "SELECT 1"),
    ]

    for db_id, sql in checks:
        result = run_sql(db_id, sql)
        print(f"{db_id:16s} {sql[:32]:34s} -> {result.rows if result.ok else result.error}")

    print()
    for sql_tool in make_sql_tools("concert_singer"):
        print(f"tool: {sql_tool.name}")
    print(make_sql_tools("concert_singer")[0].invoke({}))
