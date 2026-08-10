"""Worker agent: turns a question into SQL.

A LangGraph ReAct agent bound to the SQL tools for one database. It has no
memory or extra tools yet -- both are added by the teacher during the
improvement loop. This file works standalone with empty memory, which is
also how the ablation baseline is produced.
"""

from dataclasses import dataclass
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool, tool
from langgraph.errors import GraphRecursionError

from src.sql_executor import make_sql_tools

load_dotenv()
WORKER_MODEL = "claude-haiku-4-5"

# A hard cap on tool calls per question. Without this, a confused agent can
# loop (describe table, query, describe table again...) and never submit.
MAX_STEPS = 15

# Persona only -- no step-by-step instructions. Scripting the process here
# would be prompting around the exact thing the teacher is meant to fix.
SYSTEM_PROMPT = """\
You are a database analyst. You answer questions about a SQLite database by \
writing SQL.

You do not know the schema in advance -- use your tools to look at the \
tables before writing a query. The only way to give your answer is by \
calling submit_answer with the final SQL. This is true even after you \
already know the result from running a query -- do not respond in plain \
text, always finish by calling submit_answer.
{memory_section}"""

# baseline the ablation compares against.
NO_MEMORY = ""


@tool
def submit_answer(sql: str) -> str:
    """Submit the final SQL query that answers the question. Call this once, as your last action."""
    return "answer submitted"


@dataclass
class WorkerResult:
    """What the worker decided, plus the full run for the trace log."""

    sql: str | None
    messages: list[BaseMessage]
    submitted: bool  # False if the agent stopped without calling submit_answer


def format_memory_section(lessons: list[str]) -> str:
    """Render lessons for the system prompt. Empty list means no memory yet."""
    if not lessons:
        return NO_MEMORY
    bulleted = "\n".join(f"- {lesson}" for lesson in lessons)
    return f"\nLessons from past mistakes:\n{bulleted}"


def build_worker(
    db_id: str,
    lessons: list[str] | None = None,
    extra_tools: list[BaseTool] | None = None,
    model: str = WORKER_MODEL,
):
    """Build a fresh worker agent for one question.

    Rebuilt per episode rather than reused, so a lesson or tool the teacher
    just added takes effect immediately with no state to carry over.
    """
    tools = [*make_sql_tools(db_id), *(extra_tools or []), submit_answer]

    llm = ChatAnthropic(model=model, temperature=0, max_tokens=4096)
    system_prompt = SYSTEM_PROMPT.format(
        memory_section=format_memory_section(lessons or [])
    )

    return create_agent(llm, tools, system_prompt=system_prompt)


def _extract_submission(messages: list[BaseMessage]) -> str | None:
    """Find the SQL from the last submit_answer call, if the agent made one."""
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls:
            if call["name"] == "submit_answer":
                return call["args"].get("sql")
    return None


def run_worker(
    db_id: str,
    question: str,
    lessons: list[str] | None = None,
    extra_tools: list[BaseTool] | None = None,
    model: str = WORKER_MODEL,
) -> WorkerResult:
    """Ask the worker to answer one question and return its final SQL."""
    agent = build_worker(db_id, lessons, extra_tools, model)

    # recursion_limit counts graph steps, not tool calls, so it needs roughly
    # 2x MAX_STEPS of headroom (one step per model call, one per tool call).
    try:
        result = agent.invoke(
            {"messages": [("user", question)]},
            config={"recursion_limit": MAX_STEPS * 2 + 2},
        )
        messages = result["messages"]
    except GraphRecursionError:
        # Worker used its whole tool-call budget without ever calling
        # submit_answer. invoke() gives back no partial state on this error,
        # so record what happened as a message instead of losing the trace
        # and crashing whatever batch of questions is running.
        messages = [
            HumanMessage(content=question),
            AIMessage(
                content=f"[gave up: used all {MAX_STEPS} tool calls without submitting an answer]"
            ),
        ]

    sql = _extract_submission(messages)
    return WorkerResult(sql=sql, messages=messages, submitted=sql is not None)


if __name__ == "__main__":
    result = run_worker(
        db_id="concert_singer",
        question="What are the names of all singers, ordered by age?",
    )
    print("submitted:", result.submitted)
    print("sql:", result.sql)
    print(f"\n{len(result.messages)} messages in the run")
