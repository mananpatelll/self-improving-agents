"""Teacher agent: turns one failed trial into a lesson.
"""

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field
from typing import TYPE_CHECKING

from src.logger import summarize_steps
from src.memory import add_lesson

if TYPE_CHECKING:
    from src.main import Trial

load_dotenv()
TEACHER_MODEL = "claude-sonnet-5"

TEACHER_PROMPT = """\
A worker agent tried to answer a question by writing SQL, and got it wrong. \
Your job is to write ONE lesson that would help the worker avoid this exact \
class of mistake on ANY database, not just this one.

Question: {question}

What the worker tried:
{trace}

Worker's final query: {predicted_sql}
Correct query: {gold_sql}
Why it was wrong: {reason}

Write a lesson that is a general SQL principle or technique, not a note \
about this specific database. Do not mention any table, column, or \
database name from this question. Do not restate this question or this \
query -- the lesson must be useful for a completely different question on \
a completely different database.
"""


class LessonDraft(BaseModel):
    applies_when: str = Field(
        description=(
            "One short clause, under 20 words, describing when this lesson "
            "applies. No specific database, table, or column names. "
            "Example: 'a question asks for entities including ones with "
            "zero related rows'."
        )
    )
    rule: str = Field(
        description=(
            "One short clause, under 25 words, giving the general SQL rule "
            "or technique to apply, phrased so it works on any database. "
            "Must read naturally after 'When <applies_when>, ...' -- so "
            "start with a lowercase verb, not a full sentence. Example: "
            "'use LEFT JOIN instead of INNER JOIN, and COUNT the joined "
            "key rather than *'."
        )
    )


def _format_step(step: dict) -> str:
    if "tool" in step:
        line = f"- called {step['tool']}({step['args']})"
        if "result" in step:
            result = step["result"]
            if len(result) > 200:
                result = result[:200] + "..."
            line += f" -> {result}"
        return line
    return f"- note: {step.get('note', '')}"


def _is_generalizable(lesson: str, db_id: str, gold_sql: str) -> bool:
    """Reject a lesson that leaks specifics of this one episode.
    """
    lowered = lesson.lower()
    if db_id.lower() in lowered:
        return False
    if gold_sql.strip().lower() in lowered:
        return False
    return True


def teach(trial: "Trial") -> str | None:
    """Look at one failed trial and write a lesson to memory.
    """
    if trial.verdict.reason.startswith("gold query failed"):
        # Dataset artifact, not a worker mistake -- nothing to learn.
        return None

    steps = summarize_steps(trial.worker.messages)
    trace = "\n".join(_format_step(step) for step in steps) or "(worker took no tool actions)"

    prompt = TEACHER_PROMPT.format(
        question=trial.example.question,
        trace=trace,
        predicted_sql=trial.worker.sql or "(no query submitted)",
        gold_sql=trial.example.gold_sql,
        reason=trial.verdict.reason,
    )

    llm = ChatAnthropic(model=TEACHER_MODEL, max_tokens=1024)
    draft: LessonDraft = llm.with_structured_output(LessonDraft).invoke(prompt)

    _strip_chars = ".\"'"
    applies_when = draft.applies_when.strip().strip(_strip_chars)
    rule = draft.rule.strip().strip(_strip_chars)
    rule = rule[0].lower() + rule[1:] if rule else rule  
    lesson = f"When {applies_when}, {rule}."

    if not _is_generalizable(lesson, trial.example.db_id, trial.example.gold_sql):
        print(f"  -> teacher's lesson leaked specifics, discarding: {lesson}")
        return None

    add_lesson(lesson)
    print(f"  -> new lesson: {lesson}")
    return lesson
