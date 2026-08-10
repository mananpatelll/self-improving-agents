"""Run the worker over a set of questions and score it.

One question at a time: worker writes SQL, the verifier checks it against
gold. On failure the trial is handed to the teacher -- not built yet, so
that hook is a no-op for now, but the wiring is in place.

Works on any list of Example objects, so the same code scores the 30-question
eval set or drives the 60-question improvement loop. Change SPLIT below to
switch which one runs.
"""

from dataclasses import dataclass
from pathlib import Path

from langchain_core.tools import BaseTool

from src.agents.worker import WorkerResult, run_worker
from src.dataset import Example, load_examples, make_splits
from src.logger import append_trial, log_path
from src.sql_executor import ExecResult
from src.verifier import Verdict, verify

# Change this to switch what the script runs on: 30 fixed eval questions, or
# the 60 the improvement loop learns from.
SPLIT = "eval"  # "eval" | "improve"

DEV_PATH = "spider_data/dev.json"


@dataclass
class Trial:
    """One question, start to finish: what the worker produced and whether it was right."""

    example: Example
    worker: WorkerResult
    verdict: Verdict


def send_to_teacher(trial: Trial) -> None:
    """Hand a failed trial to the teacher for improvement."""
    print("  -> would send to teacher (not built yet)")


def run_trial(
    example: Example,
    lessons: list[str] | None = None,
    extra_tools: list[BaseTool] | None = None,
) -> Trial:
    """Run one question through the worker and verify the result."""
    worker_result = run_worker(
        db_id=example.db_id,
        question=example.question,
        lessons=lessons,
        extra_tools=extra_tools,
    )

    if not worker_result.submitted:
        verdict = Verdict(
            correct=False,
            reason="worker did not submit an answer",
            predicted=ExecResult(),
            gold=ExecResult(),
        )
    else:
        verdict = verify(example.db_id, worker_result.sql, example.gold_sql)

    return Trial(example=example, worker=worker_result, verdict=verdict)


def run_split(
    examples: list[Example],
    split_name: str = "",
    lessons: list[str] | None = None,
    extra_tools: list[BaseTool] | None = None,
    route_failures_to_teacher: bool = False,
    log_file: Path | None = None,
) -> list[Trial]:
    """Run the worker over every example, verify each, and score the split.

    Each trial is appended to log_file as it finishes, if one is given, so
    the log survives even if the run is interrupted partway through.
    """
    trials: list[Trial] = []

    for i, example in enumerate(examples, start=1):
        trial = run_trial(example, lessons, extra_tools)
        trials.append(trial)

        status = "PASS" if trial.verdict.correct else "FAIL"
        print(f"[{i}/{len(examples)}] {status}  {example.db_id:28s} {example.question[:50]}")
        if not trial.verdict.correct:
            print(f"         reason: {trial.verdict.reason}")

        if log_file is not None:
            append_trial(log_file, split_name, trial)

        if route_failures_to_teacher and not trial.verdict.correct:
            send_to_teacher(trial)

    return trials


def score(trials: list[Trial]) -> float:
    """Fraction of trials the worker got right."""
    if not trials:
        return 0.0
    return sum(1 for t in trials if t.verdict.correct) / len(trials)


if __name__ == "__main__":
    examples = load_examples(DEV_PATH)
    splits = make_splits(examples)

    questions = splits[SPLIT]
    path = log_path(SPLIT)
    print(f"running '{SPLIT}' split: {len(questions)} questions")
    print(f"logging to {path}\n")

    trials = run_split(
        questions,
        split_name=SPLIT,
        route_failures_to_teacher=(SPLIT == "improve"),
        log_file=path,
    )

    correct = sum(1 for t in trials if t.verdict.correct)
    print(f"\n{SPLIT}: {correct}/{len(trials)} correct ({score(trials):.1%})")
