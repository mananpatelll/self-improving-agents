"""Run the worker over a set of questions and score it.

One question at a time: worker writes SQL, the verifier checks it against
gold. On failure the trial is handed to the teacher, which writes a lesson
to memory for the worker to use on later questions.

Works on any list of Example objects, so the same code scores the 30-question
eval set or drives the 60-question improvement loop. Change SPLIT below to
switch which one runs.
"""

import time
from dataclasses import dataclass
from pathlib import Path

from langchain_core.tools import BaseTool

from src.agents.teacher import TEACHER_MODEL, teach
from src.agents.worker import WORKER_MODEL, WorkerResult, run_worker
from src.dataset import Example, IMPROVE_DBS, load_examples, make_splits
from src.logger import append_trial, log_path, write_run_summary
from src.memory import load_lessons
from src.sql_executor import ExecResult
from src.verifier import Verdict, verify

# Change this to switch what the script runs on: 30 fixed eval questions, or
# the 60 the improvement loop learns from.
SPLIT = "improve"  # "eval" | "improve"

USE_MEMORY = True

# Passed to dataset.make_splits so every run draws the same questions --
# needed to compare runs against each other.
SPLIT_SEED = 0

DEV_PATH = "spider_data/dev.json"


@dataclass
class Trial:
    """One question, start to finish: what the worker produced and whether it was right."""

    example: Example
    worker: WorkerResult
    verdict: Verdict


def send_to_teacher(trial: Trial) -> None:
    """Hand a failed trial to the teacher for improvement."""
    teach(trial)


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
        current_lessons = load_lessons() if lessons is None else lessons

        trial = run_trial(example, current_lessons, extra_tools)
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


def _subscore(trials: list[Trial], in_improve_dbs: bool) -> dict:

    subset = [t for t in trials if (t.example.db_id in IMPROVE_DBS) == in_improve_dbs]
    correct = sum(1 for t in subset if t.verdict.correct)
    return {"correct": correct, "total": len(subset)}


if __name__ == "__main__":
    examples = load_examples(DEV_PATH)
    splits = make_splits(examples, seed=SPLIT_SEED)

    questions = splits[SPLIT]
    path = log_path(SPLIT)

    # None means "read memory fresh before each question"; [] pins it empty.
    lessons = None if USE_MEMORY else []
    lessons_at_start = len(load_lessons()) if USE_MEMORY else 0

    print(f"running '{SPLIT}' split: {len(questions)} questions")
    print(f"memory: {'on' if USE_MEMORY else 'off'} ({lessons_at_start} lessons at start)")
    print(f"logging to {path}\n")

    started = time.monotonic()
    trials = run_split(
        questions,
        split_name=SPLIT,
        lessons=lessons,
        route_failures_to_teacher=(SPLIT == "improve"),
        log_file=path,
    )
    duration_seconds = round(time.monotonic() - started, 1)

    correct = sum(1 for t in trials if t.verdict.correct)
    seen = _subscore(trials, in_improve_dbs=True)
    unseen = _subscore(trials, in_improve_dbs=False)

    write_run_summary({
        "run_id": path.stem,
        "split": SPLIT,
        "use_memory": USE_MEMORY,
        "worker_model": WORKER_MODEL,
        "teacher_model": TEACHER_MODEL if SPLIT == "improve" else None,
        "split_seed": SPLIT_SEED,
        "num_questions": len(trials),
        "num_correct": correct,
        "accuracy": score(trials),
        "num_no_submission": sum(1 for t in trials if not t.worker.submitted),
        "lessons_at_start": lessons_at_start,
        "lessons_at_end": len(load_lessons()),
        "duration_seconds": duration_seconds,
        "trace_log": str(path),
        "seen": seen,
        "unseen": unseen,
    })

    print(f"\n{SPLIT}: {correct}/{len(trials)} correct ({score(trials):.1%})")
    if seen["total"] and unseen["total"]:
        print(f"  seen dbs:   {seen['correct']}/{seen['total']}")
        print(f"  unseen dbs: {unseen['correct']}/{unseen['total']}")
