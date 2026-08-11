"""Run the worker over a set of questions and score it.

One question at a time: worker writes SQL, the verifier checks it against
gold. On failure the trial is handed to the teacher, which writes a lesson
to memory for the worker to use on later questions.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import yaml
from langchain_core.tools import BaseTool

from src.agents.teacher import TEACHER_MODEL, teach
from src.agents.worker import WORKER_MODEL, WorkerResult, run_worker
from src.dataset import Example, IMPROVE_DBS, load_examples, make_splits
from src.logger import append_trial, log_path, write_run_summary
from src.memory import clear_memory, load_lessons
from src.sql_executor import ExecResult
from src.verifier import Verdict, verify

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

DEV_PATH = "spider_data/dev.json"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Read run settings from config.yaml."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


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
    config = load_config()

    split = config["split"]
    use_memory = config["use_memory"]
    seed = config["seed"]
    limit = config.get("limit")

    if config.get("reset_memory"):
        clear_memory()
        print("memory cleared")

    examples = load_examples(DEV_PATH)
    questions = make_splits(examples, seed=seed)[split]
    if limit is not None:
        questions = questions[:limit]

    path = log_path(split)

    # None means "read memory fresh before each question"; [] pins it empty.
    lessons = None if use_memory else []
    lessons_at_start = len(load_lessons()) if use_memory else 0

    print(f"running '{split}' split: {len(questions)} questions")
    print(f"memory: {'on' if use_memory else 'off'} ({lessons_at_start} lessons at start)")
    print(f"logging to {path}\n")

    started = time.monotonic()
    trials = run_split(
        questions,
        split_name=split,
        lessons=lessons,
        route_failures_to_teacher=(split == "improve"),
        log_file=path,
    )
    duration_seconds = round(time.monotonic() - started, 1)

    correct = sum(1 for t in trials if t.verdict.correct)
    seen = _subscore(trials, in_improve_dbs=True)
    unseen = _subscore(trials, in_improve_dbs=False)

    write_run_summary({
        "run_id": path.stem,
        "split": split,
        "use_memory": use_memory,
        "worker_model": WORKER_MODEL,
        "teacher_model": TEACHER_MODEL if split == "improve" else None,
        "split_seed": seed,
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

    print(f"\n{split}: {correct}/{len(trials)} correct ({score(trials):.1%})")
    if seen["total"] and unseen["total"]:
        print(f"  seen dbs:   {seen['correct']}/{seen['total']}")
        print(f"  unseen dbs: {unseen['correct']}/{unseen['total']}")
