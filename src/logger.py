"""Write and read trial logs.

One JSON line per trial: the question, what the worker tried, and whether
it was right. Written as each trial finishes rather than buffered, so an
interrupted run still leaves a usable log of everything that ran before it
stopped.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

if TYPE_CHECKING:
    # Only needed for the type hint below -- importing this for real would
    # create a circular import, since main.py imports this module too.
    from src.main import Trial

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


RUNS_LOG = LOG_DIR / "runs.jsonl"


def log_path(split: str) -> Path:
    """Build a log file path for one run, named so runs never collide."""
    LOG_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return LOG_DIR / f"{split}_{stamp}.jsonl"


def _flatten(content: str | list) -> str:
    """Turn LangChain message content into plain text.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return " ".join(part for part in parts if part).strip()
    return str(content)


def summarize_steps(messages: list[BaseMessage]) -> list[dict]:
    """Turn the worker's raw message list into a flat, readable step list.
    """
    results_by_call_id = {
        message.tool_call_id: _flatten(message.content)
        for message in messages
        if isinstance(message, ToolMessage)
    }

    steps = []
    for message in messages:
        if not isinstance(message, AIMessage):
            continue

        # Any prose the model wrote before its tool calls this turn.
        note = _flatten(message.content)

        if not message.tool_calls:
            if note:
                steps.append({"note": note})
            continue

        for call in message.tool_calls:
            step = {"tool": call["name"], "args": call["args"]}
            if call["id"] in results_by_call_id:
                step["result"] = results_by_call_id[call["id"]]
            if note:
                step["note"] = note
                note = ""  # only attach the preamble to the first call
            steps.append(step)

    return steps


def append_trial(path: Path, split: str, trial: "Trial") -> None:
    """Append one trial to the log as a single JSON line."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "qid": trial.example.qid,
        "db_id": trial.example.db_id,
        "question": trial.example.question,
        "gold_sql": trial.example.gold_sql,
        "predicted_sql": trial.worker.sql,
        "submitted": trial.worker.submitted,
        "correct": trial.verdict.correct,
        "reason": trial.verdict.reason,
        "steps": summarize_steps(trial.worker.messages),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_log(path: Path) -> list[dict]:
    """Read every record from a log file, in the order they were written."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_run_summary(summary: dict, path: Path = RUNS_LOG) -> None:
    """Append one run's settings and aggregate metrics as a single JSON line."""
    path.parent.mkdir(exist_ok=True)
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), **summary}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_runs(path: Path = RUNS_LOG) -> list[dict]:
    """Read every run summary, in the order they were written."""
    if not path.exists():
        return []
    return read_log(path)


def print_runs(path: Path = RUNS_LOG) -> None:
    """Print a compact table of every run, for comparing settings and scores."""
    runs = read_runs(path)
    if not runs:
        print("no runs logged yet")
        return

    header = f"{'run_id':30s} {'split':9s} {'memory':7s} {'score':8s} {'seen':8s} {'unseen':8s}"
    print(header)
    print("-" * len(header))
    for r in runs:
        seen, unseen = r.get("seen", {}), r.get("unseen", {})
        seen_s = f"{seen['correct']}/{seen['total']}" if seen.get("total") else "-"
        unseen_s = f"{unseen['correct']}/{unseen['total']}" if unseen.get("total") else "-"
        score_s = f"{r['num_correct']}/{r['num_questions']}"
        print(
            f"{r['run_id']:30s} {r['split']:9s} {str(r['use_memory']):7s} "
            f"{score_s:8s} {seen_s:8s} {unseen_s:8s}"
        )


def _truncate(text: str, limit: int = 100) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


def print_log(path: Path) -> None:
    """Print a log file as a readable trace: one block per trial."""
    records = read_log(path)
    correct = sum(1 for r in records if r["correct"])

    for record in records:
        status = "PASS" if record["correct"] else "FAIL"
        print(f"{status}  {record['db_id']:24s} {record['question']}")

        for step in record["steps"]:
            if "tool" in step:
                line = f"    {step['tool']}({step['args']})"
                if "result" in step:
                    line += f" -> {_truncate(step['result'])}"
                print(line)
                if step.get("note"):
                    print(f"      # {step['note']}")
            elif step.get("note"):
                print(f"    # {step['note']}")

        print(f"    predicted: {record['predicted_sql']}")
        print(f"    gold:      {record['gold_sql']}")
        if not record["correct"]:
            print(f"    reason:    {record['reason']}")
        print()

    print(f"{correct}/{len(records)} correct")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m src.logger <path/to/log.jsonl>")
        print("       python -m src.logger runs")
    elif sys.argv[1] == "runs":
        print_runs()
    else:
        print_log(Path(sys.argv[1]))
