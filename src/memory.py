"""Store and load lessons the teacher has written."""

import json
from pathlib import Path

MEMORY_PATH = Path(__file__).resolve().parent.parent / "memory.json"


def load_lessons(path: Path = MEMORY_PATH) -> list[str]:
    """Read all lessons currently in memory. Empty list if none exist yet."""
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def add_lesson(lesson: str, path: Path = MEMORY_PATH) -> None:
    """Append one lesson to memory."""
    lessons = load_lessons(path)
    lessons.append(lesson)
    path.write_text(json.dumps(lessons, indent=2, ensure_ascii=False), encoding="utf-8")


def clear_memory(path: Path = MEMORY_PATH) -> None:
    """Delete all lessons. Used to reset before a fresh or baseline run."""
    if path.exists():
        path.unlink()


if __name__ == "__main__":
    clear_memory()
    add_lesson(
        "when a question asks for entities including ones with zero related "
        "rows, use LEFT JOIN, not INNER JOIN"
    )
    add_lesson(
        "COUNT(*) counts NULLs, COUNT(column) does not -- match whichever "
        "the question actually asks for"
    )
    print(load_lessons())
    clear_memory()
