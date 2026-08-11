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
    lessons = load_lessons()
    print(f"{len(lessons)} lessons in {MEMORY_PATH.name}\n")
    for i, lesson in enumerate(lessons, start=1):
        print(f"{i:2d}. {lesson}")
