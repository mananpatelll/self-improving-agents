"""Load Spider dev examples and build deterministic train/eval splits."""

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# ~100MB database, not worth keeping it for this project
EXCLUDED_DBS = frozenset({"wta_1"})

# The improvement loop only draws questions from these 10 databases.
# The remaining 9 databases are held back and used only at eval time, so we can
# see whether the teacher's changes transfer to schemas it never worked on.
IMPROVE_DBS = (
    "world_1", "car_1", "cre_Doc_Template_Mgt", "dog_kennels", "flight_2",
    "student_transcripts_tracking", "tvshow", "network_1", "concert_singer", "pets_1",
)


@dataclass(frozen=True)
class Example:
    """One Spider question paired with its correct SQL."""

    qid: str
    db_id: str
    question: str
    gold_sql: str


def load_examples(dev_path: str | Path) -> list[Example]:
    """Read dev.json and return every question except the excluded databases."""
    raw = json.loads(Path(dev_path).read_text(encoding="utf-8"))

    return [
        Example(
            # Position in dev.json. Stable across runs, so traces stay comparable.
            qid=f"dev_{i}",
            db_id=item["db_id"],
            question=item["question"],
            gold_sql=item["query"],
        )
        for i, item in enumerate(raw)
        if item["db_id"] not in EXCLUDED_DBS
    ]


def _sample_round_robin(
    pool: list[Example],
    n: int,
    rng: random.Random,
) -> list[Example]:
    """Pick n examples, taking one database at a time so no database dominates.

    Raises ValueError if the pool has fewer than n examples.
    """
    if len(pool) < n:
        raise ValueError(f"pool has {len(pool)} examples, need {n}")

    # Group by database, then shuffle inside each group so the picks are random.
    by_db: dict[str, list[Example]] = defaultdict(list)
    for example in pool:
        by_db[example.db_id].append(example)
    for group in by_db.values():
        rng.shuffle(group)

    # Cycle through the databases, taking one question from each per pass.
    picked: list[Example] = []
    databases = sorted(by_db)
    while len(picked) < n:
        for db_id in databases:
            if by_db[db_id] and len(picked) < n:
                picked.append(by_db[db_id].pop())

    return picked


def make_splits(
    examples: list[Example],
    n_improve: int = 60,
    n_eval_seen: int = 15,
    n_eval_unseen: int = 15,
    seed: int = 0,
) -> dict[str, list[Example]]:
    """Split examples into an improvement set and an eval set.

    The eval set has two halves:
      eval_seen   - databases the improvement loop used, but different questions
      eval_unseen - databases the teacher never touched at all

    Comparing the two tells us whether gains are database-specific or general.
    Same seed always gives the same split.
    """
    rng = random.Random(seed)

    seen_pool = [e for e in examples if e.db_id in IMPROVE_DBS]
    unseen_pool = [e for e in examples if e.db_id not in IMPROVE_DBS]

    improve = _sample_round_robin(seen_pool, n_improve, rng)

    # Eval questions must not be ones the teacher already learned from.
    used_qids = {e.qid for e in improve}
    remaining_seen = [e for e in seen_pool if e.qid not in used_qids]

    eval_seen = _sample_round_robin(remaining_seen, n_eval_seen, rng)
    eval_unseen = _sample_round_robin(unseen_pool, n_eval_unseen, rng)

    return {
        "improve": improve,
        "eval_seen": eval_seen,
        "eval_unseen": eval_unseen,
        "eval": eval_seen + eval_unseen,
    }


if __name__ == "__main__":
    examples = load_examples("spider_data/dev.json")
    print(f"loaded {len(examples)} questions")

    for name, split in make_splits(examples).items():
        n_databases = len({e.db_id for e in split})
        print(f"{name:12s} {len(split):3d} questions across {n_databases} databases")
