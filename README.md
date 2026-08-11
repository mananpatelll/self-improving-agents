# Self-improving agent loop

Can a self-improving agentic loop improve the performance of cheaper, smaller models without any fine-tuning?

The system has two agents. The **worker** answers user questions. The **teacher** monitors the worker's failures and writes lessons into the worker's memory so it stops repeating them. Worker is `claude-haiku-4-5`, teacher is `claude-sonnet-5`, and the teacher only runs on failures.

The domain is text-to-SQL. It was chosen because the Spider dataset gives real questions paired with correct queries, so correctness can be checked by running both and comparing results — no human or LLM needed to judge. The framework itself transfers to any other domain.

## Result

The worker was given 30 questions and had to write a SQL query for each. This ran three times: once to get a baseline, then an improvement loop over 60 different questions, then the same 30 questions again to measure the change.

| | Overall | Seen databases | Unseen databases |
| --- | --- | --- | --- |
| Baseline (no memory) | 20/30 — 66.7% | 9/15 | 11/15 |
| After improvement | **24/30 — 80.0%** | 11/15 | 13/15 |
| Change | **+13.3 points** | +2 | +2 |

The improvement loop scored 38/60, and its 22 failures produced **21 lessons** (one skipped due to a repeated model error).

**Unseen databases** are the ones the worker was never tested on during the improvement loop — the teacher never wrote a lesson about them. If the teacher had been memorising schemas, the seen half would have improved and the unseen half would have stayed flat. Both moved by the same amount, so what was learned is transferable SQL reasoning rather than memorisation.

All 21 lessons are in `memory.json`.

## How it works

- **Worker** (`src/agents/worker.py`) — LangGraph agent with three tools: list tables, describe tables, run a query. Doesn't get the schema up front; has to explore. Rebuilt per question so new lessons apply immediately.
- **Verifier** (`src/verifier.py`) — runs the predicted and gold SQL, compares result sets. Pure code, no model.
- **Teacher** (`src/agents/teacher.py`) — sees a failed question, the worker's full tool trace, and why it was wrong. Writes one generalizable lesson.
- **Memory** (`src/memory.py`) — flat list of lessons, injected into the worker's system prompt. Re-read before every question, so a lesson written at question 5 is available at question 6.

The system prompt is persona only. The teacher never rewrites it.

### The loop

To get the results, I ran three stages:

```text
1. Score      eval split,    memory off   -> baseline
2. Improve    improve split, memory on    -> teacher writes lessons on failures
3. Re-score   eval split,    memory on    -> same 30 questions, teacher off
```

## Evaluation method

**Correctness** is execution accuracy: run the predicted query and the gold query, compare the rows they return. Not a text comparison — two very different queries can both be right. Row order only matters when the gold query has `ORDER BY`; otherwise rows compare as multisets so dropped or duplicated rows are still caught.

**The data** is [Spider dev](https://yale-lily.github.io/spider), filtered to hard and extra-hard questions only using Spider's own difficulty classifier. Two reasons: to test whether the loop helps on genuinely hard questions, and because with an eval set of only 30, an unfiltered sample would have been mostly easy questions the worker already gets right — leaving no room to measure an improvement.

The improvement loop uses 60 questions across 10 databases. The eval set is 30 questions: 15 on those same 10 databases, and 15 on 9 databases the loop never touched. Splits are disjoint by construction and seeded, so every run draws the same questions and the teacher never sees an eval question.

## Running it

```bash
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env           # add your ANTHROPIC_API_KEY
```

Settings live in `config.yaml`. Edit it, then run `python -m src.main`. To reproduce the experiment, run three times:

| Stage | `split` | `use_memory` | `reset_memory` |
| --- | --- | --- | --- |
| 1. Baseline | `eval` | `false` | `true` |
| 2. Improve | `improve` | `true` | `false` |
| 3. Re-score | `eval` | `true` | `false` |

Set `limit: 3` for a cheap smoke test before spending on a full run.

The Spider databases are committed, so there is nothing to download.

## Limitations

**Gold queries were used deliberately, to keep the measurement clean.** To test whether the loop itself works, the failure signal has to be exact — otherwise "the loop didn't help" can't be told apart from "the judge was wrong". Comparing against a known-correct query removes that ambiguity entirely, and costs nothing.

Production doesn't have gold queries, but nothing in the loop depends on them except the verifier. The teacher reasons from the failure reason and the tool trace, so the same architecture runs on a different signal: an offline improvement pass like this one to build up initial memory, then live improvement from real user traffic using human-in-the-loop feedback or an LLM judge. That's a change to the verifier, not to the loop.

**Other gaps:**

- 30 eval questions is small. A 4-question swing is a real signal but a wide confidence interval; a paired test over more questions would be the honest next step.
- Execution accuracy has false positives. A wrong query can return the same rows as the gold query by coincidence — most often when both return zero rows — and gets marked correct. Rare, but it means the score is a slight over-estimate.
- Lessons accumulate in one flat list injected in full. Fine at 21; retrieval would be needed beyond a few dozen.
- One of 22 failures produced no lesson: the teacher's structured-output call kept returning malformed arguments. That is a model-side error, not a system one — it is caught, counted, and skipped rather than being allowed to stop the run.

**Cut for time.** I planned for about 4 hours and ended up at roughly 5.5–6 hours of screen time, so I dropped the part where the teacher writes new *tools*, not just lessons.

A lesson fixes missing *knowledge*, a tool fixes a missing *capability*. Every failure in this run was knowledge, since `run_query` executes arbitrary SQL and can reach almost anything in the database. A tool is only needed when the answer isn't in the database at all — *"find customers who bought products whose current market price is above $1,000"* can't be answered by any query, because the stored price is the price at purchase and the current one lives outside the system. That is the case a teacher-written tool would cover, and the worker already accepts an `extra_tools` argument, so the hook is there.
