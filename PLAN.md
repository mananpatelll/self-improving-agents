# Plan

## Goal

Make a cheap model good at text-to-SQL by letting it learn from its own failures.

Sonnet 5 is already strong at this task, but expensive. Haiku 4.5 is cheap and mediocre. The bet: a self-improving loop closes enough of that gap that you don't have to pay for the big model at inference time.

Worker is `claude-haiku-4-5`. Teacher is `claude-sonnet-5`, and it only runs on failures — so the expensive model is a one-off cost, not a per-query one.

## Architecture

- **Worker** — takes a question, explores the schema with tools, writes SQL. Reads memory and skills before answering.
- **Verifier** — runs the SQL, compares the result set to the gold query's. Pure code, no LLM.
- **Teacher** — sees a failure and improves the system so that class of failure stops.

## How the loop runs

The system improves over time:

**1. Score.** 20-30 eval questions go to the worker one at a time. Count right vs wrong.

**2. Improve.** 60-80 separate questions, one at a time. Worker generates a query. Pass → next question. Fail → teacher looks at why and makes an improvement. Repeat through the set.

**3. Score again.** Same eval set, same worker, no teacher. Compare to step 1.

Steps 2 and 3 can repeat. The eval set is fixed so the numbers are comparable across rounds.

## What the teacher changes

**Memory** — a generalizable lesson.

```json
{
  "applies_when": "question asks for entities including ones with zero related rows",
  "lesson": "use LEFT JOIN, not INNER JOIN; COUNT the joined key, not *"
}
```

**Skills** — a new tool, when the failure was a missing capability rather than missing knowledge. Example: the worker filters `country = 'USA'` when the column stores `United States`. No advice fixes that — it needs a `sample_column_values` tool. The teacher writes the tool, the harness smoke-tests it against a real database, and registers it only if it runs clean.

The system prompt is just persona and stays fixed.

## Not memorizing

The lesson schema has no field for a question or a query — there's nowhere to write the answer down. Lessons are about SQL, not about a specific database or table. Anything containing verbatim question text or a full gold query is rejected before it's stored.

The eval catches it either way: the improvement set and the eval set are separate, and the eval set includes databases the teacher never touched.

## Data

Spider dev. 20 databases, small SQLite files, fast to run locally.

| Set | Size | Purpose |
|---|---|---|
| Eval | 20-30 q | Scored before and after. Teacher never sees it. |
| Improvement | 60-80 q | Teacher learns from failures here. |


The eval set spans both databases the improvement set covered and databases it didn't, so I can tell schema-specific gains from transferable ones.

## Traces

JSONL, one record per episode: question, tool calls, SQL attempts, execution result, verifier verdict. Teacher episodes also record the failure analysis and the exact memory or skill written.

## Build order

1. Spider loader, SQLite executor, result comparison
2. Trace logger
3. Worker with schema tools — baseline score
4. Memory + skill registry
5. Teacher
6. Run the loop
7. Re-score, write up
