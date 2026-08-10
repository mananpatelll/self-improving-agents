# Self-improving text-to-SQL agent

Making a cheap model good at text-to-SQL by having it learn from its own failures.

Sonnet 5 handles this task well but costs too much to run per query. Haiku 4.5 is cheap and mediocre. This system uses a self-improving loop to close the gap, so the expensive model is a one-time cost instead of a per-query one.

## Architecture

**Worker** (`claude-haiku-4-5`) — user-facing. Explores the schema with tools, writes SQL, reads memory and skills before answering.

**Verifier** — runs the SQL and compares the result set to the gold query's. Code, not an LLM.

**Teacher** (`claude-sonnet-5`) — sees a failure, works out why, and improves the system. Writes memory (generalizable lessons) or skills (new tools). Only runs on failures.

## How it improves

1. Score the worker on a fixed eval set
2. Run a separate improvement set — on each failure, the teacher makes a change
3. Score again on the same eval set

Repeatable. The eval set stays fixed so rounds are comparable.

## Status

Planning. See [PLAN.md](PLAN.md) for design and build order.
