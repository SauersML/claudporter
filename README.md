# claudporter

Extract a Claude Code session JSONL into readable plain text, with real-user vs.
headless/subagent labeling and inlined child transcripts.

## Run without cloning

```bash
curl -fsSL https://raw.githubusercontent.com/SauersML/claudporter/main/claudporter.py | python3 -
```

## What it does

- Picks the newest session under `~/.claude/projects/` (or pass one explicitly).
- Labels each message by source:
  - `USER` — you typed it in an interactive session.
  - `HEADLESS` — programmatic run (Agent SDK, benchmark, `claude -p …`).
    Detected when the session never modified a file
    (`file-history-snapshot`) and never used an interactive-only tool
    (`Agent`, `TodoWrite`, `TaskCreate`, `ScheduleWakeup`, `EnterPlanMode`, …).
  - `SUBAGENT` — `isSidechain: true`, from `<session>/subagents/agent-*.jsonl`.
  - `TASK_NOTIF` / `CMD_OUTPUT` / `SLASH_CMD` — harness-generated, hidden.
- Inlines child transcripts under the tool_use that spawned them:
  - `Agent` / `Task` subagents are matched via `description` ↔ `meta.json`.
  - `Bash` calls running `claude -p "<prompt>"` are matched to the headless
    child whose first user prompt is that string (best-effort).
- Strips thinking, Read-tool file dumps, and raw JSON event blobs.

## Usage

```
claudporter.py [SOURCE] [-o OUTPUT] [--thinking]
```

- `SOURCE` — optional `.jsonl`; defaults to newest session
- `-o / --output` — defaults to `~/.claude/most_recent_conversation_plain.txt`
- `--thinking` — include assistant thinking blocks

Example:

```bash
python3 claudporter.py
```
