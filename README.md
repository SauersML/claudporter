# claudporter

Extract the most recent Claude conversation JSONL into clean plain text.

## Run without cloning

```bash
curl -fsSL https://raw.githubusercontent.com/SauersML/claudporter/main/claudporter.py | python3 -
```

## What it does

- Finds newest `~/.claude/projects/**/*.jsonl`
- Exports readable plain text to `~/.claude/most_recent_conversation_plain.txt`
- Includes useful assistant/user/tool content
- Omits thinking blocks by default
- Removes noisy local-command/meta lines and raw JSON event blobs

## Optional flags

- `--include-thinking`
- `--keep-noise`
- `--source /path/to/session.jsonl`
- `--output /path/to/output.txt`
- `--remove-token "exact_string"` (repeatable)
