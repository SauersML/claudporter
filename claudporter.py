#!/usr/bin/env python3
"""Extract the most recent Claude conversation JSONL into plain text.

Defaults:
- Source: newest ~/.claude/projects/**/*.jsonl
- Output: ~/.claude/most_recent_conversation_plain.txt
- Includes: user/assistant text, thinking, tool_use, tool_result, nested progress messages
- Filters local-command/meta noise lines
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

DEFAULT_GIBBERISH_TOKEN = (
    "gzF7KIC/Qb0HO/CBRIiMEYaqlbhSOFXbIBPopdqk9uEIsdHksuR24ph788w7ewZYDmotC0gblAkjsKH/tHnqiqPyAIfZYrr9EQGBpXxIKaNGNs+3/"
    "okBeJqnAFwDCQP9IXfxPgfbJw3TFnXyFt9AEmbVRfWhxfCuypBabsbfDL+C0sSHYy9jXxDSQfrNLo0BMOCy6QFbcWbQOhHfIhYl32PVtbhy5AXNbhvolRZZ"
    "G7u78RCseFN6xnjrBNT3OQGQ70xiICBf/HSjqXmaSWoL+yqDVY2Xf/2zQuF487xshvXT1btKI0YN3txr8aHdRRb6uObWglMD4N1oXoddI0H45xVqBGEqgka"
    "Dj6SyHvBNMkSsxBRf/vJhE5hfL8Agtoiia6Z/X6c9zsl6RIVODFnhEIZVQb2we762UNUys0aqjW4o6GHeEYci6v+rDBE/4UtNDl"
)

NOISE_RE = re.compile(
    r"^<local-command|^<command-name>|^<local-command-stdout>|^<local-command-caveat>|^\[Request interrupted by user\]$"
)
READ_FILE_LINE_RE = re.compile(r"^\s*\d+→")

RAW_JSON_EVENT_KEYS = {
    "parentUuid",
    "isSidechain",
    "userType",
    "cwd",
    "sessionId",
    "type",
    "timestamp",
}


def parse_args() -> argparse.Namespace:
    home = Path.home()
    default_claude_dir = home / ".claude"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--claude-dir",
        type=Path,
        default=default_claude_dir,
        help="Claude data directory (default: ~/.claude)",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Optional explicit source .jsonl file; if omitted, uses newest in projects/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_claude_dir / "most_recent_conversation_plain.txt",
        help="Output plain text file",
    )
    parser.add_argument(
        "--include-thinking",
        action="store_true",
        help="Include assistant thinking content (excluded by default)",
    )
    parser.add_argument(
        "--keep-noise",
        action="store_true",
        help="Keep local-command/meta noise lines",
    )
    parser.add_argument(
        "--remove-token",
        action="append",
        default=[],
        help="Extra exact token string(s) to remove from final text (can repeat)",
    )
    parser.add_argument(
        "--omit-read-file-content",
        action="store_true",
        help="Omit Read tool file-content dumps (TOOL_RESULT blocks with numbered arrow lines)",
    )
    return parser.parse_args()


def newest_jsonl(projects_dir: Path) -> Path:
    files = [p for p in projects_dir.rglob("*.jsonl") if p.is_file()]
    if not files:
        raise FileNotFoundError(f"No .jsonl files found under {projects_dir}")
    return max(files, key=lambda p: p.stat().st_mtime)


def role_label(role: str | None) -> str:
    r = (role or "UNKNOWN").lower()
    if r == "assistant":
        return "ASSISTANT"
    if r == "user":
        return "USER"
    return r.upper()


def trim_text(text: str) -> str:
    return text.strip()


def stringify(value) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def kv_lines(obj: dict | None) -> str:
    if not isinstance(obj, dict):
        return ""
    lines: list[str] = []
    for k, v in obj.items():
        lines.append(f"{k}: {stringify(v)}")
    return "\n".join(lines)


def looks_like_raw_json_event_blob(text: str) -> bool:
    s = text.strip()
    if not (s.startswith("{") and s.endswith("}")):
        return False
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return False
    if not isinstance(obj, dict):
        return False
    keys = set(obj.keys())
    # Skip obvious internal event/progress records dumped as JSON.
    return bool(keys & RAW_JSON_EVENT_KEYS) and ("type" in keys or "timestamp" in keys)


def strip_raw_json_event_lines(text: str) -> str:
    """Remove embedded line-level JSON event blobs from larger text blocks."""
    cleaned: list[str] = []
    for line in text.splitlines():
        if looks_like_raw_json_event_blob(line):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def looks_like_read_file_dump(text: str) -> bool:
    # Claude's Read tool file dumps are typically lines like "1→...".
    return sum(1 for line in text.splitlines() if READ_FILE_LINE_RE.match(line)) >= 2


def extract_items(
    message: dict,
    include_thinking: bool,
    tool_use_names: dict[str, str],
    omit_read_file_content: bool,
) -> Iterable[tuple[str, str]]:
    content = message.get("content")
    if isinstance(content, str):
        yield "", content
        return
    if not isinstance(content, list):
        return

    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype == "text":
            yield "", part.get("text", "")
        elif ptype == "thinking":
            if include_thinking:
                yield "THINKING", part.get("thinking", "")
        elif ptype == "tool_use":
            name = part.get("name", "")
            tool_use_id = part.get("id", "")
            if isinstance(tool_use_id, str) and tool_use_id and isinstance(name, str) and name:
                tool_use_names[tool_use_id] = name
            payload = kv_lines(part.get("input"))
            text = f"name: {name}" + (f"\n{payload}" if payload else "")
            yield "TOOL_USE", text
        elif ptype == "tool_result":
            tool_use_id = part.get("tool_use_id", "")
            header = f"tool_use_id: {tool_use_id}\n" if tool_use_id else ""
            body = stringify(part.get("content", ""))
            tool_name = tool_use_names.get(tool_use_id, "").lower() if isinstance(tool_use_id, str) else ""
            if omit_read_file_content and tool_name == "read" and looks_like_read_file_dump(body):
                continue
            yield "TOOL_RESULT", header + body
        else:
            # Preserve any other content types in a readable form.
            yield ptype.upper() if isinstance(ptype, str) else "CONTENT", stringify(part)


def row_to_context(row: dict) -> tuple[str, str, dict] | None:
    rtype = row.get("type")
    if rtype in ("user", "assistant"):
        msg = row.get("message")
        if isinstance(msg, dict):
            return row.get("timestamp", ""), rtype, msg
        return None

    # Nested agent progress messages can contain useful tool and text content.
    if rtype == "progress":
        data = row.get("data")
        if not isinstance(data, dict):
            return None
        inner = data.get("message")
        if not isinstance(inner, dict):
            return None
        msg = inner.get("message")
        if not isinstance(msg, dict):
            return None
        ts = inner.get("timestamp") or row.get("timestamp", "")
        role = msg.get("role") or inner.get("type") or "assistant"
        return ts, role, msg

    return None


def extract_plain_text(
    source: Path,
    include_thinking: bool,
    keep_noise: bool,
    omit_read_file_content: bool,
) -> str:
    out_chunks: list[str] = []
    tool_use_names: dict[str, str] = {}

    with source.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue

            ctx = row_to_context(row)
            if ctx is None:
                continue

            ts, role, msg = ctx
            who = role_label(role)
            for kind, text in extract_items(
                msg,
                include_thinking=include_thinking,
                tool_use_names=tool_use_names,
                omit_read_file_content=omit_read_file_content,
            ):
                text = trim_text(str(text))
                if not text:
                    continue
                if not keep_noise and NOISE_RE.search(text):
                    continue
                text = strip_raw_json_event_lines(text)
                if not text:
                    continue
                if looks_like_raw_json_event_blob(text):
                    continue
                suffix = f" {kind}" if kind else ""
                out_chunks.append(f"[{ts}] {who}{suffix}:\n{text}\n")

    return "\n".join(out_chunks).strip() + "\n"


def main() -> int:
    args = parse_args()

    claude_dir = args.claude_dir.expanduser().resolve()
    source = args.source.expanduser().resolve() if args.source else newest_jsonl(claude_dir / "projects")
    output = args.output.expanduser().resolve()

    text = extract_plain_text(
        source=source,
        include_thinking=args.include_thinking,
        keep_noise=args.keep_noise,
        omit_read_file_content=args.omit_read_file_content,
    )

    # Remove known garbage tokens.
    tokens = [DEFAULT_GIBBERISH_TOKEN, *args.remove_token]
    for token in tokens:
        if token:
            text = text.replace(token, "")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")

    print(f"source={source}")
    print(f"output={output}")
    print(f"chars={len(text)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
