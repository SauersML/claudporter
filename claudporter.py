#!/usr/bin/env python3
"""Extract a Claude Code conversation JSONL into readable plain text.

- Picks up the newest session under ~/.claude/projects/ by default.
- Distinguishes real user typing (USER) from programmatic runs (HEADLESS)
  and Task/Agent subagents (SUBAGENT).
- Inlines subagent conversations at the point the parent spawned them.
- Tries to inline headless child sessions when the parent spawned one via
  `claude -p "..."` through the Bash tool.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator

NOISE_CONTENT_PREFIXES = (
    "<local-command-stdout>",
    "<local-command-caveat>",
    "<task-notification>",
)
NOISE_LINE_RE = re.compile(r"^\[Request interrupted by user\]$")
READ_FILE_LINE_RE = re.compile(r"^\s*\d+→")

INTERACTIVE_ONLY_TOOLS = frozenset({
    "Agent", "Task", "TodoWrite",
    "TaskCreate", "TaskUpdate", "TaskList", "TaskGet", "TaskOutput", "TaskStop",
    "ScheduleWakeup", "EnterPlanMode", "ExitPlanMode",
    "EnterWorktree", "ExitWorktree",
    "Monitor", "AskUserQuestion", "ExitPlanMode",
})

RAW_EVENT_SENTINEL_KEYS = {"parentUuid", "isSidechain", "sessionId", "type", "timestamp"}

# Long gibberish token baked into early CC versions; strip if present.
DEFAULT_GIBBERISH_TOKEN = (
    "gzF7KIC/Qb0HO/CBRIiMEYaqlbhSOFXbIBPopdqk9uEIsdHksuR24ph788w7ewZYDmotC0gblAkjsKH/tHnqiqPyAIfZYrr9EQGBpXxIKaNGNs+3/"
    "okBeJqnAFwDCQP9IXfxPgfbJw3TFnXyFt9AEmbVRfWhxfCuypBabsbfDL+C0sSHYy9jXxDSQfrNLo0BMOCy6QFbcWbQOhHfIhYl32PVtbhy5AXNbhvolRZZ"
    "G7u78RCseFN6xnjrBNT3OQGQ70xiICBf/HSjqXmaSWoL+yqDVY2Xf/2zQuF487xshvXT1btKI0YN3txr8aHdRRb6uObWglMD4N1oXoddI0H45xVqBGEqgka"
    "Dj6SyHvBNMkSsxBRf/vJhE5hfL8Agtoiia6Z/X6c9zsl6RIVODFnhEIZVQb2we762UNUys0aqjW4o6GHeEYci6v+rDBE/4UtNDl"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", nargs="?", type=Path, default=None,
                   help="Session .jsonl (default: newest under ~/.claude/projects/)")
    p.add_argument("-o", "--output", type=Path,
                   default=Path.home() / ".claude" / "most_recent_conversation_plain.txt")
    p.add_argument("--thinking", action="store_true",
                   help="Include assistant thinking blocks")
    return p.parse_args()


# ---------- session discovery ----------

def newest_session(projects_dir: Path) -> Path:
    files = [f for f in projects_dir.rglob("*.jsonl")
             if f.is_file() and "/subagents/" not in str(f)]
    if not files:
        raise FileNotFoundError(f"No session jsonl files under {projects_dir}")
    return max(files, key=lambda p: p.stat().st_mtime)


# ---------- loading ----------

def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


# ---------- session analysis ----------

def session_is_headless(rows: list[dict]) -> bool:
    """Heuristic: headless/programmatic if it never modified files and never
    used interactive-only tools (Agent/TodoWrite/ScheduleWakeup/etc.)."""
    for r in rows:
        t = r.get("type")
        if t == "file-history-snapshot":
            return False
        if t == "assistant":
            for part in (r.get("message", {}).get("content") or []):
                if isinstance(part, dict) and part.get("type") == "tool_use":
                    if part.get("name") in INTERACTIVE_ONLY_TOOLS:
                        return False
    return True


def session_header(rows: list[dict], path: Path, headless: bool) -> str:
    meta = {}
    for r in rows:
        if r.get("type") in ("user", "assistant"):
            for k in ("sessionId", "cwd", "gitBranch", "version", "permissionMode"):
                if k not in meta and r.get(k):
                    meta[k] = r[k]
        if r.get("type") == "custom-title" and r.get("customTitle"):
            meta.setdefault("title", r["customTitle"])
        if r.get("type") == "agent-name" and r.get("agentName"):
            meta.setdefault("agentName", r["agentName"])

    ts = [r.get("timestamp") for r in rows if r.get("timestamp")]
    n_user = sum(1 for r in rows
                 if r.get("type") == "user" and _user_is_textual(r))
    n_asst = sum(1 for r in rows if r.get("type") == "assistant")

    lines = [
        "=" * 72,
        f"source: {path}",
        f"session: {meta.get('sessionId', '?')}",
    ]
    if "title" in meta:
        lines.append(f"title: {meta['title']}")
    if "agentName" in meta:
        lines.append(f"agentName: {meta['agentName']}")
    if "cwd" in meta:
        lines.append(f"cwd: {meta['cwd']}")
    if "gitBranch" in meta:
        lines.append(f"gitBranch: {meta['gitBranch']}")
    if "version" in meta:
        lines.append(f"version: {meta['version']}")
    if ts:
        lines.append(f"time: {ts[0]}  →  {ts[-1]}")
    lines.append(f"messages: user={n_user} assistant={n_asst}")
    lines.append(f"classification: {'HEADLESS (programmatic)' if headless else 'INTERACTIVE'}")
    lines.append("=" * 72)
    return "\n".join(lines) + "\n"


def _user_is_textual(row: dict) -> bool:
    msg = row.get("message", {})
    c = msg.get("content")
    if isinstance(c, list):
        if any(isinstance(p, dict) and p.get("type") == "tool_result" for p in c):
            return False
    if row.get("isMeta"):
        return False
    return True


# ---------- subagent linking ----------

def find_subagents(parent_path: Path) -> dict[str, Path]:
    """Return {description: jsonl_path} for subagents of this parent session."""
    sid = parent_path.stem
    sub_dir = parent_path.parent / sid / "subagents"
    if not sub_dir.is_dir():
        return {}
    out: dict[str, Path] = {}
    for meta in sub_dir.glob("*.meta.json"):
        try:
            m = json.loads(meta.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        desc = m.get("description")
        jsonl = meta.with_suffix("").with_suffix(".jsonl")
        # with_suffix twice strips .json then adds .jsonl — above is wrong if stem ends .meta
        jsonl = sub_dir / (meta.stem.removesuffix(".meta") + ".jsonl")
        if desc and jsonl.exists():
            out.setdefault(desc, jsonl)
    return out


# ---------- headless child linking ----------

CLAUDE_P_RE = re.compile(r"""claude(?:\s+[\w\-]+)*\s+-p\s+(?P<q>["'])(?P<prompt>.*?)(?P=q)""", re.DOTALL)


def extract_claude_p_prompt(bash_cmd: str) -> str | None:
    m = CLAUDE_P_RE.search(bash_cmd or "")
    return m.group("prompt") if m else None


def index_sessions_by_first_prompt(projects_dir: Path) -> dict[tuple[str, str], Path]:
    """Map (cwd, first_user_text_prefix) -> session path, for resolving headless children."""
    idx: dict[tuple[str, str], Path] = {}
    for fp in projects_dir.rglob("*.jsonl"):
        if "/subagents/" in str(fp):
            continue
        try:
            with fp.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row.get("type") == "user" and _user_is_textual(row):
                        msg = row.get("message", {})
                        c = msg.get("content")
                        txt = c if isinstance(c, str) else (
                            c[0].get("text", "") if isinstance(c, list) and c
                            and isinstance(c[0], dict) else "")
                        key = (row.get("cwd", ""), (txt or "").strip()[:200])
                        idx.setdefault(key, fp)
                        break
        except OSError:
            continue
    return idx


# ---------- content rendering ----------

def _stringify(v) -> str:
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def _kv_block(d: dict | None) -> str:
    if not isinstance(d, dict):
        return ""
    return "\n".join(f"{k}: {_stringify(v)}" for k, v in d.items())


def _looks_like_raw_event_blob(text: str) -> bool:
    s = text.strip()
    if not (s.startswith("{") and s.endswith("}")):
        return False
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return False
    if not isinstance(obj, dict):
        return False
    return bool(set(obj) & RAW_EVENT_SENTINEL_KEYS)


def _strip_raw_event_lines(text: str) -> str:
    return "\n".join(l for l in text.splitlines()
                     if not _looks_like_raw_event_blob(l)).strip()


def _looks_like_read_dump(text: str) -> bool:
    return sum(1 for l in text.splitlines() if READ_FILE_LINE_RE.match(l)) >= 2


def _role_label(row: dict, headless_session: bool) -> str:
    t = row.get("type")
    if t == "assistant":
        return "ASSISTANT"
    if t != "user":
        return (t or "?").upper()
    # user row — distinguish:
    if row.get("isSidechain"):
        return "SUBAGENT"
    origin = row.get("origin")
    if isinstance(origin, dict) and origin.get("kind") == "task-notification":
        return "TASK_NOTIF"
    msg = row.get("message", {})
    c = msg.get("content")
    txt = c if isinstance(c, str) else (
        c[0].get("text", "") if isinstance(c, list) and c
        and isinstance(c[0], dict) else "")
    s = (txt or "").lstrip()
    if s.startswith("<task-notification>"):
        return "TASK_NOTIF"
    if s.startswith("<local-command-stdout>"):
        return "CMD_OUTPUT"
    if s.startswith("<command-name>") or s.startswith("<command-message>"):
        return "SLASH_CMD"
    if headless_session:
        return "HEADLESS"
    return "USER"


# ---------- main rendering loop ----------

def render_message_parts(
    message: dict,
    tool_names_by_id: dict[str, str],
    include_thinking: bool,
    omit_read_file_content: bool,
) -> Iterator[tuple[str, str, dict | None]]:
    """Yield (kind, text, raw_part). kind is '' for plain text/tool_use/etc."""
    content = message.get("content")
    if isinstance(content, str):
        yield "", content, None
        return
    if not isinstance(content, list):
        return
    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype == "text":
            yield "", part.get("text", ""), part
        elif ptype == "thinking":
            if include_thinking:
                yield "THINKING", part.get("thinking", ""), part
        elif ptype == "tool_use":
            name = part.get("name", "")
            tid = part.get("id", "")
            if isinstance(tid, str) and tid and name:
                tool_names_by_id[tid] = name
            body = f"name: {name}"
            inp = _kv_block(part.get("input"))
            if inp:
                body += f"\n{inp}"
            yield "TOOL_USE", body, part
        elif ptype == "tool_result":
            tid = part.get("tool_use_id", "")
            header = f"tool_use_id: {tid}\n" if tid else ""
            body = _stringify(part.get("content", ""))
            tool = tool_names_by_id.get(tid, "").lower() if isinstance(tid, str) else ""
            if omit_read_file_content and tool == "read" and _looks_like_read_dump(body):
                continue
            yield "TOOL_RESULT", header + body, part
        else:
            yield (ptype.upper() if isinstance(ptype, str) else "CONTENT"), _stringify(part), part


def row_as_message(row: dict) -> tuple[str, dict] | None:
    t = row.get("type")
    if t in ("user", "assistant"):
        msg = row.get("message")
        if isinstance(msg, dict):
            return row.get("timestamp", ""), msg
    if t == "progress":  # nested agent progress
        data = row.get("data")
        if isinstance(data, dict):
            inner = data.get("message")
            if isinstance(inner, dict):
                m = inner.get("message")
                if isinstance(m, dict):
                    return inner.get("timestamp") or row.get("timestamp", ""), m
    return None


def render_session(
    rows: list[dict],
    *,
    source_path: Path,
    include_thinking: bool,
    keep_noise: bool,
    omit_read_file_content: bool,
    inline_subagents: bool,
    inline_headless: bool,
    subagents_by_desc: dict[str, Path],
    first_prompt_index: dict[tuple[str, str], Path],
    visited: set[Path],
    depth: int,
    child_kind: str = "",
) -> str:
    headless = session_is_headless(rows)
    # Subagent sessions typically have no file-history-snapshot either; trust child_kind.
    if child_kind == "subagent":
        headless = False
    parts: list[str] = []
    if depth == 0:
        parts.append(session_header(rows, source_path, headless))
    else:
        indent_hdr = "  " * depth
        tag = child_kind.upper().replace("_", " ") if child_kind else (
            "HEADLESS CHILD" if headless else "NESTED")
        parts.append(f"{indent_hdr}┌── {tag} session: {source_path.name}\n")

    indent = "  " * depth
    tool_names: dict[str, str] = {}

    for row in rows:
        ctx = row_as_message(row)
        if ctx is None:
            continue
        ts, msg = ctx
        label = _role_label(row, headless)

        for kind, text, part in render_message_parts(
                msg, tool_names, include_thinking, omit_read_file_content):
            body = (text or "").strip()
            if not body:
                continue
            if not keep_noise:
                if NOISE_LINE_RE.match(body):
                    continue
                if label in ("TASK_NOTIF", "CMD_OUTPUT") and not keep_noise:
                    continue
                if any(body.lstrip().startswith(pfx) for pfx in NOISE_CONTENT_PREFIXES):
                    continue
            body = _strip_raw_event_lines(body)
            if not body or _looks_like_raw_event_blob(body):
                continue

            suffix = f" {kind}" if kind else ""
            indented = "\n".join(indent + l for l in body.splitlines())
            parts.append(f"{indent}[{ts}] {label}{suffix}:\n{indented}\n")

            # After a tool_use, optionally inline matched child session.
            if kind == "TOOL_USE" and isinstance(part, dict):
                child_rows, child_path, child_kind_tag = _resolve_child(
                    part, source_path, subagents_by_desc,
                    first_prompt_index,
                    inline_subagents, inline_headless,
                    visited,
                )
                if child_rows:
                    parts.append(render_session(
                        child_rows,
                        source_path=child_path,
                        include_thinking=include_thinking,
                        keep_noise=keep_noise,
                        omit_read_file_content=omit_read_file_content,
                        inline_subagents=inline_subagents,
                        inline_headless=inline_headless,
                        subagents_by_desc=find_subagents(child_path),
                        first_prompt_index=first_prompt_index,
                        visited=visited,
                        depth=depth + 1,
                        child_kind=child_kind_tag,
                    ))
                    parts.append(f"{indent}└── end {child_path.name}\n")

    return "\n".join(parts)


def _resolve_child(
    tool_use_part: dict,
    parent_path: Path,
    subagents_by_desc: dict[str, Path],
    first_prompt_index: dict[tuple[str, str], Path],
    inline_subagents: bool,
    inline_headless: bool,
    visited: set[Path],
) -> tuple[list[dict] | None, Path | None, str]:
    name = tool_use_part.get("name", "")
    inp = tool_use_part.get("input") or {}

    if inline_subagents and name in ("Agent", "Task"):
        desc = inp.get("description") or inp.get("subagent_type") or ""
        fp = subagents_by_desc.get(desc)
        if fp and fp not in visited:
            visited.add(fp)
            return load_jsonl(fp), fp, "subagent"

    if inline_headless and name == "Bash":
        prompt = extract_claude_p_prompt(inp.get("command", ""))
        if prompt:
            cwd = inp.get("cwd") or _infer_cwd(parent_path) or ""
            key = (cwd, prompt.strip()[:200])
            fp = first_prompt_index.get(key)
            if not fp:
                for (_, p), path in first_prompt_index.items():
                    if p == key[1]:
                        fp = path
                        break
            if fp and fp != parent_path and fp not in visited:
                visited.add(fp)
                return load_jsonl(fp), fp, "headless_child"

    return None, None, ""


def _infer_cwd(session_path: Path) -> str | None:
    try:
        with session_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("cwd"):
                    return row["cwd"]
    except OSError:
        pass
    return None


# ---------- entry ----------

def main() -> int:
    args = parse_args()
    projects_dir = Path.home() / ".claude" / "projects"
    source = (args.source.expanduser().resolve()
              if args.source else newest_session(projects_dir))

    rows = load_jsonl(source)
    text = render_session(
        rows,
        source_path=source,
        include_thinking=args.thinking,
        keep_noise=False,
        omit_read_file_content=True,
        inline_subagents=True,
        inline_headless=True,
        subagents_by_desc=find_subagents(source),
        first_prompt_index=index_sessions_by_first_prompt(projects_dir),
        visited={source},
        depth=0,
    )

    text = text.replace(DEFAULT_GIBBERISH_TOKEN, "")
    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")

    print(f"source={source}")
    print(f"output={out}")
    print(f"chars={len(text)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
