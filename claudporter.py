#!/usr/bin/env python3
"""Extract Claude Code session JSONLs into readable plain text.

Default (no args): port every session under ~/.claude/projects/ into
~/.claude/ported/ with two folders — `interactive/` (sessions that had a real
human user) and `headless/` (programmatic runs that couldn't be linked to a
parent). Subagents and `claude -p` children are inlined at their spawn point
inside the parent's file and not re-emitted in `headless/`.

Pass a single SOURCE .jsonl to render just that one session to an output file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timedelta, timezone
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Iterator

NOISE_CONTENT_PREFIXES = (
    "<local-command-stdout>",
    "<local-command-caveat>",
    "<task-notification>",
)
NOISE_LINE_RE = re.compile(r"^\[Request interrupted by user\]$")
READ_FILE_LINE_RE = re.compile(r"^\s*\d+→")

RAW_EVENT_SENTINEL_KEYS = {"parentUuid", "isSidechain", "sessionId", "type", "timestamp"}

DEFAULT_GIBBERISH_TOKEN = (
    "gzF7KIC/Qb0HO/CBRIiMEYaqlbhSOFXbIBPopdqk9uEIsdHksuR24ph788w7ewZYDmotC0gblAkjsKH/tHnqiqPyAIfZYrr9EQGBpXxIKaNGNs+3/"
    "okBeJqnAFwDCQP9IXfxPgfbJw3TFnXyFt9AEmbVRfWhxfCuypBabsbfDL+C0sSHYy9jXxDSQfrNLo0BMOCy6QFbcWbQOhHfIhYl32PVtbhy5AXNbhvolRZZ"
    "G7u78RCseFN6xnjrBNT3OQGQ70xiICBf/HSjqXmaSWoL+yqDVY2Xf/2zQuF487xshvXT1btKI0YN3txr8aHdRRb6uObWglMD4N1oXoddI0H45xVqBGEqgka"
    "Dj6SyHvBNMkSsxBRf/vJhE5hfL8Agtoiia6Z/X6c9zsl6RIVODFnhEIZVQb2we762UNUys0aqjW4o6GHeEYci6v+rDBE/4UtNDl"
)

CLAUDE_P_RE = re.compile(r"""claude(?:\s+[\w\-]+)*\s+-p\s+(?P<q>["'])(?P<prompt>.*?)(?P=q)""", re.DOTALL)


# ---------- jsonl loading ----------

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


# ---------- classification ----------

def _user_is_textual(row: dict) -> bool:
    msg = row.get("message", {})
    c = msg.get("content")
    if isinstance(c, list):
        if any(isinstance(p, dict) and p.get("type") == "tool_result" for p in c):
            return False
    return not row.get("isMeta")


def session_is_interactive(rows: list[dict]) -> bool:
    """A session counts as interactive iff it contains at least one
    `file-history-snapshot` event. That event is emitted when a tool edits
    a file in the workspace, which only happens when a human is driving.
    Multi-turn eval harnesses (question → score-request) look
    conversational but never touch the filesystem, so they fall through
    to headless."""
    return any(r.get("type") == "file-history-snapshot" for r in rows)


def session_is_headless(rows: list[dict]) -> bool:
    return not session_is_interactive(rows)


# ---------- subagent & headless-child linking ----------

def find_subagents(parent_path: Path) -> dict[str, Path]:
    sub_dir = parent_path.parent / parent_path.stem / "subagents"
    if not sub_dir.is_dir():
        return {}
    out: dict[str, Path] = {}
    for meta in sub_dir.glob("*.meta.json"):
        try:
            m = json.loads(meta.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        desc = m.get("description")
        jsonl = sub_dir / (meta.stem.removesuffix(".meta") + ".jsonl")
        if desc and jsonl.exists():
            out.setdefault(desc, jsonl)
    return out


def extract_claude_p_prompt(bash_cmd: str) -> str | None:
    m = CLAUDE_P_RE.search(bash_cmd or "")
    return m.group("prompt") if m else None


def index_first_prompts(session_paths: list[Path]) -> dict[tuple[str, str], Path]:
    idx: dict[tuple[str, str], Path] = {}
    for fp in session_paths:
        try:
            with fp.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row.get("type") == "user" and _user_is_textual(row):
                        c = row.get("message", {}).get("content")
                        txt = c if isinstance(c, str) else (
                            c[0].get("text", "") if isinstance(c, list) and c
                            and isinstance(c[0], dict) else "")
                        idx.setdefault((row.get("cwd", ""), (txt or "").strip()[:200]), fp)
                        break
        except OSError:
            pass
    return idx


def _parse_ts(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _collect_bash_spans(rows: list[dict]) -> list[tuple[str, datetime, datetime, str]]:
    """Return (tool_use_id, start, end, cwd) for each Bash with a matched tool_result."""
    starts: dict[str, tuple[datetime | None, str]] = {}
    spans: list[tuple[str, datetime, datetime, str]] = []
    sess_cwd = ""
    for r in rows:
        if r.get("cwd"):
            sess_cwd = r["cwd"]
        if r.get("type") == "assistant":
            sts = _parse_ts(r.get("timestamp"))
            for b in (r.get("message", {}).get("content") or []):
                if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") == "Bash":
                    starts[b.get("id", "")] = (sts, sess_cwd)
        if r.get("type") == "user":
            c = r.get("message", {}).get("content")
            if isinstance(c, list):
                for p in c:
                    if isinstance(p, dict) and p.get("type") == "tool_result":
                        tid = p.get("tool_use_id", "")
                        if tid in starts:
                            sts, cwd = starts.pop(tid)
                            ets = _parse_ts(r.get("timestamp"))
                            if sts and ets and tid:
                                spans.append((tid, sts, ets, cwd))
    return spans


def index_bash_owners(session_paths: list[Path]
                      ) -> dict[str, tuple[datetime, datetime, str, Path]]:
    """tool_use_id -> (start, end, cwd, source_jsonl) across interactive parents
    and their subagents."""
    out: dict[str, tuple[datetime, datetime, str, Path]] = {}
    for fp in session_paths:
        try:
            rows = load_jsonl(fp)
        except Exception:
            continue
        if session_is_headless(rows):
            continue
        for tid, s, e, cwd in _collect_bash_spans(rows):
            out[tid] = (s, e, cwd, fp)
        sub_dir = fp.parent / fp.stem / "subagents"
        if sub_dir.is_dir():
            for sfp in sub_dir.glob("*.jsonl"):
                try:
                    srows = load_jsonl(sfp)
                except Exception:
                    continue
                for tid, s, e, cwd in _collect_bash_spans(srows):
                    out[tid] = (s, e, cwd, sfp)
    return out


def _first_user_ts_cwd(path: Path) -> tuple[datetime | None, str]:
    try:
        with path.open(encoding="utf-8") as f:
            cwd = ""
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("cwd") and not cwd:
                    cwd = r["cwd"]
                if r.get("type") == "user" and _user_is_textual(r):
                    return _parse_ts(r.get("timestamp")), cwd or r.get("cwd", "")
    except OSError:
        pass
    return None, ""


MIN_SPAWNING_BASH_DURATION = timedelta(seconds=10)


def link_headless_to_bash(
    session_paths: list[Path],
    bash_owners: dict[str, tuple[datetime, datetime, str, Path]],
) -> dict[str, list[Path]]:
    """Return tool_use_id -> [headless paths spawned during that Bash].

    Only long-running Bash calls (≥10s) count: a command that returned in
    0-1 seconds cannot have hosted a Claude subprocess that takes longer than
    itself. The headless session's first user timestamp must fall strictly
    within [start, end]."""
    by_cwd: dict[str, list[tuple[str, datetime, datetime]]] = {}
    for tid, (s, e, cwd, _src) in bash_owners.items():
        if (e - s) < MIN_SPAWNING_BASH_DURATION:
            continue
        by_cwd.setdefault(cwd, []).append((tid, s, e))
    children: dict[str, list[tuple[datetime, Path]]] = {}
    for fp in session_paths:
        try:
            rows = load_jsonl(fp)
        except Exception:
            continue
        if not session_is_headless(rows):
            continue
        ts, cwd = _first_user_ts_cwd(fp)
        if not ts:
            continue
        best = None
        for tid, s, e in by_cwd.get(cwd, []):
            if s <= ts <= e:
                if best is None or s > best[1]:
                    best = (tid, s)
        if best:
            children.setdefault(best[0], []).append((ts, fp))
    return {tid: [p for _, p in sorted(v)] for tid, v in children.items()}


# ---------- rendering ----------

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
    return isinstance(obj, dict) and bool(set(obj) & RAW_EVENT_SENTINEL_KEYS)


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
    if row.get("isSidechain"):
        return "SUBAGENT"
    origin = row.get("origin")
    if isinstance(origin, dict) and origin.get("kind") == "task-notification":
        return "TASK_NOTIF"
    c = row.get("message", {}).get("content")
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
    return "HEADLESS" if headless_session else "USER"


def _parts(message: dict, tool_names: dict[str, str], include_thinking: bool
           ) -> Iterator[tuple[str, str, dict | None]]:
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
            tid = part.get("id", "")
            name = part.get("name", "")
            if isinstance(tid, str) and tid and name:
                tool_names[tid] = name
            body = f"name: {name}"
            inp = _kv_block(part.get("input"))
            if inp:
                body += f"\n{inp}"
            yield "TOOL_USE", body, part
        elif ptype == "tool_result":
            tid = part.get("tool_use_id", "")
            body = _stringify(part.get("content", ""))
            tool = tool_names.get(tid, "").lower() if isinstance(tid, str) else ""
            if tool == "read" and _looks_like_read_dump(body):
                continue
            yield "TOOL_RESULT", body, part
        else:
            yield (ptype.upper() if isinstance(ptype, str) else "CONTENT"), _stringify(part), part


def _row_message(row: dict) -> dict | None:
    t = row.get("type")
    if t in ("user", "assistant"):
        msg = row.get("message")
        if isinstance(msg, dict):
            return msg
    if t == "progress":
        data = row.get("data")
        if isinstance(data, dict):
            inner = data.get("message")
            if isinstance(inner, dict):
                m = inner.get("message")
                if isinstance(m, dict):
                    return m
    return None


def render_session(
    rows: list[dict],
    *,
    source_path: Path,
    include_thinking: bool = False,
    first_prompt_index: dict[tuple[str, str], Path] | None = None,
    headless_by_tool_use: dict[str, list[Path]] | None = None,
    visited: set[Path] | None = None,
    inlined_children: set[Path] | None = None,
    depth: int = 0,
    child_kind: str = "",
) -> str:
    visited = visited if visited is not None else {source_path}
    inlined_children = inlined_children if inlined_children is not None else set()
    first_prompt_index = first_prompt_index if first_prompt_index is not None else {}
    headless_by_tool_use = headless_by_tool_use if headless_by_tool_use is not None else {}

    headless = False if child_kind == "subagent" else session_is_headless(rows)
    indent = "  " * depth
    out: list[str] = []
    if depth > 0:
        tag = child_kind.upper().replace("_", " ") or ("HEADLESS CHILD" if headless else "NESTED")
        out.append(f"{indent}┌── {tag}")

    tool_names: dict[str, str] = {}
    for row in rows:
        msg = _row_message(row)
        if msg is None:
            continue
        label = _role_label(row, headless)
        for kind, text, part in _parts(msg, tool_names, include_thinking):
            body = (text or "").strip()
            if not body:
                continue
            if NOISE_LINE_RE.match(body):
                continue
            if label in ("TASK_NOTIF", "CMD_OUTPUT"):
                continue
            if any(body.lstrip().startswith(pfx) for pfx in NOISE_CONTENT_PREFIXES):
                continue
            body = _strip_raw_event_lines(body)
            if not body or _looks_like_raw_event_blob(body):
                continue
            suffix = f" {kind}" if kind else ""
            indented = "\n".join(indent + l for l in body.splitlines())
            out.append(f"{indent}{label}{suffix}:\n{indented}\n")

            if kind == "TOOL_USE" and isinstance(part, dict):
                # 1) explicit `claude -p` / Agent child
                child_rows, child_path, child_tag = _resolve_child(
                    part, source_path, first_prompt_index, visited)
                if child_rows and child_path is not None:
                    inlined_children.add(child_path)
                    out.append(render_session(
                        child_rows, source_path=child_path,
                        include_thinking=include_thinking,
                        first_prompt_index=first_prompt_index,
                        headless_by_tool_use=headless_by_tool_use,
                        visited=visited,
                        inlined_children=inlined_children,
                        depth=depth + 1, child_kind=child_tag,
                    ))
                    out.append(f"{indent}└── end")
                # 2) time-window matched headless children spawned by this Bash
                tid = part.get("id", "")
                if tid and tid in headless_by_tool_use:
                    for hp in headless_by_tool_use[tid]:
                        if hp in visited:
                            continue
                        visited.add(hp)
                        inlined_children.add(hp)
                        try:
                            hrows = load_jsonl(hp)
                        except Exception:
                            continue
                        out.append(render_session(
                            hrows, source_path=hp,
                            include_thinking=include_thinking,
                            first_prompt_index=first_prompt_index,
                            headless_by_tool_use=headless_by_tool_use,
                            visited=visited,
                            inlined_children=inlined_children,
                            depth=depth + 1, child_kind="headless_child",
                        ))
                        out.append(f"{indent}└── end")

    return "\n".join(out)


def _resolve_child(
    tool_use_part: dict,
    parent_path: Path,
    first_prompt_index: dict[tuple[str, str], Path],
    visited: set[Path],
) -> tuple[list[dict] | None, Path | None, str]:
    name = tool_use_part.get("name", "")
    inp = tool_use_part.get("input") or {}

    if name in ("Agent", "Task"):
        desc = inp.get("description") or inp.get("subagent_type") or ""
        fp = find_subagents(parent_path).get(desc)
        if fp and fp not in visited:
            visited.add(fp)
            return load_jsonl(fp), fp, "subagent"

    if name == "Bash":
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


def render_to_text(source: Path, first_prompt_index: dict,
                   headless_by_tool_use: dict[str, list[Path]] | None = None,
                   include_thinking: bool = False
                   ) -> tuple[str, set[Path]]:
    rows = load_jsonl(source)
    inlined: set[Path] = set()
    text = render_session(
        rows, source_path=source,
        include_thinking=include_thinking,
        first_prompt_index=first_prompt_index,
        headless_by_tool_use=headless_by_tool_use,
        inlined_children=inlined,
    )
    return text.replace(DEFAULT_GIBBERISH_TOKEN, "").strip() + "\n", inlined


# ---------- batch mode ----------

def _cwd_to_slug(cwd: Path) -> str:
    """Claude Code encodes cwd as project dir name: /A/B/C -> -A-B-C."""
    parts = cwd.resolve().parts
    if parts and parts[0] == "/":
        parts = parts[1:]
    return "-" + "-".join(p.replace("/", "-") for p in parts)


def scoped_sessions(projects_dir: Path, cwd: Path) -> list[Path]:
    """All session jsonl files under project dirs matching the current cwd's slug
    (includes worktree-variant project dirs like `<slug>--claude-worktrees-…`)."""
    slug = _cwd_to_slug(cwd)
    out: list[Path] = []
    for d in projects_dir.iterdir() if projects_dir.exists() else []:
        if not d.is_dir():
            continue
        if d.name == slug or d.name.startswith(slug + "-"):
            out.extend(f for f in d.glob("*.jsonl") if f.is_file())
    return out


def _render_worker(args):
    source, first_prompt_index, headless_by_tool_use, include_thinking = args
    try:
        text, inlined = render_to_text(
            source, first_prompt_index, headless_by_tool_use, include_thinking)
        return source, text, inlined, None
    except Exception as e:
        return source, "", set(), repr(e)


def batch_port(projects_dir: Path, cwd: Path, out_dir: Path, include_thinking: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    interactive_dir = out_dir / "interactive"
    headless_dir = out_dir / "headless"
    interactive_dir.mkdir(exist_ok=True)
    headless_dir.mkdir(exist_ok=True)

    sources = scoped_sessions(projects_dir, cwd)
    if not sources:
        print(f"no sessions under project slug for {cwd}")
        return
    print(f"scope: {cwd}  ({len(sources)} sessions)")
    first_prompt_index = index_first_prompts(sources)
    bash_owners = index_bash_owners(sources)
    headless_by_tool_use = link_headless_to_bash(sources, bash_owners)
    n_linked = sum(len(v) for v in headless_by_tool_use.values())
    print(f"linked headless-by-timewindow: {n_linked}")

    workers = min(cpu_count(), 8)
    jobs = [(p, first_prompt_index, headless_by_tool_use, include_thinking) for p in sources]

    results: list[tuple[Path, str, set[Path], str | None]] = []
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_render_worker, jobs, chunksize=16), 1):
            results.append(r)
            if i % 500 == 0:
                print(f"  rendered {i}/{len(jobs)}")

    all_inlined: set[Path] = set()
    for _, _, inl, _ in results:
        all_inlined |= inl

    n_int = n_head = n_skipped = n_err = 0
    for source, text, inlined, err in results:
        if err:
            n_err += 1
            continue
        if source in all_inlined:
            n_skipped += 1
            continue
        headless = session_is_headless(load_jsonl(source))
        dest_dir = headless_dir if headless else interactive_dir
        proj_slug = source.parent.name.lstrip("-").replace("-", "_")
        fname = f"{proj_slug}__{source.stem}.txt"
        (dest_dir / fname).write_text(text, encoding="utf-8")
        if headless:
            n_head += 1
        else:
            n_int += 1

    print(f"\nwrote: interactive={n_int}  headless={n_head}  "
          f"inlined-into-parent={n_skipped}  errors={n_err}")
    print(f"output: {out_dir}")


# ---------- single-file mode ----------

def newest_session(projects_dir: Path) -> Path:
    files = [f for f in projects_dir.rglob("*.jsonl")
             if f.is_file() and "/subagents/" not in str(f)]
    if not files:
        raise FileNotFoundError(f"No sessions under {projects_dir}")
    return max(files, key=lambda p: p.stat().st_mtime)


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", nargs="?", type=Path, default=None,
                   help="Single session .jsonl (omit to batch-port all projects)")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Output file (single mode) or directory (batch mode)")
    p.add_argument("--thinking", action="store_true",
                   help="Include assistant thinking blocks")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    projects_dir = Path.home() / ".claude" / "projects"

    if args.source is None:
        out_dir = (args.output.expanduser().resolve() if args.output
                   else Path.home() / ".claude" / "ported")
        batch_port(projects_dir, Path.cwd(), out_dir, args.thinking)
        return 0

    source = args.source.expanduser().resolve()
    sources_in_scope = scoped_sessions(projects_dir, Path.cwd()) or [source]
    first_prompt_index = index_first_prompts(sources_in_scope)
    bash_owners = index_bash_owners(sources_in_scope)
    headless_by_tool_use = link_headless_to_bash(sources_in_scope, bash_owners)
    text, _ = render_to_text(source, first_prompt_index, headless_by_tool_use, args.thinking)

    out = (args.output.expanduser().resolve() if args.output
           else Path.home() / ".claude" / "most_recent_conversation_plain.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"source={source}")
    print(f"output={out}")
    print(f"chars={len(text)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
