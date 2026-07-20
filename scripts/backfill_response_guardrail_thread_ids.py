#!/usr/bin/env python3
"""Backfill missing ``thread_id`` on response-guardrail rows in ``guardrail_log/events.jsonl``.

Historically ``ConversationEngine._response_guardrail_denial`` and
``ConversationRunner._evaluate_response_guardrails`` called
``Gateway.evaluate_assistant_message(..., thread_id=None)`` (fixed in
the same PR that added this script). Every response-guardrail decision
emitted before the fix landed on disk with ``thread_id: null``, which
``TraceProcessor._load_gateway_rows`` silently discards -- so the
dashboard's guardrail-hit metric shows zero for ``off_menu_recommendation``
(the only response guardrail today) even though it fires often.

This script recovers ``thread_id`` for those rows by joining each null
record against ``_all_traces.csv``:

  key = (agent_id, exact assistant-message content)

Uniqueness on that key already resolves ~95% of records. The rest are
disambiguated by picking the CSV row whose ``time:timestamp`` is closest
to the JSONL ``ts`` (and only when the second-closest is at least 5s
farther away, so we never pick a coin flip). Rows we can't resolve are
left untouched.

Usage:
    poetry run python scripts/backfill_response_guardrail_thread_ids.py \\
        --jsonl guardrail_log/events.jsonl \\
        --csv generated_event_log/_all_traces.csv

By default writes a ``.backfilled`` alongside the input; pass ``--in-place``
to overwrite (the original is copied to ``<path>.bak`` first). After
backfilling the JSONL, regenerate the CSV so downstream analytics pick
up the recovered rows::

    poetry run python -c "from src.trace_processing import TraceProcessor; TraceProcessor().process_all_traces()"
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd

RESPONSE_TOOL_NAME = "assistant_message"
TIME_TIEBREAK_MIN_GAP_S = 5.0


def build_case_index(csv_path: Path) -> dict[tuple[str, str], list[tuple[str, pd.Timestamp]]]:
    """(agent, content) -> list of (case_id, timestamp) for every AIMessage row."""
    df = pd.read_csv(csv_path, low_memory=False)
    df["_ts"] = pd.to_datetime(df["time:timestamp"], errors="coerce")
    df = df[df["message"].notna() & df["org:resource"].notna()]
    idx: dict[tuple[str, str], list[tuple[str, pd.Timestamp]]] = defaultdict(list)
    for _, row in df.iterrows():
        key = (str(row["org:resource"]), str(row["message"]))
        idx[key].append((row["case_id"], row["_ts"]))
    return idx


def resolve_thread_id(
    record: dict,
    index: dict[tuple[str, str], list[tuple[str, pd.Timestamp]]],
) -> str | None:
    agent = record.get("agent_id", "")
    content = record.get("tool_args", {}).get("content", "")
    candidates = index.get((agent, content), [])
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0][0]

    unique_cases = {cid for cid, _ in candidates}
    if len(unique_cases) == 1:
        return next(iter(unique_cases))

    ts = pd.to_datetime(record.get("ts"), unit="s", errors="coerce")
    if pd.isna(ts):
        return None

    sorted_by_dt = sorted(
        (
            (abs((cts - ts).total_seconds()), cid)
            for cid, cts in candidates
            if cts is not None and not pd.isna(cts)
        ),
        key=lambda x: x[0],
    )
    if not sorted_by_dt:
        return None
    best_dt, best_cid = sorted_by_dt[0]
    if len(sorted_by_dt) == 1:
        return best_cid
    second_dt, _ = sorted_by_dt[1]
    if second_dt - best_dt >= TIME_TIEBREAK_MIN_GAP_S:
        return best_cid
    return None


def backfill(jsonl_in: Path, jsonl_out: Path, csv_path: Path) -> dict[str, int]:
    index = build_case_index(csv_path)
    stats = {
        "total_records": 0,
        "response_null_tid": 0,
        "backfilled": 0,
        "unresolved": 0,
        "left_untouched_non_null": 0,
    }
    with jsonl_in.open("r", encoding="utf-8") as fin, jsonl_out.open("w", encoding="utf-8") as fout:
        for line in fin:
            stats["total_records"] += 1
            record = json.loads(line)
            is_response = (
                record.get("event_type") == "gateway_decision"
                and record.get("tool_name") == RESPONSE_TOOL_NAME
                and record.get("thread_id") is None
            )
            if is_response:
                stats["response_null_tid"] += 1
                resolved = resolve_thread_id(record, index)
                if resolved is not None:
                    record["thread_id"] = resolved
                    stats["backfilled"] += 1
                else:
                    stats["unresolved"] += 1
            else:
                stats["left_untouched_non_null"] += 1
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jsonl", type=Path, default=Path("guardrail_log/events.jsonl"))
    parser.add_argument("--csv", type=Path, default=Path("generated_event_log/_all_traces.csv"))
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path (default: <jsonl>.backfilled). Ignored with --in-place.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input JSONL. The original is copied to <path>.bak first.",
    )
    args = parser.parse_args()

    if not args.jsonl.exists():
        parser.error(f"JSONL not found: {args.jsonl}")
    if not args.csv.exists():
        parser.error(f"CSV not found: {args.csv}")

    if args.in_place:
        backup = args.jsonl.with_suffix(args.jsonl.suffix + ".bak")
        shutil.copy2(args.jsonl, backup)
        target = args.jsonl.with_suffix(args.jsonl.suffix + ".tmp")
    else:
        target = args.output or args.jsonl.with_suffix(args.jsonl.suffix + ".backfilled")

    stats = backfill(args.jsonl, target, args.csv)

    if args.in_place:
        target.replace(args.jsonl)
        print(f"Wrote in place; original preserved at {backup}")
    else:
        print(f"Wrote {target}")

    print("Stats:")
    for k, v in stats.items():
        print(f"  {k:28}: {v:>6}")
    if stats["response_null_tid"]:
        recovery = 100 * stats["backfilled"] / stats["response_null_tid"]
        print(f"  recovery                    : {recovery:>5.1f}%")


if __name__ == "__main__":
    main()
