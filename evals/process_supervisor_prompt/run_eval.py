"""Eval harness for the ProcessSupervisor prompt.

Loads ground_truth.jsonl, runs each labelled message against each candidate
"Process description" format via Haiku, scores exact-match accuracy on
(decision_kind, activity_id), tracks llm_unparseable_output rate and per-call
latency, prints a summary table + sample wrong answers per candidate.

Run:
    poetry run python evals/process_supervisor_prompt/run_eval.py

Output: prints to stdout and writes results.csv next to this file.

NOT a unit test — this is an evidence-gathering script. Production code
(src/control_plane/process_supervisor.py) is not modified by this harness.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

# Force Haiku regardless of any pre-exported shell var (the user's shell has
# ANTHROPIC_MODEL set to opus; the eval intentionally bypasses that).
os.environ["ANTHROPIC_MODEL"] = "anthropic--claude-4.5-haiku"

# Make src.* importable when run from the repo root.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.llm import create_chat_llm  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
GROUND_TRUTH = EVAL_DIR / "ground_truth.jsonl"
RESULTS_CSV = EVAL_DIR / "results.csv"

# Identical scaffolding for every candidate. Only PROCESS_DESCRIPTION changes.
INSTRUCTION_HEADER = (
    "You are the process supervisor for a multi-agent coffee shop. Your only job\n"
    "is to classify ONE incoming message as either an Execution of a known\n"
    "activity, a Termination of one, or a Violation."
)

OUTPUT_RULES = (
    "Reply with EXACTLY ONE line, no prose, no quotes, in one of these formats:\n"
    "  Execution:<ActivityID>:<slug>\n"
    "  Termination:<ActivityID>:<slug>:<reason>\n"
    "      reason is `terminal` for a BPMN end event, or `via_handoff_to_<agent>`.\n"
    "  Violation:<snake_case_reason_no_spaces>\n"
    "\n"
    "Use the activity ID exactly as given (e.g. `A01`, `A05b`) — never any other label.\n"
    "Use the slug (snake_case) for <slug>, not the human-readable display name.\n"
    "If the log tail is empty, this is the FIRST message of a fresh conversation —\n"
    "that is normal, not a violation."
)

CANDIDATES: dict[str, str] = {
    # --- Series 1: catalog-aware (original) ---
    "markdown":   (EVAL_DIR / "format_a_markdown.md").read_text(),
    "yaml":       (EVAL_DIR / "format_b_yaml.yaml").read_text(),
    "dsl":        (EVAL_DIR / "format_c_dsl.txt").read_text(),
    # --- Series 2: BPMN-only (no tool names, no triggers, no slugs) ---
    "bpmn_md":    (EVAL_DIR / "format_d_bpmn_md.md").read_text(),
    "bpmn_dsl":   (EVAL_DIR / "format_e_bpmn_dsl.txt").read_text(),
    "bpmn_json":  (EVAL_DIR / "format_f_bpmn_json.json").read_text(),
}

_EXEC_RE = re.compile(r"^\s*Execution\s*:\s*(?P<id>A\d+[a-z]?)\s*:\s*(?P<name>[^|\r\n]+?)\s*$")
_TERM_RE = re.compile(r"^\s*Termination\s*:\s*(?P<id>A\d+[a-z]?)\s*:\s*(?P<name>[^|\r\n:]+?)\s*:\s*(?P<reason>[a-zA-Z0-9_\-]+)\s*$")
_VIOL_RE = re.compile(r"^\s*Violation\s*:\s*(?P<reason>.+?)\s*$", re.IGNORECASE)


@dataclass
class Decision:
    kind: str            # "Execution" | "Termination" | "Violation" | "Unparseable"
    activity_id: str | None
    raw: str


def parse_decision(raw_text: str) -> Decision:
    text = (raw_text or "").strip().splitlines()[0] if raw_text else ""
    if (m := _EXEC_RE.match(text)):
        return Decision("Execution", m.group("id"), text)
    if (m := _TERM_RE.match(text)):
        return Decision("Termination", m.group("id"), text)
    if (m := _VIOL_RE.match(text)):
        return Decision("Violation", None, text)
    return Decision("Unparseable", None, text)


def render_message_brief(record: dict) -> str:
    """One-line description of the message — same shape the real supervisor uses."""
    parts = [
        f"agent={record['agent']}",
        f"trigger={record['trigger']}",
        f"tool={record.get('tool') or '-'}",
        f"target={record.get('target') or '-'}",
        f"content={record['content_excerpt']}",
    ]
    return " ".join(parts)


def expected_line(record: dict) -> str:
    """Render the ground-truth decision for a record as the supervisor would write it
    to its in-memory log — used to build the cumulative `Prior log tail` for later
    records in the same conversation."""
    exp = record["expected"]
    kind = exp["decision"]
    if kind == "Execution":
        return f"Execution:{exp['activity_id']}:{exp['activity_name']}"
    if kind == "Termination":
        return f"Termination:{exp['activity_id']}:{exp['activity_name']}:{exp['reason']}"
    if kind == "Violation":
        return f"Violation:{exp.get('reason_hint', 'unspecified')}"
    raise ValueError(f"unknown decision kind: {kind}")


def build_prompt(process_description: str, record: dict, prior_tail: list[str]) -> str:
    tail_block = "\n".join(prior_tail) if prior_tail else "(empty)"
    return (
        f"{INSTRUCTION_HEADER}\n\n"
        f"# Process model\n{process_description}\n\n"
        f"# Prior log tail\n{tail_block}\n\n"
        f"# New message\n{render_message_brief(record)}\n\n"
        f"# Output rules\n{OUTPUT_RULES}\n"
    )


def score(expected: dict, decision: Decision) -> str:
    """Return 'correct' | 'wrong' | 'unparseable'."""
    if decision.kind == "Unparseable":
        return "unparseable"
    exp_kind = expected["decision"]
    if decision.kind != exp_kind:
        return "wrong"
    if exp_kind == "Violation":
        # Any Violation is correct regardless of reason text.
        return "correct"
    if decision.activity_id != expected.get("activity_id"):
        return "wrong"
    return "correct"


def main() -> int:
    records = [json.loads(line) for line in GROUND_TRUTH.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(records)} ground-truth records from {GROUND_TRUTH.name}\n")

    llm = create_chat_llm()
    print(f"Model: {getattr(llm, 'model', '?')}\n")

    # rows accumulate per-message results for CSV.
    rows: list[dict] = []
    summary: dict[str, dict] = {}

    for cand_name, process_description in CANDIDATES.items():
        print(f"=== Candidate: {cand_name} ({len(process_description)} chars) ===")
        per_msg = []
        latencies = []
        n_correct = n_wrong = n_unparseable = 0
        prior_tail: list[str] = []  # cumulative ground-truth decisions for prior records

        for rec in records:
            prompt = build_prompt(process_description, rec, prior_tail)
            t0 = time.perf_counter()
            try:
                resp = llm.invoke(prompt)
                raw = resp.content if hasattr(resp, "content") else str(resp)
                if isinstance(raw, list):
                    raw = next(
                        (c.get("text", "") for c in raw if isinstance(c, dict) and c.get("type") == "text"),
                        "",
                    )
            except Exception as e:
                raw = f"<<error: {e}>>"
            dt = time.perf_counter() - t0
            latencies.append(dt)

            decision = parse_decision(str(raw))
            verdict = score(rec["expected"], decision)
            n_correct      += verdict == "correct"
            n_wrong        += verdict == "wrong"
            n_unparseable  += verdict == "unparseable"

            per_msg.append((rec, decision, verdict, dt))
            rows.append({
                "candidate": cand_name,
                "i": rec["i"],
                "agent": rec["agent"],
                "trigger": rec["trigger"],
                "tool": rec.get("tool") or "",
                "expected_kind": rec["expected"]["decision"],
                "expected_id":   rec["expected"].get("activity_id", ""),
                "raw":           decision.raw,
                "decision_kind": decision.kind,
                "decision_id":   decision.activity_id or "",
                "verdict":       verdict,
                "latency_s":     f"{dt:.3f}",
            })
            mark = {"correct": "✓", "wrong": "✗", "unparseable": "?"}[verdict]
            print(f"  {mark} #{rec['i']:>2} {rec['agent']:<24s} {rec['trigger']:<10s} "
                  f"tool={(rec.get('tool') or '-'):<22s}  →  {decision.raw[:80]}")

            # Feed the ground-truth (NOT the model's own answer) forward, so each
            # candidate sees the canonical history and is scored independently of
            # whether it got prior records right.
            prior_tail.append(expected_line(rec))

        n = len(records)
        summary[cand_name] = {
            "accuracy": n_correct / n,
            "unparseable_rate": n_unparseable / n,
            "wrong_rate": n_wrong / n,
            "p50_latency": median(latencies),
            "mean_latency": mean(latencies),
            "max_latency": max(latencies),
            "context_chars": len(process_description),
        }

        print(f"  → accuracy={n_correct}/{n} ({n_correct/n:.0%})  "
              f"wrong={n_wrong}  unparseable={n_unparseable}  "
              f"p50={median(latencies):.2f}s  max={max(latencies):.2f}s\n")

    # ---------- Summary table ----------
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    header = f"{'candidate':<10} {'ctx_chars':>9} {'accuracy':>9} {'wrong':>6} {'unparse':>8} {'p50':>6} {'mean':>6} {'max':>6}"
    print(header)
    print("-" * len(header))
    for name, s in summary.items():
        print(f"{name:<10} {s['context_chars']:>9d} "
              f"{s['accuracy']:>9.0%} {s['wrong_rate']:>6.0%} {s['unparseable_rate']:>8.0%} "
              f"{s['p50_latency']:>5.2f}s {s['mean_latency']:>5.2f}s {s['max_latency']:>5.2f}s")

    # Wrong-answer examples per candidate (up to 3 each).
    print("\nSample misclassifications:")
    for cand_name in CANDIDATES:
        wrongs = [(r, d) for (r, d, v, _) in per_msg_collect(rows, cand_name) if v_for(rows, cand_name, r["i"]) != "correct"]
        if not wrongs:
            print(f"  [{cand_name}] all correct")
            continue
        print(f"  [{cand_name}]")
        for r, d in wrongs[:3]:
            exp = r["expected"]
            exp_str = f"{exp['decision']}:{exp.get('activity_id','-')}"
            print(f"    #{r['i']:>2} {r['agent']:<24s} {r['trigger']:<10s} "
                  f"tool={(r.get('tool') or '-'):<22s}")
            print(f"        expected: {exp_str}")
            print(f"        got     : {d.raw[:100]}")

    # ---------- CSV ----------
    with RESULTS_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote per-message results to {RESULTS_CSV}")
    return 0


def per_msg_collect(rows, cand_name):
    """Reconstruct (record, decision) pairs for one candidate from the rows list."""
    out = []
    for r in rows:
        if r["candidate"] != cand_name:
            continue
        # Need the original record to print expected fields; rebuild a minimal one.
        rec = {
            "i": r["i"],
            "agent": r["agent"],
            "trigger": r["trigger"],
            "tool": r["tool"],
            "expected": {"decision": r["expected_kind"], "activity_id": r["expected_id"]},
        }
        decision = type("D", (), {"raw": r["raw"], "kind": r["decision_kind"], "activity_id": r["decision_id"]})()
        out.append((rec, decision, None, None))
    return out


def v_for(rows, cand_name, i):
    for r in rows:
        if r["candidate"] == cand_name and r["i"] == i:
            return r["verdict"]
    return None


if __name__ == "__main__":
    raise SystemExit(main())
