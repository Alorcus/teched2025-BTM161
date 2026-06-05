"""Series-3 eval: vary the SYSTEM PROMPT while pinning the process notation
to bpmn_dsl (format_e_bpmn_dsl.txt).

Goal: lift BPMN-only accuracy above the 82% Series-2 baseline by changing
the framing/instruction text rather than the process description.

Reuses ground_truth.jsonl, scoring, and Haiku model identical to run_eval.py.
Writes results_series3.csv next to this file.

Run:
    poetry run python evals/process_supervisor_prompt/run_eval_series3.py
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

# Force Haiku regardless of any pre-exported shell var.
os.environ["ANTHROPIC_MODEL"] = "anthropic--claude-4.5-haiku"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.llm import create_chat_llm  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
GROUND_TRUTH = EVAL_DIR / "ground_truth.jsonl"
PROMPTS_DIR = EVAL_DIR / "prompts"
PROCESS_DESCRIPTION_PATH = EVAL_DIR / "format_e_bpmn_dsl.txt"
RESULTS_CSV = EVAL_DIR / "results_series3.csv"

PROMPT_VARIANTS: dict[str, Path] = {
    "v1_baseline":         PROMPTS_DIR / "v1_baseline.txt",
    "v2_multi_message":    PROMPTS_DIR / "v2_multi_message.txt",
    "v3_tool_inventory":   PROMPTS_DIR / "v3_tool_inventory.txt",
    "v4_combined":         PROMPTS_DIR / "v4_combined.txt",
    "v5_strict_violation": PROMPTS_DIR / "v5_strict_violation.txt",
    "v6_lenient_violation":PROMPTS_DIR / "v6_lenient_violation.txt",
}

_EXEC_RE = re.compile(r"^\s*Execution\s*:\s*(?P<id>A\d+[a-z]?)\s*:\s*(?P<name>[^|\r\n]+?)\s*$")
_TERM_RE = re.compile(r"^\s*Termination\s*:\s*(?P<id>A\d+[a-z]?)\s*:\s*(?P<name>[^|\r\n:]+?)\s*:\s*(?P<reason>[a-zA-Z0-9_\-]+)\s*$")
_VIOL_RE = re.compile(r"^\s*Violation\s*:\s*(?P<reason>.+?)\s*$", re.IGNORECASE)


@dataclass
class Decision:
    kind: str
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
    parts = [
        f"agent={record['agent']}",
        f"trigger={record['trigger']}",
        f"tool={record.get('tool') or '-'}",
        f"target={record.get('target') or '-'}",
        f"content={record['content_excerpt']}",
    ]
    return " ".join(parts)


def expected_line(record: dict) -> str:
    exp = record["expected"]
    kind = exp["decision"]
    if kind == "Execution":
        return f"Execution:{exp['activity_id']}:{exp['activity_name']}"
    if kind == "Termination":
        return f"Termination:{exp['activity_id']}:{exp['activity_name']}:{exp['reason']}"
    if kind == "Violation":
        return f"Violation:{exp.get('reason_hint', 'unspecified')}"
    raise ValueError(f"unknown decision kind: {kind}")


def render_prompt(template: str, process_description: str, record: dict, prior_tail: list[str]) -> str:
    tail_block = "\n".join(prior_tail) if prior_tail else "(empty)"
    return template.format(
        process_model=process_description,
        prior_log_tail=tail_block,
        message_brief=render_message_brief(record),
    )


def score(expected: dict, decision: Decision) -> str:
    if decision.kind == "Unparseable":
        return "unparseable"
    exp_kind = expected["decision"]
    if decision.kind != exp_kind:
        return "wrong"
    if exp_kind == "Violation":
        return "correct"
    if decision.activity_id != expected.get("activity_id"):
        return "wrong"
    return "correct"


def main() -> int:
    records = [json.loads(line) for line in GROUND_TRUTH.read_text().splitlines() if line.strip()]
    process_description = PROCESS_DESCRIPTION_PATH.read_text()
    print(f"Series 3 — varying SYSTEM PROMPT, fixed format = {PROCESS_DESCRIPTION_PATH.name}")
    print(f"Loaded {len(records)} ground-truth records\n")

    llm = create_chat_llm()
    print(f"Model: {getattr(llm, 'model', '?')}\n")

    rows: list[dict] = []
    summary: dict[str, dict] = {}

    for variant_name, template_path in PROMPT_VARIANTS.items():
        template = template_path.read_text()
        print(f"=== Variant: {variant_name} ({len(template)} prompt-chars) ===")
        latencies: list[float] = []
        n_correct = n_wrong = n_unparseable = 0
        prior_tail: list[str] = []

        for rec in records:
            prompt = render_prompt(template, process_description, rec, prior_tail)
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

            rows.append({
                "variant": variant_name,
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

            prior_tail.append(expected_line(rec))

        n = len(records)
        summary[variant_name] = {
            "accuracy": n_correct / n,
            "wrong_rate": n_wrong / n,
            "unparseable_rate": n_unparseable / n,
            "p50_latency": median(latencies),
            "mean_latency": mean(latencies),
            "max_latency": max(latencies),
            "prompt_chars": len(template),
        }
        print(f"  → accuracy={n_correct}/{n} ({n_correct/n:.0%})  "
              f"wrong={n_wrong}  unparseable={n_unparseable}  "
              f"p50={median(latencies):.2f}s  max={max(latencies):.2f}s\n")

    # Summary table
    print("=" * 80)
    print("SUMMARY — Series 3 (system prompt × bpmn_dsl process notation)")
    print("=" * 80)
    header = f"{'variant':<22} {'pchars':>6} {'accuracy':>9} {'wrong':>6} {'unparse':>8} {'p50':>6} {'max':>6}"
    print(header)
    print("-" * len(header))
    for name, s in summary.items():
        print(f"{name:<22} {s['prompt_chars']:>6d} "
              f"{s['accuracy']:>9.0%} {s['wrong_rate']:>6.0%} {s['unparseable_rate']:>8.0%} "
              f"{s['p50_latency']:>5.2f}s {s['max_latency']:>5.2f}s")

    # Recommend a winner: highest accuracy, tie-break on shortest p50.
    ranked = sorted(
        summary.items(),
        key=lambda kv: (-kv[1]["accuracy"], kv[1]["p50_latency"]),
    )
    winner_name, winner_stats = ranked[0]
    print(f"\nRecommended winner: {winner_name} "
          f"(accuracy={winner_stats['accuracy']:.0%}, p50={winner_stats['p50_latency']:.2f}s)")

    # Wrong-answer details for each variant.
    print("\nMisclassifications per variant:")
    for variant_name in PROMPT_VARIANTS:
        wrongs = [r for r in rows if r["variant"] == variant_name and r["verdict"] != "correct"]
        if not wrongs:
            print(f"  [{variant_name}] all correct")
            continue
        print(f"  [{variant_name}]")
        for r in wrongs:
            print(f"    #{r['i']:>2} {r['agent']:<24s} {r['trigger']:<10s} tool={r['tool']:<22s}")
            print(f"        expected: {r['expected_kind']}:{r['expected_id'] or '-'}")
            print(f"        got     : {r['raw'][:100]}")

    # CSV
    with RESULTS_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote per-message results to {RESULTS_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
