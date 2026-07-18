#!/usr/bin/env python3
"""Extract customer feedback rows from _all_traces.csv, sorted by scenario and setup."""
import argparse
import csv
from pathlib import Path

FIELDS = [
    "case_setup",
    "case_scenario_index",
    "case_id",
    "time:timestamp",
    "drink",
    "feedback_score",
    "feedback_reason",
    "feedback_valid",
]


def scenario_key(row):
    idx = row.get("case_scenario_index", "")
    try:
        return (int(idx), row.get("case_setup", ""))
    except ValueError:
        return (10**9, row.get("case_setup", ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("generated_event_log/_all_traces.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("generated_event_log/_customer_feedback.csv"),
    )
    args = parser.parse_args()

    with args.input.open(newline="") as f:
        rows = [
            row for row in csv.DictReader(f)
            if row.get("concept:name") == "user_feedback"
        ]

    rows.sort(key=scenario_key)

    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} feedback rows to {args.output}")


if __name__ == "__main__":
    main()
