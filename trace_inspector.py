# inspect_trace.py
import json
from pathlib import Path

# Find the most recent trace
mlruns_path = Path("mlruns/1/traces")
trace_dirs = [d for d in mlruns_path.iterdir() if d.is_dir()]
latest_trace = max(trace_dirs, key=lambda d: d.stat().st_mtime)
trace_file = latest_trace / "artifacts" / "traces.json"

print(f"Reading: {trace_file}")

with open(trace_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"\nTop-level keys: {data.keys()}")

# Check structure
if 'spans' in data:
    print(f"\nFound {len(data['spans'])} spans")
    print("\nSpan names found:")
    span_names = [span.get('name', 'unknown') for span in data['spans']]
    for name in set(span_names):
        count = span_names.count(name)
        print(f"  - {name}: {count} spans")
    
    # Show first span structure
    print("\nFirst span keys:", data['spans'][0].keys() if data['spans'] else "No spans")
    
elif 'data' in data and 'spans' in data['data']:
    print(f"\nFound spans in data['data']: {len(data['data']['spans'])} spans")
    
else:
    print("\nUnexpected structure. Looking for spans...")
    import pprint
    pprint.pprint(list(data.keys())[:10])