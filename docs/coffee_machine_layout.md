# Coffee Machine Panel Layout

```
┌─────────────────────────────────────┐
│  Coffee Machine                     │
│                                     │
│         ┌───────────┐               │
│         │   ☕ SVG   │               │
│         └───────────┘               │
│  ▓▓▓▓▓▓▓▓░░░░░░░ (progress bar)    │
│  ⏳ Brewing latte...                │
│                                     │
│  [INIT] ▶ [SUCC][SUCC][FAIL][SUCC] [♻️] │
└─────────────────────────────────────┘
```

## Notes

- Everything is inside **one** bordered frame
- Queue badges take ~80% of the bottom row, regenerate button takes ~20% (flush right)
- Queue row is the last element inside the box, separated from status by a thin divider line
