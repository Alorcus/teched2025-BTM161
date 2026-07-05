"""Setup discovery and selection.

A "setup" is a self-contained `config/setups/<name>/` directory containing
`agents/*.yaml` and `guidelines/*.yaml`. Setups are not merged or composed —
each one is a complete, independent configuration of the coffee shop.
"""

from pathlib import Path

SETUPS_ROOT = Path("config/setups")
DEFAULT_SETUP = "baseline"


def list_setups() -> list[str]:
    if not SETUPS_ROOT.exists():
        return []
    return sorted(p.name for p in SETUPS_ROOT.iterdir() if p.is_dir())


def setup_dir(name: str) -> Path:
    """Return the directory for `name`, validating it looks like a setup."""
    d = SETUPS_ROOT / name
    if not d.is_dir():
        available = list_setups()
        raise ValueError(
            f"Setup {name!r} not found in {SETUPS_ROOT}. "
            f"Available: {available or '(none)'}"
        )
    if not (d / "agents").is_dir():
        raise ValueError(f"Setup {name!r} is missing required 'agents/' directory")
    if not (d / "guidelines").is_dir():
        raise ValueError(f"Setup {name!r} is missing required 'guidelines/' directory")
    return d


def resolve_setup_name(cli_value: str | None) -> str:
    """Pick a single setup name. Falls back to the default setup if available."""
    if cli_value:
        return cli_value
    available = list_setups()
    if DEFAULT_SETUP in available:
        return DEFAULT_SETUP
    raise SystemExit(
        f"No setup selected. Pass --setup <name>. Available: {available or '(none)'}"
    )


def resolve_setup_names(cli_values: list[str] | None) -> list[str]:
    """Resolve the ordered list of setups to run.

    - If cli_values is non-empty, return it (order preserved; duplicates allowed).
    - Else fall back to the default via resolve_setup_name(None).
    """
    if cli_values:
        return list(cli_values)
    return [resolve_setup_name(None)]
