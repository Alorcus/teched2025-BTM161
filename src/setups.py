"""Setup discovery and selection.

A "setup" is a self-contained `config/setups/<name>/` directory containing
`agents/*.yaml` and `guidelines/*.yaml`. Setups are not merged or composed —
each one is a complete, independent configuration of the coffee shop.
"""
import os
from pathlib import Path

SETUPS_ROOT = Path("config/setups")
ENV_VAR = "COFFEE_SHOP_SETUP"
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
    """Pick the setup name. Env var supersedes CLI flag. Falls back to the default setup if available."""
    env_value = os.environ.get(ENV_VAR)
    name = env_value or cli_value
    if not name:
        available = list_setups()
        if DEFAULT_SETUP in available:
            return DEFAULT_SETUP
        raise SystemExit(
            f"No setup selected. Pass --setup <name> or set {ENV_VAR}=<name>. "
            f"Available: {available or '(none)'}"
        )
    return name
