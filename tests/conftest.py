"""Pytest fixtures shared across the suite."""
import pytest

from src.agents.barista_agent import (
    COFFEE_MACHINE_PORT,
    _kill_process_on_port,
    check_port_in_use,
)


@pytest.fixture(scope="session", autouse=True)
def _kill_leaked_coffee_machine():
    """Safety net: kill any process bound to the coffee machine port at session
    end. Belt-and-suspenders behind stop_coffee_machine() — catches leaks from
    tests that crash before reaching their own cleanup."""
    yield
    if check_port_in_use(COFFEE_MACHINE_PORT):
        _kill_process_on_port(COFFEE_MACHINE_PORT)
