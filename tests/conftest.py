"""Pytest configuration shared by the whole suite."""

from __future__ import annotations

import importlib.util

# Módulos de teste que exercitam o extra opcional ``contextmap[ros1]``/``[ros2]``
# e importam o ``rosbags`` já na coleta. Em uma instalação base (só NumPy) eles não
# são coletados em vez de quebrar a suíte inteira; o cabeçalho do relatório
# torna a omissão explícita. Caminhos relativos a este diretório.
_REQUIRES_ROSBAGS = (
    "ingestion/adapters/test_ros1_bag.py",
    "ingestion/adapters/test_ros2_bag.py",
    "state_estimation/test_state_estimation_fast_lio_process.py",
)
_ROSBAGS_INSTALLED = importlib.util.find_spec("rosbags") is not None

collect_ignore: list[str] = [] if _ROSBAGS_INSTALLED else list(_REQUIRES_ROSBAGS)


def pytest_report_header() -> str | None:
    """Report which modules the base install leaves out, so the omission is visible."""
    if _ROSBAGS_INSTALLED:
        return None
    return "rosbags not installed (base install); not collected: " + ", ".join(_REQUIRES_ROSBAGS)
