"""Test module for CLI functionality."""

from unittest.mock import patch

from pydantic_settings import CliApp

from aind_behavior_vr_foraging_curricula.cli import make_entry_point
from aind_behavior_vr_foraging_curricula.template.__main__ import (
    run_curriculum,
)


def dummy_runner(args):
    """Dummy runner function for testing."""
    return run_curriculum(args)


def test_make_entry_point_with_extra_cli_args():
    """Test make_entry_point with additional CLI arguments."""
    injected_cli_args = [
        "run",
        "--data-directory",
        "demo",
        "--input-trainer-state",
        "state.json",
    ]

    def patched_run(cls, *args, **kwargs):
        existing_args = kwargs.get("cli_args", [])
        combined_args = existing_args + injected_cli_args
        kwargs["cli_args"] = combined_args
        return original_run(cls, *args, **kwargs)

    original_run = CliApp.run

    with patch("aind_behavior_vr_foraging_curricula.cli.CliApp.run", side_effect=patched_run):
        entry_point = make_entry_point(dummy_runner, "0.0.0")
        trainer_state = entry_point()
        assert trainer_state is not None
