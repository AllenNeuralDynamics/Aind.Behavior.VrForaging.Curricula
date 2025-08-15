from unittest.mock import patch

from pydantic_settings import CliApp

from aind_behavior_vr_foraging_curricula.cli import make_entry_point


def run_cli_with_args(dummy_runner, version, injected_cli_args):
    """
    Helper to patch CliApp.run and call make_entry_point with injected CLI args.
    """
    original_run = CliApp.run

    def patched_run(cls, *args, **kwargs):
        existing_args = kwargs.get("cli_args", [])
        combined_args = existing_args + injected_cli_args
        kwargs["cli_args"] = combined_args
        return original_run(cls, *args, **kwargs)

    with patch("aind_behavior_vr_foraging_curricula.cli.CliApp.run", side_effect=patched_run):
        entry_point = make_entry_point(dummy_runner, version)
        return entry_point()
