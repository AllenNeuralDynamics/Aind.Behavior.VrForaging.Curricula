import importlib
import logging
from pathlib import Path

from aind_behavior_curriculum import __version__

from aind_behavior_vr_foraging_curricula import curricula_logger
from tests import REPO_ROOT
from tests.conftest import run_cli_with_args


def test_all_curricula_have_cli(caplog):
    curricula_logger.handlers.clear()  # the stdout pollutes the test

    for module_dir in [
        p
        for p in Path(REPO_ROOT / "src/aind_behavior_vr_foraging_curricula").iterdir()
        if p.is_dir() and not p.name.startswith("_")
    ]:
        module = importlib.import_module(f"aind_behavior_vr_foraging_curricula.{module_dir.name}")
        runner = getattr(module, "run_curriculum", None)
        assert runner is not None, f"Module {module_dir.name} does not have a 'run_curriculum' function."
        with caplog.at_level(logging.INFO, logger="aind_behavior_vr_foraging_curricula"):
            run_cli_with_args(runner, "0.0.0", ["dsl-version"])
        _ = run_cli_with_args(runner, "0.0.0", ["dsl-version"])
        assert any(__version__ == record.getMessage() for record in caplog.records)
