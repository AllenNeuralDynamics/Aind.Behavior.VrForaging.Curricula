import importlib
import logging
import os
import tempfile

from aind_behavior_curriculum import __version__ as dsl_version
from pydantic_settings import CliApp

from aind_behavior_vr_foraging_curricula import __version__ as version
from aind_behavior_vr_foraging_curricula import curricula_logger
from aind_behavior_vr_foraging_curricula.cli import _KNOWN_CURRICULA, CurriculumAppCliArgs
from aind_behavior_vr_foraging_curricula.template import __test_placeholder


def test_known_curricula(caplog):
    curricula_logger.handlers.clear()  # the stdout pollutes the test
    CliApp.run(CurriculumAppCliArgs, cli_args=["list"])
    with caplog.at_level(logging.INFO, logger="aind_behavior_vr_foraging_curricula"):
        msgs = [m.getMessage() for m in caplog.records]
        recovered = [m.strip(" -") for m in msgs if m.strip().startswith("-")]
        assert set(recovered) == set(_KNOWN_CURRICULA)


def test_version(caplog):
    curricula_logger.handlers.clear()  # the stdout pollutes the test
    CliApp.run(CurriculumAppCliArgs, cli_args=["version"])
    with caplog.at_level(logging.INFO, logger="aind_behavior_vr_foraging_curricula"):
        msgs = [m.getMessage() for m in caplog.records]
        assert msgs == [version]


def test_dsl_version(caplog):
    curricula_logger.handlers.clear()  # the stdout pollutes the test
    CliApp.run(CurriculumAppCliArgs, cli_args=["dsl-version"])
    with caplog.at_level(logging.INFO, logger="aind_behavior_vr_foraging_curricula"):
        msgs = [m.getMessage() for m in caplog.records]
        assert msgs == [dsl_version]


def test_known_curricula_implement_runner():
    for curriculum in _KNOWN_CURRICULA:
        module = importlib.import_module(f"aind_behavior_vr_foraging_curricula.{curriculum}")
        runner = getattr(module, "run_curriculum")
        del module, runner


def test_curriculum_name_inference(caplog):
    curricula_logger.handlers.clear()  # the stdout pollutes the test
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp_file:
        trainer_state, _ = __test_placeholder.make()
        tmp_file.write(trainer_state.model_dump_json(indent=2))
        tmp_file_path = tmp_file.name
    try:
        assert os.path.exists(tmp_file_path)
        CliApp.run(
            CurriculumAppCliArgs, cli_args=["run", "--data-directory", "demo", "--input-trainer-state", tmp_file_path]
        )
        with caplog.at_level(logging.ERROR, logger="aind_behavior_vr_foraging_curricula"):
            error_msgs = [m.getMessage() for m in caplog.records if m.levelno >= logging.ERROR]
            assert len(error_msgs) == 0, f"Unexpected errors: {error_msgs}"
    finally:
        os.unlink(tmp_file_path)
