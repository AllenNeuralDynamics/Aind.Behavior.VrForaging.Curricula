import aind_behavior_curriculum

from aind_behavior_vr_foraging_curricula.template.__main__ import run_curriculum
from aind_behavior_vr_foraging_curricula.template.__test_placeholder import make as make_placeholder
from tests.conftest import run_cli_with_args


def dummy_runner(args):
    return run_curriculum(args)


def test_make_entry_point_with_extra_cli_args():
    injected_cli_args = [
        "run",
        "--data-directory",
        "demo",
        "--input-trainer-state",
        "state.json",
    ]

    suggestion = run_cli_with_args(dummy_runner, "0.0.0", injected_cli_args)
    trainer_state_expected, metrics_expected = make_placeholder()
    assert suggestion is not None
    assert suggestion.trainer_state == trainer_state_expected
    assert suggestion.metrics == metrics_expected
    assert suggestion.version == "0.0.0"
    assert suggestion.dsl_version == aind_behavior_curriculum.__version__
