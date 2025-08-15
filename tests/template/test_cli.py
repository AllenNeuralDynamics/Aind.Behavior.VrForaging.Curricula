from typing import cast

import aind_behavior_curriculum
from aind_behavior_vr_foraging.task_logic import AindVrForagingTaskLogic

from aind_behavior_vr_foraging_curricula.template import run_curriculum
from aind_behavior_vr_foraging_curricula.template.__test_placeholder import make as make_placeholder
from aind_behavior_vr_foraging_curricula.template.stages import s_stage_b
from tests import suppress_all_logging
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

    with suppress_all_logging():
        suggestion = run_cli_with_args(dummy_runner, "0.0.0", injected_cli_args)
        trainer_state_expected, metrics_expected = make_placeholder()

    assert suggestion is not None

    task = cast(AindVrForagingTaskLogic, suggestion.trainer_state.stage.task)

    assert suggestion.trainer_state.curriculum == trainer_state_expected.curriculum
    assert suggestion.trainer_state.stage == s_stage_b
    assert suggestion.metrics == metrics_expected
    assert task.task_parameters.rng_seed == s_stage_b.task.task_parameters.rng_seed
    assert suggestion.version == "0.0.0"
    assert suggestion.dsl_version == aind_behavior_curriculum.__version__
