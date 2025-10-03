import logging
from typing import cast

import aind_behavior_curriculum
from aind_behavior_vr_foraging.task_logic import AindVrForagingTaskLogic
from pydantic_settings import CliApp

from aind_behavior_vr_foraging_curricula.cli import CurriculumAppCliArgs, CurriculumSuggestion
from aind_behavior_vr_foraging_curricula.template import CURRICULUM_VERSION, __test_placeholder
from aind_behavior_vr_foraging_curricula.template.curriculum import s_stage_b


def test_cli(caplog):
    trainer_state, metrics = __test_placeholder.make()
    trainer_state = aind_behavior_curriculum.TrainerState.model_validate_json(trainer_state.model_dump_json())
    metrics = aind_behavior_curriculum.Metrics.model_validate_json(metrics.model_dump_json())
    CliApp.run(
        CurriculumAppCliArgs,
        cli_args=[
            "run",
            "--data-directory",
            "demo",  # import to trigger test mode
            "--input-trainer-state",
            "state.json",
            "--curriculum",
            "template",
        ],
    )
    with caplog.at_level(logging.INFO, logger="aind_behavior_vr_foraging_curricula"):
        msgs = [m.getMessage() for m in caplog.records]
        suggestion = CurriculumSuggestion.model_validate_json(msgs[0])

    assert suggestion is not None

    task = cast(AindVrForagingTaskLogic, suggestion.trainer_state.stage.task)

    assert suggestion.trainer_state.curriculum == trainer_state.curriculum
    assert suggestion.trainer_state.stage == s_stage_b
    assert suggestion.metrics == metrics
    assert task.task_parameters.rng_seed == s_stage_b.task.task_parameters.rng_seed
    assert suggestion.version == CURRICULUM_VERSION
    assert suggestion.dsl_version == aind_behavior_curriculum.__version__
