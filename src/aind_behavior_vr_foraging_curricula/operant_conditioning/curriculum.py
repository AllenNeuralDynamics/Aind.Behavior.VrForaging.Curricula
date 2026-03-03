from typing import Any, Type, TypeVar

import aind_behavior_curriculum
import pydantic
from aind_behavior_curriculum import (
    StageTransition,
    Trainer,
    TrainerState,
    create_curriculum,
)
from aind_behavior_vr_foraging.task_logic import AindVrForagingTaskLogic

from .. import __semver__
from ..cli import CurriculumCliArgs, CurriculumSuggestion
from ..depletion.curriculum import (
    metrics_from_dataset_path,
    st_s_stage_one_odor_no_depletion_s_stage_one_odor_w_depletion_day_0,
    trainer_state_from_file,
)
from ..depletion.metrics import DepletionCurriculumMetrics
from ..depletion.stages import (
    make_s_stage_one_odor_no_depletion,
)
from .stages import make_s_stage_a100_b100_c0, make_s_stage_b100_c0, make_s_stage_reversals

CURRICULUM_NAME = "OperantConditioning"
PKG_LOCATION = ".".join(__name__.split(".")[:-1])

TModel = TypeVar("TModel", bound=pydantic.BaseModel)


# ============================================================
# Curriculum definition
# ============================================================

curriculum_class: Type[aind_behavior_curriculum.Curriculum[AindVrForagingTaskLogic]] = create_curriculum(
    CURRICULUM_NAME, __semver__, (AindVrForagingTaskLogic,), pkg_location=PKG_LOCATION
)
CURRICULUM = curriculum_class()


def st_never(metrics: DepletionCurriculumMetrics) -> bool:
    return False


CURRICULUM.add_stage_transition(
    make_s_stage_one_odor_no_depletion(),
    make_s_stage_a100_b100_c0(),
    StageTransition(st_s_stage_one_odor_no_depletion_s_stage_one_odor_w_depletion_day_0),
)

CURRICULUM.add_stage_transition(
    make_s_stage_a100_b100_c0(),
    make_s_stage_b100_c0(),
    StageTransition(st_never),
)

CURRICULUM.add_stage_transition(
    make_s_stage_b100_c0(),
    make_s_stage_reversals(),
    StageTransition(st_never),
)

# ==============================================================================
# Create a Trainer that uses the curriculum to bootstrap suggestions
# ==============================================================================

TRAINER = Trainer(CURRICULUM)


def run_curriculum(args: CurriculumCliArgs) -> CurriculumSuggestion[TrainerState[Any], Any]:
    metrics: aind_behavior_curriculum.Metrics
    trainer_state = trainer_state_from_file(args.input_trainer_state, TRAINER)
    metrics = metrics_from_dataset_path(args.data_directory, trainer_state)
    trainer_state = TRAINER.evaluate(trainer_state, metrics)
    return CurriculumSuggestion(trainer_state=trainer_state, metrics=metrics, version=__semver__)
