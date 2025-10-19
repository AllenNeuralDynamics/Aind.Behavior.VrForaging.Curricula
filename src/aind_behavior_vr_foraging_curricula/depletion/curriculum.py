import os
from typing import Any, Type, TypeVar, Union

import aind_behavior_curriculum
import pydantic
from aind_behavior_curriculum import (
    Metrics,
    StageTransition,
    Trainer,
    TrainerState,
    create_curriculum,
)
from aind_behavior_vr_foraging.task_logic import AindVrForagingTaskLogic

from ..cli import CurriculumCliArgs, CurriculumSuggestion, model_from_json_file
from .metrics import DepletionCurriculumMetrics
from .stages import (
    s_stage_all_odors_rewarded,
    s_stage_graduation,
    s_stage_one_odor_no_depletion,
    s_stage_one_odor_w_depletion_day_0,
    s_stage_one_odor_w_depletion_day_1,
)

CURRICULUM_VERSION = "0.1.0"
CURRICULUM_NAME = "Depletion"
PKG_LOCATION = ".".join(__name__.split(".")[:-1])

TModel = TypeVar("TModel", bound=pydantic.BaseModel)


# ============================================================
# Stage transitions
# ============================================================


def st_s_stage_one_odor_no_depletion_s_stage_one_odor_w_depletion_day_0(metrics: DepletionCurriculumMetrics) -> bool:
    if metrics.last_reward_site_length is None:
        raise ValueError("last_reward_site_length is None")
    if metrics.last_stop_duration is None:
        raise ValueError("last_stop_duration is None")
    return (
        (metrics.n_reward_sites_travelled > 200)
        and (metrics.n_choices > 150)
        and (metrics.last_reward_site_length >= 50)
        and (metrics.last_stop_duration >= 0.4)
    )


def st_s_stage_one_odor_w_depletion_day_0_s_stage_one_odor_w_depletion_day_1(
    metrics: DepletionCurriculumMetrics,
) -> bool:
    return metrics.n_patches_visited > 20


def st_s_stage_one_odor_w_depletion_day_1_s_stage_one_odor_w_depletion_day_0(
    metrics: DepletionCurriculumMetrics,
) -> bool:
    return metrics.n_patches_visited <= 20


def st_s_stage_one_odor_w_depletion_day_1_s_stage_all_odors_rewarded(metrics: DepletionCurriculumMetrics) -> bool:
    return metrics.n_patches_visited > 20


def st_s_stage_all_odors_rewarded_s_stage_graduation(metrics: DepletionCurriculumMetrics) -> bool:
    patches = metrics.n_patches_visited_per_patch
    return patches.get(0, 0) > 15 and patches.get(1, 0) > 15


# ============================================================
# Curriculum definition
# ============================================================

curriculum_class: Type[aind_behavior_curriculum.Curriculum[AindVrForagingTaskLogic]] = create_curriculum(
    CURRICULUM_NAME, CURRICULUM_VERSION, (AindVrForagingTaskLogic,), pkg_location=PKG_LOCATION
)
CURRICULUM = curriculum_class()


CURRICULUM.add_stage_transition(
    s_stage_one_odor_no_depletion,
    s_stage_one_odor_w_depletion_day_0,
    StageTransition(st_s_stage_one_odor_no_depletion_s_stage_one_odor_w_depletion_day_0),
)

CURRICULUM.add_stage_transition(
    s_stage_one_odor_w_depletion_day_0,
    s_stage_one_odor_w_depletion_day_1,
    StageTransition(st_s_stage_one_odor_w_depletion_day_0_s_stage_one_odor_w_depletion_day_1),
)

CURRICULUM.add_stage_transition(
    s_stage_one_odor_w_depletion_day_1,
    s_stage_one_odor_w_depletion_day_0,
    StageTransition(st_s_stage_one_odor_w_depletion_day_1_s_stage_one_odor_w_depletion_day_0),
)

CURRICULUM.add_stage_transition(
    s_stage_one_odor_w_depletion_day_1,
    s_stage_all_odors_rewarded,
    StageTransition(st_s_stage_one_odor_w_depletion_day_1_s_stage_all_odors_rewarded),
)

CURRICULUM.add_stage_transition(
    s_stage_all_odors_rewarded, s_stage_graduation, StageTransition(st_s_stage_all_odors_rewarded_s_stage_graduation)
)

# ==============================================================================
# Create a Trainer that uses the curriculum to bootstrap suggestions
# ==============================================================================

TRAINER = Trainer(CURRICULUM)


def trainer_state_from_file(path: Union[str, os.PathLike], trainer: Trainer = TRAINER) -> TrainerState:
    return model_from_json_file(path, trainer.trainer_state_model)


def metrics_from_dataset_path(dataset_path: Union[str, os.PathLike], trainer_state: TrainerState) -> Metrics:
    stage = trainer_state.stage
    if stage is None:
        raise ValueError("Trainer state does not have a stage")
    if stage.metrics_provider is None:
        raise ValueError("Stage does not have a metrics provider")
    metrics_provider = stage.metrics_provider
    return metrics_provider.callable(dataset_path)


def run_curriculum(args: CurriculumCliArgs) -> CurriculumSuggestion[TrainerState[Any], Any]:
    metrics: aind_behavior_curriculum.Metrics
    trainer_state = trainer_state_from_file(args.input_trainer_state)
    metrics = metrics_from_dataset_path(args.data_directory, trainer_state)
    trainer_state = TRAINER.evaluate(trainer_state, metrics)
    return CurriculumSuggestion(trainer_state=trainer_state, metrics=metrics, version=CURRICULUM_VERSION)
