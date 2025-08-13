import os
import pathlib
from typing import TypeVar, Union

import pydantic

# This curriculum only has 2 stages and a single transition from stage 1 to stage 2
# The first stage has a single policy that update suggestions while stage 1 is active
from aind_behavior_curriculum import (
    Metrics,
    StageTransition,
    Trainer,
    TrainerState,
    create_curriculum,
)
from aind_behavior_vr_foraging.task_logic import AindVrForagingTaskLogic

from . import CURRICULUM_NAME, CURRICULUM_VERSION
from .metrics import VrForagingTemplateMetrics
from .stages import s_stage_a, s_stage_b

TModel = TypeVar("TModel", bound=pydantic.BaseModel)


def model_from_json_file(json_path: os.PathLike | str, model: type[TModel]) -> TModel:
    with open(pathlib.Path(json_path), "r", encoding="utf-8") as file:
        return model.model_validate_json(file.read())


# ============================================================
# Stage transitions
# ============================================================


def st_s_stage_a_s_stage_b(metrics: VrForagingTemplateMetrics) -> bool:
    return metrics.metric1 > 1


# ============================================================
# Curriculum definition
# ============================================================

curriculum_class = create_curriculum(CURRICULUM_NAME, CURRICULUM_VERSION, (AindVrForagingTaskLogic,), __package__)
CURRICULUM = curriculum_class()


CURRICULUM.add_stage_transition(s_stage_a, s_stage_b, StageTransition(st_s_stage_a_s_stage_b))

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
