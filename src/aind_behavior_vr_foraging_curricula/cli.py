import os
import typing

import aind_behavior_curriculum
from pydantic import BaseModel, Field, RootModel, SerializeAsAny
from pydantic_settings import (
    BaseSettings,
    CliImplicitFlag,
    CliSubCommand,
)

from . import __version__

TModel = typing.TypeVar("TModel", bound=BaseModel)
TTrainerState = typing.TypeVar("TTrainerState", bound=aind_behavior_curriculum.TrainerState)
TMetrics = typing.TypeVar("TMetrics", bound=aind_behavior_curriculum.Metrics)


class CurriculumCliArgs(BaseSettings):
    data_directory: os.PathLike = Field(description="Path to the session data directory.")
    input_trainer_state: os.PathLike = Field(description="Path to a deserializable trainer state.")
    mute_suggestion: CliImplicitFlag[bool] = Field(default=False, description="Disables the suggestion output")
    output_suggestion: typing.Optional[os.PathLike] = Field(
        default=None,
        description="A path to save the suggestion. If not provided, the suggestion will not be serialized to a file.",
    )


class _AnyRoot(RootModel):
    root: typing.Any


class CurriculumAppCliArgs(BaseSettings, cli_prog_name="curriculum", cli_kebab_case=True):
    run: CliSubCommand[CurriculumCliArgs]
    version: CliSubCommand[_AnyRoot]
    dsl_version: CliSubCommand[_AnyRoot]


class CurriculumSuggestion(BaseModel, typing.Generic[TTrainerState, TMetrics]):
    trainer_state: SerializeAsAny[TTrainerState] = Field(description="The TrainerState suggestion.")
    metrics: SerializeAsAny[TMetrics] = Field(description="The calculated metrics.")
    version: str = Field(default=__version__, description="The version of the curriculum.")
    dsl_version: str = Field(
        default=aind_behavior_curriculum.__version__, description="The version of the curriculum library."
    )
