import os
import typing as t
from pathlib import Path

import aind_behavior_curriculum
from pydantic import BaseModel, Field, RootModel, SerializeAsAny
from pydantic_settings import BaseSettings, CliApp, CliImplicitFlag, CliSubCommand

from . import __version__, curricula_logger

TModel = t.TypeVar("TModel", bound=BaseModel)
TTrainerState = t.TypeVar("TTrainerState", bound=aind_behavior_curriculum.TrainerState)
TMetrics = t.TypeVar("TMetrics", bound=aind_behavior_curriculum.Metrics)


class CurriculumCliArgs(BaseSettings):
    data_directory: os.PathLike = Field(description="Path to the session data directory.")
    input_trainer_state: os.PathLike = Field(description="Path to a deserialized trainer state.")
    mute_suggestion: CliImplicitFlag[bool] = Field(default=False, description="Disables the suggestion output")
    output_suggestion: t.Optional[os.PathLike] = Field(
        default=None,
        description="A path to save the suggestion. If not provided, the suggestion will not be serialized to a file.",
    )


class _AnyRoot(RootModel):
    root: t.Any


class CurriculumAppCliArgs(BaseSettings, cli_prog_name="curriculum", cli_kebab_case=True):
    run: CliSubCommand[CurriculumCliArgs]
    version: CliSubCommand[_AnyRoot]
    dsl_version: CliSubCommand[_AnyRoot]


class CurriculumSuggestion(BaseModel, t.Generic[TTrainerState, TMetrics]):
    trainer_state: SerializeAsAny[TTrainerState] = Field(description="The TrainerState suggestion.")
    metrics: SerializeAsAny[TMetrics] = Field(description="The calculated metrics.")
    version: str = Field(default=__version__, description="The version of the curriculum.")
    dsl_version: str = Field(
        default=aind_behavior_curriculum.__version__, description="The version of the curriculum library."
    )


def make_entry_point(
    runner: t.Callable[[CurriculumCliArgs], CurriculumSuggestion], curriculum_version: str
) -> t.Callable[[], t.Optional[CurriculumSuggestion]]:
    """
    Creates an entry point for the curriculum CLI application.
    """

    def _wrapped() -> t.Optional[CurriculumSuggestion]:
        args = CliApp.run(CurriculumAppCliArgs, cli_exit_on_error=True)

        if args.run is not None:
            try:
                suggestion = runner(args.run)
                suggestion.version = curriculum_version
                suggestion.dsl_version = aind_behavior_curriculum.__version__

                if not args.run.mute_suggestion:
                    curricula_logger.info(suggestion.model_dump_json())

                if args.run.output_suggestion is not None:
                    with open(Path(args.run.output_suggestion) / "suggestion.json", "w", encoding="utf-8") as file:
                        file.write(suggestion.model_dump_json(indent=2))

                return suggestion

            except Exception as e:
                curricula_logger.error(f"Error occurred while running curriculum: {e}")
                raise e

        if args.version:
            curricula_logger.info(curriculum_version)
            return None
        if args.dsl_version:
            curricula_logger.info(aind_behavior_curriculum.__version__)
            return None

        raise RuntimeError("No valid subcommand provided.")

    return _wrapped
