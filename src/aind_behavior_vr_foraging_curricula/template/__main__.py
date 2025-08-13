from pathlib import Path

import aind_behavior_curriculum
from pydantic_settings import CliApp

from .. import logger
from ..cli import CurriculumAppCliArgs, CurriculumCliArgs, CurriculumSuggestion
from . import CURRICULUM_VERSION
from .curriculum import metrics_from_dataset_path, trainer_state_from_file


def run_curriculum(args: CurriculumCliArgs):
    if not args.data_directory == Path("demo"):
        # This is just for tests
        trainer_state = trainer_state_from_file(args.input_trainer_state)
        metrics = metrics_from_dataset_path(args.data_directory, trainer_state)
    else:
        # This is a demo mode for unittest only
        from .curriculum import (
            TRAINER,
            Policy,
            s_stage_a,
        )
        from .metrics import VrForagingTemplateMetrics
        from .stages import p_set_mode_from_metric1

        trainer_state = TRAINER.create_trainer_state(
            stage=s_stage_a,
            is_on_curriculum=True,
            active_policies=tuple([Policy(x) for x in [p_set_mode_from_metric1]]),
        )
        metrics = VrForagingTemplateMetrics(metric1=50, metric2_history=[1.0, 2.0, 3.0])

    suggestion = CurriculumSuggestion(trainer_state=TRAINER.evaluate(trainer_state, metrics), metrics=metrics)

    # Outputs
    if not args.mute_suggestion:
        logger.info(suggestion.model_dump_json())

    if args.output_suggestion is not None:
        with open(Path(args.output_suggestion) / "suggestion.json", "w", encoding="utf-8") as file:
            file.write(suggestion.model_dump_json(indent=2))


def main():
    args = CliApp.run(CurriculumAppCliArgs)
    if args.run is not None:
        run_curriculum(args.run)
    if args.version:
        logger.info(CURRICULUM_VERSION)
    if args.dsl_version:
        logger.info(aind_behavior_curriculum.__version__)


if __name__ == "__main__":
    main()
