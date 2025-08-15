from pathlib import Path

import aind_behavior_curriculum

from ..cli import CurriculumCliArgs, CurriculumSuggestion, make_entry_point
from .curriculum import CURRICULUM_VERSION, TRAINER, metrics_from_dataset_path, trainer_state_from_file


def run_curriculum(args: CurriculumCliArgs):
    metrics: aind_behavior_curriculum.Metrics
    if args.data_directory == Path("demo"):
        from . import __test_placeholder

        trainer_state, metrics = __test_placeholder.make()

    else:
        trainer_state = trainer_state_from_file(args.input_trainer_state)
        metrics = metrics_from_dataset_path(args.data_directory, trainer_state)
        return CurriculumSuggestion(
            trainer_state=trainer_state, metrics=metrics, dsl_version=aind_behavior_curriculum.__version__
        )

    return CurriculumSuggestion(trainer_state=TRAINER.evaluate(trainer_state, metrics), metrics=metrics)


def main():
    runner = make_entry_point(run_curriculum, CURRICULUM_VERSION)
    _ = runner()


if __name__ == "__main__":
    main()
