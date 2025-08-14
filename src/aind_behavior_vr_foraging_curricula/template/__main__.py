from pathlib import Path

import aind_behavior_curriculum

from ..cli import CurriculumCliArgs, CurriculumSuggestion, make_entry_point
from .curriculum import CURRICULUM_VERSION, metrics_from_dataset_path, trainer_state_from_file


def run_curriculum(args: CurriculumCliArgs):
    if not args.data_directory == Path("demo"):
        trainer_state = trainer_state_from_file(args.input_trainer_state)
        metrics = metrics_from_dataset_path(args.data_directory, trainer_state)
        return CurriculumSuggestion(
            trainer_state=trainer_state, metrics=metrics, dsl_version=aind_behavior_curriculum.__version__
        )
    else:
        # This is a demo mode for unittest only
        from aind_behavior_curriculum import Policy

        from .curriculum import (
            TRAINER,
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

    return suggestion


def main():
    runner = make_entry_point(run_curriculum, CURRICULUM_VERSION)
    _ = runner()


if __name__ == "__main__":
    main()
