from typing import Callable

from aind_behavior_services.task_logic import distributions
from aind_behavior_vr_foraging import task_logic
from aind_behavior_vr_foraging.task_logic import AindVrForagingTaskLogic

from . import helpers
from .metrics import DepletionCurriculumMetrics

# ============================================================
# Policies to update task parameters based on metrics
# ============================================================

# Useful type hints for generic policies
PolicyType = Callable[
    [DepletionCurriculumMetrics, AindVrForagingTaskLogic], AindVrForagingTaskLogic
]  # This should generally work for type hinting


def p_stochastic_reward(metrics: DepletionCurriculumMetrics, task: AindVrForagingTaskLogic) -> AindVrForagingTaskLogic:
    if metrics.total_water_consumed > 750:
        task.task_parameters.environment.blocks[0].environment_statistics.patches[
            0
        ].reward_specification.probability = task_logic.scalar_value(0.9)
    return task


def p_learn_to_run(metrics: DepletionCurriculumMetrics, task: AindVrForagingTaskLogic) -> AindVrForagingTaskLogic:
    if metrics.n_reward_sites_travelled > 200:
        patch_gen = (
            task.task_parameters.environment.blocks[0].environment_statistics.patches[0].patch_virtual_sites_generator
        )

        assert isinstance(patch_gen.inter_site.length_distribution, distributions.ExponentialDistribution)
        assert patch_gen.inter_site.length_distribution.truncation_parameters is not None
        patch_gen.inter_site.length_distribution.truncation_parameters.min = helpers.clamp(
            patch_gen.inter_site.length_distribution.truncation_parameters.min * 1.5,
            minimum=10,
            maximum=20,
        )
        patch_gen.inter_site.length_distribution.truncation_parameters.max = helpers.clamp(
            patch_gen.inter_site.length_distribution.truncation_parameters.max * 1.5,
            minimum=30,
            maximum=100,
        )

        assert isinstance(patch_gen.inter_patch.length_distribution, distributions.ExponentialDistribution)
        assert patch_gen.inter_patch.length_distribution.truncation_parameters is not None
        patch_gen.inter_patch.length_distribution.truncation_parameters.min = helpers.clamp(
            patch_gen.inter_patch.length_distribution.truncation_parameters.min * 2,
            minimum=25,
            maximum=200,
        )
        patch_gen.inter_patch.length_distribution.truncation_parameters.max = helpers.clamp(
            patch_gen.inter_patch.length_distribution.truncation_parameters.max * 2,
            minimum=75,
            maximum=600,
        )

        assert isinstance(patch_gen.reward_site.length_distribution, distributions.Scalar)
        patch_gen.reward_site.length_distribution.distribution_parameters.value = helpers.clamp(
            patch_gen.reward_site.length_distribution.distribution_parameters.value + 10,
            minimum=20,
            maximum=50,
        )
        # This should not be needed, but just in case...
        task.task_parameters.environment.blocks[0].environment_statistics.patches[
            0
        ].patch_virtual_sites_generator = patch_gen

    return task


def p_learn_to_stop(metrics: DepletionCurriculumMetrics, task: AindVrForagingTaskLogic) -> AindVrForagingTaskLogic:
    if metrics.n_choices > 150:
        task.task_parameters.updaters[task_logic.UpdaterTarget.STOP_VELOCITY_THRESHOLD].parameters.initial_value = (
            task.task_parameters.updaters[task_logic.UpdaterTarget.STOP_VELOCITY_THRESHOLD].parameters.initial_value
            - 16.6
        )

        task.task_parameters.updaters[task_logic.UpdaterTarget.REWARD_DELAY_OFFSET].parameters.initial_value = (
            task.task_parameters.updaters[task_logic.UpdaterTarget.REWARD_DELAY_OFFSET].parameters.initial_value + 0.1
        )

        task.task_parameters.updaters[task_logic.UpdaterTarget.STOP_DURATION_OFFSET].parameters.initial_value = (
            task.task_parameters.updaters[task_logic.UpdaterTarget.STOP_DURATION_OFFSET].parameters.initial_value + 0.1
        )

    return task
