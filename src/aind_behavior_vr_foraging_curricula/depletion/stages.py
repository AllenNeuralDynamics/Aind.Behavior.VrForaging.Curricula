from typing import Callable

from aind_behavior_curriculum import MetricsProvider, Policy, Stage
from aind_behavior_services.task_logic import distributions
from aind_behavior_vr_foraging import task_logic
from aind_behavior_vr_foraging.task_logic import AindVrForagingTaskLogic, AindVrForagingTaskParameters

from . import helpers
from .metrics import DepletionCurriculumMetrics, metrics_from_dataset

# ============================================================
# Policies to update task parameters based on metrics
# ============================================================

# Useful type hints for generic policies
PolicyType = Callable[
    [DepletionCurriculumMetrics, AindVrForagingTaskLogic], AindVrForagingTaskLogic
]  # This should generally work for type hinting


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def p_stochastic_reward(metrics: DepletionCurriculumMetrics, task: AindVrForagingTaskLogic) -> AindVrForagingTaskLogic:
    if metrics.total_water_consumed > 0.75:
        task.task_parameters.environment.blocks[0].environment_statistics.patches[
            0
        ].reward_specification.probability = task_logic.scalar_value(0.9)
    return task


def p_learn_to_run(metrics: DepletionCurriculumMetrics, task: AindVrForagingTaskLogic) -> AindVrForagingTaskLogic:
    if metrics.n_patches_visited > 200:
        patch_gen = task.task_parameters.environment.blocks[0].environment_statistics.patches[0].patch_virtual_sites_generator
        
        #assert isinstance(patch_gen.inter_patch.length_distribution, distributions.ExponentialDistribution)
        #assert isinstance(patch_gen.inter_site.length_distribution, distributions.ExponentialDistribution)
        #assert isinstance(patch_gen.reward_site.length_distribution, distributions.ExponentialDistribution)
        assert patch_gen.inter_site.length_distribution.truncation_parameters is not None
        patch_gen.inter_site.length_distribution.truncation_parameters.min = _clamp(
            patch_gen.inter_site.length_distribution.truncation_parameters.min * 1.5,
            minimum=10,
            maximum=30,
        )
        patch_gen.inter_site.length_distribution.truncation_parameters.max = _clamp(
            patch_gen.inter_site.length_distribution.truncation_parameters.max * 1.5,
            minimum=30,
            maximum=100,
        )
        
        assert patch_gen.inter_patch.length_distribution.truncation_parameters is not None
        patch_gen.inter_patch.length_distribution.truncation_parameters.min = _clamp(
            patch_gen.inter_patch.length_distribution.truncation_parameters.min + 10,
            minimum=20,
            maximum=75,
        )
        patch_gen.inter_patch.length_distribution.truncation_parameters.max = _clamp(
            patch_gen.inter_patch.length_distribution.truncation_parameters.max + 10,
            minimum=30,
            maximum=100,
        )

    #Intersite: min start at 10, maximum at 30. if value <20 = value + 10. If value >30, value + (value – 20). Maximum of 20, 100.  This one is too messy!

    #Odorsite: start = 20; +10 every time rule 1 is met. Maximum of 50.  

    #Interpatch: start min 25, start max 75. x2 both every time rule 1 is met. Maximum of 200,600.

def p_learn_to_stop(metrics: DepletionCurriculumMetrics, task: AindVrForagingTaskLogic) -> AindVrForagingTaskLogic:
    if metrics.n_choices > 100:
        task.task_parameters.updaters[
            task_logic.UpdaterTarget.STOP_VELOCITY_THRESHOLD
        ].parameters.initial_value = _clamp(
            task.task_parameters.updaters[task_logic.UpdaterTarget.STOP_VELOCITY_THRESHOLD].parameters.initial_value
            - 16.6,
            minimum=10,
            maximum=60,
        )
        task.task_parameters.updaters[
            task_logic.UpdaterTarget.STOP_DURATION_OFFSET
        ].parameters.maximum = task.task_parameters.updaters[
            task_logic.UpdaterTarget.STOP_DURATION_OFFSET
        ].parameters.initial_value

        task.task_parameters.updaters[task_logic.UpdaterTarget.REWARD_DELAY_OFFSET].parameters.initial_value = _clamp(
            task.task_parameters.updaters[task_logic.UpdaterTarget.REWARD_DELAY_OFFSET].parameters.initial_value + 0.1,
            minimum=0,
            maximum=0.5,
        )
        task.task_parameters.updaters[
            task_logic.UpdaterTarget.REWARD_DELAY_OFFSET
        ].parameters.maximum = task.task_parameters.updaters[
            task_logic.UpdaterTarget.REWARD_DELAY_OFFSET
        ].parameters.initial_value

        task.task_parameters.updaters[task_logic.UpdaterTarget.STOP_DURATION_OFFSET].parameters.initial_value = _clamp(
            task.task_parameters.updaters[task_logic.UpdaterTarget.STOP_DURATION_OFFSET].parameters.initial_value + 0.1,
            minimum=0,
            maximum=0.5,
        )
    return task


# ============================================================
# Stage definition
# ============================================================

s_stage_one_odor_no_depletion = Stage(
    name="one_odor_no_depletion",
    task=AindVrForagingTaskLogic(
        stage_name="one_odor_no_depletion",
        task_parameters=AindVrForagingTaskParameters(
            updaters={
                task_logic.UpdaterTarget.STOP_DURATION_OFFSET: task_logic.NumericalUpdater(
                    operation=task_logic.NumericalUpdaterOperation.OFFSET,
                    parameters=task_logic.NumericalUpdaterParameters(
                        initial_value=0, on_success=0.003, minimum=0, maximum=0.6
                    ),
                ),
                task_logic.UpdaterTarget.REWARD_DELAY_OFFSET: task_logic.NumericalUpdater(
                    operation=task_logic.NumericalUpdaterOperation.OFFSET,
                    parameters=task_logic.NumericalUpdaterParameters(
                        initial_value=0,
                        on_success=-0.002,
                        minimum=0,
                        maximum=0.4,
                    ),
                ),
                task_logic.UpdaterTarget.STOP_VELOCITY_THRESHOLD: task_logic.NumericalUpdater(
                    operation=task_logic.NumericalUpdaterOperation.GAIN,
                    parameters=task_logic.NumericalUpdaterParameters(
                        initial_value=60,
                        on_success=0.99,
                        minimum=10,
                        maximum=60,
                    ),
                ),
            },
            operation_control=helpers.make_default_operation_control(time_to_collect=99999, velocity_threshold=0),
            environment=task_logic.BlockStructure(
                blocks=[
                    task_logic.Block(
                        environment_statistics=task_logic.EnvironmentStatistics(
                            patches=[
                                task_logic.Patch(
                                    label="odor_0",
                                    state_index=0,
                                    odor_specification=task_logic.OdorSpecification(index=0, concentration=1),
                                    reward_specification=task_logic.RewardSpecification(
                                        amount=task_logic.scalar_value(5),
                                        probability=task_logic.scalar_value(1),
                                        available=task_logic.scalar_value(9999),
                                        reward_function=[],
                                    ),
                                    patch_virtual_sites_generator=task_logic.PatchVirtualSitesGenerator(
                                        inter_patch=helpers.make_interpatch(
                                            length_distribution=helpers.uniform_distribution(100, 150)
                                        ),  # TODO
                                        inter_site=helpers.make_intersite(
                                            length_distribution=helpers.uniform_distribution(15, 25)
                                        ),
                                        reward_site=helpers.make_reward_site(
                                            length_distribution=helpers.normal_distribution(60, 5)
                                        ),
                                    ),
                                )
                            ]
                        ),
                        end_conditions=[],
                    )
                ],
            ),
        ),
    ),
    start_policies=[Policy(x) for x in [p_learn_to_run, p_learn_to_stop, p_stochastic_reward]],
    metrics_provider=MetricsProvider(metrics_from_dataset),
)

s_stage_one_odor_w_depletion_day_0 = Stage(
    name="one_odor_w_depletion_day_0",
    task=AindVrForagingTaskLogic(),
    metrics_provider=MetricsProvider(metrics_from_dataset),
)

s_stage_one_odor_w_depletion_day_1 = Stage(
    name="one_odor_w_depletion_day_1",
    task=AindVrForagingTaskLogic(),
    metrics_provider=MetricsProvider(metrics_from_dataset),
)

s_stage_all_odors_rewarded = Stage(
    name="all_odors_rewarded",
    task=AindVrForagingTaskLogic(),
    metrics_provider=MetricsProvider(metrics_from_dataset),
)

s_stage_graduation = Stage(
    name="graduation",
    task=AindVrForagingTaskLogic(),
    metrics_provider=MetricsProvider(metrics_from_dataset),
)
