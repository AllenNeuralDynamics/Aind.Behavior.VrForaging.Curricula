from aind_behavior_curriculum import MetricsProvider, Policy, Stage
from aind_behavior_vr_foraging import task_logic
from aind_behavior_vr_foraging.task_logic import AindVrForagingTaskLogic, AindVrForagingTaskParameters

from . import helpers
from .metrics import metrics_from_dataset
from .policies import p_learn_to_run, p_learn_to_stop, p_stochastic_reward

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
                                            length_distribution=helpers.exponential_distribution(100, 150)  # TODO
                                        ),
                                        inter_site=helpers.make_intersite(
                                            length_distribution=helpers.exponential_distribution(15, 25)  # TODO
                                        ),
                                        reward_site=helpers.make_reward_site(
                                            length_distribution=task_logic.scalar_value(20)
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
