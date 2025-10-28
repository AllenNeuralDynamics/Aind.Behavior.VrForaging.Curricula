from aind_behavior_curriculum import MetricsProvider, Policy, Stage
from aind_behavior_vr_foraging import task_logic
from aind_behavior_vr_foraging.task_logic import AindVrForagingTaskLogic, AindVrForagingTaskParameters

from ..depletion import helpers
from ..depletion.metrics import metrics_from_dataset

# ============================================================
# Stage definition
# ============================================================

s_stage_all_odors_rewarded = Stage(
    name="all_odors_rewarded",
    task=AindVrForagingTaskLogic(
        stage_name="all_odors_rewarded_stops_rate",
        task_parameters=AindVrForagingTaskParameters(
            operation_control=helpers.make_default_operation_control(time_to_collect=99999, velocity_threshold=8),
            environment=task_logic.BlockStructure(
                blocks=[
                    task_logic.Block(
                        environment_statistics=task_logic.EnvironmentStatistics(
                            first_state_occupancy=[0.5, 0.5],
                            transition_matrix=[[0.5, 0.5], [0.5, 0.5]],
                            patches=[
                                helpers.make_graduated_patch(
                                    label="odor_slow", state_index=0, odor_index=1, max_reward_probability=0.9, rate_reward_probability=0.969, rule="ON_CHOICE"
                                ),
                                helpers.make_graduated_patch(
                                    label="odor_fast", state_index=1, odor_index=2, max_reward_probability=0.9, rate_reward_probability=0.9405088708, rule="ON_CHOICE"
                                ),
                            ],
                        ),
                        end_conditions=[],
                    )
                ],
            ),
        ),
    ),
    metrics_provider=MetricsProvider(metrics_from_dataset),
)

s_stage_graduation = Stage(
    name="graduation",
    task=AindVrForagingTaskLogic(
        stage_name="graduation_stops_rate",
        task_parameters=AindVrForagingTaskParameters(
            operation_control=helpers.make_default_operation_control(time_to_collect=99999, velocity_threshold=8),
            environment=task_logic.BlockStructure(
                blocks=[
                    task_logic.Block(
                        environment_statistics=task_logic.EnvironmentStatistics(
                            first_state_occupancy=[0.45, 0.45, 0.1],
                            transition_matrix=[[0.45, 0.45, 0.1], [0.45, 0.45, 0.1], [0.45, 0.45, 0.1]],
                            patches=[
                                helpers.make_graduated_patch(
                                    label="odor_0", state_index=2, odor_index=0, max_reward_probability=0.0, rate_reward_probability=0.0, rule="ON_CHOICE"
                                ),
                               helpers.make_graduated_patch(
                                    label="odor_slow", state_index=0, odor_index=1, max_reward_probability=0.9, rate_reward_probability=0.969, rule="ON_CHOICE"
                                ),
                                helpers.make_graduated_patch(
                                    label="odor_fast", state_index=1, odor_index=2, max_reward_probability=0.9, rate_reward_probability=0.9405088708, rule="ON_CHOICE"
                                ),
                            ],
                        ),
                        end_conditions=[],
                    )
                ],
            ),
        ),
    ),
    metrics_provider=MetricsProvider(metrics_from_dataset),
)
