from aind_behavior_curriculum import Stage
from aind_behavior_vr_foraging import task_logic as vr_task_logic
from aind_behavior_vr_foraging.task_logic import (
    AindVrForagingTaskLogic,
    AindVrForagingTaskParameters,
)

from ..depletion import helpers
from ..depletion.metrics import metrics_from_dataset


# ============================================================
# Stage definition
# ============================================================
def make_default_operation_control(
    velocity_threshold: float,
) -> vr_task_logic.OperationControl:
    return vr_task_logic.OperationControl(
        movable_spout_control=vr_task_logic.MovableSpoutControl(
            enabled=False,
        ),
        audio_control=vr_task_logic.AudioControl(duration=0.2, frequency=9999),
        odor_control=vr_task_logic.OdorControl(valve_max_open_time=10000),
        position_control=vr_task_logic.PositionControl(
            frequency_filter_cutoff=5,
            velocity_threshold=velocity_threshold,
        ),
    )


def make_patch(
    label: str,
    state_index: int,
    odor_index: int,
    p_reward: float,
    reward_amount: float = 5.0,
    stop_duration: float = 0.5,
    delay_to_reward: float = 0.5,
):
    return vr_task_logic.Patch(
        label=label,
        state_index=state_index,
        odor_specification=vr_task_logic.OdorSpecification(index=odor_index, concentration=1),
        reward_specification=vr_task_logic.RewardSpecification(
            amount=vr_task_logic.scalar_value(reward_amount),
            probability=vr_task_logic.scalar_value(p_reward),
            available=vr_task_logic.scalar_value(999999),
            delay=vr_task_logic.scalar_value(delay_to_reward),
            operant_logic=vr_task_logic.OperantLogic(
                is_operant=False,
                stop_duration=vr_task_logic.scalar_value(stop_duration),
                time_to_collect_reward=100000,
                grace_distance_threshold=10,
            ),
        ),
        patch_virtual_sites_generator=vr_task_logic.PatchVirtualSitesGenerator(
            inter_patch=vr_task_logic.VirtualSiteGenerator(
                render_specification=vr_task_logic.RenderSpecification(contrast=1),
                label=vr_task_logic.VirtualSiteLabels.INTERPATCH,
                length_distribution=vr_task_logic.distributions.ExponentialDistribution(
                    distribution_parameters=vr_task_logic.distributions.ExponentialDistributionParameters(
                        rate=1.0 / 40
                    ),
                    truncation_parameters=vr_task_logic.distributions.TruncationParameters(
                        min=200,
                        max=400,
                    ),
                    scaling_parameters=vr_task_logic.distributions.ScalingParameters(offset=200),
                ),
            ),
            inter_site=vr_task_logic.VirtualSiteGenerator(
                render_specification=vr_task_logic.RenderSpecification(contrast=0.5),
                label=vr_task_logic.VirtualSiteLabels.INTERSITE,
                length_distribution=vr_task_logic.scalar_value(40.0),
                treadmill_specification=None,
            ),
            reward_site=vr_task_logic.VirtualSiteGenerator(
                render_specification=vr_task_logic.RenderSpecification(contrast=0.5),
                label=vr_task_logic.VirtualSiteLabels.REWARDSITE,
                length_distribution=vr_task_logic.scalar_value(40.0),
            ),
        ),
        patch_terminators=[vr_task_logic.PatchTerminatorOnChoice(count=1)],
    )


def make_environment(
    reward_probability: tuple[float, float, float],
    state_occupancy: tuple[float, float, float],
) -> vr_task_logic.EnvironmentStatistics:
    _state_occupancy = list(state_occupancy)
    _state_occupancy = [x / sum(_state_occupancy) for x in _state_occupancy]
    patch_labels = ["A", "B", "C"]
    patches = [
        make_patch(
            label=patch_labels[i],
            state_index=i,
            odor_index=i,
            p_reward=reward_probability[i],
        )
        for i in range(3)
    ]
    environment_statistics = vr_task_logic.EnvironmentStatistics(
        first_state_occupancy=_state_occupancy,
        transition_matrix=[_state_occupancy for _ in range(len(_state_occupancy))],
        patches=patches,
    )
    return environment_statistics


def make_s_stage_a100_b100_c0() -> Stage:
    env = make_environment(reward_probability=(1.0, 1.0, 0.0), state_occupancy=(0.2, 0.4, 0.4))
    task_logic = AindVrForagingTaskLogic(
        task_parameters=AindVrForagingTaskParameters(
            rng_seed=None,
            environment=vr_task_logic.BlockStructure(
                blocks=[vr_task_logic.Block(environment_statistics=env, end_conditions=[])],
            ),
            operation_control=helpers.make_default_operation_control(velocity_threshold=8),
        ),
        stage_name="stage_a100_b100_c0",
    )
    return Stage(
        name="stage_a100_b100_c0",
        task=task_logic,
        metrics_provider=metrics_from_dataset,
    )


def make_s_stage_b100_c0() -> Stage:
    env = make_environment(reward_probability=(1.0, 1.0, 0.0), state_occupancy=(0.0, 0.5, 0.5))
    task_logic = AindVrForagingTaskLogic(
        task_parameters=AindVrForagingTaskParameters(
            rng_seed=None,
            environment=vr_task_logic.BlockStructure(
                blocks=[vr_task_logic.Block(environment_statistics=env, end_conditions=[])],
            ),
            operation_control=helpers.make_default_operation_control(velocity_threshold=8),
        ),
        stage_name="stage_b100_c0",
    )
    return Stage(
        name="stage_b100_c0",
        task=task_logic,
        metrics_provider=metrics_from_dataset,
    )


def _make_block_end_condition() -> vr_task_logic.BlockEndCondition:
    return vr_task_logic.BlockEndConditionPatchCount(
        value=vr_task_logic.distributions.ExponentialDistribution(
            distribution_parameters=vr_task_logic.distributions.ExponentialDistributionParameters(rate=1 / 10),
            scaling_parameters=vr_task_logic.distributions.ScalingParameters(offset=40),
            truncation_parameters=vr_task_logic.distributions.TruncationParameters(min=40, max=80),
        )
    )


def make_s_stage_reversals() -> Stage:
    env_high = make_environment(reward_probability=(1.0, 1.0, 0.0), state_occupancy=(0.0, 0.5, 0.5))
    env_low = make_environment(reward_probability=(1.0, 0.0, 1.0), state_occupancy=(0.0, 0.5, 0.5))
    task_logic = AindVrForagingTaskLogic(
        task_parameters=AindVrForagingTaskParameters(
            rng_seed=None,
            environment=vr_task_logic.BlockStructure(
                sampling_mode="Sequential",
                blocks=[
                    vr_task_logic.Block(
                        environment_statistics=env_high,
                        end_conditions=[_make_block_end_condition()],
                    ),
                    vr_task_logic.Block(
                        environment_statistics=env_low,
                        end_conditions=[_make_block_end_condition()],
                    ),
                ],
            ),
            operation_control=helpers.make_default_operation_control(velocity_threshold=8),
        ),
        stage_name="stage_reversals",
    )
    return Stage(
        name="stage_reversals",
        task=task_logic,
        metrics_provider=metrics_from_dataset,
    )
