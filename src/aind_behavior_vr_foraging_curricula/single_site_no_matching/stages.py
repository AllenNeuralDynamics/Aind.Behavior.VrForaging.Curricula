from typing import Any, Optional

import aind_behavior_services.task_logic.distributions as distributions
import aind_behavior_vr_foraging.task_logic as vr_task_logic
from aind_behavior_curriculum import MetricsProvider, Policy, Stage
from aind_behavior_vr_foraging.task_logic import AindVrForagingTaskLogic, AindVrForagingTaskParameters

from .metrics import metrics_from_dataset
from .policies import p_learn_to_stop, p_seed_reward_delay


def make_patch(
    label: str,
    state_index: int,
    odor_index: int,
    p_reward: float,
    stop_duration: float = 0.5,
    reward_amount: float = 5.0,
    inter_site_length: float = 15,
    reward_site_length: float = 40,
    inter_patch_min_length: float = 50,
    inter_patch_mean_length: float = 150,
    inter_patch_max_length: float = 500,
    delay: Optional[distributions.Distribution] = None,
):
    if delay is None:
        delay = vr_task_logic.scalar_value(0.5)
    return vr_task_logic.Patch(
        label=label,
        state_index=state_index,
        odor_specification=vr_task_logic.OdorSpecification(index=odor_index, concentration=1),
        patch_terminators=[
            vr_task_logic.PatchTerminatorOnChoice(count=vr_task_logic.scalar_value(1)),
            vr_task_logic.PatchTerminatorOnRejection(count=vr_task_logic.scalar_value(1)),
        ],
        reward_specification=vr_task_logic.RewardSpecification(
            amount=vr_task_logic.scalar_value(reward_amount),
            probability=vr_task_logic.scalar_value(p_reward),
            available=vr_task_logic.scalar_value(999999),
            delay=delay,
            operant_logic=vr_task_logic.OperantLogic(
                is_operant=False,
                stop_duration=stop_duration,
                time_to_collect_reward=100000,
                grace_distance_threshold=10,
            ),
        ),
        patch_virtual_sites_generator=vr_task_logic.PatchVirtualSitesGenerator(
            inter_patch=vr_task_logic.VirtualSiteGenerator(
                render_specification=vr_task_logic.RenderSpecification(contrast=1),
                label=vr_task_logic.VirtualSiteLabels.INTERPATCH,
                length_distribution=distributions.ExponentialDistribution(
                    distribution_parameters=distributions.ExponentialDistributionParameters(
                        rate=1.0 / inter_patch_mean_length
                    ),
                    scaling_parameters=distributions.ScalingParameters(offset=inter_patch_min_length),
                    truncation_parameters=distributions.TruncationParameters(
                        min=inter_patch_min_length,
                        max=inter_patch_max_length,
                    ),
                ),
            ),
            inter_site=vr_task_logic.VirtualSiteGenerator(
                render_specification=vr_task_logic.RenderSpecification(contrast=0.5),
                label=vr_task_logic.VirtualSiteLabels.INTERSITE,
                length_distribution=vr_task_logic.scalar_value(inter_site_length),
            ),
            reward_site=vr_task_logic.VirtualSiteGenerator(
                render_specification=vr_task_logic.RenderSpecification(contrast=0.5),
                label=vr_task_logic.VirtualSiteLabels.REWARDSITE,
                length_distribution=vr_task_logic.scalar_value(reward_site_length),
            ),
        ),
    )


def make_block(
    p_rew: tuple[float, Optional[float], Optional[float]],
    n_min_patches: int = 100,
    block_length_exp_mean: float = 25,
    block_length_max: Optional[float] = None,
    first_state_occupancy: Optional[list[float]] = None,
    make_patch_kwargs: Optional[dict[str, Any]] = None,
) -> vr_task_logic.Block:
    make_patch_kwargs = make_patch_kwargs or {}
    patches = [
        make_patch(
            label="OdorA",
            state_index=0,
            odor_index=0,
            p_reward=p_rew[0],
            **make_patch_kwargs,
        )
    ]
    if p_rew[1] is not None:
        patches.append(
            make_patch(
                label="OdorB",
                state_index=1,
                odor_index=1,
                p_reward=p_rew[1],
                **make_patch_kwargs,
            )
        )
    if p_rew[2] is not None:
        patches.append(
            make_patch(
                label="OdorC",
                state_index=2,
                odor_index=2,
                p_reward=p_rew[2],
                **make_patch_kwargs,
            )
        )

    if first_state_occupancy is None:
        per_p = 1.0 / len(patches)
        first_state_occupancy = [per_p] * len(patches)
    if block_length_max is None:
        block_length_max = n_min_patches + 50
    return vr_task_logic.Block(
        environment_statistics=vr_task_logic.EnvironmentStatistics(
            first_state_occupancy=list(first_state_occupancy),
            transition_matrix=[list(first_state_occupancy) for _ in range(len(patches))],
            patches=patches,
        ),
        end_conditions=[
            vr_task_logic.BlockEndConditionPatchCount(
                value=distributions.ExponentialDistribution(
                    distribution_parameters=distributions.ExponentialDistributionParameters(
                        rate=1 / block_length_exp_mean
                    ),
                    scaling_parameters=distributions.ScalingParameters(offset=n_min_patches),
                    truncation_parameters=distributions.TruncationParameters(min=n_min_patches, max=block_length_max),
                )
            )
        ],
    )


def make_operation_control(velocity_threshold: float) -> vr_task_logic.OperationControl:
    return vr_task_logic.OperationControl(
        movable_spout_control=vr_task_logic.MovableSpoutControl(enabled=False),
        audio_control=vr_task_logic.AudioControl(duration=0.2, frequency=9999),
        odor_control=vr_task_logic.OdorControl(),
        position_control=vr_task_logic.PositionControl(
            frequency_filter_cutoff=5,
            velocity_threshold=velocity_threshold,
        ),
    )


# ============================================================
# Stage definition
# ============================================================


def make_s_learn_to_stop() -> Stage:
    return Stage(
        name="learn_to_stop",
        task=AindVrForagingTaskLogic(
            stage_name="learn_to_stop",
            task_parameters=AindVrForagingTaskParameters(
                rng_seed=None,
                updaters={
                    vr_task_logic.UpdaterTarget.STOP_DURATION_OFFSET: vr_task_logic.NumericalUpdater(
                        operation=vr_task_logic.NumericalUpdaterOperation.OFFSET,
                        parameters=vr_task_logic.NumericalUpdaterParameters(
                            initial_value=0, on_success=0.005, minimum=0, maximum=1.0
                        ),
                    ),
                    vr_task_logic.UpdaterTarget.STOP_VELOCITY_THRESHOLD: vr_task_logic.NumericalUpdater(
                        operation=vr_task_logic.NumericalUpdaterOperation.GAIN,
                        parameters=vr_task_logic.NumericalUpdaterParameters(
                            initial_value=60,
                            on_success=0.995,
                            minimum=8,
                            maximum=60,
                        ),
                    ),
                },
                environment=vr_task_logic.BlockStructure(
                    blocks=[
                        make_block(
                            p_rew=(1, 1, None),
                            n_min_patches=100000,
                            make_patch_kwargs={
                                "inter_patch_min_length": 50,
                                "inter_patch_mean_length": 120,
                                "inter_patch_max_length": 150,
                                "inter_site_length": 15,
                                "reward_site_length": 40,
                            },
                        ),
                    ],
                    sampling_mode="Sequential",
                ),
                operation_control=make_operation_control(velocity_threshold=60),
            ),
        ),
        start_policies=[Policy(x) for x in [p_learn_to_stop]],
        metrics_provider=MetricsProvider(metrics_from_dataset),
    )


def make_s_learn_to_choose() -> Stage:
    """High-contrast discrimination stage. Two odors, alternating blocks with
    p_rew=(0.9, 0.1) and (0.1, 0.9). REWARD_DELAY_OFFSET ramps 0 -> 0.3 s within
    session; carried forward across S2 sessions via p_seed_reward_delay."""
    _make_patch_kwargs = {
        "inter_patch_min_length": 30,
        "inter_patch_mean_length": 60,
        "inter_patch_max_length": 190,
        "inter_site_length": 15,
        "reward_site_length": 50,
        "reward_amount": 5.0,
        "stop_duration": 1.5,
    }
    return Stage(
        name="learn_to_choose",
        task=AindVrForagingTaskLogic(
            stage_name="learn_to_choose",
            task_parameters=AindVrForagingTaskParameters(
                rng_seed=None,
                updaters={
                    vr_task_logic.UpdaterTarget.REWARD_DELAY_OFFSET: vr_task_logic.NumericalUpdater(
                        operation=vr_task_logic.NumericalUpdaterOperation.OFFSET,
                        parameters=vr_task_logic.NumericalUpdaterParameters(
                            initial_value=0, on_success=0.002, minimum=0, maximum=0.3
                        ),
                    ),
                },
                environment=vr_task_logic.BlockStructure(
                    blocks=[
                        make_block(
                            p_rew=(0.9, 0.1, None),
                            n_min_patches=40,
                            block_length_exp_mean=25,
                            block_length_max=90,
                            make_patch_kwargs=_make_patch_kwargs,
                        ),
                        make_block(
                            p_rew=(0.1, 0.9, None),
                            n_min_patches=40,
                            block_length_exp_mean=25,
                            block_length_max=90,
                            make_patch_kwargs=_make_patch_kwargs,
                        ),
                    ],
                    sampling_mode="Random",
                ),
                operation_control=make_operation_control(velocity_threshold=8),
            ),
        ),
        start_policies=[Policy(p_seed_reward_delay)],
        metrics_provider=MetricsProvider(metrics_from_dataset),
    )


THREE_VALUE_REWARD_PROBABILITIES: tuple[float, ...] = (0.10, 0.40, 0.70)
GRADUATED_REWARD_PROBABILITIES: tuple[float, ...] = (0.10, 0.30, 0.50, 0.70, 0.90)
# Applied to both the 3-value and 5-value grids: skip (p_A, p_B) pairs whose sum is
# below this floor. With 0.4 the only excluded combo is (0.1, 0.1).
GRADUATED_MIN_PROBABILITY_SUM: float = 0.4


def make_s_three_value_grid() -> Stage:
    """Three-odor intermediate stage. Reward-yielding odors A and B sampled per-block
    uncoupled from {0.1, 0.4, 0.7}; distractor odor C always 0%. Occupancy
    [A=0.475, B=0.475, C=0.05]. REWARD_DELAY_OFFSET ramps 0 -> 1.5 s within session,
    carried forward via p_seed_reward_delay."""
    _make_patch_kwargs = {
        "inter_patch_min_length": 30,
        "inter_patch_mean_length": 60,
        "inter_patch_max_length": 190,
        "inter_site_length": 15,
        "reward_site_length": 50,
        "reward_amount": 5.0,
        "stop_duration": 1.5,
    }

    blocks = [
        make_block(
            p_rew=(p_a, p_b, 0.0),
            n_min_patches=40,
            block_length_exp_mean=25,
            block_length_max=90,
            first_state_occupancy=[0.475, 0.475, 0.05],
            make_patch_kwargs=_make_patch_kwargs,
        )
        for p_a in THREE_VALUE_REWARD_PROBABILITIES
        for p_b in THREE_VALUE_REWARD_PROBABILITIES
        if p_a + p_b >= GRADUATED_MIN_PROBABILITY_SUM
    ]

    return Stage(
        name="three_value_grid",
        task=AindVrForagingTaskLogic(
            stage_name="three_value_grid",
            task_parameters=AindVrForagingTaskParameters(
                rng_seed=None,
                updaters={
                    vr_task_logic.UpdaterTarget.REWARD_DELAY_OFFSET: vr_task_logic.NumericalUpdater(
                        operation=vr_task_logic.NumericalUpdaterOperation.OFFSET,
                        parameters=vr_task_logic.NumericalUpdaterParameters(
                            initial_value=0, on_success=0.005, minimum=0, maximum=1.5
                        ),
                    ),
                },
                environment=vr_task_logic.BlockStructure(
                    blocks=blocks,
                    sampling_mode="Random",
                ),
                operation_control=make_operation_control(velocity_threshold=8),
            ),
        ),
        start_policies=[Policy(p_seed_reward_delay)],
        metrics_provider=MetricsProvider(metrics_from_dataset),
    )


def _graduated_reward_delay(max_delay: float = 5.25, char: float = 2.0) -> distributions.Distribution:
    return distributions.ExponentialDistribution(
        distribution_parameters=distributions.ExponentialDistributionParameters(rate=1.0 / char),
        scaling_parameters=distributions.ScalingParameters(offset=0.5),
        truncation_parameters=distributions.TruncationParameters(min=0.5, max=max_delay),
    )


def make_s_graduated_narrow_delay() -> Stage:
    """Variable-delay preview of the graduated stage. Same patches and grid as
    `make_s_graduated_stage`, but the base reward-delay distribution is truncated
    at 5.25 s (mean ~2.0 s, matching S3's end-of-session effective delay). A small
    REWARD_DELAY_OFFSET updater ramps 0 -> 0.1 within each session so the effective
    mean climbs to ~2.1 s by end-of-session, matching S5's fixed mean. No start
    policies: every S4 session ramps fresh from offset 0 (avoids the cross-stage
    base mismatch with S3's scalar-base offset)."""
    _make_patch_kwargs = {
        "inter_patch_min_length": 30,
        "inter_patch_mean_length": 60,
        "inter_patch_max_length": 190,
        "inter_site_length": 15,
        "reward_site_length": 50,
        "reward_amount": 5.0,
        "stop_duration": 1.5,
        "delay": _graduated_reward_delay(max_delay=5.25, char=1.5),
    }

    blocks = [
        make_block(
            p_rew=(p_a, p_b, 0.0),
            n_min_patches=40,
            block_length_exp_mean=25,
            block_length_max=90,
            first_state_occupancy=[0.475, 0.475, 0.05],
            make_patch_kwargs=_make_patch_kwargs,
        )
        for p_a in GRADUATED_REWARD_PROBABILITIES
        for p_b in GRADUATED_REWARD_PROBABILITIES
        if p_a + p_b >= GRADUATED_MIN_PROBABILITY_SUM
    ]

    return Stage(
        name="graduated_narrow_delay",
        task=AindVrForagingTaskLogic(
            stage_name="graduated_narrow_delay",
            task_parameters=AindVrForagingTaskParameters(
                rng_seed=None,
                updaters={
                    vr_task_logic.UpdaterTarget.REWARD_DELAY_OFFSET: vr_task_logic.NumericalUpdater(
                        operation=vr_task_logic.NumericalUpdaterOperation.OFFSET,
                        parameters=vr_task_logic.NumericalUpdaterParameters(
                            initial_value=0, on_success=0.001, minimum=0, maximum=0.21
                        ),
                    ),
                },
                environment=vr_task_logic.BlockStructure(
                    blocks=blocks,
                    sampling_mode="Random",
                ),
                operation_control=make_operation_control(velocity_threshold=8),
            ),
        ),
        metrics_provider=MetricsProvider(metrics_from_dataset),
    )


def make_s_graduated_stage() -> Stage:
    _graduated_make_patch_kwargs = {
        "inter_patch_min_length": 30,
        "inter_patch_mean_length": 60,
        "inter_patch_max_length": 190,
        "inter_site_length": 15,
        "reward_site_length": 50,
        "reward_amount": 5.0,
        "stop_duration": 1.5,
        "delay": _graduated_reward_delay(),
    }

    blocks = [
        make_block(
            p_rew=(p_a, p_b, 0.0),
            n_min_patches=40,
            block_length_exp_mean=25,
            block_length_max=90,
            first_state_occupancy=[0.475, 0.475, 0.05],
            make_patch_kwargs=_graduated_make_patch_kwargs,
        )
        for p_a in GRADUATED_REWARD_PROBABILITIES
        for p_b in GRADUATED_REWARD_PROBABILITIES
        if p_a + p_b >= GRADUATED_MIN_PROBABILITY_SUM
    ]

    return Stage(
        name="graduated_stage",
        task=AindVrForagingTaskLogic(
            stage_name="graduated_stage",
            task_parameters=AindVrForagingTaskParameters(
                rng_seed=None,
                environment=vr_task_logic.BlockStructure(
                    blocks=blocks,
                    sampling_mode="Random",
                ),
                operation_control=make_operation_control(velocity_threshold=8),
            ),
        ),
        metrics_provider=MetricsProvider(metrics_from_dataset),
    )
