from aind_behavior_services.task_logic import distributions
from aind_behavior_vr_foraging import task_logic


def make_default_operation_control(time_to_collect: float, velocity_threshold: float) -> task_logic.OperationControl:
    return task_logic.OperationControl(
        movable_spout_control=task_logic.MovableSpoutControl(
            time_to_collect_after_reward=time_to_collect,
        ),
        audio_control=task_logic.AudioControl(duration=0.2, frequency=9999),
        odor_control=task_logic.OdorControl(valve_max_open_time=10),
        position_control=task_logic.PositionControl(
            frequency_filter_cutoff=5,
            velocity_threshold=velocity_threshold,
        ),
    )


def operant_logic(stop_duration: float = 0.5, is_operant: bool = False):
    return task_logic.OperantLogic(
        is_operant=is_operant,
        stop_duration=stop_duration,
        time_to_collect_reward=100000,
        grace_distance_threshold=10,
    )


def normal_distribution(
    mean: float, standard_deviation: float, minimum: float = 0, maximum: float = 9999999
) -> distributions.NormalDistribution:
    return distributions.NormalDistribution(
        distribution_parameters=distributions.NormalDistributionParameters(mean=mean, std=standard_deviation),
        truncation_parameters=distributions.TruncationParameters(min=minimum, max=maximum, is_truncated=True),
        scaling_parameters=distributions.ScalingParameters(scale=1.0, offset=0.0),
    )


def uniform_distribution(minimum: float, maximum: float) -> distributions.UniformDistribution:
    return distributions.UniformDistribution(
        distribution_parameters=distributions.UniformDistributionParameters(min=minimum, max=maximum)
    )


def exponential_distribution(
    rate: float, minimum: float = 0, maximum: float = 9999999
) -> distributions.ExponentialDistribution:
    return distributions.ExponentialDistribution(
        distribution_parameters=distributions.ExponentialDistributionParameters(rate=rate),
        truncation_parameters=distributions.TruncationParameters(min=minimum, max=maximum, is_truncated=True),
    )


def make_reward_site(length_distribution: distributions.Distribution) -> task_logic.VirtualSiteGenerator:
    return task_logic.VirtualSiteGenerator(
        render_specification=task_logic.RenderSpecification(contrast=0.5),
        label=task_logic.VirtualSiteLabels.REWARDSITE,
        length_distribution=length_distribution,
        treadmill_specification=task_logic.TreadmillSpecification(friction=task_logic.scalar_value(0)),
    )


def make_intersite(length_distribution: distributions.Distribution) -> task_logic.VirtualSiteGenerator:
    return task_logic.VirtualSiteGenerator(
        render_specification=task_logic.RenderSpecification(contrast=0.5),
        label=task_logic.VirtualSiteLabels.INTERSITE,
        length_distribution=length_distribution,
        treadmill_specification=task_logic.TreadmillSpecification(friction=task_logic.scalar_value(0)),
    )


def make_interpatch(length_distribution: distributions.Distribution) -> task_logic.VirtualSiteGenerator:
    return task_logic.VirtualSiteGenerator(
        render_specification=task_logic.RenderSpecification(contrast=1),
        label=task_logic.VirtualSiteLabels.INTERPATCH,
        length_distribution=length_distribution,
        treadmill_specification=task_logic.TreadmillSpecification(friction=task_logic.scalar_value(0)),
    )


def make_virtualsites(
    rewardsite: float = 50,
    interpatch_min: float = 200,
    interpatch_max: float = 600,
    intersite_min: float = 20,
    intersite_max: float = 100,
):
    return task_logic.PatchVirtualSitesGenerator(
        inter_patch=make_interpatch(
            length_distribution=exponential_distribution(rate=0.01, minimum=interpatch_min, maximum=interpatch_max)
        ),
        inter_site=make_intersite(
            length_distribution=exponential_distribution(rate=0.05, minimum=intersite_min, maximum=intersite_max)
        ),
        reward_site=make_reward_site(length_distribution=task_logic.scalar_value(rewardsite)),
    )


def ExponentialProbabilityRewardCount(
    amount_drop: int = 5,
    maximum_p: float = 0.9,
    available_water: int = 50,
    c: float = -0.9,
    stop_duration: float = 0.0,
    rule: str = "ON_REWARD",
):
    reward_function = task_logic.PatchRewardFunction(
        available=task_logic.ClampedRateFunction(minimum=0, maximum=maximum_p, rate=task_logic.scalar_value(c)),
        rule=task_logic.RewardFunctionRule[rule],
    )

    reset_function = task_logic.OnThisPatchEntryRewardFunction(
        available=task_logic.SetValueFunction(value=task_logic.scalar_value(available_water))
    )

    agent = task_logic.RewardSpecification(
        operant_logic=operant_logic(stop_duration=stop_duration, is_operant=False),
        delay=normal_distribution(0.25, 0.15, 0.0, 0.75),
        amount=task_logic.scalar_value(value=amount_drop),
        probability=task_logic.scalar_value(maximum_p),
        available=task_logic.scalar_value(available_water),
        reward_function=[reset_function, reward_function],
    )

    return agent


def ExponentialProbabilityReward(
    amount_drop: int = 5,
    available_water: int = 50,
    c=-0.9,
    maximum_p=0.9,
    stop_duration: float = 0.5,
    delay_mean: float = 0.5,
    rule="ON_REWARD",
):
    reward_function = task_logic.PatchRewardFunction(
        probability=task_logic.ClampedMultiplicativeRateFunction(
            minimum=0, maximum=maximum_p, rate=task_logic.scalar_value(c)
        ),
        rule=task_logic.RewardFunctionRule[rule],
    )

    reset_function = task_logic.OnThisPatchEntryRewardFunction(
        probability=task_logic.SetValueFunction(value=task_logic.scalar_value(maximum_p))
    )

    agent = task_logic.RewardSpecification(
        operant_logic=operant_logic(stop_duration=stop_duration, is_operant=False),
        delay=normal_distribution(delay_mean, 0.15, 0.0, 1),
        amount=task_logic.scalar_value(value=amount_drop),
        probability=task_logic.scalar_value(maximum_p),
        available=task_logic.scalar_value(available_water),
        reward_function=[reset_function, reward_function],
    )
    return agent


def CountUntilDepleted(
    available_water: int = 21,
    max_p: float = 0.9,
    amount_drop: int = 5,
    stop_duration: float = 0.5,
    rule: str = "ON_REWARD",
):
    reward_function = task_logic.PatchRewardFunction(
        available=task_logic.ClampedRateFunction(
            rate=task_logic.scalar_value(-amount_drop), minimum=0, maximum=available_water
        ),
        rule=task_logic.RewardFunctionRule[rule],
    )

    reset_function = task_logic.OnThisPatchEntryRewardFunction(
        available=task_logic.SetValueFunction(value=task_logic.scalar_value(available_water))
    )

    agent = task_logic.RewardSpecification(
        operant_logic=operant_logic(stop_duration=stop_duration, is_operant=False),
        delay=normal_distribution(0.0, 0.15, 0.0, 0.75),
        amount=task_logic.scalar_value(value=amount_drop),
        probability=task_logic.scalar_value(max_p),
        available=task_logic.scalar_value(available_water),
        reward_function=[reset_function, reward_function],
    )

    return agent


def make_graduated_patch(
    label: str,
    state_index: int,
    odor_index: int,
    max_reward_probability: float = 0.9,
    rate_reward_probability: float = 0.8795015081718721,
    reward_amount: float = 5.0,
    reward_available: float = 9999,
    stop_duration: float = 0.5,
    delay_mean: float = 0.5,
    rule="ON_REWARD",
):
    reward_function = task_logic.PatchRewardFunction(
        probability=task_logic.ClampedMultiplicativeRateFunction(
            minimum=0, maximum=max_reward_probability, rate=task_logic.scalar_value(rate_reward_probability)
        ),
        rule=task_logic.RewardFunctionRule[rule],
    )

    reset_function = task_logic.OnThisPatchEntryRewardFunction(
        probability=task_logic.SetValueFunction(value=task_logic.scalar_value(max_reward_probability))
    )

    agent = task_logic.RewardSpecification(
        operant_logic=operant_logic(stop_duration=stop_duration, is_operant=False),
        delay=normal_distribution(delay_mean, 0.15, 0.0, 1),
        amount=task_logic.scalar_value(value=reward_amount),
        probability=task_logic.scalar_value(max_reward_probability),
        available=task_logic.scalar_value(reward_available),
        reward_function=[reset_function, reward_function],
    )

    return task_logic.Patch(
        label=label,
        state_index=state_index,
        odor_specification=task_logic.OdorSpecification(index=odor_index, concentration=1),
        reward_specification=agent,
        patch_virtual_sites_generator=make_virtualsites(),
    )


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))
