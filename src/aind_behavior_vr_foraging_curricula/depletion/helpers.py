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
        render_specification=task_logic.RenderSpecification(contrast=0.5),
        label=task_logic.VirtualSiteLabels.INTERPATCH,
        length_distribution=length_distribution,
        treadmill_specification=task_logic.TreadmillSpecification(friction=task_logic.scalar_value(0)),
    )


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))
