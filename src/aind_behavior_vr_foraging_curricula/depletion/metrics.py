import logging
import os

import pandas as pd
from aind_behavior_curriculum import Metrics
from aind_behavior_vr_foraging.data_contract import dataset as vr_foraging_dataset
from aind_behavior_vr_foraging.task_logic import AindVrForagingTaskLogic
from contraqctor.contract.json import SoftwareEvents
from pydantic import Field, NonNegativeFloat, NonNegativeInt

logger = logging.getLogger(__name__)


class DepletionCurriculumMetrics(Metrics):
    total_water_consumed: NonNegativeFloat = Field(description="Total water (in milliliters) consumed in the session.")

    n_reward_sites_travelled: NonNegativeInt = Field(
        description="Number of reward sites travelled during the session.",
    )

    n_choices: NonNegativeInt = Field(
        description="Total number of choices (i.e. harvest attempts) made by the subject."
    )

    n_patches_visited: NonNegativeInt = Field(description="Total number of patches visited during the session.")

    n_patches_visited_per_patch: dict[NonNegativeInt, NonNegativeInt] = Field(
        description="Total number of patches visited during the session aggregated by patch index."
    )

    last_stop_duration: NonNegativeFloat | None = Field(
        description="Minimum stop duration (in seconds) currently implemented."
    )
    last_reward_site_length: NonNegativeFloat | None = Field(
        description="Length (in cm) of the reward site currently implemented."
    )


def _try_get_datastream_as_dataframe(datastream: SoftwareEvents) -> pd.DataFrame | None:
    try:
        datastream.load()
        return datastream.data
    except FileNotFoundError:
        return None


def metrics_from_dataset(data_directory: os.PathLike) -> DepletionCurriculumMetrics:
    dataset = vr_foraging_dataset(data_directory)

    task_logic = dataset["Behavior"]["InputSchemas"]["TaskLogic"].load().data
    if isinstance(task_logic, dict):
        task_logic = AindVrForagingTaskLogic.model_validate(task_logic)

    total_water_consumed = _try_get_datastream_as_dataframe(dataset["Behavior"]["SoftwareEvents"]["GiveReward"])
    choices = _try_get_datastream_as_dataframe(dataset["Behavior"]["SoftwareEvents"]["ChoiceFeedback"])
    patches = _try_get_datastream_as_dataframe(dataset["Behavior"]["SoftwareEvents"]["ActivePatch"])

    patches_visited = (
        pd.concat(
            [
                choices[["name"]],
                patches.assign(
                    label=pd.json_normalize(patches["data"])["state_index"].values,
                    patch_number=range(1, len(patches) + 1),
                )[["patch_number", "label"]],
            ]
        )
        .sort_index()
        .assign(label=lambda df: df["label"].shift(1), patch_number=lambda df: df["patch_number"].shift(1))
        .loc[lambda df: ~df["name"].isna()]
    )

    n_patches_visited_per_patch = (
        patches_visited.groupby("label").patch_number.nunique().fillna(0).astype(int).to_dict()
    )

    sites_visited = _try_get_datastream_as_dataframe(dataset["Behavior"]["SoftwareEvents"]["ActiveSite"])

    if sites_visited is None:
        reward_sites_travelled = pd.DataFrame()
    else:
        reward_sites_travelled = sites_visited[sites_visited["data"].apply(lambda x: x["label"] == "RewardSite")]

    if len(reward_sites_travelled) > 0:
        last_stop_duration = reward_sites_travelled["data"].iloc[-1]["reward_specification"]["operant_logic"][
            "stop_duration"
        ]
        last_reward_site_length = reward_sites_travelled["data"].iloc[-1]["length"]
    else:
        last_stop_duration = None
        last_reward_site_length = None

    DepletionCurriculumMetrics(
        total_water_consumed=(total_water_consumed["data"].sum() if total_water_consumed is not None else 0.0),
        n_choices=len(choices) if choices is not None else 0,
        n_reward_sites_travelled=len(reward_sites_travelled),
        last_stop_duration=last_stop_duration,
        last_reward_site_length=last_reward_site_length,
        n_patches_visited=sum(n_patches_visited_per_patch.values()),
        n_patches_visited_per_patch={int(k): int(v) for k, v in n_patches_visited_per_patch.items()},
    )
