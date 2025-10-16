from typing import Any, cast

import pytest
from aind_behavior_curriculum import Curriculum, Trainer, TrainerState
from aind_behavior_vr_foraging import task_logic
from aind_behavior_vr_foraging.task_logic import AindVrForagingTaskLogic

from aind_behavior_vr_foraging_curricula.depletion import CURRICULUM, TRAINER
from aind_behavior_vr_foraging_curricula.depletion.curriculum import (
    st_s_stage_one_odor_no_depletion_s_stage_one_odor_w_depletion_day_0,
    st_s_stage_one_odor_w_depletion_day_1_s_stage_one_odor_w_depletion_day_0,
    st_s_stage_one_odor_w_depletion_day_0_s_stage_one_odor_w_depletion_day_1,
    st_s_stage_one_odor_w_depletion_day_1_s_stage_all_odors_rewarded,
    st_s_stage_all_odors_rewarded_s_stage_graduation,
)

from aind_behavior_vr_foraging_curricula.depletion.stages import (
    s_stage_all_odors_rewarded,
    s_stage_graduation,
    s_stage_one_odor_no_depletion,
    s_stage_one_odor_w_depletion_day_0,
    s_stage_one_odor_w_depletion_day_1,
)

from aind_behavior_vr_foraging_curricula.depletion.metrics import DepletionCurriculumMetrics
from aind_behavior_vr_foraging_curricula.depletion.policies import p_learn_to_stop, p_learn_to_run, p_stochastic_reward


@pytest.fixture
def trainer() -> Trainer[Any]:
    return TRAINER


@pytest.fixture
def curriculum() -> Curriculum[AindVrForagingTaskLogic]:
    return CURRICULUM


@pytest.fixture
def init_state() -> TrainerState[Any]:
    return TRAINER.create_trainer_state(stage= s_stage_all_odors_rewarded, is_on_curriculum=True, active_policies=[p_learn_to_run, p_learn_to_stop, p_stochastic_reward])

@pytest.fixture
def fail_metrics() -> DepletionCurriculumMetrics:
    return DepletionCurriculumMetrics(
            total_water_consumed=0,
            n_choices=0,
            n_reward_sites_travelled=5,
            n_patches_visited=0,
            n_patches_visited_per_patch={0: 0},
            last_stop_duration=0.3,
            last_reward_site_length=30,
        )


@pytest.fixture
def ok_metrics() -> DepletionCurriculumMetrics:
    return DepletionCurriculumMetrics(
            total_water_consumed=750,
            n_choices=150,
            n_reward_sites_travelled=5,
            n_patches_visited=50,
            n_patches_visited_per_patch={0: 25},
            last_stop_duration=0.5,
            last_reward_site_length=50,
        )

@pytest.mark.usefixtures("init_state", "ok_metrics", "fail_metrics", "trainer")
class TestCurriculumProgression:
    def test_p_learn_to_stop(self, init_state: TrainerState):
        metrics = DepletionCurriculumMetrics(
            total_water_consumed=750,
            n_choices=150,
            n_reward_sites_travelled=5,
            n_patches_visited=50,
            n_patches_visited_per_patch={0: 25, 1:25},
            last_stop_duration=0.5,
            last_reward_site_length=50,
        )

        assert init_state.stage is not None
        init_state.stage.set_start_policies(start_policies=[p_learn_to_run, p_learn_to_stop, p_stochastic_reward])
    
        init_settings = cast(AindVrForagingTaskLogic, init_state.stage.task)
        
        updated = p_learn_to_stop(metrics, init_settings.model_copy(deep=True))
        assert updated is not None
        assert updated.task_parameters.updaters is not None
        assert (
            updated.task_parameters.updaters[task_logic.UpdaterTarget.STOP_VELOCITY_THRESHOLD].parameters.initial_value
            == metrics.last_stop_threshold_updater * 1.2
        )
        assert (
            updated.task_parameters.updaters[task_logic.UpdaterTarget.STOP_DURATION_OFFSET].parameters.initial_value
            == metrics.last_stop_duration_offset_updater * 0.8
        )

    def test_p_learn_to_run(self, init_state: TrainerState):
        metrics = DepletionCurriculumMetrics(
            total_water_consumed=750,
            n_choices=150,
            n_reward_sites_travelled=5,
            n_patches_visited=50,
            n_patches_visited_per_patch={0: 25, 1:25},
            last_stop_duration=0.5,
            last_reward_site_length=50,
        )

        assert init_state.stage is not None
        init_settings = cast(AindVrForagingTaskLogic, init_state.stage.task)

        updated = p_learn_to_run(metrics, init_settings.model_copy(deep=True))
        assert updated is not None
        assert updated.task_parameters.updaters is not None
        # assert (
        #     updated.task_parameters.updaters[task_logic.UpdaterTarget.RUN_VELOCITY_THRESHOLD].parameters.initial_value
        #     == metrics.last_run_threshold_updater * 1.2
        # )
        # assert (
        #     updated.task_parameters.updaters[task_logic.UpdaterTarget.RUN_DURATION_OFFSET].parameters.initial_value
        #     == metrics.last_run_duration_offset_updater * 0.8
        # )

    def test_st_s_stage_one_odor_no_depletion_s_stage_one_odor_w_depletion_day_0_pass(self, ok_metrics: DepletionCurriculumMetrics):
        assert st_s_stage_one_odor_no_depletion_s_stage_one_odor_w_depletion_day_0(ok_metrics) is True

    def test_st_s_stage_one_odor_no_depletion_s_stage_one_odor_w_depletion_day_0_fail(self, fail_metrics: DepletionCurriculumMetrics):
        assert st_s_stage_one_odor_no_depletion_s_stage_one_odor_w_depletion_day_0(fail_metrics) is False

    def test_st_s_stage_one_odor_w_depletion_day_0_s_stage_one_odor_w_depletion_day_1_pass(self, ok_metrics: DepletionCurriculumMetrics):
        assert st_s_stage_one_odor_w_depletion_day_0_s_stage_one_odor_w_depletion_day_1(ok_metrics) is True

    def test_st_s_stage_one_odor_w_depletion_day_0_s_stage_one_odor_w_depletion_day_1_fail(self, fail_metrics: DepletionCurriculumMetrics):
        assert st_s_stage_one_odor_w_depletion_day_0_s_stage_one_odor_w_depletion_day_1(fail_metrics) is False
    
    def test_st_s_stage_one_odor_w_depletion_day_1_s_stage_one_odor_w_depletion_day_0_pass(self, fail_metrics: DepletionCurriculumMetrics):
        assert st_s_stage_one_odor_w_depletion_day_1_s_stage_one_odor_w_depletion_day_0(fail_metrics) is True
        
    def test_st_s_stage_one_odor_w_depletion_day_1_s_stage_one_odor_w_depletion_day_0_fail(self, ok_metrics: DepletionCurriculumMetrics):
        assert st_s_stage_one_odor_w_depletion_day_1_s_stage_one_odor_w_depletion_day_0(ok_metrics) is False
    
    def test_st_s_stage_one_odor_w_depletion_day_1_s_stage_all_odors_rewarded_pass(self, ok_metrics: DepletionCurriculumMetrics):
        assert st_s_stage_one_odor_w_depletion_day_1_s_stage_all_odors_rewarded(ok_metrics) is True
    
    def test_st_s_stage_one_odor_w_depletion_day_1_s_stage_all_odors_rewarded_fail(self, fail_metrics: DepletionCurriculumMetrics):
        assert st_s_stage_one_odor_w_depletion_day_1_s_stage_all_odors_rewarded(fail_metrics) is False
    
    def test_progression_pass(self, trainer: Trainer, init_state: TrainerState, ok_metrics: DepletionCurriculumMetrics):
        proposal = trainer.evaluate(init_state, ok_metrics)
        assert proposal.is_on_curriculum is True
        assert proposal.curriculum == trainer.curriculum
        assert proposal.stage is not None
        assert proposal.stage.name == trainer.curriculum.see_stages()[1].name
        assert proposal.stage.task == trainer.curriculum.see_stages()[1].task

    def test_progression_fail(
        self, trainer: Trainer, init_state: TrainerState, fail_metrics: DepletionCurriculumMetrics
    ):
        proposal = trainer.evaluate(init_state, fail_metrics)
        assert proposal.is_on_curriculum is True
        assert proposal.curriculum == trainer.curriculum
        assert proposal.stage is not None
        assert proposal.stage.name == trainer.curriculum.see_stages()[0].name
        # Cannot evaluate the task as it is subject to change by policies