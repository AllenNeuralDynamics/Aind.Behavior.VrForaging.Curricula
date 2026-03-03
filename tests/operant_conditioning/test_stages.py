from aind_behavior_vr_foraging_curricula.operant_conditioning.stages import (
    make_s_stage_a100_b100_c0,
    make_s_stage_b100_c0,
    make_s_stage_reversals,
)


class TestStageCreation:
    """Test that all stages in the operant_conditioning module can be created without crashing."""

    def test_make_s_stage_a100_b100_c0(self):
        """Test that stage_a100_b100_c0 can be instantiated."""
        make_s_stage_a100_b100_c0()

    def test_make_s_stage_b100_c0(self):
        """Test that stage_b100_c0 can be instantiated."""
        make_s_stage_b100_c0()

    def test_make_s_stage_reversals(self):
        """Test that stage_reversals can be instantiated."""
        make_s_stage_reversals()
