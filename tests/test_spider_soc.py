import math
import numpy as np
from curvilinear_spider import (
    CurvilinearALUSlice,
    CurvilinearDecoder,
    CurvilinearNeuralSoC,
    CholeskyGaitSynthesizer,
    CurvilinearSpiderOrganism,
)


def test_curvilinear_alu_and_decoder_exactness():
    alu_slice = CurvilinearALUSlice(decimals=2)
    for a in (0, 1):
        for b in (0, 1):
            for cin in (0, 1):
                res, cout = alu_slice.evaluate(a, b, cin, 0)  # ADD
                assert res == (a + b + cin) % 2
                assert cout == int((a + b + cin) >= 2)

    decoder = CurvilinearDecoder(decimals=2)
    ctrl_synth = decoder.decode(13)  # SYNTH_GAIT opcode
    assert ctrl_synth["RegWrite"] == 1


def test_cholesky_gait_self_repair_equilibrium():
    synth = CholeskyGaitSynthesizer()
    # Sever 3 left-side legs: Leg 0, Leg 1, Leg 2 (severe asymmetric amputation!)
    damage_mask = (1 << 0) | (1 << 1) | (1 << 2)
    res = synth.synthesize_compensatory_gait(damage_mask)

    weights = res["load_weights"]
    assert weights[0] == 0.0 and weights[1] == 0.0 and weights[2] == 0.0
    assert abs(np.sum(weights) - 1.0) < 1e-5
    # Verify massive roll imbalance reduction vs uncompensated uniform weights
    uncomp_w = np.array([0.0 if ((damage_mask >> i) & 1) else 0.2 for i in range(8)])
    uncomp_roll = abs(float(np.sum(synth.leg_coords[:, 0] * uncomp_w)))
    assert res["roll_imbalance"] < uncomp_roll * 0.15


def test_spider_soc_reflex_and_limb_amputation_recovery():
    prey_pos = np.array([25.0, 10.0])

    # 1. Spider WITH Curvilinear SoC SYNTH_GAIT self-repair
    spider_repaired = CurvilinearSpiderOrganism()
    spider_repaired.sever_limbs([0, 1, 2])  # Sever 3 left legs

    # 2. Spider WITHOUT self-repair (static tetrapod CPG)
    spider_broken = CurvilinearSpiderOrganism()
    spider_broken.sever_limbs([0, 1, 2])

    for _ in range(80):
        info_rep = spider_repaired.step_physics(0.05, prey_pos, enable_self_repair=True)
        info_brk = spider_broken.step_physics(0.05, prey_pos, enable_self_repair=False)

    assert info_rep["synth_triggered"] is True
    assert info_rep["roll_angle"] < info_brk["roll_angle"] * 0.20
    assert info_rep["dist_prey"] < info_brk["dist_prey"]
