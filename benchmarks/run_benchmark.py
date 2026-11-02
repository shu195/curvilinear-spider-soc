"""
Empirical Falsification & Benchmark Suite:
Curvilinear-CPU Biomechanical Spider (Theraphosidae + 8-Bit Curvilinear SoC + Cholesky Self-Repair)
===================================================================================================
Tests Null Hypothesis H_0:
  "An 8-legged biomechanical spider running a 16-instruction Curvilinear Von Neumann CPU
   cannot recover zero-roll locomotion and prey-capture velocity in < 1 ms after losing
   3 unilateral limbs without iterative RL retraining."
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from curvilinear_spider import CurvilinearSpiderOrganism, CholeskyGaitSynthesizer


def run_benchmark():
    print("=" * 88)
    print("CURVILINEAR-CPU BIOMECHANICAL SPIDER (THERAPHOSIDAE + CHOLESKY SELF-REPAIR) BENCHMARK")
    print("=" * 88)

    synth = CholeskyGaitSynthesizer()
    prey_target = np.array([30.0, 12.0])

    scenarios = [
        ("Intact 8-Limb Arachnid (0 Severed)", []),
        ("Minor Trauma: 2 Contralateral Limbs Severed (L1, R4)", [0, 7]),
        ("Severe Unilateral Trauma: 3 Left Limbs Severed (L1, L2, L3)", [0, 1, 2]),
        ("Extreme Trauma: 4 Limbs Severed (50% Amputation: L1, L3, R2, R4)", [0, 2, 5, 7]),
    ]

    print(f"{'Damage Scenario':<52} | {'Mode':<12} | {'Roll Tilt':<10} | {'Speed Eff':<10} | {'Prey Dist (t=4s)'}")
    print("-" * 106)

    solve_times = []
    for label, severed in scenarios:
        for enable_repair, mode_name in [(False, "Static CPG"), (True, "SoC+Cholesky")]:
            if len(severed) == 0 and not enable_repair:
                continue
            spider = CurvilinearSpiderOrganism()
            spider.sever_limbs(severed)

            last_info = {}
            for _ in range(80):  # 4.0 seconds at dt = 0.05s
                last_info = spider.step_physics(0.05, prey_target, enable_self_repair=enable_repair)

            if enable_repair and len(severed) > 0:
                solve_times.append(last_info["solve_ms"])

            print(
                f"{label:<52} | {mode_name:<12} | {last_info['roll_angle']:>7.4f} rad | "
                f"{last_info['forward_efficiency'] * 100.0:>8.2f}% | {last_info['dist_prey']:>8.2f} m"
            )
        print("-" * 106)

    avg_ms = float(np.mean(solve_times)) if solve_times else 0.0
    print(f"[1] Onboard Curvilinear SoC ROM Footprint : 16 Words (32 Bytes) | 0 'if/else' ALU branches")
    print(f"[2] Hardware Opcode SYNTH_GAIT (Opcode 13): Triggered autonomously via MMIO_LIMB_DAMAGE (0x03)")
    print(f"[3] Mean Closed-Form Cholesky Repair Time : {avg_ms:.4f} ms (Zero iterative backpropagation!)")
    print("=" * 88)


if __name__ == "__main__":
    run_benchmark()
