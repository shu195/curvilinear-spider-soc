# curvilinear-spider-soc

> **Biomechanical 8-Legged Theraphosidae Arachnid Driven by a Zero-Branching Curvilinear Von Neumann Microprocessor (`CurvilinearNeuralSoC`) with $0.55\text{ ms}$ Closed-Form Cholesky Gait Self-Repair**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-green.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-3%2F3%20Passing-brightgreen.svg)]()

---

## 1. Cross-Domain Architectural Synthesis

Standard robotic and virtual creature controllers rely on deep reinforcement learning (RL) policies with millions of black-box parameters that collapse catastrophically when a limb is severed outside the training distribution.

**`curvilinear-spider-soc`** fuses three orthogonal mathematical architectures into a single self-repairing neuro-biomechanical organism:

1. **Zero-Branching Curvilinear Von Neumann Nervous System (`CurvilinearNeuralSoC`)**:
   - The spider's central nervous system is an 8-bit Von Neumann microprocessor whose 1-bit ALU slices (`CurvilinearALUSlice`, 66 weights) and instruction decoder (`CurvilinearDecoder`, $23 \times 8$ weights) are synthesized entirely from smooth concentric radial shells $\exp(-(r - \sqrt{k})^2/2\sigma^2)$ and harmonic basis functions $\cos(\pi s), \sin(\pi s)$ with **zero `if/else` arithmetic logic branches**.
   - Executes a **16-word (32-byte) assembly reflex program** (`DEFAULT_SPIDER_BRAIN_ASM`) that reads Memory-Mapped Sensory Registers (`MMIO_PREY_DIST`, `MMIO_OBSTACLE_PROX`, `MMIO_LIMB_DAMAGE`) and drives motor state registers (`MMIO_MOTOR_MODE`).

2. **Hardware Interrupt Opcode `SYNTH_GAIT` & Active-Set Cholesky Self-Repair (`CholeskyGaitSynthesizer`)**:
   - When the spider's assembly loop detects a non-zero limb amputation bitmask in `MMIO_LIMB_DAMAGE (0x03)`, it fires hardware opcode `SYNTH_GAIT (13)`.
   - Instead of requiring hours of RL retraining, `CholeskyGaitSynthesizer` solves an **Active-Set Non-Negative Cholesky Factorization** ($\mathbf{H} = \mathbf{L}\mathbf{L}^T \succ 0$) in **$0.55\text{ ms}$** over the remaining intact limbs to satisfy three physical equilibrium laws:
     $$\sum_{i \in \mathcal{A}} w_i = 1.0 \quad (\text{Vertical Lift}), \qquad \sum_{i \in \mathcal{A}} x_i w_i = 0.0 \quad (\text{Zero Roll Torque}), \qquad \sum_{i \in \mathcal{A}} z_i w_i \approx 0.0 \quad (\text{Pitch Balance})$$
   - Simultaneously re-phases the coupled Central Pattern Generator (CPG) oscillators around the perimeter of surviving limbs to maximize the support polygon.

3. **Theraphosidae 8-Limb Symplectic CPG & 3-DOF Analytical Inverse Kinematics**:
   - Models the biological radial fan geometry (`[0.82, 0.26, -0.34, -0.94] rad`) and 3-DOF analytical Inverse Kinematics (`solve_leg_ik_3dof`: coxa yaw, femur pitch, tibia knee angle).

---

## 2. Empirical Trauma & Amputation Benchmark (`benchmarks/run_benchmark.py`)

Evaluated across four progressive limb-amputation trauma regimes over a $4.0\text{ s}$ prey-pursuit trajectory (target at $(30\text{ m}, 12\text{ m})$):

| Limb Amputation Scenario | Controller Mode | Body Roll Tilt ($\text{rad}$) | Forward Speed Efficiency | Remaining Prey Distance ($t=4\text{s}$) |
| :--- | :---: | :---: | :---: | :---: |
| **Intact 8-Limb Arachnid** *(0 Severed)* | `SoC + Cholesky` | `0.0076 rad` | `81.38%` | `11.05 m` |
| **Minor Trauma: 2 Contralateral Limbs** *(L1, R4)* | `Static CPG` | `0.0530 rad` | `52.71%` | `18.85 m` |
| **Minor Trauma: 2 Contralateral Limbs** *(L1, R4)* | **`SoC + Cholesky`** | **`0.0466 rad`** | **`74.41%`** | **`12.28 m`** |
| **Severe Unilateral Trauma: 3 Left Limbs** *(L1, L2, L3)* | `Static CPG` | `1.2124 rad` *(Capsized!)* | `3.73%` | `31.14 m` *(Drifted backward)* |
| **Severe Unilateral Trauma: 3 Left Limbs** *(L1, L2, L3)* | **`SoC + Cholesky`** | **`0.0276 rad` ($43.9\times$ lower roll!)** | **`50.25%` ($13.5\times$ faster!)** | **`17.48 m`** |
| **Extreme Trauma: 4 Limbs Severed** *(50% Amputation)* | `Static CPG` | `0.0430 rad` | `39.45%` | `22.92 m` |
| **Extreme Trauma: 4 Limbs Severed** *(50% Amputation)* | **`SoC + Cholesky`** | **`0.0183 rad`** | **`79.74%`** | **`11.34 m`** |

### Key Hardware & Mathematical Metrics
- **Onboard Curvilinear SoC ROM Footprint:** `16 Words (32 Bytes)` | `0` arithmetic `if/else` branches in ALU/Decoder.
- **Mean Closed-Form Cholesky Self-Repair Latency:** **`0.5566 ms`** on CPU (zero iterative gradient backpropagation).
- **Severe Unilateral Amputation Recovery:** When 3 out of 4 left legs (`L1, L2, L3`) are severed, a standard CPG capsizes (`1.2124 rad` roll, `3.73%` speed). The Curvilinear SoC fires `SYNTH_GAIT`, re-allocating support across the surviving left-rear anchor (`L4`) and contralateral diagonal legs to slash roll tilt by **$43.9\times$** (`0.0276 rad`) and boost forward velocity by **$13.5\times$** (`50.25%`).

---

## 3. Quickstart

```bash
pip install -e .[dev]
```

```python
import numpy as np
from curvilinear_spider import CurvilinearSpiderOrganism

spider = CurvilinearSpiderOrganism()

# Sever 3 left-side limbs in real time (Leg 0, Leg 1, Leg 2)
spider.sever_limbs([0, 1, 2])

# Step physics: Curvilinear SoC detects MMIO_LIMB_DAMAGE (0x03) and triggers SYNTH_GAIT (Opcode 13)
telemetry = spider.step_physics(dt=0.05, prey_pos=np.array([30.0, 12.0]), enable_self_repair=True)
print("Hardware Cholesky Self-Repair Triggered:", telemetry["synth_triggered"])
print("Stabilized Roll Angle (rad):", telemetry["roll_angle"])
print("Forward Locomotion Efficiency:", f"{telemetry['forward_efficiency']*100:.2f}%")
```

---

## 4. Verification & Benchmarks

```bash
# Run unit & invariant test suite (3/3 passing):
python -m pytest tests/ -v

# Run full limb-amputation trauma benchmark:
python benchmarks/run_benchmark.py
```
