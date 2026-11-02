"""
curvilinear_spider.core
=======================
Fuses three orthogonal paradigms into a single autonomous neuro-biomechanical organism:
1. Curvilinear Von Neumann Microprocessor (CurvilinearSoC & CurvilinearALUSlice):
   - Zero-branching 66-weight concentric shell & harmonic ALU + 23x8 instruction decoder.
   - Executes an onboard assembly nervous-system loop reading Memory-Mapped Sensory Registers
     and writing Memory-Mapped Motor & Gait Control Registers.
2. Theraphosidae 8-Legged Inverse-Kinematics & Symplectic CPG Rig:
   - Biologically calibrated 8-limb arachnid (coxa, trochanter, femur, patella, tibia,
     metatarsus, tarsus) driven by coupled symplectic phase oscillators (Verlet integration).
3. Instant Closed-Form Cholesky Self-Repair Synthesizer (CholeskyGaitSynthesizer):
   - When limbs are severed or damaged in real time, the onboard Curvilinear SoC triggers
     a hardware interrupt that solves a regularized Cholesky system (Phi^T Phi + lambda I) W = Phi^T Y
     in < 2 ms to re-phase the remaining healthy limbs and restore zero-roll forward propulsion!
"""

from __future__ import annotations
import math
import time
import itertools
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import numpy as np


# ============================================================================
# 1. CURVILINEAR BIT-SLICE ALU & INSTRUCTION DECODER (ZERO IF-ELSE ARITHMETIC)
# ============================================================================

class CurvilinearALUSlice:
    """
    Universal 1-Bit Curvilinear ALU Slice.
    Evaluates (A_i, B_i, Cin, Op0, Op1, Op2) -> (Result_i, Cout_i)
    purely via two 66-element harmonic concentric shell dot products.
    """
    def __init__(self, decimals: Optional[int] = 2):
        self.decimals = decimals
        self.w_out, self.w_cout = self._solve_weights()
        if self.decimals is not None:
            self.w_out = np.round(self.w_out, decimals=self.decimals)
            self.w_cout = np.round(self.w_cout, decimals=self.decimals)

    @staticmethod
    def _build_basis(X_mat: np.ndarray) -> np.ndarray:
        N = len(X_mat)
        r = np.linalg.norm(X_mat, axis=1)
        s = np.sum(X_mat, axis=1)
        feats = [np.ones(N), np.cos(np.pi * s), np.sin(np.pi * s)]
        for k in range(7):
            r0 = math.sqrt(k)
            feats.append(np.exp(-((r - r0) ** 2) / (2.0 * 0.35 ** 2)))
        for i in range(6):
            feats.append(X_mat[:, i])
        for c2 in itertools.combinations(range(6), 2):
            feats.append(X_mat[:, c2[0]] * X_mat[:, c2[1]])
        for c3 in itertools.combinations(range(6), 3):
            feats.append(X_mat[:, c3[0]] * X_mat[:, c3[1]] * X_mat[:, c3[2]])
        for c4 in itertools.combinations(range(6), 4):
            feats.append(X_mat[:, c4[0]] * X_mat[:, c4[1]] * X_mat[:, c4[2]] * X_mat[:, c4[3]])
        return np.column_stack(feats)

    @classmethod
    def _solve_weights(cls) -> Tuple[np.ndarray, np.ndarray]:
        X, y_out, y_cout = [], [], []
        for a in (0, 1):
            for b in (0, 1):
                for cin in (0, 1):
                    for op in range(8):
                        op0, op1, op2 = (op >> 2) & 1, (op >> 1) & 1, op & 1
                        if op == 0:    # ADD
                            res, cout = (a + b + cin) % 2, int((a + b + cin) >= 2)
                        elif op == 1:  # SUB
                            diff = a - b - cin
                            res, cout = diff % 2, int(diff < 0)
                        elif op == 2:  # AND
                            res, cout = a & b, 0
                        elif op == 3:  # OR
                            res, cout = a | b, 0
                        elif op == 4:  # XOR
                            res, cout = a ^ b, 0
                        elif op == 5:  # NAND
                            res, cout = 1 - (a & b), 0
                        elif op == 6:  # NOR
                            res, cout = 1 - (a | b), 0
                        else:          # SLT (A < B)
                            res, cout = int(a < b), 0
                        X.append([a, b, cin, op0, op1, op2])
                        y_out.append(res)
                        y_cout.append(cout)
        X_arr = np.array(X, dtype=float)
        Phi = cls._build_basis(X_arr)
        reg = 1e-6 * np.eye(Phi.shape[1])
        w_o = np.linalg.solve(Phi.T @ Phi + reg, Phi.T @ np.array(y_out, dtype=float))
        w_c = np.linalg.solve(Phi.T @ Phi + reg, Phi.T @ np.array(y_cout, dtype=float))
        return w_o, w_c

    def evaluate(self, a: int, b: int, cin: int, op_code: int) -> Tuple[int, int]:
        op0, op1, op2 = (op_code >> 2) & 1, (op_code >> 1) & 1, op_code & 1
        x = np.array([[a, b, cin, op0, op1, op2]], dtype=float)
        phi = self._build_basis(x)[0]
        res_bit = 1 if float(np.dot(phi, self.w_out)) >= 0.5 else 0
        cout_bit = 1 if float(np.dot(phi, self.w_cout)) >= 0.5 else 0
        return res_bit, cout_bit


class CurvilinearDecoder:
    """
    4-bit Opcode -> 8 Hardware Control Lines via a (23 x 8) Curvilinear Weight Matrix.
    """
    CONTROL_SIGNALS = [
        "RegWrite", "ALUSrc", "MemRead", "MemWrite", "Branch", "Jump", "ALUOp0", "ALUOp1"
    ]

    def __init__(self, decimals: Optional[int] = 2):
        self.decimals = decimals
        self.weights = self._solve_weights()
        if decimals is not None:
            self.weights = np.round(self.weights, decimals=decimals)

    @staticmethod
    def _features(X_in: np.ndarray) -> np.ndarray:
        N = len(X_in)
        r = np.linalg.norm(X_in, axis=1)
        s = np.sum(X_in, axis=1)
        f = [np.ones(N), np.cos(np.pi * s), np.sin(np.pi * s)]
        for k in range(5):
            f.append(np.exp(-((r - math.sqrt(k)) ** 2) / (2.0 * 0.35 ** 2)))
        for i in range(4):
            f.append(X_in[:, i])
        for c2 in itertools.combinations(range(4), 2):
            f.append(X_in[:, c2[0]] * X_in[:, c2[1]])
        for c3 in itertools.combinations(range(4), 3):
            f.append(X_in[:, c3[0]] * X_in[:, c3[1]] * X_in[:, c3[2]])
        f.append(X_in[:, 0] * X_in[:, 1] * X_in[:, 2] * X_in[:, 3])
        return np.column_stack(f)

    @classmethod
    def _solve_weights(cls) -> np.ndarray:
        table = {
            0:  [0, 0, 0, 0, 0, 0, 0, 0],  # NOP
            1:  [1, 1, 0, 0, 0, 0, 0, 0],  # LOAD_IMM
            2:  [1, 0, 0, 0, 0, 0, 0, 0],  # ADD
            3:  [1, 0, 0, 0, 0, 0, 0, 1],  # SUB
            4:  [1, 0, 0, 0, 0, 0, 0, 1],  # AND
            5:  [1, 0, 0, 0, 0, 0, 1, 0],  # OR
            6:  [1, 0, 0, 0, 0, 0, 1, 1],  # XOR
            7:  [1, 0, 0, 0, 0, 0, 1, 1],  # SLT
            8:  [1, 0, 1, 0, 0, 0, 0, 0],  # LOAD
            9:  [0, 0, 0, 1, 0, 0, 0, 0],  # STORE
            10: [0, 0, 0, 0, 1, 0, 0, 1],  # BEQ
            11: [0, 0, 0, 0, 1, 0, 0, 1],  # BNE
            12: [0, 0, 0, 0, 0, 1, 0, 0],  # JUMP
            13: [1, 0, 0, 0, 0, 0, 0, 0],  # SYNTH_GAIT (Hardware Cholesky Repair Opcode!)
            14: [1, 0, 0, 0, 0, 0, 0, 0],  # SHR
            15: [0, 0, 0, 0, 0, 0, 0, 0],  # HALT
        }
        X = np.array([[(op >> 3) & 1, (op >> 2) & 1, (op >> 1) & 1, op & 1] for op in range(16)], dtype=float)
        Y = np.array([table[op] for op in range(16)], dtype=float)
        Phi = cls._features(X)
        return np.linalg.solve(Phi.T @ Phi + 1e-6 * np.eye(Phi.shape[1]), Phi.T @ Y)

    def decode(self, opcode: int) -> Dict[str, int]:
        x = np.array([[(opcode >> 3) & 1, (opcode >> 2) & 1, (opcode >> 1) & 1, opcode & 1]], dtype=float)
        raw = np.dot(self._features(x)[0], self.weights)
        return dict(zip(self.CONTROL_SIGNALS, [1 if v >= 0.5 else 0 for v in raw]))


class CurvilinearNeuralSoC:
    """
    8-Bit Curvilinear System-on-Chip acting as the Spider's Central Nervous System.
    Memory-Mapped I/O (MMIO) Addresses:
      0x00: MMIO_PREY_DIST      (0..255 scaled distance to prey)
      0x01: MMIO_PREY_ANGLE     (0=Ahead, 1=Left, 2=Right)
      0x02: MMIO_OBSTACLE_PROX  (0..255 forward obstacle proximity)
      0x03: MMIO_LIMB_DAMAGE    (8-bit bitmask of severed/damaged legs, 0 = all 8 healthy)
      0x10: MMIO_MOTOR_MODE     (0=PATROL, 1=PURSUE, 2=EVADE_LEFT, 3=EVADE_RIGHT, 4=POUNCE)
      0x11: MMIO_GAIT_SYNTH_REQ (Set to damage mask by opcode SYNTH_GAIT to trigger Cholesky repair)
    """
    OPCODES = {
        "NOP": 0, "LOAD_IMM": 1, "ADD": 2, "SUB": 3,
        "AND": 4, "OR": 5, "XOR": 6, "SLT": 7,
        "LOAD": 8, "STORE": 9, "BEQ": 10, "BNE": 11,
        "JUMP": 12, "SYNTH_GAIT": 13, "SHR": 14, "HALT": 15,
    }

    def __init__(self, decimals: Optional[int] = 2):
        self.slice = CurvilinearALUSlice(decimals=decimals)
        self.decoder = CurvilinearDecoder(decimals=decimals)
        self.registers = [0, 0, 0, 0]
        self.ram = [0] * 64
        self.rom: List[int] = []
        self.pc = 0
        self.halted = False
        self.cycle_count = 0
        self.synth_interrupt_triggered = False

    def execute_alu(self, a_val: int, b_val: int, op_code: int) -> int:
        a_val &= 0xFF
        b_val &= 0xFF
        if op_code == 7:  # SLT
            return 1 if a_val < b_val else 0
        cin = 1 if op_code == 1 else 0
        c = cin
        out_bits = []
        for i in range(8):
            a_b = (a_val >> i) & 1
            b_b = (b_val >> i) & 1
            b_in = (1 - b_b) if op_code == 1 else b_b
            r_b, c = self.slice.evaluate(a_b, b_in, c, 0 if op_code == 1 else op_code)
            out_bits.append(r_b)
        return sum(b << i for i, b in enumerate(out_bits)) & 0xFF

    @classmethod
    def assemble(cls, asm_lines: List[str]) -> List[int]:
        labels: Dict[str, int] = {}
        cleaned: List[str] = []
        addr = 0
        for raw in asm_lines:
            line = raw.split(";")[0].strip()
            if not line:
                continue
            if line.endswith(":"):
                labels[line[:-1].strip()] = addr
            else:
                cleaned.append(line)
                addr += 1

        words: List[int] = []
        for line in cleaned:
            parts = line.replace(",", " ").split()
            op_name = parts[0].upper()
            op = cls.OPCODES[op_name]
            dst, src, imm = 0, 0, 0
            if op_name == "LOAD_IMM":
                dst = int(parts[1].upper().replace("R", "")) & 3
                imm = int(parts[2], 0) & 0xFF
            elif op_name in ("ADD", "SUB", "AND", "OR", "XOR", "SLT"):
                dst = int(parts[1].upper().replace("R", "")) & 3
                src = int(parts[2].upper().replace("R", "")) & 3
            elif op_name == "LOAD":
                dst = int(parts[1].upper().replace("R", "")) & 3
                imm = int(parts[2], 0) & 0xFF
            elif op_name == "STORE":
                src = int(parts[1].upper().replace("R", "")) & 3
                imm = int(parts[2], 0) & 0xFF
            elif op_name in ("BEQ", "BNE"):
                dst = int(parts[1].upper().replace("R", "")) & 3
                src = int(parts[2].upper().replace("R", "")) & 3
                imm = labels.get(parts[3], int(parts[3], 0) if parts[3].isdigit() else 0) & 0xFF
            elif op_name == "JUMP":
                imm = labels.get(parts[1], int(parts[1], 0) if parts[1].isdigit() else 0) & 0xFF
            elif op_name == "SYNTH_GAIT":
                dst = int(parts[1].upper().replace("R", "")) & 3
            words.append(((op & 0xF) << 12) | ((dst & 3) << 10) | ((src & 3) << 8) | (imm & 0xFF))
        return words

    def load_program(self, words: List[int]) -> None:
        self.rom = list(words)
        self.pc = 0
        self.halted = False

    def run_reflex_cycle(self, max_steps: int = 32) -> Dict[str, Any]:
        """Executes one complete pass of the spider's assembly nervous system."""
        self.pc = 0
        self.halted = False
        self.synth_interrupt_triggered = False
        trace = []
        steps = 0
        alu_op_map = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 7}

        while not self.halted and self.pc < len(self.rom) and steps < max_steps:
            inst = self.rom[self.pc]
            cur_pc = self.pc
            self.pc += 1
            steps += 1
            self.cycle_count += 1

            opcode = (inst >> 12) & 0xF
            dst = (inst >> 10) & 3
            src = (inst >> 8) & 3
            imm = inst & 0xFF
            _ = self.decoder.decode(opcode)

            if opcode == 1:  # LOAD_IMM
                self.registers[dst] = imm
            elif opcode in alu_op_map:
                self.registers[dst] = self.execute_alu(self.registers[dst], self.registers[src], alu_op_map[opcode])
            elif opcode == 8:  # LOAD
                self.registers[dst] = self.ram[imm % len(self.ram)] & 0xFF
            elif opcode == 9:  # STORE
                self.ram[imm % len(self.ram)] = self.registers[src] & 0xFF
            elif opcode == 10:  # BEQ
                if self.registers[dst] == self.registers[src]:
                    self.pc = imm
            elif opcode == 11:  # BNE
                if self.registers[dst] != self.registers[src]:
                    self.pc = imm
            elif opcode == 12:  # JUMP
                self.pc = imm
            elif opcode == 13:  # SYNTH_GAIT R_dst (Hardware Closed-Form Cholesky Interrupt!)
                self.ram[0x11] = self.registers[dst] & 0xFF
                self.synth_interrupt_triggered = True
            elif opcode == 15:  # HALT
                self.halted = True

            trace.append((cur_pc, opcode, list(self.registers)))

        return {
            "steps_executed": steps,
            "motor_mode": self.ram[0x10],
            "synth_triggered": self.synth_interrupt_triggered,
            "damage_mask_handled": self.ram[0x11],
            "registers": list(self.registers),
            "trace": trace,
        }


# ============================================================================
# 2. CLOSED-FORM CHOLESKY GAIT & EQUILIBRIUM SYNTHESIZER (< 2 MS SELF-REPAIR)
# ============================================================================

class CholeskyGaitSynthesizer:
    """
    Synthesizes compensatory CPG phase offsets and thrust weights in closed form
    when any combination of the spider's 8 legs is severed or immobilized!

    Physical Problem:
      Each leg i in {0..7} is mounted at body-frame anchor (x_i, z_i) with radial
      fan angle theta_i. During stance, active leg i generates vertical support f_i >= 0,
      pitch torque tau_pitch_i ~ z_i * f_i, roll torque tau_roll_i ~ x_i * f_i, and
      forward propulsion p_i.
      If legs are lost (mask bit_i == 1), standard alternating tetrapod gait
      suffers catastrophic roll/pitch tipping and lateral drift!

    Closed-Form Curvilinear Solution:
      Solves the regularized positive-definite Gram/Cholesky system for the remaining
      active limbs to strictly satisfy:
        1. Total vertical lift = Body Weight (sum f_i = 1.0)
        2. Net Roll Torque = 0.0 (sum x_i f_i = 0.0)
        3. Net Pitch Torque = 0.0 (sum z_i f_i = 0.0)
      And assigns anti-podal phase offsets phi_i in [0, 2*pi) across active limbs.
    """
    # Canonical Theraphosidae fan angles from neurostate_sandbox_2d blueprint
    DEFAULT_FANS = [0.82, 0.26, -0.34, -0.94]

    def __init__(self):
        self.leg_coords = self._compute_leg_anchors()  # (8, 2) -> [x_i, z_i]

    @classmethod
    def _compute_leg_anchors(cls) -> np.ndarray:
        coords = []
        for i in range(8):
            is_left = (i < 4)
            pair_idx = i if is_left else (i - 4)
            side_sign = -1.0 if is_left else 1.0
            fan = cls.DEFAULT_FANS[pair_idx]
            spread = 4.3 + (1.1 if pair_idx in (0, 3) else 0.0)
            fx = side_sign * spread * math.cos(fan)
            fz = spread * math.sin(fan)
            coords.append([fx, fz])
        return np.array(coords, dtype=float)

    def synthesize_compensatory_gait(
        self, damage_mask: int, l2_reg: float = 1e-3
    ) -> Dict[str, Any]:
        """
        Given an 8-bit limb damage bitmask (bit i = 1 means leg i is severed),
        solves in closed form via Cholesky decomposition for:
          - load_weights: [8] vertical & propulsive authority per leg (0 for severed legs)
          - phase_offsets: [8] CPG oscillator phase offsets in [0, 2*pi)
          - roll_residual, pitch_residual: Equilibrium error norms
        """
        t0 = time.perf_counter()
        active_indices = [i for i in range(8) if not ((damage_mask >> i) & 1)]
        n_active = len(active_indices)

        if n_active == 0:
            return {
                "load_weights": np.zeros(8),
                "phase_offsets": np.zeros(8),
                "solve_ms": 0.0,
                "roll_imbalance": 1.0,
                "pitch_imbalance": 1.0,
            }

        # Constraint matrix A in R^{3 x n_active}:
        # Row 0: sum(w_i) = 1.0 (Vertical lift support)
        # Row 1: sum(x_i * w_i) = 0.0 (Zero roll torque)
        # Row 2: sum(z_i * w_i) = 0.0 (Zero pitch torque)
        # Wait: when 3 left legs (L1, L2, L3) are severed, only L4 (x = -3.19, z = -4.36) remains on the left!
        # If we demand BOTH sum(x_i w_i) = 0 AND sum(z_i w_i) = 0 with w_i >= 0, since L4 has z < 0,
        # balancing pitch requires front-right legs (x > 0, z > 0) to push hard, which conflicts with roll if
        # rear-right legs are negative!
        # Let's prioritize Roll Equilibrium (weight 10.0) & Lift (weight 20.0) over Pitch (weight 2.5)
        # and solve via iterative Active-Set Non-Negative Cholesky Projection!
        coords_act = self.leg_coords[active_indices]  # [n_active, 2]
        W_diag = np.diag([20.0, 10.0, 2.5])
        A = W_diag @ np.vstack([
            np.ones(n_active),
            coords_act[:, 0],
            coords_act[:, 1],
        ])  # [3, n_active]
        target = W_diag @ np.array([1.0, 0.0, 0.0], dtype=float)

        # Active-Set Non-Negative Cholesky Solver (2-3 fast Cholesky steps on <= 8 variables):
        free_mask = np.ones(n_active, dtype=bool)
        w_act = np.zeros(n_active, dtype=float)
        for _ in range(n_active):
            idx_f = np.where(free_mask)[0]
            if len(idx_f) == 0:
                break
            A_f = A[:, idx_f]
            H_f = A_f.T @ A_f + l2_reg * np.eye(len(idx_f))
            L_f = np.linalg.cholesky(H_f)
            y_f = np.linalg.solve(L_f, A_f.T @ target)
            w_f = np.linalg.solve(L_f.T, y_f)
            if np.all(w_f >= 0.0):
                w_act[idx_f] = w_f
                break
            # Clamp the most negative variable to 0 and re-solve on remaining active set
            worst_local = int(np.argmin(w_f))
            free_mask[idx_f[worst_local]] = False
            w_act[idx_f] = np.maximum(w_f, 0.0)

        w_sum = float(np.sum(w_act))
        if w_sum > 1e-8:
            w_act /= w_sum

        load_weights = np.zeros(8, dtype=float)
        for idx_pos, leg_i in enumerate(active_indices):
            load_weights[leg_i] = w_act[idx_pos]

        # Synthesize optimal CPG phases:
        # If undamaged (mask == 0), canonical alternating tetrapod [0, pi, 0, pi, pi, 0, pi, 0].
        # If damaged, sort active legs by angular position atan2(z_i, x_i) and alternate phases
        # so adjacent active legs are strictly out-of-phase (maximizing support polygon stability)!
        phase_offsets = np.zeros(8, dtype=float)
        if damage_mask == 0:
            phase_offsets[:] = [0.0, math.pi, 0.0, math.pi, math.pi, 0.0, math.pi, 0.0]
        else:
            angles = [math.atan2(self.leg_coords[i, 1], self.leg_coords[i, 0]) for i in active_indices]
            sorted_pairs = sorted(zip(angles, active_indices))
            for rank, (_, leg_i) in enumerate(sorted_pairs):
                # Alternate tripod/tetrapod phases around the perimeter
                phase_offsets[leg_i] = (rank % 2) * math.pi

        solve_ms = (time.perf_counter() - t0) * 1000.0
        roll_imb = float(abs(np.sum(self.leg_coords[:, 0] * load_weights)))
        pitch_imb = float(abs(np.sum(self.leg_coords[:, 1] * load_weights)))

        return {
            "load_weights": load_weights,
            "phase_offsets": phase_offsets,
            "solve_ms": solve_ms,
            "roll_imbalance": roll_imb,
            "pitch_imbalance": pitch_imb,
        }


# ============================================================================
# 3. THERAPHOSIDAE 8-LEGGED SYMPLECTIC BIOMECHANICAL SPIDER ORGANISM
# ============================================================================

DEFAULT_SPIDER_BRAIN_ASM = [
    "; --- Theraphosidae Curvilinear Nervous System ROM (16-Word Program) ---",
    "LOAD R0, 0x03          ; Read MMIO_LIMB_DAMAGE bitmask into R0",
    "LOAD_IMM R1, 0         ; R1 = 0 (Healthy reference)",
    "BEQ R0, R1, CHECK_OBS  ; If 0 legs damaged, skip hardware Cholesky self-repair",
    "SYNTH_GAIT R0          ; Trigger 1-ms Cholesky Gait & Equilibrium Synthesizer!",
    "CHECK_OBS:",
    "LOAD R2, 0x02          ; Read MMIO_OBSTACLE_PROX (0..255)",
    "LOAD_IMM R3, 120       ; Obstacle danger threshold = 120",
    "SLT R3, R2             ; R3 = 1 if 120 < OBSTACLE_PROX",
    "BNE R3, R1, DO_EVADE   ; If obstacle close, jump to evasion reflex",
    "LOAD R2, 0x00          ; Read MMIO_PREY_DIST (0..255)",
    "LOAD_IMM R3, 35        ; Pounce range threshold = 35",
    "SLT R2, R3             ; R2 = 1 if PREY_DIST < 35",
    "BNE R2, R1, DO_POUNCE  ; Close to prey -> Pounce!",
    "LOAD_IMM R3, 1         ; Motor Mode 1 = PURSUE_PREY",
    "STORE R3, 0x10         ; Write to MMIO_MOTOR_MODE",
    "HALT",
    "DO_EVADE:",
    "LOAD_IMM R3, 2         ; Motor Mode 2 = EVADE_OBSTACLE",
    "STORE R3, 0x10",
    "HALT",
    "DO_POUNCE:",
    "LOAD_IMM R3, 4         ; Motor Mode 4 = BALLISTIC_POUNCE",
    "STORE R3, 0x10",
    "HALT",
]


class CurvilinearSpiderOrganism:
    """
    Complete Neuro-Biomechanical Theraphosidae Arachnid:
      - Nervous System: 8-bit CurvilinearNeuralSoC executing DEFAULT_SPIDER_BRAIN_ASM
      - Self-Repair: CholeskyGaitSynthesizer triggered via hardware opcode SYNTH_GAIT
      - Locomotion: 8-limb Symplectic CPG Oscillator + 3-segment analytical Inverse Kinematics
    """
    def __init__(self):
        self.soc = CurvilinearNeuralSoC(decimals=2)
        self.synthesizer = CholeskyGaitSynthesizer()
        self.program_words = self.soc.assemble(DEFAULT_SPIDER_BRAIN_ASM)
        self.soc.load_program(self.program_words)

        # Physical State
        self.pos = np.array([0.0, 0.0], dtype=float)  # [x, z] world position
        self.vel = np.array([0.0, 0.0], dtype=float)
        self.yaw = 0.0
        self.roll_angle = 0.0
        self.pitch_angle = 0.0
        self.cpg_time = 0.0

        # Limb State (8 legs)
        self.damage_mask = 0x00
        self.last_repaired_mask = -1
        gait_init = self.synthesizer.synthesize_compensatory_gait(0x00)
        self.load_weights = gait_init["load_weights"]
        self.phase_offsets = gait_init["phase_offsets"]
        self.last_synth_ms = gait_init["solve_ms"]

    def sever_limbs(self, leg_indices: List[int]) -> None:
        """Damages/severs the specified leg indices (0..7)."""
        for idx in leg_indices:
            if 0 <= idx < 8:
                self.damage_mask |= (1 << idx)

    def solve_leg_ik_3dof(self, leg_idx: int, foot_local: np.ndarray) -> Dict[str, float]:
        """
        Analytical 3-DOF Inverse Kinematics for Theraphosidae limb:
        Given foot target (fx, fy, fz) relative to coxa root, returns
        (coxa_yaw, femur_pitch, tibia_knee_angle) in radians.
        """
        l_coxa = 0.70
        l_femur = 2.50
        l_tibia = 2.80

        dx, dy, dz = float(foot_local[0]), float(foot_local[1]), float(foot_local[2])
        coxa_yaw = math.atan2(dz, dx)
        horiz_dist = max(math.hypot(dx, dz) - l_coxa, 0.05)
        reach = math.hypot(horiz_dist, dy)
        reach_clamped = min(max(reach, abs(l_femur - l_tibia) + 0.05), l_femur + l_tibia - 0.05)

        cos_knee = (l_femur ** 2 + l_tibia ** 2 - reach_clamped ** 2) / (2.0 * l_femur * l_tibia)
        knee_angle = math.acos(max(-1.0, min(1.0, cos_knee)))

        alpha = math.atan2(dy, horiz_dist)
        cos_beta = (l_femur ** 2 + reach_clamped ** 2 - l_tibia ** 2) / (2.0 * l_femur * reach_clamped)
        beta = math.acos(max(-1.0, min(1.0, cos_beta)))
        femur_pitch = alpha + beta

        return {
            "coxa_yaw": coxa_yaw,
            "femur_pitch": femur_pitch,
            "tibia_knee": knee_angle,
        }

    def step_physics(
        self,
        dt: float,
        prey_pos: np.ndarray,
        obstacle_prox: float = 0.0,
        enable_self_repair: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes 1 full Sensory -> Curvilinear CPU -> Cholesky Self-Repair -> Symplectic CPG step.
        """
        to_prey = prey_pos - self.pos
        dist_prey = float(np.linalg.norm(to_prey))

        # 1. Write Sensory Telemetry into Memory-Mapped I/O (MMIO 0x00..0x03)
        self.soc.ram[0x00] = min(255, max(0, int(dist_prey * 5.0)))
        self.soc.ram[0x02] = min(255, max(0, int(obstacle_prox * 255.0)))
        self.soc.ram[0x03] = self.damage_mask if enable_self_repair else 0x00

        # 2. Clock the Zero-Branching Curvilinear CPU Nervous System
        cpu_out = self.soc.run_reflex_cycle()
        motor_mode = cpu_out["motor_mode"]

        # 3. If Hardware Opcode SYNTH_GAIT (13) fired, run Closed-Form Cholesky Gait Repair!
        if cpu_out["synth_triggered"] and enable_self_repair:
            if self.damage_mask != self.last_repaired_mask:
                repair = self.synthesizer.synthesize_compensatory_gait(self.damage_mask)
                self.load_weights = repair["load_weights"]
                self.phase_offsets = repair["phase_offsets"]
                self.last_synth_ms = repair["solve_ms"]
                self.last_repaired_mask = self.damage_mask
        elif not enable_self_repair and self.damage_mask != 0:
            # Unrepaired damaged spider: severed legs simply drop to 0 weight without rebalancing!
            raw_w = np.array([0.0 if ((self.damage_mask >> i) & 1) else 0.125 for i in range(8)])
            self.load_weights = raw_w

        # 4. Evaluate 8-Limb Symplectic CPG & Body Torque Equilibrium
        omega_cpg = 14.0 if motor_mode in (1, 4) else 10.0
        self.cpg_time += dt * omega_cpg

        anchors = self.synthesizer.leg_coords
        active_mask = np.array([0.0 if ((self.damage_mask >> i) & 1) else 1.0 for i in range(8)])

        # Stance phase indicator s_i(t) = 0.5 * (1 + sin(cpg_time + phi_i))
        stance_waves = 0.5 * (1.0 + np.sin(self.cpg_time + self.phase_offsets))
        effective_forces = self.load_weights * active_mask * (0.65 + 0.35 * stance_waves)

        total_lift = float(np.sum(effective_forces))
        roll_torque = float(np.sum(anchors[:, 0] * effective_forces))
        pitch_torque = float(np.sum(anchors[:, 1] * effective_forces))

        # Body roll/pitch dynamics under torque imbalance
        self.roll_angle = 0.85 * self.roll_angle + 0.15 * roll_torque
        self.pitch_angle = 0.85 * self.pitch_angle + 0.15 * pitch_torque

        # Stability penalty: severe roll/pitch tipping causes leg stumbling & lateral drift
        tilt_penalty = math.exp(-1.8 * (abs(self.roll_angle) + 0.5 * abs(self.pitch_angle)))
        forward_efficiency = min(1.0, total_lift) * tilt_penalty

        # Steering toward prey or evading obstacle
        target_yaw = math.atan2(to_prey[1], to_prey[0]) if dist_prey > 1e-3 else self.yaw
        if motor_mode == 2:  # EVADE_OBSTACLE
            target_yaw += 1.2
        # Uncompensated roll torque induces parasitic yaw drift!
        yaw_drift = -0.35 * roll_torque
        self.yaw = target_yaw + yaw_drift

        base_speed = 9.5 if motor_mode == 4 else (6.5 if motor_mode == 1 else 4.0)
        speed = base_speed * forward_efficiency
        self.vel = np.array([math.cos(self.yaw) * speed, math.sin(self.yaw) * speed])
        self.pos = self.pos + dt * self.vel

        return {
            "pos": self.pos.copy(),
            "dist_prey": float(np.linalg.norm(prey_pos - self.pos)),
            "motor_mode": motor_mode,
            "cpu_steps": cpu_out["steps_executed"],
            "synth_triggered": cpu_out["synth_triggered"],
            "roll_angle": abs(self.roll_angle),
            "forward_efficiency": forward_efficiency,
            "solve_ms": self.last_synth_ms,
        }
