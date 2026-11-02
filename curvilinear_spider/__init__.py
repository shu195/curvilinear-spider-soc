"""
curvilinear_spider: Differentiable Curvilinear-CPU Biomechanical Arachnid
=========================================================================
8-legged Theraphosidae spider driven by a zero-branching Curvilinear Von Neumann SoC
with < 1 ms closed-form Cholesky gait self-repair under limb amputation.
"""

from .core import (
    CurvilinearALUSlice,
    CurvilinearDecoder,
    CurvilinearNeuralSoC,
    CholeskyGaitSynthesizer,
    CurvilinearSpiderOrganism,
    DEFAULT_SPIDER_BRAIN_ASM,
)

__version__ = "0.1.0"
__all__ = [
    "CurvilinearALUSlice",
    "CurvilinearDecoder",
    "CurvilinearNeuralSoC",
    "CholeskyGaitSynthesizer",
    "CurvilinearSpiderOrganism",
    "DEFAULT_SPIDER_BRAIN_ASM",
]
