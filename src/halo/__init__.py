"""HALO (Human-AI Layered Oversight).

Separates oversight allocation (how much review a case gets) from release
authorization (whether the required review actually happened for the exact
output being released).

    halo.gate          the release gate
    halo.adapter       places the gate in front of any generate(prompt) callable
    halo.verification  oracle, static checker, scenarios, mutants, randomized suite
    halo.simulation    discrete-event simulator and allocation policies
    halo.experiments   the paper's analyses
"""

from .adapter import AgentAdapter
from .gate import Contract, HoldReason, Receipt, ReleaseGate, output_hash

__version__ = "1.1.0"
__all__ = ["AgentAdapter", "Contract", "HoldReason", "Receipt", "ReleaseGate", "output_hash"]
