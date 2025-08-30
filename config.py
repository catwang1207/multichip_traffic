#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone configuration for the heterogeneous chiplet framework.

IMPORTANT:
- Do NOT import anything here. This avoids circular imports.
- Other modules (solver, tests) import *from* this file.
"""

# Capacity per chiplet (number of PE slots available on a chiplet)
MAX_PES_PER_CHIPLET = 576  # adjust to your platform

# Inter-chiplet bandwidth used to compute transfer duration (bytes per cycle)
INTER_CHIPLET_BANDWIDTH = 1024  # example value; tune to your model

# Default PE-type strategy/ratio
DEFAULT_PE_TYPE_STRATEGY = "mixed"   # or "separated"
DEFAULT_IMC_RATIO = 15 / 16          # e.g., 0.9375 for mixed strategy defaults

# Cost weights for the objective
COST_WEIGHTS = {
    "cycle_weight": 100,                # weight on makespan (cycles)
    "exponential_penalty_base": 5000,   # base for extra-chiplet exponential penalty
    "violation_penalty": 1000000,        # penalty per constraint violation
}

# Preset for larger dataset runs used in the sweep script
LARGE_DATASET = {
    "timeout_seconds": 60.0,  # per run
    "rcl_size": 1,
    "max_chiplets": 50,
}

def get_imc_dig_counts(ratio: float, K: int):
    """
    Convert an IMC ratio (0..1) and per-chiplet capacity K into integer quotas.

    Returns:
        (imc_max, dig_max): maximum IMC and DIG PEs allowed per chiplet.
    """
    # Round to nearest while keeping bounds
    imc = int(round(ratio * K))
    if imc < 0:
        imc = 0
    if imc > K:
        imc = K
    dig = K - imc
    return imc, dig
