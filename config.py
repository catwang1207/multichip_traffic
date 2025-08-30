#!/usr/bin/env python3
"""
Centralized configuration for chiplet assignment optimization
"""

# =============================================================================
# CHIPLET HARDWARE PARAMETERS
# =============================================================================

# Maximum PEs per chiplet (hardware constraint)
MAX_PES_PER_CHIPLET = 144

# Inter-chiplet communication bandwidth (bytes per cycle)
INTER_CHIPLET_BANDWIDTH = 4608

# PE type strategy parameters
DEFAULT_IMC_RATIO = 15/16  # Default ratio of IMC PEs per chiplet for mixed strategy (66.7%)
DEFAULT_PE_TYPE_STRATEGY = 'mixed'  # Default strategy: 'separated' or 'mixed'

# =============================================================================
# GRASP OPTIMIZATION PARAMETERS  
# =============================================================================

# Small dataset parameters
MeEDIUM_DATASET = {
    'max_chiplets': 30,
    'timeout_seconds': 60,
    'rcl_size': 4,
    'ls_max_passes': 4,
    'pair_swap_samples': 500
}

# Large dataset parameters  
LARGE_DATASET = {
    'max_chiplets': 50,
    'timeout_seconds':60,
    'rcl_size': 4,
    'ls_max_passes': 4,
    'pair_swap_samples': 500
}

# ILP parameters (legacy - kept for backward compatibility)
ILP_CONFIG = {
    'max_pes_per_chiplet': 32,  # Legacy ILP used 32
    'inter_chiplet_bandwidth': 8192,  # Legacy ILP used 8192
    'timeout_seconds': 300
}

# =============================================================================
# COST FUNCTION WEIGHTS
# =============================================================================

# Cost function: cycle_weight * max_time + chiplet_penalty + violation_penalty * violations
COST_WEIGHTS = {
    'cycle_weight': 100,           # Cost per cycle
    'exponential_penalty_base': 5000,  # Base for exponential chiplet penalty
    'violation_penalty': 1000000   # Cost per constraint violation
}

# =============================================================================
# VALIDATION PARAMETERS
# =============================================================================

VALIDATION_CONFIG = {
    'max_pes_per_chiplet': MAX_PES_PER_CHIPLET,
    'inter_chiplet_bandwidth': INTER_CHIPLET_BANDWIDTH
}

# =============================================================================
# DATA FILES
# =============================================================================

DATA_FILES = {
    'small_dataset': 'gpt2_transformer_small.txt',
    'large_dataset': 'gpt2_transformer.txt'
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_config_for_dataset(dataset_size):
    """Get configuration parameters for a specific dataset size"""
    if dataset_size.lower() in ['small', 'gpt2_transformer_small.txt']:
        return SMALL_DATASET
    elif dataset_size.lower() in ['large', 'gpt2_transformer.txt']:
        return LARGE_DATASET
    else:
        # Default to small dataset config
        return SMALL_DATASET

def print_current_config():
    """Print current configuration parameters"""
    print("=== CURRENT CONFIGURATION ===")
    print(f"Max PEs per chiplet: {MAX_PES_PER_CHIPLET}")
    print(f"Inter-chiplet bandwidth: {INTER_CHIPLET_BANDWIDTH} bytes/cycle")
    print(f"Default PE type strategy: {DEFAULT_PE_TYPE_STRATEGY}")
    print(f"Default IMC ratio: {DEFAULT_IMC_RATIO:.1%}")
    print(f"Small dataset max chiplets: {SMALL_DATASET['max_chiplets']}")
    print(f"Large dataset max chiplets: {LARGE_DATASET['max_chiplets']}")
    print(f"Cycle cost weight: {COST_WEIGHTS['cycle_weight']}")
    print(f"Violation penalty: {COST_WEIGHTS['violation_penalty']}")

def get_imc_dig_counts(imc_ratio=DEFAULT_IMC_RATIO, max_pes=MAX_PES_PER_CHIPLET):
    """Calculate IMC and DIG PE counts per chiplet based on ratio"""
    imc_count = int(round(imc_ratio * max_pes))
    dig_count = max_pes - imc_count
    return imc_count, dig_count

def print_ratio_info(imc_ratio=DEFAULT_IMC_RATIO):
    """Print information about IMC/DIG ratio"""
    imc_count, dig_count = get_imc_dig_counts(imc_ratio)
    print(f"IMC ratio {imc_ratio:.1%} = {imc_count} IMC + {dig_count} DIG per {MAX_PES_PER_CHIPLET}-PE chiplet")

if __name__ == "__main__":
    print_current_config()
    print("\nRatio breakdown:")
    print_ratio_info()