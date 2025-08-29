#!/usr/bin/env python3
"""
Centralized configuration for chiplet assignment optimization
"""

# =============================================================================
# CHIPLET HARDWARE PARAMETERS
# =============================================================================

# Maximum PEs per chiplet (hardware constraint)
MAX_PES_PER_CHIPLET = 36

# Inter-chiplet communication bandwidth (bytes per cycle)
INTER_CHIPLET_BANDWIDTH = 4096

# =============================================================================
# OPTIMIZATION PARAMETERS  
# =============================================================================

# Small dataset parameters
SMALL_DATASET = {
    'max_chiplets': 6,
    'timeout_seconds': 10,
    'max_iterations': 100000,
    'initial_temp': 2000.0,
    'cooling_rate': 0.998,
    'max_no_improvement': 10000
}

# Large dataset parameters  
LARGE_DATASET = {
    'max_chiplets': 8,
    'timeout_seconds': 10,
    'max_iterations': 100000,
    'initial_temp': 2000.0,
    'cooling_rate': 0.998,
    'max_no_improvement': 10000
}

# ILP parameters (legacy)
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
    'base_chiplet_cost': 500,      # Cost per chiplet (minimum configuration)
    'exponential_penalty_base': 2000,  # Base for exponential chiplet penalty
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
    print(f"Small dataset max chiplets: {SMALL_DATASET['max_chiplets']}")
    print(f"Large dataset max chiplets: {LARGE_DATASET['max_chiplets']}")
    print(f"Cycle cost weight: {COST_WEIGHTS['cycle_weight']}")
    print(f"Violation penalty: {COST_WEIGHTS['violation_penalty']}")

if __name__ == "__main__":
    print_current_config()