#!/usr/bin/env python3

"""
Test to show type mixing violations by restricting local search.
"""

from gr_hetro_chiplet_framework import ChipletProblemGRASP

def test_with_violations():
    """Test pure separation with limited local search to show violations"""
    print("=" * 80)
    print("TESTING PURE SEPARATION WITH TYPE MIXING VIOLATIONS")
    print("Using only 1 local search pass to show violation penalties")
    print("=" * 80)
    
    traffic_file = "traffic_table_gpt2_seq32_layers1_20250829_172134.csv"
    
    problem = ChipletProblemGRASP(traffic_file)
    
    result = problem.solve(
        timeout=30,
        max_chiplets=4,  # Force constraint on chiplets 
        pe_type_strategy='separated',
        rcl_size=3,
        starts=1,
        ls_max_passes=1,  # Minimal local search to preserve violations
        pair_swap_samples=100  # Fewer samples to avoid fixing all violations
    )
    
    print(f"Pure Separation with Limited Local Search:")
    print(f"  Status: {result['status']}")
    print(f"  Total Time: {result.get('total_time', 'N/A')} cycles")
    print(f"  Chiplets Used: {result.get('num_chiplets', 'N/A')}")
    print(f"  Solve Time: {result.get('solve_time', 'N/A'):.2f} seconds")
    
    if 'violations' in result:
        print(f"  Violations: {result['violations']}")
        total_violations = sum(result['violations'].values())
        print(f"  Total Violations: {total_violations}")
        if 'pe_type_separation' in result['violations']:
            print(f"  Type Mixing Violations: {result['violations']['pe_type_separation']}")

if __name__ == "__main__":
    test_with_violations()