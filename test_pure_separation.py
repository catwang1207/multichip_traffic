#!/usr/bin/env python3

"""
Test pure separation by allowing enough chiplets for type separation.
"""

from gr_hetro_chiplet_framework import ChipletProblemGRASP

def test_with_more_chiplets():
    """Test with higher chiplet limit to allow pure separation"""
    print("=" * 80)
    print("TESTING WITH HIGHER CHIPLET LIMIT TO ALLOW PURE SEPARATION")
    print("Using 8 max chiplets to allow pure IMC/DIG separation")
    print("=" * 80)
    
    traffic_file = "traffic_table_gpt2_seq32_layers1_20250829_172134.csv"
    
    results = {}
    
    for strategy, name in [('ignore', 'Baseline'), ('separated', 'Strategy 1 (Pure Sep)'), ('mixed', 'Strategy 2 (Mixed)')]:
        print(f"\n--- Testing {name} ---")
        
        problem = ChipletProblemGRASP(traffic_file)
        
        result = problem.solve(
            timeout=30,
            max_chiplets=8,  # Allow up to 8 chiplets
            pe_type_strategy=strategy,
            imc_ratio=0.67,  # 6 IMC + 3 DIG for mixed strategy
            rcl_size=3,
            starts=2
        )
        
        results[strategy] = result
        print(f"{name} Result:")
        print(f"  Status: {result['status']}")
        print(f"  Total Time: {result.get('total_time', 'N/A')} cycles")
        print(f"  Chiplets Used: {result.get('num_chiplets', 'N/A')}")
        print(f"  Solve Time: {result.get('solve_time', 'N/A'):.2f} seconds")
        
    print("\n" + "=" * 80)
    print("COMPARISON WITH 8 MAX CHIPLETS")
    print("=" * 80)
    for strategy, name in [('ignore', 'Baseline'), ('separated', 'Pure Separation'), ('mixed', 'Mixed Strategy')]:
        result = results[strategy]
        print(f"{name:15}: {result.get('total_time', 'N/A'):>6} cycles, {result.get('num_chiplets', 'N/A'):>2} chiplets")

if __name__ == "__main__":
    test_with_more_chiplets()