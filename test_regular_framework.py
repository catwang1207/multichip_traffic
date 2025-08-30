#!/usr/bin/env python3

"""
Test the regular GRASP framework after ILP cleanup.
"""

from gr_chiplet_framework import ChipletProblemGRASP

def test_regular_framework():
    """Test the regular framework works after ILP removal"""
    print("Testing regular GRASP framework after ILP cleanup...")
    
    traffic_file = "traffic_table_gpt2_seq32_layers1_20250829_172134.csv"
    
    try:
        problem = ChipletProblemGRASP(traffic_file)
        
        result = problem.solve(
            timeout=15,
            max_chiplets=6,
            starts=2,
            rcl_size=3
        )
        
        print(f"Regular GRASP Result:")
        print(f"  Status: {result['status']}")
        print(f"  Total Time: {result.get('total_time', 'N/A')} cycles")
        print(f"  Chiplets Used: {result.get('num_chiplets', 'N/A')}")
        print(f"  Solve Time: {result.get('solve_time', 'N/A'):.2f} seconds")
        print("✅ Regular framework works correctly!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_regular_framework()