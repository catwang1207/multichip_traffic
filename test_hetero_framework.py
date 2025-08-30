#!/usr/bin/env python3

"""
Test script for the modified heterogeneous chiplet framework with PE type support.
Tests both Strategy 1 (separated) and Strategy 2 (mixed) approaches.
"""

import sys
import os
from gr_hetro_chiplet_framework import ChipletProblemGRASP
from config import LARGE_DATASET, DEFAULT_IMC_RATIO

def test_strategy_1():
    """Test Strategy 1: Pure separation of IMC and DIG PEs"""
    print("=" * 80)
    print("TESTING STRATEGY 1: Pure Separation (IMC-only and DIG-only chiplets)")
    print("=" * 80)
    
    traffic_file = "traffic_table_large.csv"
    
    if not os.path.exists(traffic_file):
        print(f"Error: Traffic file {traffic_file} not found!")
        return None
        
    try:
        # Create problem instance
        problem = ChipletProblemGRASP(traffic_file)
        
        # Solve with Strategy 1 (separated)
        result = problem.solve(
            timeout=LARGE_DATASET['timeout_seconds'],
            max_chiplets=LARGE_DATASET['max_chiplets'],  # Use config value (30)
            pe_type_strategy='separated',
            rcl_size=LARGE_DATASET['rcl_size'],
            starts=3,  # Just a few starts for testing
            solution_file_suffix='_separated'
        )
        
        print(f"Strategy 1 Result:")
        print(f"  Status: {result['status']}")
        print(f"  Total Time: {result.get('total_time', 'N/A')} cycles")
        print(f"  Chiplets Used: {result.get('num_chiplets', 'N/A')}")
        print(f"  Solve Time: {result.get('solve_time', 'N/A'):.2f} seconds")
        if 'violations' in result:
            print(f"  Violations: {result['violations']}")
            
        return result
        
    except Exception as e:
        print(f"Error testing Strategy 1: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_strategy_2():
    """Test Strategy 2: Mixed chiplets with 27 IMC + 9 DIG per chiplet"""
    print("=" * 80)
    print("TESTING STRATEGY 2: Mixed Chiplets")
    print("=" * 80)
    
    traffic_file = "traffic_table_large.csv"
    
    if not os.path.exists(traffic_file):
        print(f"Error: Traffic file {traffic_file} not found!")
        return None
        
    try:
        # Create problem instance
        problem = ChipletProblemGRASP(traffic_file)
        
        # Solve with Strategy 2 (mixed using config ratio)
        result = problem.solve(
            timeout=LARGE_DATASET['timeout_seconds'],
            max_chiplets=LARGE_DATASET['max_chiplets'],  # Use config value (50)
            pe_type_strategy='mixed',
            imc_ratio=DEFAULT_IMC_RATIO,  # Use config value (15/16 = 93.75%)
            rcl_size=LARGE_DATASET['rcl_size'],
            starts=3,  # Just a few starts for testing
            solution_file_suffix='_mixed'
        )
        
        print(f"Strategy 2 Result:")
        print(f"  Status: {result['status']}")
        print(f"  Total Time: {result.get('total_time', 'N/A')} cycles")
        print(f"  Chiplets Used: {result.get('num_chiplets', 'N/A')}")
        print(f"  Solve Time: {result.get('solve_time', 'N/A'):.2f} seconds")
        if 'violations' in result:
            print(f"  Violations: {result['violations']}")
            
        return result
        
    except Exception as e:
        print(f"Error testing Strategy 2: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("Testing the heterogeneous chiplet framework")
    print("=" * 80)
    
    # Test only mixed strategy
    strategy2_result = test_strategy_2()
    
    # Summary
    print("=" * 80)
    print("STRATEGY 2 RESULT")
    print("=" * 80)
    
    if strategy2_result:
        print(f"Strategy 2 (mixed): {strategy2_result.get('total_time', 'N/A'):>8} cycles, "
              f"{strategy2_result.get('num_chiplets', 'N/A'):>2} chiplets, "
              f"{strategy2_result.get('solve_time', 0):>6.2f}s")
    else:
        print("Strategy 2 (mixed): FAILED")

if __name__ == "__main__":
    main()