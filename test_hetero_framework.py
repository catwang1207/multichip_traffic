#!/usr/bin/env python3

"""
Test script for the modified heterogeneous chiplet framework with PE type support.
Tests both Strategy 1 (separated) and Strategy 2 (mixed) approaches.
"""

import sys
import os
from gr_hetro_chiplet_framework import ChipletProblemGRASP

def test_strategy_1():
    """Test Strategy 1: Pure separation of IMC and DIG PEs"""
    print("=" * 80)
    print("TESTING STRATEGY 1: Pure Separation (IMC-only and DIG-only chiplets)")
    print("=" * 80)
    
    traffic_file = "traffic_table_gpt2_seq32_layers1_20250829_172134.csv"
    
    if not os.path.exists(traffic_file):
        print(f"Error: Traffic file {traffic_file} not found!")
        return None
        
    try:
        # Create problem instance
        problem = ChipletProblemGRASP(traffic_file)
        
        # Solve with Strategy 1 (separated)
        result = problem.solve(
            timeout=30,  # Short timeout for testing
            max_chiplets=8,
            pe_type_strategy='separated',
            rcl_size=3,
            starts=3  # Just a few starts for testing
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
    print("TESTING STRATEGY 2: Mixed Chiplets (27 IMC + 9 DIG per chiplet)")
    print("=" * 80)
    
    traffic_file = "traffic_table_gpt2_seq32_layers1_20250829_172134.csv"
    
    if not os.path.exists(traffic_file):
        print(f"Error: Traffic file {traffic_file} not found!")
        return None
        
    try:
        # Create problem instance
        problem = ChipletProblemGRASP(traffic_file)
        
        # Solve with Strategy 2 (mixed with 75% IMC, 25% DIG)
        result = problem.solve(
            timeout=30,  # Short timeout for testing
            max_chiplets=8,
            pe_type_strategy='mixed',
            imc_ratio=0.75,  # 27 out of 36 PEs per chiplet
            rcl_size=3,
            starts=3  # Just a few starts for testing
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

def test_baseline():
    """Test baseline: Original algorithm ignoring PE types"""
    print("=" * 80)
    print("TESTING BASELINE: Original Algorithm (ignoring PE types)")
    print("=" * 80)
    
    traffic_file = "traffic_table_gpt2_seq32_layers1_20250829_172134.csv"
    
    if not os.path.exists(traffic_file):
        print(f"Error: Traffic file {traffic_file} not found!")
        return None
        
    try:
        # Create problem instance
        problem = ChipletProblemGRASP(traffic_file)
        
        # Solve with baseline (ignore types)
        result = problem.solve(
            timeout=30,  # Short timeout for testing
            max_chiplets=8,
            pe_type_strategy='ignore',
            rcl_size=3,
            starts=3  # Just a few starts for testing
        )
        
        print(f"Baseline Result:")
        print(f"  Status: {result['status']}")
        print(f"  Total Time: {result.get('total_time', 'N/A')} cycles")
        print(f"  Chiplets Used: {result.get('num_chiplets', 'N/A')}")
        print(f"  Solve Time: {result.get('solve_time', 'N/A'):.2f} seconds")
        if 'violations' in result:
            print(f"  Violations: {result['violations']}")
            
        return result
        
    except Exception as e:
        print(f"Error testing baseline: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("Testing the modified heterogeneous chiplet framework")
    print("=" * 80)
    
    # Test all three approaches
    baseline_result = test_baseline()
    strategy1_result = test_strategy_1()
    strategy2_result = test_strategy_2()
    
    # Summary comparison
    print("=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    
    results = [
        ("Baseline (ignore types)", baseline_result),
        ("Strategy 1 (separated)", strategy1_result),
        ("Strategy 2 (mixed 27:9)", strategy2_result)
    ]
    
    for name, result in results:
        if result:
            print(f"{name:25}: {result.get('total_time', 'N/A'):>8} cycles, "
                  f"{result.get('num_chiplets', 'N/A'):>2} chiplets, "
                  f"{result.get('solve_time', 0):>6.2f}s")
        else:
            print(f"{name:25}: FAILED")

if __name__ == "__main__":
    main()