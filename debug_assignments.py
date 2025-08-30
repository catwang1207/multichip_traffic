#!/usr/bin/env python3

"""
Debug script to examine actual PE assignments and verify if pure separation is really happening.
"""

import sys
import os
from gr_hetro_chiplet_framework import ChipletProblemGRASP

def debug_pe_assignments():
    """Debug what's actually happening with PE assignments"""
    print("=" * 80)
    print("DEBUGGING PE ASSIGNMENTS - IS PURE SEPARATION REALLY HAPPENING?")
    print("=" * 80)
    
    traffic_file = "traffic_table_gpt2_seq32_layers1_20250829_172134.csv"
    
    problem = ChipletProblemGRASP(traffic_file)
    
    print(f"PE Types loaded: {problem.pe_types}")
    
    # Count PEs by type
    imc_count = sum(1 for t in problem.pe_types.values() if t == 'IMC')
    dig_count = sum(1 for t in problem.pe_types.values() if t == 'DIG')
    print(f"IMC PEs: {imc_count}, DIG PEs: {dig_count}")
    print(f"Pure separation needs: {(imc_count + 8) // 9} IMC chiplets + {(dig_count + 8) // 9} DIG chiplets = {(imc_count + 8) // 9 + (dig_count + 8) // 9} total chiplets")
    
    # Check which PEs are which types
    imc_pes = [pe for pe, t in problem.pe_types.items() if t == 'IMC']
    dig_pes = [pe for pe, t in problem.pe_types.items() if t == 'DIG']
    print(f"IMC PE IDs: {sorted(imc_pes)}")
    print(f"DIG PE IDs: {sorted(dig_pes)}")
    print(f"Total PEs in dataset: {len(problem.pes)}")
    print(f"Total PEs with types: {len(problem.pe_types)}")
    
    # Test with pure separation strategy
    from gr_hetro_chiplet_framework import HeuristicSolution
    import random
    
    sol = HeuristicSolution(problem, max_chiplets=4, rng=random.Random(42), 
                           pe_type_strategy='separated', imc_ratio=0.75)
    sol.grasp_construct(rcl_size=3)
    
    print(f"\\nActual solution uses {sol.pe_assignments}")
    print(f"Chiplet assignments:")
    
    for chiplet in range(4):
        pes_on_chiplet = [pe for pe, chip in sol.pe_assignments.items() if chip == chiplet]
        if pes_on_chiplet:
            types_on_chiplet = [problem.pe_types.get(pe, 'UNKNOWN') for pe in pes_on_chiplet]
            type_counts = {}
            for t in types_on_chiplet:
                type_counts[t] = type_counts.get(t, 0) + 1
            print(f"  Chiplet {chiplet}: {len(pes_on_chiplet)} PEs - {type_counts}")
            
            # Check if this chiplet has mixed types
            unique_types = set(types_on_chiplet)
            if len(unique_types) > 1:
                print(f"    ❌ TYPE MIXING DETECTED on chiplet {chiplet}!")
            else:
                print(f"    ✅ Pure type: {unique_types}")
    
    # Check violations manually
    violations = sol._count_type_mixing_violations()
    print(f"\\nType mixing violations found: {violations}")
    
    if violations == 0 and len(set(problem.pe_types.values())) > 1:
        print("\\n🚨 BUG DETECTED: Claims 0 violations but this should be impossible with 4 chiplets!")

if __name__ == "__main__":
    debug_pe_assignments()