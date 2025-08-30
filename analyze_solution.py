#!/usr/bin/env python3

"""
Analyze the PE distribution and constraint satisfaction in heterogeneous chiplet solutions
"""

import json
import os
from collections import defaultdict

def analyze_solution(solution_file, strategy_name):
    """Analyze a solution file and show detailed PE distribution"""
    
    if not os.path.exists(solution_file):
        print(f"Solution file {solution_file} not found!")
        return
    
    print(f"\n{'='*80}")
    print(f"ANALYZING {strategy_name.upper()}")
    print(f"{'='*80}")
    
    # Load solution
    with open(solution_file, 'r') as f:
        solution = json.load(f)
    
    # Handle different solution file formats
    metadata = solution.get('metadata', {})
    
    status = metadata.get('status') or solution.get('status', 'unknown')
    total_time = metadata.get('total_cycles') or solution.get('total_time', 'N/A')
    num_chiplets = metadata.get('num_chiplets') or solution.get('num_chiplets', 'N/A')
    algorithm = metadata.get('solver') or solution.get('algorithm', 'unknown')
    
    print(f"Status: {status}")
    print(f"Total Time: {total_time} cycles")
    print(f"Chiplets Used: {num_chiplets}")
    print(f"Algorithm: {algorithm}")
    
    # Analyze PE assignments
    pe_assignments = solution.get('pe_assignments', {})
    task_assignments = solution.get('task_assignments', {})
    
    # Group PEs by chiplet
    chiplets = defaultdict(list)
    for pe, chiplet in pe_assignments.items():
        chiplets[chiplet].append(int(pe))
    
    print(f"\nPE DISTRIBUTION:")
    print(f"{'Chiplet':<8} {'PEs':<20} {'Count':<6} {'IMC':<4} {'DIG':<4} {'Ratio':<10}")
    print("-" * 60)
    
    # Load PE types from CSV (assuming it exists)
    pe_types = {}
    csv_file = "traffic_table_gpt2_seq32_layers1_20250829_172134.csv"
    if os.path.exists(csv_file):
        import pandas as pd
        df = pd.read_csv(csv_file)
        
        # Check available columns
        cols = df.columns.tolist()
        print(f"CSV columns: {cols[:5]}...")  # Show first 5 columns for debugging
        
        # Try different column name formats
        if 'src_type' in df.columns and 'dst_type' in df.columns:
            # Current format with src_type/dst_type
            for _, row in df.iterrows():
                # Extract PE number from "P0", "P1", etc. format
                src_pe_str = str(row['src_pe'])
                dst_pe_str = str(row['dest_pe'])
                
                if src_pe_str.startswith('P') and src_pe_str != 'P-1':
                    pe_id = int(src_pe_str[1:])  # Remove 'P' prefix
                    if pd.notna(row['src_type']):
                        pe_types[pe_id] = row['src_type']
                        
                if dst_pe_str.startswith('P') and dst_pe_str != 'P-1':
                    pe_id = int(dst_pe_str[1:])  # Remove 'P' prefix
                    if pd.notna(row['dst_type']):
                        pe_types[pe_id] = row['dst_type']
                        
        elif 'source_pe_type' in df.columns and 'dest_pe_type' in df.columns:
            # Legacy v1 format
            for _, row in df.iterrows():
                if pd.notna(row['source_pe_type']) and row['source_pe'] != -1:
                    pe_types[int(row['source_pe'])] = row['source_pe_type']
                if pd.notna(row['dest_pe_type']) and row['dest_pe'] != -1:
                    pe_types[int(row['dest_pe'])] = row['dest_pe_type']
        elif 'pe_type_src' in df.columns and 'pe_type_dst' in df.columns:
            # Legacy v2 format
            for _, row in df.iterrows():
                if pd.notna(row['pe_type_src']) and row['source_pe'] != -1:
                    pe_types[int(row['source_pe'])] = row['pe_type_src']
                if pd.notna(row['pe_type_dst']) and row['dest_pe'] != -1:
                    pe_types[int(row['dest_pe'])] = row['pe_type_dst']
    
    total_imc = 0
    total_dig = 0
    
    for chiplet_id in sorted(chiplets.keys()):
        pes = sorted(chiplets[chiplet_id])
        count = len(pes)
        
        # Count IMC and DIG
        imc_count = sum(1 for pe in pes if pe_types.get(pe) == 'IMC')
        dig_count = sum(1 for pe in pes if pe_types.get(pe) == 'DIG')
        other_count = count - imc_count - dig_count
        
        ratio = f"{imc_count}/{count}" if count > 0 else "0/0"
        if count > 0:
            ratio += f" ({100*imc_count/count:.1f}%)"
        
        total_imc += imc_count
        total_dig += dig_count
        
        pes_str = str(pes[:5])  # Show first 5 PEs
        if len(pes) > 5:
            pes_str = pes_str[:-1] + f", ...+{len(pes)-5}]"
            
        print(f"{chiplet_id:<8} {pes_str:<20} {count:<6} {imc_count:<4} {dig_count:<4} {ratio:<10}")
        
        if other_count > 0:
            print(f"         Warning: {other_count} PEs with unknown/other type")
    
    print("-" * 60)
    total_typed = total_imc + total_dig
    print(f"{'TOTAL':<8} {'':<20} {sum(len(pes) for pes in chiplets.values()):<6} {total_imc:<4} {total_dig:<4}")
    if total_typed > 0:
        print(f"Overall ratio: {total_imc}/{total_typed} ({100*total_imc/total_typed:.1f}% IMC)")
    else:
        print(f"Overall ratio: No PE types detected")
        
    print(f"PE types loaded: {len(pe_types)} PEs with types: {set(pe_types.values()) if pe_types else 'None'}")
    
    # Check bandwidth constraints by analyzing the tasks
    print(f"\nBANDWITH CONSTRAINT ANALYSIS:")
    
    # Load original task data to check inter vs intra chiplet communication
    csv_file = "traffic_table_gpt2_seq32_layers1_20250829_172134.csv"
    inter_chiplet_tasks = 0
    intra_chiplet_tasks = 0
    
    if os.path.exists(csv_file):
        import pandas as pd
        df = pd.read_csv(csv_file)
        
        for _, row in df.iterrows():
            # Extract PE numbers from "P0", "P1", etc. format
            src_pe_str = str(row['src_pe'])
            dst_pe_str = str(row['dest_pe'])
            
            # Skip external DRAM (P-1)
            if src_pe_str == 'P-1' or dst_pe_str == 'P-1':
                continue
            
            if not (src_pe_str.startswith('P') and dst_pe_str.startswith('P')):
                continue
                
            src_pe = int(src_pe_str[1:])  # Remove 'P' prefix
            dst_pe = int(dst_pe_str[1:])  # Remove 'P' prefix
                
            src_chiplet = pe_assignments.get(str(src_pe))
            dst_chiplet = pe_assignments.get(str(dst_pe))
            
            if src_chiplet is not None and dst_chiplet is not None:
                if src_chiplet == dst_chiplet:
                    intra_chiplet_tasks += 1
                else:
                    inter_chiplet_tasks += 1
        
        print(f"Intra-chiplet tasks (same chiplet): {intra_chiplet_tasks}")
        print(f"Inter-chiplet tasks (different chiplets): {inter_chiplet_tasks}")
        
        if inter_chiplet_tasks > 0:
            bandwidth_efficiency = intra_chiplet_tasks / (intra_chiplet_tasks + inter_chiplet_tasks) * 100
            print(f"Bandwidth efficiency: {bandwidth_efficiency:.1f}% (higher = more intra-chiplet)")
    else:
        print("Could not analyze bandwidth - CSV file not found")
    
    # Capacity constraints
    print(f"\nCAPACITY CONSTRAINT ANALYSIS:")
    max_pes_per_chiplet = 9  # From config
    capacity_violations = 0
    
    for chiplet_id, pes in chiplets.items():
        if len(pes) > max_pes_per_chiplet:
            capacity_violations += len(pes) - max_pes_per_chiplet
            print(f"  Chiplet {chiplet_id}: {len(pes)} PEs (exceeds {max_pes_per_chiplet} limit)")
    
    if capacity_violations == 0:
        print("  ✅ All chiplets within capacity limits")
    else:
        print(f"  ❌ {capacity_violations} capacity violations")
    
    return {
        'chiplets': dict(chiplets),
        'pe_types': pe_types,
        'total_imc': total_imc,
        'total_dig': total_dig,
        'capacity_violations': capacity_violations
    }

def main():
    """Analyze solutions from different strategies"""
    
    solution_file = "traffic_table_gpt2_seq32_layers1_20250829_172134_gr_solution.json"
    
    # The test saves the last solution, so we need to run each strategy separately
    # or modify the test to save different files
    
    print("To get separate analyses for each strategy, we need to run them individually.")
    print("Let's analyze the current solution file:")
    
    if os.path.exists(solution_file):
        analyze_solution(solution_file, "Last Run (Mixed Strategy)")
    else:
        print(f"Solution file {solution_file} not found!")
        print("Please run the test first: python test_hetero_framework.py")

if __name__ == "__main__":
    main()