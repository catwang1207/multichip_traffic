#!/usr/bin/env python3

import pandas as pd
from config import get_imc_dig_counts, MAX_PES_PER_CHIPLET, DEFAULT_IMC_RATIO

print("=== DEBUGGING MIXED STRATEGY MIN CHIPLETS CALCULATION ===")
print(f"Config values:")
print(f"  MAX_PES_PER_CHIPLET = {MAX_PES_PER_CHIPLET}")
print(f"  DEFAULT_IMC_RATIO = {DEFAULT_IMC_RATIO}")

# Load PE types from large dataset
try:
    df = pd.read_csv('traffic_table_large.csv')
    
    def _norm_type(s):
        u = str(s).strip().upper()
        if u in ('DIGITAL', 'DIG'):
            return 'DIG'
        if u == 'IMC':
            return 'IMC'
        return u

    def _parse_pe_id(pe_str):
        pe_str = str(pe_str).strip()
        if pe_str.startswith('P'):
            return int(pe_str[1:])
        else:
            return int(pe_str)

    pe_types = {}
    for _, row in df.iterrows():
        source_pe = _parse_pe_id(row['src_pe'])
        dest_pe = _parse_pe_id(row['dest_pe'])
        src_type = _norm_type(row['src_type'])
        dst_type = _norm_type(row['dst_type'])
        pe_types[source_pe] = src_type
        pe_types[dest_pe] = dst_type

    # Count PEs by type
    imc_count = sum(1 for pe_type in pe_types.values() if pe_type == 'IMC')
    dig_count = sum(1 for pe_type in pe_types.values() if pe_type == 'DIG')
    total_pes = len(pe_types)
    other_count = total_pes - imc_count - dig_count

    print(f"\nPE counts from large dataset:")
    print(f"  IMC PEs: {imc_count}")
    print(f"  DIG PEs: {dig_count}")
    print(f"  Other PEs: {other_count}")
    print(f"  Total PEs: {total_pes}")

    # Calculate mixed strategy quotas
    imc_per, dig_per = get_imc_dig_counts(DEFAULT_IMC_RATIO, MAX_PES_PER_CHIPLET)
    print(f"\nMixed strategy quotas per chiplet:")
    print(f"  IMC per chiplet: {imc_per}")
    print(f"  DIG per chiplet: {dig_per}")
    print(f"  Total per chiplet: {imc_per + dig_per}")

    # Current calculation
    imc_per = max(0, min(imc_per, MAX_PES_PER_CHIPLET))
    dig_per = max(0, min(dig_per, MAX_PES_PER_CHIPLET))
    need_imc = (imc_count + max(1, imc_per) - 1) // max(1, imc_per) if imc_count > 0 else 0
    need_dig = (dig_count + max(1, dig_per) - 1) // max(1, dig_per) if dig_count > 0 else 0
    need_other = (other_count + MAX_PES_PER_CHIPLET - 1) // MAX_PES_PER_CHIPLET

    print(f"\nMinimum chiplets needed:")
    print(f"  For {imc_count} IMC PEs at {imc_per}/chiplet: {need_imc} chiplets")
    print(f"  For {dig_count} DIG PEs at {dig_per}/chiplet: {need_dig} chiplets")
    print(f"  For {other_count} other PEs at {MAX_PES_PER_CHIPLET}/chiplet: {need_other} chiplets")
    
    current_result = max(1, need_imc, need_dig, need_other)
    print(f"  Current result: max({need_imc}, {need_dig}, {need_other}) = {current_result}")

    # Check if this makes sense
    total_capacity_needed = imc_count + dig_count + other_count
    simple_min = (total_capacity_needed + MAX_PES_PER_CHIPLET - 1) // MAX_PES_PER_CHIPLET
    
    print(f"\nSanity check:")
    print(f"  Total PEs needing assignment: {total_capacity_needed}")
    print(f"  Simple capacity-based minimum: {simple_min} chiplets")
    print(f"  Mixed strategy minimum: {current_result} chiplets")
    
    if current_result < simple_min:
        print(f"  ⚠️  WARNING: Mixed strategy minimum ({current_result}) < capacity minimum ({simple_min})")
        print(f"      This suggests the calculation might be incorrect!")
    else:
        print(f"  ✅ Mixed strategy minimum >= capacity minimum (OK)")

except Exception as e:
    print(f"Error: {e}")