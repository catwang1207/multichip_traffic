#!/usr/bin/env python3

import json
import pandas as pd
from collections import defaultdict
from config import get_imc_dig_counts, MAX_PES_PER_CHIPLET, DEFAULT_IMC_RATIO

# Load the mixed solution
with open('traffic_table_medium_gr_mixed_solution.json', 'r') as f:
    solution = json.load(f)

# Load PE types
df = pd.read_csv('traffic_table_medium.csv')
pe_types = {}

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

for _, row in df.iterrows():
    source_pe = _parse_pe_id(row['src_pe'])
    dest_pe = _parse_pe_id(row['dest_pe'])
    src_type = _norm_type(row['src_type'])
    dst_type = _norm_type(row['dst_type'])
    pe_types[source_pe] = src_type
    pe_types[dest_pe] = dst_type

# Analyze PE assignments by chiplet
pe_assignments = solution['pe_assignments']
chiplets = defaultdict(list)

for pe_str, chiplet in pe_assignments.items():
    pe = int(pe_str)
    chiplets[chiplet].append(pe)

# Get mixed strategy limits
max_imc, max_dig = get_imc_dig_counts(DEFAULT_IMC_RATIO, MAX_PES_PER_CHIPLET)

print(f"Mixed Strategy Constraints: Max {max_imc} IMC, Max {max_dig} DIG per chiplet")
print(f"Total chiplets in solution: {len(chiplets)}")
print()

violations = 0
for chiplet_id in sorted(chiplets.keys()):
    pes = chiplets[chiplet_id]
    imc_count = sum(1 for pe in pes if pe_types.get(pe) == 'IMC')
    dig_count = sum(1 for pe in pes if pe_types.get(pe) == 'DIG')
    other_count = len(pes) - imc_count - dig_count
    
    # Check violations
    imc_violation = max(0, imc_count - max_imc)
    dig_violation = max(0, dig_count - max_dig)
    total_violation = imc_violation + dig_violation
    
    status = "✅" if total_violation == 0 else "❌"
    print(f"Chiplet {chiplet_id:2d}: {len(pes)} PEs ({imc_count} IMC, {dig_count} DIG, {other_count} other) {status}")
    
    if total_violation > 0:
        violations += total_violation
        if imc_violation > 0:
            print(f"    ❌ IMC violation: {imc_count} > {max_imc} (excess: {imc_violation})")
        if dig_violation > 0:
            print(f"    ❌ DIG violation: {dig_count} > {max_dig} (excess: {dig_violation})")

print()
print(f"Total constraint violations: {violations}")
if violations == 0:
    print("🎉 Solution is truly VALID for mixed strategy!")
else:
    print("❌ Solution has real constraint violations!")