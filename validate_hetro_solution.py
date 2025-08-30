#!/usr/bin/env python3

"""
Heterogeneous Chiplet Solution Validator
Validates solutions from gr_hetro_chiplet_framework.py with PE type constraints
"""

import json
import pandas as pd
import sys
import os
from collections import defaultdict
from config import VALIDATION_CONFIG, get_imc_dig_counts, MAX_PES_PER_CHIPLET, DEFAULT_IMC_RATIO

class HeteroSolutionValidator:
    """Validator for heterogeneous chiplet solutions with PE type constraints"""
    
    def __init__(self, solution_file, csv_file=None, strategy=None):
        self.solution_file = solution_file
        self.csv_file = csv_file
        self.strategy = strategy
        self.solution = None
        self.traffic_data = None
        self.pe_types = {}
        self.violations = []
        
    def load_solution(self):
        """Load solution from JSON file"""
        try:
            with open(self.solution_file, 'r') as f:
                self.solution = json.load(f)
            return True
        except Exception as e:
            print(f"❌ Error loading solution file {self.solution_file}: {e}")
            return False
    
    def load_traffic_data(self):
        """Load traffic data from CSV file with PE types"""
        if not self.csv_file:
            print("ℹ️  No CSV file provided - skipping PE type validation")
            return True
            
        try:
            self.traffic_data = pd.read_csv(self.csv_file)
            
            # Extract PE types from CSV
            if 'src_type' in self.traffic_data.columns and 'dst_type' in self.traffic_data.columns:
                for _, row in self.traffic_data.iterrows():
                    # Extract PE number from "P0", "P1", etc. format
                    src_pe_str = str(row['src_pe'])
                    dst_pe_str = str(row['dest_pe'])
                    
                    if src_pe_str.startswith('P') and src_pe_str != 'P-1':
                        pe_id = int(src_pe_str[1:])  # Remove 'P' prefix
                        if pd.notna(row['src_type']):
                            self.pe_types[pe_id] = row['src_type']
                            
                    if dst_pe_str.startswith('P') and dst_pe_str != 'P-1':
                        pe_id = int(dst_pe_str[1:])  # Remove 'P' prefix
                        if pd.notna(row['dst_type']):
                            self.pe_types[pe_id] = row['dst_type']
            
            print(f"✅ Loaded PE types: {len(self.pe_types)} PEs ({dict(pd.Series(list(self.pe_types.values())).value_counts())})")
            return True
            
        except Exception as e:
            print(f"❌ Error loading CSV file {self.csv_file}: {e}")
            return False
    
    def validate_basic_constraints(self):
        """Validate basic solution structure and constraints"""
        print("\\n=== BASIC CONSTRAINT VALIDATION ===")
        
        # Check solution structure
        pe_assignments = self.solution.get('pe_assignments', {})
        if not pe_assignments:
            self.violations.append("No PE assignments found in solution")
            return False
            
        metadata = self.solution.get('metadata', {})
        status = metadata.get('status', 'unknown')
        total_time = metadata.get('total_cycles', 'unknown')
        num_chiplets = metadata.get('num_chiplets', 'unknown')
        
        print(f"Status: {status}")
        print(f"Total Time: {total_time} cycles")
        print(f"Chiplets Used: {num_chiplets}")
        print(f"PEs Assigned: {len(pe_assignments)}")
        
        # Validate capacity constraints (MAX 9 PEs per chiplet)
        chiplet_pe_counts = defaultdict(int)
        for pe, chiplet in pe_assignments.items():
            chiplet_pe_counts[chiplet] += 1
        
        capacity_violations = 0
        max_pes_per_chiplet = VALIDATION_CONFIG['max_pes_per_chiplet']
        
        print(f"\\n📊 Chiplet PE Distribution:")
        for chiplet in sorted(chiplet_pe_counts.keys()):
            count = chiplet_pe_counts[chiplet]
            status_icon = "✅" if count <= max_pes_per_chiplet else "❌"
            print(f"  Chiplet {chiplet}: {count} PEs {status_icon}")
            if count > max_pes_per_chiplet:
                capacity_violations += count - max_pes_per_chiplet
        
        if capacity_violations > 0:
            self.violations.append(f"Capacity constraint violations: {capacity_violations} excess PEs")
            
        return capacity_violations == 0
    
    def validate_pe_type_constraints(self):
        """Validate PE type constraints based on strategy"""
        if not self.pe_types:
            print("\\n⚠️  No PE types available - skipping PE type constraint validation")
            return True
            
        print("\\n=== PE TYPE CONSTRAINT VALIDATION ===")
        
        pe_assignments = self.solution.get('pe_assignments', {})
        chiplet_violations = 0
        
        # Group PEs by chiplet and analyze types
        chiplets = defaultdict(list)
        for pe, chiplet in pe_assignments.items():
            chiplets[chiplet].append(int(pe))
        
        print(f"Strategy: {self.strategy or 'Unknown'}")
        print(f"\\n{'Chiplet':<8} {'PEs':<6} {'IMC':<4} {'DIG':<4} {'Status':<10} {'Constraint Check'}")
        print("-" * 70)
        
        for chiplet_id in sorted(chiplets.keys()):
            pes = sorted(chiplets[chiplet_id])
            imc_count = sum(1 for pe in pes if self.pe_types.get(pe) == 'IMC')
            dig_count = sum(1 for pe in pes if self.pe_types.get(pe) == 'DIG')
            total_count = len(pes)
            
            # Check constraints based on strategy
            constraint_ok = True
            constraint_msg = ""
            
            if self.strategy == 'separated':
                # Separated: Each chiplet must be homogeneous (all IMC or all DIG)
                chiplet_types = set()
                for pe in pes:
                    if pe in self.pe_types:
                        chiplet_types.add(self.pe_types[pe])
                
                if len(chiplet_types) > 1:
                    constraint_ok = False
                    constraint_msg = f"Mixed types: {chiplet_types}"
                    chiplet_violations += 1
                else:
                    constraint_msg = f"Pure {list(chiplet_types)[0] if chiplet_types else 'Unknown'}"
                    
            elif self.strategy == 'mixed':
                # Mixed: Use configurable IMC/DIG ratios from config
                max_imc, max_dig = get_imc_dig_counts(DEFAULT_IMC_RATIO, MAX_PES_PER_CHIPLET)
                imc_ok = imc_count <= max_imc
                dig_ok = dig_count <= max_dig
                
                if not (imc_ok and dig_ok):
                    constraint_ok = False
                    issues = []
                    if not imc_ok:
                        issues.append(f"IMC>{max_imc}")
                    if not dig_ok:
                        issues.append(f"DIG>{max_dig}")
                    constraint_msg = ", ".join(issues)
                    chiplet_violations += 1
                else:
                    constraint_msg = f"IMC≤{max_imc}, DIG≤{max_dig} ✓"
            else:
                # Unknown strategy or ignore
                constraint_msg = "No constraints"
            
            status_icon = "✅" if constraint_ok else "❌"
            print(f"{chiplet_id:<8} {total_count:<6} {imc_count:<4} {dig_count:<4} {status_icon:<10} {constraint_msg}")
        
        print("-" * 70)
        
        if chiplet_violations > 0:
            self.violations.append(f"PE type constraint violations: {chiplet_violations} chiplets")
            
        return chiplet_violations == 0
    
    def validate_bandwidth_efficiency(self):
        """Analyze bandwidth efficiency (intra vs inter-chiplet communication)"""
        if self.traffic_data is None:
            print("\\n⚠️  No traffic data - skipping bandwidth analysis")
            return True
            
        print("\\n=== BANDWIDTH EFFICIENCY ANALYSIS ===")
        
        pe_assignments = self.solution.get('pe_assignments', {})
        inter_chiplet_tasks = 0
        intra_chiplet_tasks = 0
        
        for _, row in self.traffic_data.iterrows():
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
        
        total_tasks = intra_chiplet_tasks + inter_chiplet_tasks
        if total_tasks > 0:
            bandwidth_efficiency = intra_chiplet_tasks / total_tasks * 100
            print(f"Intra-chiplet tasks: {intra_chiplet_tasks}")
            print(f"Inter-chiplet tasks: {inter_chiplet_tasks}")
            print(f"Bandwidth efficiency: {bandwidth_efficiency:.1f}% (higher = better)")
        else:
            print("No valid task pairs found for bandwidth analysis")
            
        return True
    
    def validate_solution(self):
        """Run complete validation"""
        print(f"🔍 VALIDATING HETEROGENEOUS SOLUTION: {os.path.basename(self.solution_file)}")
        print("=" * 80)
        
        if not self.load_solution():
            return False
            
        if not self.load_traffic_data():
            return False
        
        # Run all validations
        basic_ok = self.validate_basic_constraints()
        pe_type_ok = self.validate_pe_type_constraints()
        bandwidth_ok = self.validate_bandwidth_efficiency()
        
        # Summary
        print("\\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        
        if not self.violations:
            print("🎉 ALL CONSTRAINTS SATISFIED!")
            print("✅ Solution is VALID")
            return True
        else:
            print("❌ CONSTRAINT VIOLATIONS DETECTED:")
            for i, violation in enumerate(self.violations, 1):
                print(f"  {i}. {violation}")
            print("\\n❌ Solution is INVALID")
            return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_hetro_solution.py <solution_file> [csv_file] [strategy]")
        print("  solution_file: JSON solution file to validate")
        print("  csv_file: Optional CSV traffic data file for PE type validation")
        print("  strategy: Optional strategy type (separated, mixed, ignore)")
        print("\\nExample:")
        print("  python validate_hetro_solution.py solution.json traffic.csv mixed")
        sys.exit(1)
    
    solution_file = sys.argv[1]
    csv_file = sys.argv[2] if len(sys.argv) > 2 else None
    strategy = sys.argv[3] if len(sys.argv) > 3 else None
    
    if not os.path.exists(solution_file):
        print(f"❌ Solution file {solution_file} not found!")
        sys.exit(1)
        
    if csv_file and not os.path.exists(csv_file):
        print(f"❌ CSV file {csv_file} not found!")
        sys.exit(1)
    
    validator = HeteroSolutionValidator(solution_file, csv_file, strategy)
    is_valid = validator.validate_solution()
    
    sys.exit(0 if is_valid else 1)

if __name__ == "__main__":
    main()