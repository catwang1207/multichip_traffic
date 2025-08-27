#!/usr/bin/env python3

import json
import pandas as pd
import sys
import os
from collections import defaultdict

class SolutionValidator:
    """Standalone solution validator that reads solution files and validates constraints"""
    
    def __init__(self, solution_file, data_file):
        self.solution_file = solution_file
        self.data_file = data_file
        self.solution = None
        self.traffic_data = None
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
        """Load traffic data"""
        try:
            self.traffic_data = pd.read_csv(self.data_file, sep='\t', comment='#', 
                                          names=['task_id', 'source_pe', 'dest_pe', 'data_size', 'wait_ids'])
            return True
        except Exception as e:
            print(f"❌ Error loading traffic data file {self.data_file}: {e}")
            return False
    
    def extract_solution_data(self):
        """Extract key data from solution for validation"""
        metadata = self.solution.get('metadata', {})
        schedule = self.solution.get('schedule', [])
        
        # Extract task assignments and times
        task_assignments = {}
        task_times = {}
        
        for cycle_data in schedule:
            cycle = cycle_data['cycle']
            chiplets = cycle_data.get('chiplets', {})
            
            for chiplet_id, tasks in chiplets.items():
                chiplet_id = int(chiplet_id)
                for task_info in tasks:
                    task_id = task_info['task_id']
                    task_assignments[task_id] = chiplet_id
                    task_times[task_id] = cycle
        
        return task_assignments, task_times, metadata
    
    def validate_task_assignment_constraint(self, task_assignments):
        """Each task must be assigned to exactly one chiplet"""
        print("Validating Task Assignment Constraint...")
        
        # Check that all tasks in traffic data are assigned
        traffic_tasks = set(self.traffic_data['task_id'].values)
        assigned_tasks = set(task_assignments.keys())
        
        unassigned = traffic_tasks - assigned_tasks
        if unassigned:
            self.violations.append(f"Tasks not assigned to any chiplet: {sorted(list(unassigned)[:10])}{'...' if len(unassigned) > 10 else ''}")
        
        # Check for duplicate assignments (shouldn't happen with proper data structure)
        if len(assigned_tasks) != len(task_assignments):
            self.violations.append("Some tasks appear to be assigned multiple times")
        
        print(f"  ✅ {len(assigned_tasks)} tasks assigned to chiplets")
        return len(unassigned) == 0
    
    def validate_task_dependency_constraint(self, task_assignments, task_times):
        """Dependent tasks must execute in correct order"""
        print("Validating Task Dependency Constraint...")
        
        dependency_violations = []
        
        for _, row in self.traffic_data.iterrows():
            if pd.notna(row['wait_ids']) and row['wait_ids'] != 'None':
                current_task = row['task_id']
                if current_task in task_times:
                    wait_task_ids = [int(x.strip()) for x in str(row['wait_ids']).split(',')]
                    for wait_task_id in wait_task_ids:
                        if wait_task_id in task_times:
                            current_time = task_times[current_task]
                            wait_time = task_times[wait_task_id]
                            
                            if current_time <= wait_time:
                                dependency_violations.append(
                                    f"Task {current_task} (t={current_time}) starts before dependency {wait_task_id} (t={wait_time}) finishes"
                                )
        
        if dependency_violations:
            self.violations.extend(dependency_violations[:5])  # Show first 5
            if len(dependency_violations) > 5:
                self.violations.append(f"... and {len(dependency_violations) - 5} more dependency violations")
        
        print(f"  {'✅' if not dependency_violations else '❌'} Dependency violations: {len(dependency_violations)}")
        return len(dependency_violations) == 0
    
    def validate_inter_chiplet_comm_constraint(self, task_assignments, task_times, bandwidth=8192):
        """Inter-chiplet communication with large data should take 2 cycles"""
        print("Validating Inter-Chiplet Communication Constraint...")
        
        comm_violations = []
        
        for _, row in self.traffic_data.iterrows():
            if pd.notna(row['wait_ids']) and row['wait_ids'] != 'None':
                current_task = row['task_id']
                if current_task in task_times and row['data_size'] > bandwidth:
                    wait_task_ids = [int(x.strip()) for x in str(row['wait_ids']).split(',')]
                    for wait_task_id in wait_task_ids:
                        if wait_task_id in task_times:
                            current_chiplet = task_assignments.get(current_task, -1)
                            wait_chiplet = task_assignments.get(wait_task_id, -1)
                            
                            if current_chiplet != wait_chiplet and current_chiplet != -1 and wait_chiplet != -1:
                                current_time = task_times[current_task]
                                wait_time = task_times[wait_task_id]
                                delay = current_time - wait_time
                                
                                if delay < 2:
                                    comm_violations.append(
                                        f"Task {current_task} (chiplet {current_chiplet}, t={current_time}) should wait 2+ cycles after task {wait_task_id} (chiplet {wait_chiplet}, t={wait_time}) for large inter-chiplet transfer ({row['data_size']} bytes), actual delay: {delay}"
                                    )
        
        if comm_violations:
            self.violations.extend(comm_violations[:3])  # Show first 3
            if len(comm_violations) > 3:
                self.violations.append(f"... and {len(comm_violations) - 3} more inter-chiplet communication violations")
        
        print(f"  {'✅' if not comm_violations else '❌'} Inter-chiplet communication violations: {len(comm_violations)}")
        return len(comm_violations) == 0
    
    def validate_chiplet_capacity_constraint(self, task_assignments, max_pes=32):
        """Each chiplet can use at most max_pes PEs"""
        print("Validating Chiplet Capacity Constraint...")
        
        # Group tasks by chiplet and find PEs used
        chiplet_pes = defaultdict(set)
        
        for _, row in self.traffic_data.iterrows():
            task_id = row['task_id']
            if task_id in task_assignments:
                chiplet = task_assignments[task_id]
                dest_pe = row['dest_pe']
                chiplet_pes[chiplet].add(dest_pe)
        
        capacity_violations = []
        for chiplet, pes in chiplet_pes.items():
            if len(pes) > max_pes:
                capacity_violations.append(f"Chiplet {chiplet} uses {len(pes)} PEs (max allowed: {max_pes})")
        
        if capacity_violations:
            self.violations.extend(capacity_violations)
        
        print(f"  {'✅' if not capacity_violations else '❌'} Capacity violations: {len(capacity_violations)}")
        for chiplet in sorted(chiplet_pes.keys()):
            print(f"    Chiplet {chiplet}: {len(chiplet_pes[chiplet])} PEs")
        
        return len(capacity_violations) == 0
    
    def validate_pe_exclusivity_constraint(self, task_assignments):
        """Each PE belongs to at most one chiplet"""
        print("Validating PE Exclusivity Constraint...")
        
        # Track which chiplet each PE belongs to
        pe_chiplets = {}
        
        for _, row in self.traffic_data.iterrows():
            task_id = row['task_id']
            if task_id in task_assignments:
                chiplet = task_assignments[task_id]
                dest_pe = row['dest_pe']
                
                if dest_pe in pe_chiplets:
                    if pe_chiplets[dest_pe] != chiplet:
                        self.violations.append(f"PE {dest_pe} is used by multiple chiplets: {pe_chiplets[dest_pe]} and {chiplet}")
                else:
                    pe_chiplets[dest_pe] = chiplet
        
        exclusivity_violations = len([v for v in self.violations if "used by multiple chiplets" in v])
        print(f"  {'✅' if exclusivity_violations == 0 else '❌'} PE exclusivity violations: {exclusivity_violations}")
        
        return exclusivity_violations == 0
    
    def generate_summary_statistics(self, task_assignments, task_times, metadata):
        """Generate summary statistics about the solution"""
        print("\n=== SOLUTION SUMMARY ===")
        
        print(f"Solver: {metadata.get('solver', 'Unknown')}")
        print(f"Status: {metadata.get('status', 'Unknown')}")
        print(f"Total cycles: {metadata.get('total_cycles', 'Unknown')}")
        print(f"Number of chiplets: {metadata.get('num_chiplets', 'Unknown')}")
        print(f"Solve time: {metadata.get('solve_time_seconds', 'Unknown'):.3f}s")
        print(f"Total tasks: {len(task_assignments)}")
        
        # Task distribution per chiplet
        chiplet_task_count = defaultdict(int)
        for task, chiplet in task_assignments.items():
            chiplet_task_count[chiplet] += 1
        
        print(f"Tasks per chiplet: {dict(sorted(chiplet_task_count.items()))}")
        
        # Communication analysis
        total_deps = 0
        intra_count = 0
        inter_small_count = 0 
        inter_large_count = 0
        
        for _, row in self.traffic_data.iterrows():
            if pd.notna(row['wait_ids']) and row['wait_ids'] != 'None':
                current_task = row['task_id']
                if current_task in task_times:
                    wait_task_ids = [int(x.strip()) for x in str(row['wait_ids']).split(',')]
                    for wait_task_id in wait_task_ids:
                        if wait_task_id in task_times:
                            total_deps += 1
                            current_chiplet = task_assignments.get(current_task, -1)
                            wait_chiplet = task_assignments.get(wait_task_id, -1)
                            
                            if current_chiplet == wait_chiplet:
                                intra_count += 1
                            else:
                                if row['data_size'] > 8192:
                                    inter_large_count += 1
                                else:
                                    inter_small_count += 1
        
        if total_deps > 0:
            print(f"\nCommunication analysis ({total_deps} total dependencies):")
            print(f"  Intra-chiplet: {intra_count} ({100*intra_count/total_deps:.1f}%)")
            print(f"  Inter-chiplet small (≤8192B): {inter_small_count} ({100*inter_small_count/total_deps:.1f}%)")
            print(f"  Inter-chiplet large (>8192B): {inter_large_count} ({100*inter_large_count/total_deps:.1f}%)")
    
    def validate(self):
        """Run all validations and return overall result"""
        print(f"=== VALIDATING SOLUTION: {self.solution_file} ===")
        
        # Load data
        if not self.load_solution() or not self.load_traffic_data():
            return False
        
        # Extract solution data
        task_assignments, task_times, metadata = self.extract_solution_data()
        
        # Run all constraint validations
        validations = [
            self.validate_task_assignment_constraint(task_assignments),
            self.validate_task_dependency_constraint(task_assignments, task_times),
            self.validate_inter_chiplet_comm_constraint(task_assignments, task_times),
            self.validate_chiplet_capacity_constraint(task_assignments),
            self.validate_pe_exclusivity_constraint(task_assignments)
        ]
        
        # Generate summary
        self.generate_summary_statistics(task_assignments, task_times, metadata)
        
        # Report violations
        if self.violations:
            print(f"\n❌ VALIDATION FAILED - {len(self.violations)} constraint violations found:")
            for i, violation in enumerate(self.violations, 1):
                print(f"  {i}. {violation}")
            return False
        else:
            print(f"\n✅ VALIDATION PASSED - All constraints satisfied!")
            return True

def main():
    if len(sys.argv) != 3:
        print("Usage: python validate_solution.py <solution_file.json> <data_file.txt>")
        print("Example: python validate_solution.py gpt2_transformer_small_ilp_solution.json gpt2_transformer_small.txt")
        sys.exit(1)
    
    solution_file = sys.argv[1]
    data_file = sys.argv[2]
    
    # Check if files exist
    if not os.path.exists(solution_file):
        print(f"❌ Solution file not found: {solution_file}")
        sys.exit(1)
    
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        sys.exit(1)
    
    # Run validation
    validator = SolutionValidator(solution_file, data_file)
    is_valid = validator.validate()
    
    sys.exit(0 if is_valid else 1)

if __name__ == "__main__":
    main()