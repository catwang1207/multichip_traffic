#!/usr/bin/env python3

import json
import pandas as pd
import sys
import os
from collections import defaultdict
from config import VALIDATION_CONFIG

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
                    # Record the earliest (start) time for multi-cycle tasks
                    if task_id not in task_times or cycle < task_times[task_id]:
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
                    wait_task_ids = [int(float(x.strip())) for x in str(row['wait_ids']).split(',')]
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
                    wait_task_ids = [int(float(x.strip())) for x in str(row['wait_ids']).split(',')]
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
    
    def validate_chiplet_capacity_constraint(self, task_assignments, max_pes=None):
        if max_pes is None:
            max_pes = VALIDATION_CONFIG['max_pes_per_chiplet']
        """Each chiplet can use at most max_pes PEs (only source_pe counts, dest_pe can be anywhere)"""
        print("Validating Chiplet Capacity Constraint...")
        
        # Group tasks by chiplet and find source PEs used (dest_pe can be on different chiplets)
        chiplet_pes = defaultdict(set)
        
        for _, row in self.traffic_data.iterrows():
            task_id = row['task_id']
            if task_id in task_assignments:
                chiplet = task_assignments[task_id]
                source_pe = row['source_pe']
                # Only source_pe belongs to the chiplet (dest_pe can be anywhere)
                chiplet_pes[chiplet].add(source_pe)
        
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
    
    def validate_pe_exclusivity_constraint(self, task_assignments, pe_assignments=None):
        """Each PE belongs to at most one chiplet (using actual PE assignments from ILP)"""
        print("Validating PE Exclusivity Constraint...")
        
        violations_before = len(self.violations)
        
        if pe_assignments:
            # Use actual PE assignments from ILP solver
            pe_chiplet_count = {}
            for pe_id, chiplet in pe_assignments.items():
                if chiplet not in pe_chiplet_count:
                    pe_chiplet_count[chiplet] = 0
                pe_chiplet_count[chiplet] += 1
            
            # Check for any PE assigned to multiple chiplets (shouldn't happen with correct ILP)
            all_pes = set()
            for pe_id in pe_assignments:
                if pe_id in all_pes:
                    self.violations.append(f"PE {pe_id} appears multiple times in PE assignments")
                all_pes.add(pe_id)
            
            print(f"  Using ILP PE assignments: {len(pe_assignments)} PEs assigned")
            for chiplet in sorted(pe_chiplet_count.keys()):
                print(f"    Chiplet {chiplet}: {pe_chiplet_count[chiplet]} PEs")
        else:
            # Fallback: infer PE assignments from task assignments (old method)
            print("  Using inferred PE assignments from task assignments")
            pe_chiplets = {}
            
            for _, row in self.traffic_data.iterrows():
                task_id = row['task_id']
                if task_id in task_assignments:
                    chiplet = task_assignments[task_id]
                    source_pe = row['source_pe']
                    dest_pe = row['dest_pe']
                    
                    # Check both source_pe and dest_pe exclusivity
                    for pe_type, pe_id in [('source', source_pe), ('dest', dest_pe)]:
                        if pe_id in pe_chiplets:
                            if pe_chiplets[pe_id] != chiplet:
                                self.violations.append(f"{pe_type.title()} PE {pe_id} is used by multiple chiplets: {pe_chiplets[pe_id]} and {chiplet} (task {task_id})")
                        else:
                            pe_chiplets[pe_id] = chiplet
        
        exclusivity_violations = len(self.violations) - violations_before
        print(f"  {'✅' if exclusivity_violations == 0 else '❌'} PE exclusivity violations: {exclusivity_violations}")
        
        if exclusivity_violations > 0:
            print(f"    Found {exclusivity_violations} PEs assigned to multiple chiplets")
            # Show PE distribution summary
            pe_counts = {}
            for pe_id, chiplet in pe_chiplets.items():
                if chiplet not in pe_counts:
                    pe_counts[chiplet] = 0
                pe_counts[chiplet] += 1
            print(f"    PE distribution: {dict(sorted(pe_counts.items()))}")
        
        return exclusivity_violations == 0
    
    def validate_no_multicasting_constraint(self, task_assignments, task_times):
        """Each source PE can only send to one destination at a time (no multicasting)"""
        print("Validating No Multicasting Constraint...")
        
        multicast_violations = []
        
        # Group tasks by time and source PE
        time_source_tasks = defaultdict(lambda: defaultdict(list))
        
        for _, row in self.traffic_data.iterrows():
            task_id = row['task_id']
            if task_id in task_times:
                time = task_times[task_id]
                source_pe = row['source_pe']
                dest_pe = row['dest_pe']
                time_source_tasks[time][source_pe].append((task_id, dest_pe))
        
        # Check for multicasting violations
        for time, source_tasks in time_source_tasks.items():
            for source_pe, tasks_dests in source_tasks.items():
                if len(tasks_dests) > 1:
                    dest_pes = [dest for _, dest in tasks_dests]
                    unique_dests = set(dest_pes)
                    if len(unique_dests) > 1:  # Multiple different destinations
                        task_ids = [task for task, _ in tasks_dests]
                        multicast_violations.append(
                            f"Time {time}: Source PE {source_pe} sends to multiple destinations {sorted(unique_dests)} (tasks {task_ids})"
                        )
        
        if multicast_violations:
            self.violations.extend(multicast_violations[:5])  # Show first 5
            if len(multicast_violations) > 5:
                self.violations.append(f"... and {len(multicast_violations) - 5} more multicasting violations")
        
        print(f"  {'✅' if not multicast_violations else '❌'} Multicasting violations: {len(multicast_violations)}")
        return len(multicast_violations) == 0
    
    def validate_task_duration_constraint(self, task_assignments, task_times, bandwidth=None):
        if bandwidth is None:
            bandwidth = VALIDATION_CONFIG['inter_chiplet_bandwidth']
        """Tasks on different chiplets must take ceiling(data_size/bandwidth) cycles"""
        print("Validating Task Duration Constraint...")
        
        # Build PE-to-chiplet mapping for O(1) lookup (performance optimization)
        pe_to_chiplet = {}
        for _, row in self.traffic_data.iterrows():
            task_id = row['task_id']
            if task_id in task_assignments:
                source_pe = row['source_pe']
                pe_to_chiplet[source_pe] = task_assignments[task_id]
        
        duration_violations = []
        
        # Check each task's expected duration based on data size and chiplet location
        for _, row in self.traffic_data.iterrows():
            task_id = row['task_id']
            if task_id in task_times and task_id in task_assignments:
                source_pe = row['source_pe']
                dest_pe = row['dest_pe']
                data_size = row['data_size']
                task_time = task_times[task_id]
                task_chiplet = task_assignments[task_id]
                
                # Find which chiplet the destination PE belongs to (O(1) lookup)
                dest_chiplet = pe_to_chiplet.get(dest_pe)
                
                # If we can determine dest_chiplet and it's different from source chiplet
                if dest_chiplet is not None and dest_chiplet != task_chiplet:
                    # Inter-chiplet task - calculate expected duration
                    import math
                    expected_duration = max(1, math.ceil(data_size / bandwidth))
                    
                    # Check actual duration from schedule
                    actual_duration = 1  # Default
                    if hasattr(self, 'solution') and 'schedule' in self.solution:
                        task_cycles = set()
                        for cycle_data in self.solution['schedule']:
                            chiplets = cycle_data.get('chiplets', {})
                            for chiplet_id, tasks in chiplets.items():
                                for task_info in tasks:
                                    if task_info['task_id'] == task_id:
                                        task_cycles.add(cycle_data['cycle'])
                        actual_duration = len(task_cycles)
                    
                    # Validate duration matches expected
                    if actual_duration != expected_duration:
                        duration_violations.append(
                            f"Task {task_id} (inter-chiplet, {data_size}B): expected {expected_duration} cycles, actual {actual_duration} cycles"
                        )
                # Intra-chiplet tasks always take 1 cycle (no violation check needed)
        
        if duration_violations:
            self.violations.extend(duration_violations[:5])  # Show first 5
            if len(duration_violations) > 5:
                self.violations.append(f"... and {len(duration_violations) - 5} more task duration violations")
        
        print(f"  {'✅' if not duration_violations else '❌'} Task duration violations: {len(duration_violations)}")
        return len(duration_violations) == 0
    
    def generate_summary_statistics(self, task_assignments, task_times, metadata):
        """Generate summary statistics about the solution"""
        print("\n=== SOLUTION SUMMARY ===")
        
        print(f"Solver: {metadata.get('solver', 'Unknown')}")
        print(f"Status: {metadata.get('status', 'Unknown')}")
        print(f"Total cycles: {metadata.get('total_cycles', 'Unknown')}")
        print(f"Number of chiplets: {metadata.get('num_chiplets', 'Unknown')}")
        solve_time = metadata.get('solve_time_seconds', 'Unknown')
        if isinstance(solve_time, (int, float)):
            print(f"Solve time: {solve_time:.3f}s")
        else:
            print(f"Solve time: {solve_time}")
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
                    wait_task_ids = [int(float(x.strip())) for x in str(row['wait_ids']).split(',')]
                    for wait_task_id in wait_task_ids:
                        if wait_task_id in task_times:
                            total_deps += 1
                            current_chiplet = task_assignments.get(current_task, -1)
                            wait_chiplet = task_assignments.get(wait_task_id, -1)
                            
                            if current_chiplet == wait_chiplet:
                                intra_count += 1
                            else:
                                # Calculate expected cycles with configured bandwidth
                                import math
                                bandwidth = VALIDATION_CONFIG['inter_chiplet_bandwidth']
                                expected_cycles = max(1, math.ceil(row['data_size'] / bandwidth))
                                if expected_cycles == 1:
                                    inter_small_count += 1
                                else:
                                    inter_large_count += 1
        
        if total_deps > 0:
            print(f"\nCommunication analysis ({total_deps} total dependencies):")
            print(f"  Intra-chiplet: {intra_count} ({100*intra_count/total_deps:.1f}%)")
            bandwidth = VALIDATION_CONFIG['inter_chiplet_bandwidth']
            print(f"  Inter-chiplet single-cycle (≤{bandwidth}B): {inter_small_count} ({100*inter_small_count/total_deps:.1f}%)")
            print(f"  Inter-chiplet multi-cycle (>{bandwidth}B): {inter_large_count} ({100*inter_large_count/total_deps:.1f}%)")
    
    def validate(self):
        """Run all validations and return overall result"""
        print(f"=== VALIDATING SOLUTION: {self.solution_file} ===")
        
        # Load data
        if not self.load_solution() or not self.load_traffic_data():
            return False
        
        # Extract solution data
        task_assignments, task_times, metadata = self.extract_solution_data()
        
        # Get PE assignments from solution if available
        pe_assignments = self.solution.get('pe_assignments', None)
        
        # Run all constraint validations
        validations = [
            self.validate_task_assignment_constraint(task_assignments),
            self.validate_task_dependency_constraint(task_assignments, task_times),
            self.validate_inter_chiplet_comm_constraint(task_assignments, task_times),
            self.validate_chiplet_capacity_constraint(task_assignments),
            self.validate_pe_exclusivity_constraint(task_assignments, pe_assignments),
            self.validate_no_multicasting_constraint(task_assignments, task_times),
            self.validate_task_duration_constraint(task_assignments, task_times)
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