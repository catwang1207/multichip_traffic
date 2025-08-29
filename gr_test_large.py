#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gr_chiplet_framework import *
from config import MAX_PES_PER_CHIPLET, INTER_CHIPLET_BANDWIDTH, LARGE_DATASET

if __name__ == "__main__":
    print("=== TESTING GRASP (LARGE DATASET) ===")
    print("Testing GRASP with large dataset that has multicasting...")
    print("Parameters: 60s timeout, max 8 chiplets, RCL=1")
    
    # Load problem
    problem = ChipletProblemGRASP('gpt2_transformer.txt')
    
    # Add constraints INCLUDING the new no-multicasting constraint
    problem.add_constraint(TaskAssignmentConstraint())
    problem.add_constraint(ChipletUsageConstraint()) 
    problem.add_constraint(TaskDependencyConstraint())
    problem.add_constraint(ChipletCapacityConstraint(max_pes=MAX_PES_PER_CHIPLET))
    problem.add_constraint(PEExclusivityConstraint())
    problem.add_constraint(InterChipletCommConstraint(bandwidth=INTER_CHIPLET_BANDWIDTH))
    problem.add_constraint(TimeBoundsConstraint())
    problem.add_constraint(NoMulticastingConstraint())  # NEW CONSTRAINT
    
    print(f"Problem: {len(problem.tasks)} tasks, {len(problem.pes)} PEs")
    print(f"Constraints: {len(problem.constraints)} (including NoMulticastingConstraint)")
    
    # Check for multicasting in input data (show limited output)
    multicast_sources = {}
    for task in problem.tasks:
        source_pe = problem.task_data[task]['source_pe']
        dest_pe = problem.task_data[task]['dest_pe']
        
        if source_pe not in multicast_sources:
            multicast_sources[source_pe] = []
        multicast_sources[source_pe].append((task, dest_pe))
    
    print(f"\nMulticasting in input data:")
    multicast_count = 0
    for source_pe, destinations in multicast_sources.items():
        unique_dests = set(dest for _, dest in destinations)
        if len(unique_dests) > 1:
            multicast_count += 1
            # Only show first 10 to avoid spam
            if multicast_count <= 10:
                print(f"  PE {source_pe} sends to {len(unique_dests)} destinations: {sorted(unique_dests)}")
    
    if multicast_count > 10:
        print(f"  ... and {multicast_count - 10} more PEs with multicasting")
    print(f"Total source PEs with multicasting: {multicast_count}")
    
    # Solve with GRASP (60s timeout, optimized parameters for large dataset)
    print(f"\nSolving with GRASP (60s timeout, optimized search)...")
    solution = problem.solve(
        timeout=LARGE_DATASET['timeout_seconds'],
        max_chiplets=LARGE_DATASET['max_chiplets'], 
        save_solution_file=True,
        rcl_size=LARGE_DATASET['rcl_size'],
        ls_max_passes=LARGE_DATASET['ls_max_passes'],
        pair_swap_samples=LARGE_DATASET['pair_swap_samples']
    )
    
    # Print solution details
    status = solution['status']
    total_time = solution.get('total_time', 'N/A')
    num_chiplets = solution.get('num_chiplets', 'N/A')
    task_assignments = solution.get('task_assignments', {})
    
    print(f"Solution: {status}, {total_time} cycles, {num_chiplets} chiplets")
    print(f"Tasks assigned: {len(task_assignments)}")
    print(f"Solve time: {solution.get('solve_time', 0):.2f}s")
    print(f"Iterations: {solution.get('iterations', 0)}")
    
    if 'violations' in solution:
        violations = solution['violations']
        total_violations = sum(violations.values())
        print(f"Constraint violations ({total_violations} total): {violations}")
        
        # Check specific violations
        if violations['no_multicasting'] == 0:
            print("✅ No multicasting violations - constraint working!")
        else:
            print(f"❌ {violations['no_multicasting']} multicasting violations found")
            
        if violations['task_assignment'] == 0:
            print("✅ All tasks assigned")
        else:
            print(f"❌ {violations['task_assignment']} task assignment violations")
    else:
        print("✅ No constraint violations!")
    
    if 'solution_file' in solution:
        print(f"Solution file: {solution['solution_file']}")
    
    if len(task_assignments) == len(problem.tasks):
        print("✅ ALL TASKS ASSIGNED!")
    else:
        missing = len(problem.tasks) - len(task_assignments)
        print(f"❌ {missing} tasks still missing")
        