#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ilp_chiplet_framework import *

if __name__ == "__main__":
    print("=== TESTING NO MULTICASTING CONSTRAINT (LARGE DATASET) ===")
    print("Testing with large dataset that has multicasting...")
    print("Parameters: 300s timeout, max 12 chiplets")
    
    # Load problem
    problem = ChipletProblem('gpt2_transformer.txt')
    
    # Add constraints INCLUDING the new no-multicasting constraint
    problem.add_constraint(TaskAssignmentConstraint())
    problem.add_constraint(ChipletUsageConstraint()) 
    problem.add_constraint(TaskDependencyConstraint())
    problem.add_constraint(ChipletCapacityConstraint(max_pes=32))
    problem.add_constraint(PEExclusivityConstraint())
    problem.add_constraint(InterChipletCommConstraint(bandwidth=8192))
    problem.add_constraint(TimeBoundsConstraint())
    problem.add_constraint(NoMulticastingConstraint())  # NEW CONSTRAINT
    
    print(f"Problem: {len(problem.tasks)} tasks, {len(problem.pes)} PEs")
    print(f"Constraints: {len(problem.constraints)} (including NoMulticastingConstraint)")
    
    # Check for multicasting in input data
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
    
    # Solve with 300s timeout 
    print(f"\nSolving with NoMulticastingConstraint (300s timeout)...")
    solution = problem.solve(timeout=300, max_chiplets=12, save_solution_file=True)
    
    # Print solution details with safe access to optional fields
    status = solution['status']
    total_time = solution.get('total_time', 'N/A')
    num_chiplets = solution.get('num_chiplets', 'N/A')
    task_assignments = solution.get('task_assignments', {})
    
    print(f"Solution: {status}, {total_time} cycles, {num_chiplets} chiplets")
    print(f"Tasks assigned: {len(task_assignments)}")
    
    if 'note' in solution:
        print(f"Note: {solution['note']}")
        
    if 'solution_file' in solution:
        print(f"Solution file: {solution['solution_file']}")
    
    # Check if solution violates multicasting constraint
    if solution['status'] != 'infeasible' and solution.get('task_assignments'):
        task_assignments = solution['task_assignments']
        task_times = solution.get('task_times', {})
        
        print(f"\nChecking solution for multicasting violations...")
        violations = []
        
        # Group tasks by time and source PE
        time_source_tasks = {}
        for task, chiplet in task_assignments.items():
            if task in task_times:
                time = task_times[task]
                source_pe = problem.task_data[task]['source_pe']
                dest_pe = problem.task_data[task]['dest_pe']
                
                if time not in time_source_tasks:
                    time_source_tasks[time] = {}
                if source_pe not in time_source_tasks[time]:
                    time_source_tasks[time][source_pe] = []
                
                time_source_tasks[time][source_pe].append((task, dest_pe))
        
        # Check for violations
        for time, source_tasks in time_source_tasks.items():
            for source_pe, tasks_dests in source_tasks.items():
                if len(tasks_dests) > 1:
                    dest_pes = [dest for _, dest in tasks_dests]
                    unique_dests = set(dest_pes)
                    if len(unique_dests) > 1:  # Multiple different destinations
                        task_ids = [task for task, _ in tasks_dests]
                        violations.append(f"Time {time}: PE {source_pe} sends to multiple destinations {sorted(unique_dests)} (tasks {task_ids})")
        
        if violations:
            print(f"❌ Found {len(violations)} multicasting violations:")
            for violation in violations[:3]:  # Show first 3
                print(f"  {violation}")
            if len(violations) > 3:
                print(f"  ... and {len(violations) - 3} more violations")
        else:
            print(f"✅ No multicasting violations found in solution!")
    
    if len(task_assignments) == len(problem.tasks):
        print("✅ ALL TASKS ASSIGNED!")
    else:
        missing = len(problem.tasks) - len(task_assignments)
        print(f"❌ {missing} tasks still missing")