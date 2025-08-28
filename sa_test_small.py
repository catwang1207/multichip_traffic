#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sa_chiplet_framework import *

if __name__ == "__main__":
    print("=== TESTING SIMULATED ANNEALING (SMALL DATASET) ===")
    print("Testing SA with small dataset that has multicasting...")
    print("Parameters: 60s timeout, max 6 chiplets")
    
    # Load problem
    problem = ChipletProblemSA('gpt2_transformer_small.txt')
    
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
            print(f"  PE {source_pe} sends to {len(unique_dests)} destinations: {sorted(unique_dests)}")
    
    print(f"Total source PEs with multicasting: {multicast_count}")
    
    # Solve with SA (60s timeout, extended parameters)
    print(f"\nSolving with Simulated Annealing (60s timeout, extended search)...")
    solution = problem.solve(
        timeout=60, 
        max_chiplets=6, 
        save_solution_file=True,
        solution_prefix='solution_gpt2_small_sa',
        initial_temp=2000.0,      # Higher initial temperature
        max_iterations=100000,    # 10x more iterations
        cooling_rate=0.99,        # Slower cooling for better exploration
        max_no_improvement=15000  # More patience before stopping
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
    
    if 'solution_file' in solution:
        print(f"Solution file: {solution['solution_file']}")
    
    if len(task_assignments) == len(problem.tasks):
        print("✅ ALL TASKS ASSIGNED!")
    else:
        missing = len(problem.tasks) - len(task_assignments)
        print(f"❌ {missing} tasks still missing")
        
    # Compare with ILP
    print(f"\n=== COMPARISON WITH ILP ===")
    print(f"SA: {total_time} cycles, {num_chiplets} chiplets, {solution.get('solve_time', 0):.1f}s")
    print(f"ILP: 134 cycles, 6 chiplets, ~59s (from previous test)")
    print(f"SA is {'faster' if solution.get('solve_time', 0) < 59 else 'slower'} but may have constraint violations")