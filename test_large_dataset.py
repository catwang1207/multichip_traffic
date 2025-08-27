#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chiplet_framework import *

if __name__ == "__main__":
    print("=== TESTING LARGE DATASET (gpt2_transformer.txt) ===")
    print("Parameters: 60s timeout, max 12 chiplets")
    
    # Load problem
    problem = ChipletProblem('gpt2_transformer.txt')
    
    # Add constraints
    problem.add_constraint(TaskAssignmentConstraint())
    problem.add_constraint(ChipletUsageConstraint()) 
    problem.add_constraint(TaskDependencyConstraint())
    problem.add_constraint(ChipletCapacityConstraint(max_pes=32))
    problem.add_constraint(PEExclusivityConstraint())  # This should now work correctly
    problem.add_constraint(InterChipletCommConstraint(bandwidth=8192))
    problem.add_constraint(TimeBoundsConstraint())
    
    print(f"Problem: {len(problem.tasks)} tasks, {len(problem.pes)} PEs")
    
    # Test ILP solver with timeout and upper bound fallback
    print("Solving large dataset with 60s timeout...")
    solution = problem.solve(timeout=60, max_chiplets=12, save_solution_file=True)
    
    print(f"Solution: {solution['status']}, {solution['total_time']} cycles, {solution['num_chiplets']} chiplets")
    print(f"Tasks assigned: {len(solution.get('task_assignments', {}))}")
    
    if 'note' in solution:
        print(f"Note: {solution['note']}")
        
    if 'solution_file' in solution:
        print(f"Solution file: {solution['solution_file']}")
    
    if len(solution.get('task_assignments', {})) == len(problem.tasks):
        print("✅ ALL TASKS ASSIGNED!")
    else:
        missing = len(problem.tasks) - len(solution.get('task_assignments', {}))
        print(f"❌ {missing} tasks still missing")