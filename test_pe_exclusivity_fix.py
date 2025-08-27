#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chiplet_framework import *

if __name__ == "__main__":
    print("Testing FIXED PE Exclusivity Constraint...")
    
    # Load problem
    problem = ChipletProblem('gpt2_transformer_small.txt')
    
    # Add constraints
    problem.add_constraint(TaskAssignmentConstraint())
    problem.add_constraint(ChipletUsageConstraint()) 
    problem.add_constraint(TaskDependencyConstraint())
    problem.add_constraint(ChipletCapacityConstraint(max_pes=32))
    problem.add_constraint(PEExclusivityConstraint())  # This should now work correctly
    problem.add_constraint(InterChipletCommConstraint(bandwidth=8192))
    problem.add_constraint(TimeBoundsConstraint())
    
    print(f"Problem: {len(problem.tasks)} tasks, {len(problem.pes)} PEs")
    
    # Test ILP solver with fixed constraint
    print("Solving with FIXED PE Exclusivity constraint...")
    ilp_solution = problem.solve('ilp', max_chiplets=6, timeout=60, save_solution_file=True)
    
    print(f"ILP: {ilp_solution['status']}, {ilp_solution['total_time']} cycles, {ilp_solution['num_chiplets']} chiplets")
    print(f"Tasks assigned: {len(ilp_solution.get('task_assignments', {}))}")
    if 'solution_file' in ilp_solution:
        print(f"New solution file: {ilp_solution['solution_file']}")
    
    if len(ilp_solution.get('task_assignments', {})) == len(problem.tasks):
        print("✅ ALL TASKS ASSIGNED!")
    else:
        missing = len(problem.tasks) - len(ilp_solution.get('task_assignments', {}))
        print(f"❌ {missing} tasks still missing")