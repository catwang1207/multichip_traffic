import pandas as pd
import time as time_module
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from collections import defaultdict
from pulp import *

# ================== CONSTRAINT DEFINITIONS ==================

@dataclass
class TaskAssignmentConstraint:
    """Each task must be assigned to exactly one chiplet"""
    name: str = "task_assignment"
    description: str = "Each task assigned to exactly one chiplet"

@dataclass 
class ChipletUsageConstraint:
    """Chiplets must be activated if tasks are assigned to them"""
    name: str = "chiplet_usage" 
    description: str = "Chiplets must be activated if tasks are assigned"

@dataclass
class TaskDependencyConstraint:
    """Dependent tasks must execute in correct order"""
    name: str = "task_dependencies"
    description: str = "Dependent tasks must execute in correct order"

@dataclass
class ChipletCapacityConstraint:
    """Each chiplet can use at most max_pes unique PEs"""
    max_pes: int = 32
    name: str = "chiplet_capacity"
    description: str = "Each chiplet can use at most max_pes unique PEs"

@dataclass
class PEExclusivityConstraint:
    """Each PE belongs to at most one chiplet"""
    name: str = "pe_exclusivity"
    description: str = "Each PE belongs to at most one chiplet"

@dataclass
class InterChipletCommConstraint:
    """Inter-chiplet communication with bandwidth limits"""
    bandwidth: int = 8192
    extra_cycles_per_overflow: int = 1
    name: str = "inter_chiplet_comm"
    description: str = "Inter-chiplet communication bandwidth constraint"

@dataclass
class TimeBoundsConstraint:
    """Total time must cover all task completion times"""
    name: str = "time_bounds"
    description: str = "Total time must cover all task completion times"

# ================== SOLUTION FILE GENERATOR ==================

def generate_solution_file(solution, solver_name, data_file, task_data):
    """Generate a solution file showing task assignments per clock cycle with detailed PE assignments"""
    
    if solution['status'] == 'infeasible':
        print(f"Cannot generate solution file - {solver_name} solution is infeasible")
        return None
    
    # Extract solution data
    task_assignments = solution.get('task_assignments', {})
    task_times = solution.get('task_times', {})
    total_time = solution.get('total_time', 0)
    num_chiplets = solution.get('num_chiplets', 0)
    pe_assignments = solution.get('pe_assignments', {})
    
    # Create solution filename
    base_name = os.path.splitext(os.path.basename(data_file))[0]
    solution_file = f"{base_name}_{solver_name.lower()}_solution.json"
    
    # Organize tasks by clock cycle and chiplet
    schedule = {}
    max_chiplet = max(task_assignments.values()) if task_assignments else 0
    
    for cycle in range(total_time):
        schedule[cycle] = {}
        for chiplet in range(max_chiplet + 1):
            schedule[cycle][chiplet] = []
    
    # Assign tasks to their execution cycles and chiplets
    for task, chiplet in task_assignments.items():
        if task in task_times:
            cycle = task_times[task]
            if cycle < total_time:
                # Ensure the chiplet exists in this cycle
                if chiplet not in schedule[cycle]:
                    schedule[cycle][chiplet] = []
                    
                source_pe = task_data[task]['source_pe']
                dest_pe = task_data[task]['dest_pe']
                
                task_info = {
                    'task_id': task,
                    'source_pe': source_pe,
                    'dest_pe': dest_pe,
                    'data_size': task_data[task]['data_size'],
                    'source_pe_chiplet': pe_assignments.get(source_pe, chiplet),  # Should match task chiplet
                    'dest_pe_chiplet': pe_assignments.get(dest_pe, -1),  # May be on different chiplet
                    'execution_cycle': cycle
                }
                schedule[cycle][chiplet].append(task_info)
    
    # Create solution data structure
    solution_data = {
        'metadata': {
            'solver': solver_name,
            'data_file': data_file,
            'timestamp': time_module.strftime('%Y-%m-%d %H:%M:%S'),
            'status': solution['status'],
            'total_cycles': total_time,
            'num_chiplets': num_chiplets,
            'solve_time_seconds': solution.get('solve_time', 0),
            'total_tasks': len(task_assignments)
        },
        'pe_assignments': pe_assignments,  # Include detailed PE-to-chiplet assignments
        'schedule': []
    }
    
    # Convert schedule to list format - use all chiplets that actually exist in schedule
    all_chiplets = set()
    for cycle in range(total_time):
        all_chiplets.update(schedule[cycle].keys())
    
    for cycle in range(total_time):
        cycle_data = {
            'cycle': cycle,
            'chiplets': {}
        }
        
        for chiplet in sorted(all_chiplets):
            if chiplet in schedule[cycle]:
                cycle_data['chiplets'][str(chiplet)] = schedule[cycle][chiplet]
            else:
                cycle_data['chiplets'][str(chiplet)] = []
        
        solution_data['schedule'].append(cycle_data)
    
    # Write solution file
    try:
        with open(solution_file, 'w') as f:
            json.dump(solution_data, f, indent=2)
        
        return solution_file
        
    except Exception as e:
        print(f"❌ Error generating solution file: {e}")
        return None

# ================== PROBLEM DEFINITION ==================

class ChipletProblem:
    """
    Declarative problem definition with constraint specification
    """
    
    def __init__(self, data_file: str):
        self.data_file = data_file
        self.constraints = []
        self.tasks = []
        self.task_data = {}
        self.dependencies = {}
        self.pes = []
        self._load_data()
        
    def _load_data(self):
        """Load and parse the problem data"""
        df = pd.read_csv(self.data_file, sep='\t', comment='#', 
                         names=['task_id', 'source_pe', 'dest_pe', 'data_size', 'wait_ids'])
        
        self.tasks = df['task_id'].tolist()
        
        # Store task data
        for _, row in df.iterrows():
            self.task_data[row['task_id']] = {
                'source_pe': row['source_pe'],
                'dest_pe': row['dest_pe'], 
                'data_size': row['data_size'],
                'wait_ids': row['wait_ids']
            }
        
        # Extract unique PEs
        all_pes = set(df['source_pe'].tolist() + df['dest_pe'].tolist())
        self.pes = sorted(list(all_pes))
        
        # Build dependency graph
        for task in self.tasks:
            self.dependencies[task] = []
            wait_ids = self.task_data[task]['wait_ids']
            if pd.notna(wait_ids) and str(wait_ids) != 'None':
                deps = [int(x.strip()) for x in str(wait_ids).split(',')]
                self.dependencies[task] = [d for d in deps if d in self.tasks]
    
    def add_constraint(self, constraint):
        """Add a constraint to the problem"""
        self.constraints.append(constraint)
        return self
        
    def get_problem_stats(self):
        """Get problem statistics"""
        return {
            'num_tasks': len(self.tasks),
            'num_pes': len(self.pes),
            'num_dependencies': sum(len(deps) for deps in self.dependencies.values()),
            'num_constraints': len(self.constraints)
        }
    
    def solve(self, timeout=60, save_solution_file=True, **kwargs):
        """Solve using ILP only with timeout solution extraction"""
        # Force ILP solver only
        solution = ILPSolver(self).solve(timeout=timeout, **kwargs)
        solver_name = "ILP"
        
        # Generate solution file if requested
        if save_solution_file and solution['status'] != 'infeasible':
            solution_file = generate_solution_file(
                solution, solver_name, self.data_file, self.task_data
            )
            solution['solution_file'] = solution_file
        
        return solution

# ================== SOLVER INTERFACE ==================

class Solver(ABC):
    """Abstract base class for all solvers"""
    
    def __init__(self, problem: ChipletProblem):
        self.problem = problem
        
    @abstractmethod
    def solve(self, **kwargs) -> Dict[str, Any]:
        """Solve the problem and return solution"""
        pass

# ================== ILP SOLVER ==================

class ILPSolver(Solver):
    """ILP solver using PuLP - translates constraints to mathematical formulation"""
    
    def solve(self, max_chiplets=10, timeout=300):
        """Solve using ILP formulation"""
        
        print(f"=== ILP SOLVER ===")
        print(f"Translating {len(self.problem.constraints)} constraints to ILP formulation...")
        
        tasks = self.problem.tasks
        pes = self.problem.pes  
        task_data = self.problem.task_data
        dependencies = self.problem.dependencies
        
        max_time = len(tasks) * 2
        
        # DECISION VARIABLES
        print("Creating variables...")
        
        # z[task][chiplet] = 1 if task assigned to chiplet
        z = {}
        for task in tasks:
            for c in range(max_chiplets):
                z[task, c] = LpVariable(f"z_{task}_{c}", cat='Binary')
        
        # y[chiplet] = 1 if chiplet is used
        y = {}
        for c in range(max_chiplets):
            y[c] = LpVariable(f"y_{c}", cat='Binary')
        
        # Task execution time variables
        task_time = {}
        for task in tasks:
            task_time[task] = LpVariable(f"t_{task}", lowBound=0, upBound=max_time, cat='Integer')
        
        # Total execution time
        total_time_var = LpVariable("total_time", lowBound=0, cat='Integer')
        
        # PE usage variables
        pe_used = {}
        for c in range(max_chiplets):
            for pe in pes:
                pe_used[pe, c] = LpVariable(f"pe_used_{pe}_{c}", cat='Binary')
        
        # Chiplet PE usage variables (1 if chiplet c has any PEs assigned to it)
        chiplet_pe_used = {}
        for c in range(max_chiplets):
            chiplet_pe_used[c] = LpVariable(f"chiplet_pe_used_{c}", cat='Binary')
        
        # CREATE PROBLEM
        prob = LpProblem("ChipletAssignment", LpMinimize)
        
        # OBJECTIVE: Minimize time + heavily penalize chiplet usage
        prob += 1000 * total_time_var + 500 * lpSum([chiplet_pe_used[c] for c in range(max_chiplets)])
        
        # TRANSLATE CONSTRAINTS
        print("Translating constraints...")
        
        for constraint in self.problem.constraints:
            
            if isinstance(constraint, TaskAssignmentConstraint):
                # Each task assigned to exactly one chiplet
                for task in tasks:
                    prob += lpSum([z[task, c] for c in range(max_chiplets)]) == 1
                    
            elif isinstance(constraint, ChipletUsageConstraint):
                # If task assigned to chiplet c, then chiplet c must be used
                for task in tasks:
                    for c in range(max_chiplets):
                        prob += z[task, c] <= y[c]
                        
            elif isinstance(constraint, TaskDependencyConstraint):
                # Dependent tasks must execute in correct order
                for task in tasks:
                    for dep_task in dependencies[task]:
                        prob += task_time[task] >= task_time[dep_task] + 1
                        
            elif isinstance(constraint, ChipletCapacityConstraint):
                # Enforce PE-task co-location: tasks are assigned to chiplet where their source_pe is located
                for task in tasks:
                    source_pe = task_data[task]['source_pe']
                    for c in range(max_chiplets):
                        # If task is on chiplet c, its source_pe must be on chiplet c
                        prob += pe_used[source_pe, c] >= z[task, c]
                        # If source_pe is on chiplet c, all tasks with that source_pe must be on chiplet c
                        prob += z[task, c] >= pe_used[source_pe, c]
                
                # Each chiplet can use at most max_pes PEs  
                for c in range(max_chiplets):
                    prob += lpSum([pe_used[pe, c] for pe in pes]) <= constraint.max_pes
                    
            elif isinstance(constraint, PEExclusivityConstraint):
                # Each PE belongs to exactly one chiplet if used, at most one chiplet if unused
                used_pes = set()
                for task in tasks:
                    used_pes.add(task_data[task]['source_pe'])
                    used_pes.add(task_data[task]['dest_pe'])
                
                for pe in pes:
                    if pe in used_pes:
                        # PEs used in any task must belong to exactly one chiplet
                        prob += lpSum([pe_used[pe, c] for c in range(max_chiplets)]) == 1
                    else:
                        # Unused PEs can belong to at most one chiplet (but don't have to)
                        prob += lpSum([pe_used[pe, c] for c in range(max_chiplets)]) <= 1
                    
            elif isinstance(constraint, InterChipletCommConstraint):
                # Add explicit timing constraints for inter-chiplet communication
                comm_penalty = 0
                pe_comm_volume = {}
                
                # For each task with inter-chiplet communication
                for task in tasks:
                    source_pe = task_data[task]['source_pe']
                    dest_pe = task_data[task]['dest_pe']
                    data_size = task_data[task]['data_size']
                    
                    # Check if this is inter-chiplet communication (different PEs)
                    if source_pe != dest_pe:
                        # Track communication volume for penalty
                        for pe in [source_pe, dest_pe]:
                            if pe not in pe_comm_volume:
                                pe_comm_volume[pe] = 0
                            pe_comm_volume[pe] += data_size
                        
                        # Add explicit timing constraint for large data transfers
                        if data_size > constraint.bandwidth:  # > 8192 bytes
                            # Find tasks that depend on this task
                            for dep_task in tasks:
                                if task in dependencies[dep_task]:
                                    # Check if source and dest PEs are in different chiplets
                                    # Add binary variables to detect inter-chiplet communication
                                    inter_chiplet_var = LpVariable(f"inter_chiplet_{task}_{dep_task}", cat='Binary')
                                    
                                    # inter_chiplet_var = 1 if source_pe and dest_pe are in different chiplets
                                    for c1 in range(max_chiplets):
                                        for c2 in range(max_chiplets):
                                            if c1 != c2:
                                                # If source_pe in chiplet c1 AND dest_pe in chiplet c2, then inter_chiplet = 1
                                                prob += inter_chiplet_var >= pe_used[source_pe, c1] + pe_used[dest_pe, c2] - 1
                                    
                                    # If inter-chiplet communication with large data, add 1 extra cycle
                                    prob += task_time[dep_task] >= task_time[task] + 1 + inter_chiplet_var
                
                # Add communication penalty to objective (keep existing penalty system)
                for pe, volume in pe_comm_volume.items():
                    if volume > constraint.bandwidth:
                        penalty_weight = (volume // constraint.bandwidth) * 0.1
                        for c in range(max_chiplets):
                            comm_penalty += penalty_weight * pe_used[pe, c]
                
                # Update objective with communication penalty
                prob.objective += comm_penalty
                
            elif isinstance(constraint, TimeBoundsConstraint):
                # Total time must cover all task completion times
                for task in tasks:
                    prob += total_time_var >= task_time[task] + 1
        
        # Add chiplet PE usage constraints: chiplet_pe_used[c] = 1 if any PE is assigned to chiplet c
        for c in range(max_chiplets):
            # If any PE is assigned to chiplet c, then chiplet_pe_used[c] must be 1
            prob += chiplet_pe_used[c] >= lpSum([pe_used[pe, c] for pe in pes]) / len(pes)
            # If no PEs are assigned to chiplet c, then chiplet_pe_used[c] can be 0
            for pe in pes:
                prob += chiplet_pe_used[c] >= pe_used[pe, c]
        
        print(f"ILP formulation complete. Solving with timeout={timeout}s...")
        
        # SOLVE with built-in timeout parameter using CBC solver
        solve_start = time_module.time()
        
        # Use PuLP's CBC solver with timeout - try to get solution file
        solver = PULP_CBC_CMD(timeLimit=timeout, msg=True, keepFiles=1)
        status = prob.solve(solver)
        
        solve_time = time_module.time() - solve_start
        
        
        # Extract solution regardless of status (optimal, feasible, or timed out)  
        # CBC returns LpStatusNotSolved when timed out but may have found a feasible solution
        if (status == LpStatusOptimal or 
            status == LpStatusNotSolved or
            (status is None and total_time_var.value() is not None)):
            # Extract solution found by ILP
            task_assignments = {}
            chiplet_tasks = defaultdict(list)
            
            for task in tasks:
                for c in range(max_chiplets):
                    if z[task, c].value() and z[task, c].value() > 0.5:  # Handle numerical precision
                        task_assignments[task] = c
                        chiplet_tasks[c].append(task)
                        break
            
            # Only proceed if we have ALL tasks assigned (complete solution)
            if len(task_assignments) == len(tasks):
                # Extract PE assignments from ILP solution - both source and dest PEs
                pe_assignments = {}
                for pe in pes:
                    for c in range(max_chiplets):
                        if pe_used[pe, c].value() and pe_used[pe, c].value() > 0.5:
                            pe_assignments[pe] = c
                            break
                
                # Task assignments should already be correct since they're based on z variables
                # which are constrained to match source_pe assignments. Keep original task_assignments.
                # The PE assignments will be used for validation, not for changing task placement.
                
                # Extract task times
                task_times = {}
                for task in tasks:
                    if task_time[task].value() is not None:
                        task_times[task] = int(round(task_time[task].value()))
                    else:
                        task_times[task] = 0
                
                active_chiplets = len([c for c in chiplet_tasks if chiplet_tasks[c]])
                
                # Calculate total_time as the maximum task completion time + 1
                if task_times:
                    total_time = max(task_times.values()) + 1
                elif total_time_var.value() is not None:
                    total_time = int(round(total_time_var.value()))
                else:
                    total_time = max_time
                
                # Determine status - CBC returns 'optimal' even when timed out
                if status == LpStatusOptimal and solve_time < timeout * 0.95:
                    solution_status = 'optimal'
                elif status == LpStatusOptimal and solve_time >= timeout * 0.95:
                    solution_status = 'timeout_feasible' 
                elif status == LpStatusNotSolved:
                    solution_status = 'timeout_feasible'
                else:
                    solution_status = 'feasible'
                
                return {
                    'status': solution_status,
                    'pe_assignments': pe_assignments,  # Include PE assignments for validation
                    'total_time': total_time,
                    'num_chiplets': active_chiplets,
                    'solve_time': solve_time,
                    'task_assignments': task_assignments,
                    'task_times': task_times,
                    'chiplet_tasks': dict(chiplet_tasks)
                }
            else:
                # Return infeasible since we don't have a complete solution
                return {
                    'status': 'infeasible',
                    'solve_time': solve_time,
                    'note': f'Incomplete solution extraction: only {len(task_assignments)}/{len(tasks)} tasks assigned'
                }
        
        # No solution found
        return {
            'status': 'infeasible',
            'solve_time': solve_time,
            'note': f'ILP failed with status {LpStatus[status]}'
        }
    

# ================== EXAMPLE USAGE ==================

if __name__ == "__main__":
    # Create problem
    # problem = ChipletProblem('gpt2_transformer.txt')
    problem = ChipletProblem('gpt2_transformer_small.txt')
    
    # Add constraints declaratively
    problem.add_constraint(TaskAssignmentConstraint())
    problem.add_constraint(ChipletUsageConstraint()) 
    problem.add_constraint(TaskDependencyConstraint())
    problem.add_constraint(ChipletCapacityConstraint(max_pes=32))
    problem.add_constraint(PEExclusivityConstraint())
    problem.add_constraint(InterChipletCommConstraint(bandwidth=8192))
    problem.add_constraint(TimeBoundsConstraint())
    
    print("Problem loaded:")
    stats = problem.get_problem_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*50)
    
    # Solve with ILP only (with timeout solution extraction)
    print("Solving with ILP only (with 60s timeout solution extraction)...")
    try:
        solution = problem.solve(timeout=60, max_chiplets=10)
        
        print(f"Solution:")
        print(f"  Status: {solution['status']}")
        if solution['status'] != 'infeasible':
            print(f"  Total time: {solution['total_time']} cycles")
            print(f"  Chiplets used: {solution['num_chiplets']}")
            print(f"  Tasks assigned: {len(solution.get('task_assignments', {}))}")
        print(f"  Solve time: {solution['solve_time']:.3f}s")
        
        if 'note' in solution:
            print(f"  Note: {solution['note']}")
            
        if 'solution_file' in solution:
            print(f"  Solution file: {solution['solution_file']}")
        
    except Exception as e:
        print(f"Solver failed: {e}")