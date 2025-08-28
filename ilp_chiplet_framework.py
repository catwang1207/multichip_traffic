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

@dataclass
class NoMulticastingConstraint:
    """Forbid multicasting - each source PE can only send to one destination PE per time cycle"""
    name: str = "no_multicasting"
    description: str = "Forbid multicasting - each source PE can only send to one destination PE per time cycle"

# ================== SOLUTION FILE GENERATOR ==================

def generate_solution_file(solution, solver_name, data_file, task_data):
    """Generate a solution file showing task assignments per clock cycle with detailed PE assignments"""
    
    if solution['status'] == 'infeasible':
        print(f"Cannot generate solution file - {solver_name} solution is infeasible")
        return None
    
    # Extract solution data
    task_assignments = solution.get('task_assignments', {})
    task_times = solution.get('task_times', {})
    task_durations = solution.get('task_durations', {})  # Extract task durations
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
    
    # Assign tasks to their execution cycles and chiplets (accounting for task duration)
    for task, chiplet in task_assignments.items():
        if task in task_times:
            start_cycle = task_times[task]
            duration = task_durations.get(task, 1)  # Default to 1 cycle if not found
            
            # Add task to all cycles it's active: [start_cycle, start_cycle + duration)
            for cycle_offset in range(duration):
                cycle = start_cycle + cycle_offset
                if cycle < total_time:
                    source_pe = task_data[task]['source_pe']
                    dest_pe = task_data[task]['dest_pe']
                    
                    task_info = {
                        'task_id': task,
                        'source_pe': source_pe,
                        'dest_pe': dest_pe,
                        'data_size': task_data[task]['data_size'],
                        'source_pe_chiplet': pe_assignments.get(source_pe, chiplet),
                        'dest_pe_chiplet': pe_assignments.get(dest_pe, -1),
                        'execution_cycle': cycle,
                        'start_cycle': start_cycle,
                        'duration': duration,
                        'cycle_offset': cycle_offset  # 0 for first cycle, 1 for second cycle, etc.
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
        try:
            df = pd.read_csv(self.data_file, sep='\t', comment='#', 
                           names=['task_id', 'source_pe', 'dest_pe', 'data_size', 'wait_ids'])
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        except pd.errors.EmptyDataError:
            raise ValueError(f"Data file is empty: {self.data_file}")
        except Exception as e:
            raise ValueError(f"Error reading data file {self.data_file}: {e}")
        
        if df.empty:
            raise ValueError(f"Data file contains no data: {self.data_file}")
        
        # Validate required columns exist
        required_columns = ['task_id', 'source_pe', 'dest_pe', 'data_size', 'wait_ids']
        if len(df.columns) != len(required_columns):
            raise ValueError(f"Data file must have exactly {len(required_columns)} columns, got {len(df.columns)}")
        
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
        
        # GLOBAL TASK DURATION AND INTER-CHIP VARIABLES
        # These will be used by multiple constraints
        task_duration = {}
        task_inter_chip = {}
        task_both_in_chiplet = {}  # task_both_in_chiplet[task, c] = both PEs of task in chiplet c
        
        for task in tasks:
            source_pe = task_data[task]['source_pe']
            dest_pe = task_data[task]['dest_pe']
            data_size = task_data[task]['data_size']
            
            # Task duration: 1 cycle (intra/small) or 2 cycles (large inter-chip)
            task_duration[task] = LpVariable(f"duration_{task}", lowBound=1, upBound=2, cat='Integer')
            
            # Inter-chip detection: 1 if source_pe and dest_pe in different chiplets
            task_inter_chip[task] = LpVariable(f"inter_chip_{task}", cat='Binary')
            
            # Auxiliary variables for detecting both PEs in same chiplet
            # Only create these for tasks that actually need inter-chip detection
            if source_pe != dest_pe:
                for c in range(max_chiplets):
                    task_both_in_chiplet[task, c] = LpVariable(f"both_in_c{c}_{task}", cat='Binary')
        
        # CREATE PROBLEM
        prob = LpProblem("ChipletAssignment", LpMinimize)
        
        # Add global task duration constraints AFTER creating the problem
        for task in tasks:
            source_pe = task_data[task]['source_pe']
            dest_pe = task_data[task]['dest_pe']
            data_size = task_data[task]['data_size']
            
            if source_pe == dest_pe:
                # Intra-PE task: always intra-chip, no need for complex inter-chip detection
                prob += task_inter_chip[task] == 0
                prob += task_duration[task] == 1  # Always 1 cycle for intra-PE
            else:
                # Inter-PE task: need to determine if PEs are in same or different chiplets
                for c in range(max_chiplets):
                    # both_in_chiplet[task, c] = 1 iff both source_pe and dest_pe are in chiplet c
                    prob += task_both_in_chiplet[task, c] <= pe_used[source_pe, c]
                    prob += task_both_in_chiplet[task, c] <= pe_used[dest_pe, c]  
                    prob += task_both_in_chiplet[task, c] >= pe_used[source_pe, c] + pe_used[dest_pe, c] - 1
                    
                    # If both PEs in this chiplet, then NOT inter-chip
                    prob += task_inter_chip[task] <= 1 - task_both_in_chiplet[task, c]
                
                # Force inter_chip = 1 if no chiplet contains both PEs
                # Only sum over chiplets where the variable exists (task has different PEs)
                both_vars = [task_both_in_chiplet[task, c] for c in range(max_chiplets) if (task, c) in task_both_in_chiplet]
                prob += task_inter_chip[task] >= 1 - lpSum(both_vars)
                
                # Link duration to data size and inter-chip status
                if data_size > 8192:  # Large data transfers
                    prob += task_duration[task] == 1 + task_inter_chip[task]  # 1 intra, 2 inter
                else:  # Small data transfers
                    prob += task_duration[task] == 1  # Always 1 cycle
        
        # OBJECTIVE: Minimize time + heavily penalize chiplet usage
        prob.objective = 1000 * total_time_var + 500 * lpSum([chiplet_pe_used[c] for c in range(max_chiplets)])
        
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
                # Dependent tasks must execute in correct order with actual task durations
                for task in tasks:
                    for dep_task in dependencies[task]:
                        # Task must wait for dependency to complete: start_time >= dep_start_time + dep_duration
                        prob += task_time[task] >= task_time[dep_task] + task_duration[dep_task]
                        
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
                        
                        # Note: Timing constraints for inter-chiplet communication are now handled
                        # by the global task_duration[task] variables, so no additional timing
                        # constraints are needed here. The task_duration already accounts for 
                        # inter-chip delays based on data size and chiplet placement.
                
                # Add communication penalty to objective (keep existing penalty system)
                for pe, volume in pe_comm_volume.items():
                    if constraint.bandwidth > 0 and volume > constraint.bandwidth:
                        penalty_weight = (volume // constraint.bandwidth) * 0.1
                        for c in range(max_chiplets):
                            comm_penalty += penalty_weight * pe_used[pe, c]
                
                # Update objective with communication penalty
                prob.objective += comm_penalty
                
            elif isinstance(constraint, TimeBoundsConstraint):
                # Total time must cover all task completion times with actual task durations
                for task in tasks:
                    # Total time must be at least when each task finishes: start_time + duration
                    prob += total_time_var >= task_time[task] + task_duration[task]
                    
            elif isinstance(constraint, NoMulticastingConstraint):
                # Forbid multicasting: each source PE can only send to one destination at a time
                # Use the global task_duration variables that account for inter-chip communication
                
                # Group tasks by source PE
                source_pe_tasks = defaultdict(list)
                for task in tasks:
                    source_pe = task_data[task]['source_pe']
                    source_pe_tasks[source_pe].append(task)
                
                # For each source PE with multiple tasks, prevent overlapping execution
                for source_pe, pe_tasks in source_pe_tasks.items():
                    if len(pe_tasks) > 1:
                        for i, task1 in enumerate(pe_tasks):
                            for task2 in pe_tasks[i+1:]:
                                # Prevent overlap: either task1 finishes before task2 starts, or vice versa
                                # task1 finishes before task2 starts: task_time[task1] + duration1 <= task_time[task2]
                                # task2 finishes before task1 starts: task_time[task2] + duration2 <= task_time[task1]
                                
                                # Use binary variable to choose one of the two orderings
                                task1_before_task2 = LpVariable(f"order_{task1}_before_{task2}", cat='Binary')
                                
                                # If task1_before_task2 = 1, then task1 finishes before task2 starts
                                prob += task_time[task2] >= task_time[task1] + task_duration[task1] - max_time * (1 - task1_before_task2)
                                
                                # If task1_before_task2 = 0, then task2 finishes before task1 starts  
                                prob += task_time[task1] >= task_time[task2] + task_duration[task2] - max_time * task1_before_task2
        
        # Add chiplet PE usage constraints: chiplet_pe_used[c] = 1 if any PE is assigned to chiplet c
        for c in range(max_chiplets):
            # If any PE is assigned to chiplet c, then chiplet_pe_used[c] must be 1
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
                    if z[task, c].value() is not None and z[task, c].value() > 0.5:  # Handle numerical precision
                        task_assignments[task] = c
                        chiplet_tasks[c].append(task)
                        break
            
            # Only proceed if we have ALL tasks assigned (complete solution)
            if len(task_assignments) == len(tasks):
                # Extract PE assignments from ILP solution - both source and dest PEs
                pe_assignments = {}
                for pe in pes:
                    for c in range(max_chiplets):
                        if pe_used[pe, c].value() is not None and pe_used[pe, c].value() > 0.5:
                            pe_assignments[pe] = c
                            break
                
                # Task assignments should already be correct since they're based on z variables
                # which are constrained to match source_pe assignments. Keep original task_assignments.
                # The PE assignments will be used for validation, not for changing task placement.
                
                # Extract task times and durations
                task_times = {}
                task_durations = {}
                for task in tasks:
                    if task_time[task].value() is not None:
                        task_times[task] = int(round(task_time[task].value()))
                    else:
                        task_times[task] = 0
                    
                    if task_duration[task].value() is not None:
                        task_durations[task] = int(round(task_duration[task].value()))
                    else:
                        task_durations[task] = 1
                
                active_chiplets = len([c for c in chiplet_tasks if chiplet_tasks[c]])
                
                # Calculate total_time as the maximum task completion time (start + duration)
                if task_times and task_durations:
                    # Only consider tasks that are actually assigned and have valid times/durations
                    assigned_tasks = [task for task in tasks if task in task_times and task in task_durations]
                    if assigned_tasks:
                        max_completion_time = max(task_times[task] + task_durations[task] for task in assigned_tasks)
                        total_time = max_completion_time
                    else:
                        total_time = max_time
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
                    'task_durations': task_durations,  # Include task durations for validation
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
        status_name = LpStatus.get(status, f'Unknown({status})')
        return {
            'status': 'infeasible',
            'solve_time': solve_time,
            'note': f'ILP failed with status {status_name}'
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
    problem.add_constraint(NoMulticastingConstraint())  # Add the new constraint
    
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