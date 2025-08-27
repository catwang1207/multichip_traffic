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
    """Generate a solution file showing task assignments per clock cycle"""
    
    if solution['status'] == 'infeasible':
        print(f"Cannot generate solution file - {solver_name} solution is infeasible")
        return None
    
    # Extract solution data
    task_assignments = solution.get('task_assignments', {})
    task_times = solution.get('task_times', {})
    total_time = solution.get('total_time', 0)
    num_chiplets = solution.get('num_chiplets', 0)
    
    # Create solution filename
    base_name = os.path.splitext(os.path.basename(data_file))[0]
    solution_file = f"{base_name}_{solver_name.lower()}_solution.json"
    
    # Organize tasks by clock cycle and chiplet
    schedule = {}
    for cycle in range(total_time):
        schedule[cycle] = {}
        for chiplet in range(max(task_assignments.values()) + 1 if task_assignments else 0):
            schedule[cycle][chiplet] = []
    
    # Assign tasks to their execution cycles and chiplets
    for task, chiplet in task_assignments.items():
        if task in task_times:
            cycle = task_times[task]
            if cycle < total_time:
                task_info = {
                    'task_id': task,
                    'source_pe': task_data[task]['source_pe'],
                    'dest_pe': task_data[task]['dest_pe'],
                    'data_size': task_data[task]['data_size']
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
        'schedule': []
    }
    
    # Convert schedule to list format
    for cycle in range(total_time):
        cycle_data = {
            'cycle': cycle,
            'chiplets': {}
        }
        
        for chiplet in range(num_chiplets):
            if chiplet in schedule[cycle]:
                cycle_data['chiplets'][chiplet] = schedule[cycle][chiplet]
            else:
                cycle_data['chiplets'][chiplet] = []
        
        solution_data['schedule'].append(cycle_data)
    
    # Write solution file
    try:
        with open(solution_file, 'w') as f:
            json.dump(solution_data, f, indent=2)
        
        print(f"✅ Solution file generated: {solution_file}")
        print(f"   Total cycles: {total_time}, Chiplets: {num_chiplets}, Tasks: {len(task_assignments)}")
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
    
    def solve(self, solver_type='greedy', save_solution_file=True, **kwargs):
        """Solve using specified solver"""
        # Solve with the specified solver
        if solver_type == 'ilp':
            solution = ILPSolver(self).solve(**kwargs)
            solver_name = "ILP"
        elif solver_type == 'greedy':
            solution = GreedySolver(self).solve(**kwargs)
            solver_name = "Greedy"
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")
        
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
    
    def solve(self, max_chiplets=10, timeout=300, **kwargs):
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
        
        # CREATE PROBLEM
        prob = LpProblem("ChipletAssignment", LpMinimize)
        
        # OBJECTIVE
        prob += 1000 * total_time_var + lpSum([y[c] for c in range(max_chiplets)])
        
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
                # Link task assignment to PE usage - only dest_pe (where task executes)
                for task in tasks:
                    dest_pe = task_data[task]['dest_pe']
                    for c in range(max_chiplets):
                        # If task is assigned to chiplet c, dest_pe must be used by chiplet c
                        prob += pe_used[dest_pe, c] >= z[task, c]
                
                # Each chiplet can use at most max_pes PEs
                for c in range(max_chiplets):
                    prob += lpSum([pe_used[pe, c] for pe in pes]) <= constraint.max_pes
                    
            elif isinstance(constraint, PEExclusivityConstraint):
                # Each PE belongs to exactly one chiplet (if used by any task)
                for pe in pes:
                    prob += lpSum([pe_used[pe, c] for c in range(max_chiplets)]) == 1
                    
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
        
        print(f"ILP formulation complete. Solving with timeout={timeout}s...")
        
        # SOLVE
        solve_start = time_module.time()
        status = prob.solve()
        solve_time = time_module.time() - solve_start
        
        if status == LpStatusOptimal:
            # Extract solution
            task_assignments = {}
            chiplet_tasks = defaultdict(list)
            
            for task in tasks:
                for c in range(max_chiplets):
                    if z[task, c].value() == 1:
                        task_assignments[task] = c
                        chiplet_tasks[c].append(task)
                        break
            
            # Extract task times
            task_times = {}
            for task in tasks:
                task_times[task] = int(task_time[task].value())
            
            active_chiplets = len([c for c in chiplet_tasks if chiplet_tasks[c]])
            
            return {
                'status': 'optimal' if status == LpStatusOptimal else 'feasible',
                'total_time': int(total_time_var.value()),
                'num_chiplets': active_chiplets,
                'solve_time': solve_time,
                'task_assignments': task_assignments,
                'task_times': task_times,
                'chiplet_tasks': dict(chiplet_tasks)
            }
        else:
            return {
                'status': 'infeasible',
                'solve_time': solve_time
            }

# ================== GREEDY SOLVER ==================

class GreedySolver(Solver):
    """Greedy heuristic solver - interprets constraints as heuristic rules"""
    
    def solve(self, **kwargs):
        """Solve using greedy heuristic guided by constraints"""
        
        print(f"=== GREEDY SOLVER ===")
        print(f"Applying {len(self.problem.constraints)} constraints as heuristic rules...")
        
        tasks = self.problem.tasks
        task_data = self.problem.task_data
        dependencies = self.problem.dependencies
        
        # Extract constraint parameters
        max_pes = 32
        bandwidth = 8192
        
        for constraint in self.problem.constraints:
            if isinstance(constraint, ChipletCapacityConstraint):
                max_pes = constraint.max_pes
            elif isinstance(constraint, InterChipletCommConstraint):
                bandwidth = constraint.bandwidth
        
        # GREEDY ALGORITHM WITH CONSTRAINT-GUIDED HEURISTICS
        
        class Chiplet:
            def __init__(self, id):
                self.id = id
                self.tasks = []
                self.pes = set()
                self.current_time = 0
                
            def can_add_pe(self, pe):
                return len(self.pes) < max_pes or pe in self.pes
                
            def add_task(self, task, pe, source_pe, data_size, start_time):
                self.tasks.append(task)
                self.pes.add(pe)
                
                # Apply InterChipletCommConstraint
                if source_pe != pe and data_size > bandwidth:
                    duration = 2  # Extra cycle for large inter-chiplet transfer
                else:
                    duration = 1
                    
                self.current_time = max(self.current_time, start_time + duration)
                return self.current_time
        
        # Initialize
        chiplets = []
        task_assignments = {}
        task_completion_times = {}
        
        def get_ready_tasks(scheduled_tasks):
            ready = []
            for task in tasks:
                if task not in scheduled_tasks:
                    # Apply TaskDependencyConstraint
                    if all(dep in scheduled_tasks for dep in dependencies[task]):
                        ready.append(task)
            return ready
        
        # MAIN GREEDY LOOP
        scheduled_tasks = set()
        
        while len(scheduled_tasks) < len(tasks):
            ready_tasks = get_ready_tasks(scheduled_tasks)
            
            if not ready_tasks:
                break
                
            for task in ready_tasks:
                pe = task_data[task]['dest_pe']
                source_pe = task_data[task]['source_pe']
                data_size = task_data[task]['data_size']
                
                # Calculate earliest start time (TaskDependencyConstraint)
                earliest_start = 0
                for dep_task in dependencies[task]:
                    if dep_task in task_completion_times:
                        earliest_start = max(earliest_start, task_completion_times[dep_task])
                
                # Find best chiplet (guided by constraints)
                best_chiplet = None
                best_score = float('inf')
                
                # Try existing chiplets
                for chiplet in chiplets:
                    if chiplet.can_add_pe(pe):  # ChipletCapacityConstraint
                        start_time = max(earliest_start, chiplet.current_time)
                        
                        # Communication cost (InterChipletCommConstraint)
                        comm_cost = 0
                        if source_pe != pe:
                            if source_pe in chiplet.pes:
                                comm_cost = 0  # Intra-chiplet
                            else:
                                if data_size > bandwidth:
                                    comm_cost = 10 * (data_size // bandwidth)
                                else:
                                    comm_cost = 5
                        
                        duration = 2 if (source_pe != pe and data_size > bandwidth) else 1
                        finish_time = start_time + duration
                        score = finish_time + comm_cost
                        
                        if score < best_score:
                            best_score = score
                            best_chiplet = chiplet
                
                # Try creating new chiplet (ChipletUsageConstraint)
                if len(chiplets) < 20:  # Reasonable limit
                    start_time = earliest_start
                    duration = 2 if (source_pe != pe and data_size > bandwidth) else 1
                    finish_time = start_time + duration
                    score = finish_time + 50  # Penalty for new chiplet
                    
                    if score < best_score:
                        best_chiplet = None  # Signal to create new
                
                # Assign task
                if best_chiplet is None:
                    new_chiplet = Chiplet(len(chiplets))
                    chiplets.append(new_chiplet)
                    best_chiplet = new_chiplet
                
                start_time = max(earliest_start, best_chiplet.current_time)
                completion_time = best_chiplet.add_task(task, pe, source_pe, data_size, start_time)
                
                task_assignments[task] = best_chiplet.id
                task_completion_times[task] = completion_time
                scheduled_tasks.add(task)
        
        # TimeBoundsConstraint
        total_time = max(task_completion_times.values()) if task_completion_times else 0
        
        chiplet_tasks = defaultdict(list)
        for task, chiplet_id in task_assignments.items():
            chiplet_tasks[chiplet_id].append(task)
        
        return {
            'status': 'feasible',
            'total_time': total_time,
            'num_chiplets': len(chiplets),
            'solve_time': 0.0,  # Very fast
            'task_assignments': task_assignments,
            'task_times': task_completion_times,
            'chiplet_tasks': dict(chiplet_tasks),
            'chiplets': chiplets  # Additional info
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
    
    # Solve with greedy heuristic
    # print("Solving with GREEDY heuristic...")
    # greedy_solution = problem.solve('greedy')
    
    # print(f"Greedy solution:")
    # print(f"  Status: {greedy_solution['status']}")
    # print(f"  Total time: {greedy_solution['total_time']} cycles")
    # print(f"  Chiplets used: {greedy_solution['num_chiplets']}")
    # print(f"  Solve time: {greedy_solution.get('solve_time', 0):.3f}s")
    
    print("\n" + "="*50)
    
    # Solve with ILP (smaller problem)
    print("Solving with ILP (may take longer)...")
    try:
        ilp_solution = problem.solve('ilp', max_chiplets=10, timeout=60)
        
        print(f"ILP solution:")
        print(f"  Status: {ilp_solution['status']}")
        if ilp_solution['status'] != 'infeasible':
            print(f"  Total time: {ilp_solution['total_time']} cycles")
            print(f"  Chiplets used: {ilp_solution['num_chiplets']}")
        print(f"  Solve time: {ilp_solution['solve_time']:.3f}s")
        
    except Exception as e:
        print(f"ILP failed: {e}")