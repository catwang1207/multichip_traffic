#!/usr/bin/env python3

import pandas as pd
import time as time_module
import json
import os
import random
import math
import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from ilp_chiplet_framework import *  # Import constraint definitions
from config import MAX_PES_PER_CHIPLET, INTER_CHIPLET_BANDWIDTH, COST_WEIGHTS

# ================== SIMULATED ANNEALING SOLUTION ==================

class SimulatedAnnealingSolution:
    """Represents a solution for the chiplet assignment problem using simulated annealing"""
    
    def __init__(self, problem, max_chiplets: int):
        self.problem = problem
        self.max_chiplets = max_chiplets
        self.tasks = list(problem.tasks)
        self.pes = list(problem.pes)
        
        # Solution representation
        self.task_assignments = {}  # task -> chiplet_id
        self.task_times = {}        # task -> start_time
        self.task_durations = {}    # task -> duration (1 or 2 cycles)
        self.pe_assignments = {}    # pe -> chiplet_id
        
        # Cached values for efficiency
        self._cached_cost = None
        self._cached_violations = None
        self._dirty = True
        
    def random_initialize(self):
        """Initialize with MINIMUM chiplets, expanding only if cycle improvement justifies cost"""
        print(f"Chiplet-minimizing initialization for {len(self.pes)} PEs")
        
        # STEP 1: Calculate theoretical minimum chiplets needed
        min_chiplets_capacity = (len(self.pes) + MAX_PES_PER_CHIPLET - 1) // MAX_PES_PER_CHIPLET  # Ceiling division for PE capacity
        
        # STEP 2: Analyze communication patterns
        pe_communication = {}  # (pe1, pe2) -> frequency
        for task in self.tasks:
            src = self.problem.task_data[task]['source_pe']
            dst = self.problem.task_data[task]['dest_pe']
            pair = tuple(sorted([src, dst]))
            pe_communication[pair] = pe_communication.get(pair, 0) + 1
        
        print(f"Minimum chiplets by capacity: {min_chiplets_capacity}")
        print(f"Found {len(pe_communication)} PE communication pairs")
        
        # STEP 3: Try increasing chiplet counts, starting from minimum
        best_solution = None
        best_cost = float('inf')
        
        for target_chiplets in range(min_chiplets_capacity, self.max_chiplets + 1):
            print(f"Trying {target_chiplets} chiplets...")
            
            # Create clusters for this chiplet count
            clusters = self._create_communication_clusters(pe_communication, target_chiplets)
            
            if not clusters:
                continue
                
            # Test this configuration
            test_slots = self._assign_clusters_to_slots(clusters, target_chiplets)
            test_cost = self._evaluate_chiplet_configuration(test_slots, target_chiplets)
            
            print(f"  {target_chiplets} chiplets → cost={test_cost}")
            
            # EXPONENTIAL penalty for additional chiplets
            extra_chiplets = max(0, target_chiplets - min_chiplets_capacity)
            if extra_chiplets == 0:
                chiplet_penalty = 0
            else:
                chiplet_penalty = COST_WEIGHTS['exponential_penalty_base'] * (2 ** extra_chiplets - 1)  # Exponential growth
            adjusted_cost = test_cost + chiplet_penalty
            
            if adjusted_cost < best_cost:
                best_cost = adjusted_cost
                best_solution = (test_slots, target_chiplets)
                print(f"  ✅ New best: {target_chiplets} chiplets, adjusted_cost={adjusted_cost}")
            else:
                print(f"  ❌ Not worth it: extra penalty={chiplet_penalty}, adjusted_cost={adjusted_cost}")
                # If cost is getting worse, stop expanding
                break
        
        # STEP 4: Use the best configuration found
        if best_solution:
            self.pe_slots, optimal_chiplets = best_solution
            print(f"OPTIMAL: {optimal_chiplets} chiplets selected")
        else:
            # Fallback to minimum
            print(f"FALLBACK: Using minimum {min_chiplets_capacity} chiplets")
            clusters = self._create_communication_clusters(pe_communication, min_chiplets_capacity)
            self.pe_slots = self._assign_clusters_to_slots(clusters, min_chiplets_capacity)
        
        # Create PE to chiplet mapping from slots
        self.pe_assignments = {}
        for chiplet in range(self.max_chiplets):
            for pe in self.pe_slots[chiplet]:
                if pe != -1:
                    self.pe_assignments[pe] = chiplet
        
        # Assign tasks based on PE assignments
        for task in self.tasks:
            source_pe = self.problem.task_data[task]['source_pe']
            dest_pe = self.problem.task_data[task]['dest_pe']
            
            # Both source and dest PE must be in the same chiplet
            source_chiplet = self.pe_assignments.get(source_pe)
            dest_chiplet = self.pe_assignments.get(dest_pe)
            
            if source_chiplet == dest_chiplet and source_chiplet is not None:
                self.task_assignments[task] = source_chiplet
            else:
                # Handle edge case - assign to source PE's chiplet
                if source_chiplet is not None:
                    self.task_assignments[task] = source_chiplet
                elif dest_chiplet is not None:
                    self.task_assignments[task] = dest_chiplet
                else:
                    # Fallback - assign to chiplet 0
                    self.task_assignments[task] = 0
        
        # Create a feasible schedule respecting dependencies
        self._create_feasible_schedule()
        
        self._dirty = True
    
    def _create_communication_clusters(self, pe_communication, target_chiplets):
        """Create communication-aware clusters for target chiplet count"""
        clusters = []
        used_pes = set()
        
        # Sort PE pairs by communication frequency
        sorted_pairs = sorted(pe_communication.items(), key=lambda x: x[1], reverse=True)
        
        # Create clusters starting with highest-communicating pairs
        for (pe1, pe2), _ in sorted_pairs:  # freq not used, using _ to indicate
            if pe1 in used_pes or pe2 in used_pes or len(clusters) >= target_chiplets:
                continue
                
            # Start new cluster
            cluster = {pe1, pe2}
            used_pes.add(pe1)
            used_pes.add(pe2)
            
            # Expand cluster greedily
            max_size = min(MAX_PES_PER_CHIPLET, len(self.pes) // target_chiplets + 5)  # Target balanced sizes
            for other_pe in self.pes:
                if (other_pe in used_pes or len(cluster) >= max_size):
                    continue
                    
                # Check communication with cluster
                comm_count = sum(pe_communication.get(tuple(sorted([other_pe, cluster_pe])), 0) 
                               for cluster_pe in cluster)
                if comm_count > 0:
                    cluster.add(other_pe)
                    used_pes.add(other_pe)
                    
            clusters.append(cluster)
        
        # Handle remaining PEs
        remaining_pes = [pe for pe in self.pes if pe not in used_pes]
        for pe in remaining_pes:
            # Find best cluster (smallest that has room)
            best_cluster = None
            for cluster in clusters:
                if len(cluster) < MAX_PES_PER_CHIPLET:
                    if best_cluster is None or len(cluster) < len(best_cluster):
                        best_cluster = cluster
            
            if best_cluster:
                best_cluster.add(pe)
            else:
                # Create new cluster if needed
                if len(clusters) < target_chiplets:
                    clusters.append({pe})
                else:
                    # Force into smallest cluster (violate capacity if needed)
                    min_cluster = min(clusters, key=len)
                    min_cluster.add(pe)
        
        return clusters
    
    def _assign_clusters_to_slots(self, clusters, target_chiplets):
        """Assign clusters to chiplet slots"""
        pe_slots = {}
        for chiplet in range(self.max_chiplets):
            pe_slots[chiplet] = [-1] * MAX_PES_PER_CHIPLET  # slots per chiplet
        
        for chiplet_idx, cluster in enumerate(clusters[:target_chiplets]):
            slot_idx = 0
            for pe in cluster:
                if slot_idx < MAX_PES_PER_CHIPLET:
                    pe_slots[chiplet_idx][slot_idx] = pe
                    slot_idx += 1
        
        return pe_slots
    
    def _evaluate_chiplet_configuration(self, test_slots, target_chiplets):
        """Evaluate a chiplet configuration without modifying current state"""
        # Temporarily store current state (if exists)
        old_pe_slots = getattr(self, 'pe_slots', None)
        old_pe_assignments = getattr(self, 'pe_assignments', {})
        old_task_assignments = getattr(self, 'task_assignments', {})
        old_task_times = getattr(self, 'task_times', {})
        old_task_durations = getattr(self, 'task_durations', {})
        
        try:
            # Apply test configuration
            self.pe_slots = test_slots
            
            # Update assignments and schedule
            self.pe_assignments = {}
            for chiplet in range(target_chiplets):
                for pe in self.pe_slots[chiplet]:
                    if pe != -1:
                        self.pe_assignments[pe] = chiplet
            
            # Update task assignments
            for task in self.tasks:
                source_pe = self.problem.task_data[task]['source_pe']
                dest_pe = self.problem.task_data[task]['dest_pe']
                
                source_chiplet = self.pe_assignments.get(source_pe)
                dest_chiplet = self.pe_assignments.get(dest_pe)
                
                if source_chiplet == dest_chiplet and source_chiplet is not None:
                    self.task_assignments[task] = source_chiplet
                else:
                    if source_chiplet is not None:
                        self.task_assignments[task] = source_chiplet
                    elif dest_chiplet is not None:
                        self.task_assignments[task] = dest_chiplet
                    else:
                        self.task_assignments[task] = 0
            
            # Create schedule and evaluate
            self._create_feasible_schedule()
            cost = self.evaluate_cost()
            
            return cost
            
        finally:
            # Restore original state
            self.pe_slots = old_pe_slots
            self.pe_assignments = old_pe_assignments
            self.task_assignments = old_task_assignments
            self.task_times = old_task_times
            self.task_durations = old_task_durations
        
                
    def _calculate_task_duration(self, task):
        """Calculate duration for a task based on chiplet assignments and bandwidth constraint"""
        source_pe = self.problem.task_data[task]['source_pe']
        dest_pe = self.problem.task_data[task]['dest_pe']
        data_size = self.problem.task_data[task]['data_size']
        
        source_chiplet = self.pe_assignments.get(source_pe)
        dest_chiplet = self.pe_assignments.get(dest_pe)
        
        # Intra-chiplet tasks always take 1 cycle
        if (source_chiplet is None or dest_chiplet is None or 
            source_chiplet == dest_chiplet):
            return 1
        
        # Inter-chiplet tasks: calculate based on bandwidth constraint
        # Find the bandwidth constraint
        inter_chiplet_bandwidth = INTER_CHIPLET_BANDWIDTH  # Default, but should find from constraints
        for constraint in self.problem.constraints:
            if hasattr(constraint, 'bandwidth'):
                inter_chiplet_bandwidth = constraint.bandwidth
                break
        
        # Duration = ceiling(data_size / bandwidth)
        import math
        duration = math.ceil(data_size / inter_chiplet_bandwidth)
        return max(1, duration)  # At least 1 cycle

    def _create_feasible_schedule(self):
        """Create a feasible schedule respecting task dependencies and timing constraints"""
        dependencies = self.problem.dependencies
        
        # Pre-calculate all task durations
        self.task_durations = {}
        for task in self.tasks:
            self.task_durations[task] = self._calculate_task_duration(task)
        
        # Topological sort with timing
        visited = set()
        self.task_times = {}
        
        def schedule_task(task):
            if task in visited:
                return self.task_times.get(task, 0)
                
            visited.add(task)
            
            # Calculate earliest start time based on dependencies
            earliest_start = 0
            
            for dep_task in dependencies[task]:
                dep_end_time = schedule_task(dep_task)
                dep_duration = self.task_durations[dep_task]  # Now guaranteed to exist
                
                # Add communication delay for inter-chiplet dependencies
                communication_delay = 0
                if (self.task_assignments.get(dep_task, 0) != self.task_assignments.get(task, 0)):
                    communication_delay = 1  # One cycle delay for inter-chiplet
                    
                required_end_time = dep_end_time + dep_duration + communication_delay
                earliest_start = max(earliest_start, required_end_time)
            
            # Handle NoMulticasting constraint - serialize tasks from same source PE
            source_pe = self.problem.task_data[task]['source_pe']
            
            # Find other tasks from same PE and ensure they don't overlap
            for other_task in self.tasks:
                if (other_task != task and 
                    other_task in self.task_times and 
                    self.problem.task_data[other_task]['source_pe'] == source_pe):
                    
                    other_end = self.task_times[other_task] + self.task_durations[other_task]
                    earliest_start = max(earliest_start, other_end)
            
            self.task_times[task] = earliest_start
            return self.task_times[task]
        
        # Schedule all tasks
        for task in self.tasks:
            schedule_task(task)
    
    def evaluate_cost(self):
        """Evaluate the cost of the current solution"""
        if not self._dirty and self._cached_cost is not None:
            return self._cached_cost
            
        total_violations = 0
        max_time = 0
        
        if self.task_times:
            max_time = max(self.task_times[task] + self.task_durations.get(task, 1) 
                          for task in self.tasks if task in self.task_times)
        
        # Count constraint violations
        violations = self.count_violations()
        total_violations = sum(violations.values())
        
        # Objective: 100*max_time + EXPONENTIAL_chiplet_penalty + 1000000*violations
        used_chiplets = len(set(self.task_assignments.values()))
        
        # Calculate exponential chiplet penalty
        min_chiplets_capacity = (len(self.pes) + MAX_PES_PER_CHIPLET - 1) // MAX_PES_PER_CHIPLET  # Theoretical minimum
        extra_chiplets = max(0, used_chiplets - min_chiplets_capacity)
        if extra_chiplets == 0:
            chiplet_penalty = 500 * used_chiplets  # Base cost for minimum chiplets
        else:
            base_cost = 500 * min_chiplets_capacity
            exponential_penalty = COST_WEIGHTS['exponential_penalty_base'] * (2 ** extra_chiplets - 1)
            chiplet_penalty = base_cost + exponential_penalty
        
        cost = COST_WEIGHTS['cycle_weight'] * max_time + chiplet_penalty + COST_WEIGHTS['violation_penalty'] * total_violations
        
        self._cached_cost = cost
        self._cached_violations = violations
        self._dirty = False
        
        return cost
    
    def count_violations(self):
        """Count violations of each constraint type"""
        violations = {
            'task_assignment': 0,
            'chiplet_capacity': 0, 
            'pe_exclusivity': 0,
            'task_dependencies': 0,
            'no_multicasting': 0,
            'inter_chiplet_comm': 0
        }
        
        # Task assignment violations (should be 0 by construction)
        unassigned_tasks = [task for task in self.tasks if task not in self.task_assignments]
        violations['task_assignment'] = len(unassigned_tasks)
        
        # Chiplet capacity violations - get max_pes from constraint
        max_pes_limit = MAX_PES_PER_CHIPLET  # Default from config
        for constraint in self.problem.constraints:
            if hasattr(constraint, 'max_pes'):
                max_pes_limit = constraint.max_pes
                break
                
        if hasattr(self, 'pe_slots'):
            # Slot-based approach - capacity is guaranteed by construction
            for chiplet in range(self.max_chiplets):
                used_slots = sum(1 for pe in self.pe_slots[chiplet] if pe != -1)
                if used_slots > max_pes_limit:
                    violations['chiplet_capacity'] += used_slots - max_pes_limit
        else:
            # Fallback to old method if pe_slots not available
            chiplet_pe_count = defaultdict(set)
            for task in self.tasks:
                if task in self.task_assignments:
                    chiplet = self.task_assignments[task]
                    source_pe = self.problem.task_data[task]['source_pe']
                    dest_pe = self.problem.task_data[task]['dest_pe']
                    chiplet_pe_count[chiplet].add(source_pe)
                    chiplet_pe_count[chiplet].add(dest_pe)
                    
            for chiplet, pes in chiplet_pe_count.items():
                if len(pes) > max_pes_limit:
                    violations['chiplet_capacity'] += len(pes) - max_pes_limit
        
        # PE exclusivity violations
        pe_chiplet_count = defaultdict(set)
        for pe, chiplet in self.pe_assignments.items():
            pe_chiplet_count[pe].add(chiplet)
            
        for pe, chiplets in pe_chiplet_count.items():
            if len(chiplets) > 1:
                violations['pe_exclusivity'] += len(chiplets) - 1
        
        # Task dependency violations - simplified to match schedule logic
        dependencies = self.problem.dependencies
        for task in self.tasks:
            if task in self.task_times:
                task_start = self.task_times[task]
                for dep_task in dependencies[task]:
                    if dep_task in self.task_times:
                        dep_start = self.task_times[dep_task]
                        dep_duration = self.task_durations.get(dep_task, 1)
                        dep_end = dep_start + dep_duration
                        
                        # Add communication delay for inter-chiplet dependencies
                        communication_delay = 0
                        if (self.task_assignments.get(dep_task, 0) != self.task_assignments.get(task, 0)):
                            communication_delay = 1
                        
                        required_start = dep_end + communication_delay
                        
                        # Violation if task starts before required time
                        if task_start < required_start:
                            violations['task_dependencies'] += 1
        
        # No multicasting violations
        time_pe_tasks = defaultdict(lambda: defaultdict(list))
        for task in self.tasks:
            if task in self.task_times:
                start_time = self.task_times[task]
                duration = self.task_durations.get(task, 1)
                source_pe = self.problem.task_data[task]['source_pe']
                dest_pe = self.problem.task_data[task]['dest_pe']
                
                # Check each cycle the task is active
                for cycle in range(start_time, start_time + duration):
                    time_pe_tasks[cycle][source_pe].append((task, dest_pe))
        
        for cycle, pe_tasks in time_pe_tasks.items():
            for source_pe, tasks_dests in pe_tasks.items():
                if len(tasks_dests) > 1:
                    unique_dests = set(dest for _, dest in tasks_dests)
                    if len(unique_dests) > 1:
                        violations['no_multicasting'] += len(unique_dests) - 1
        
        return violations
    
    def get_neighbors(self):
        """Generate ONE neighbor solution quickly (SPEED OPTIMIZED)"""
        # SPEED OPTIMIZATION: Generate only 1 neighbor per iteration
        neighbor = copy.deepcopy(self)
        
        # Simple random PE swap between chiplets
        chiplet1 = random.randint(0, self.max_chiplets - 1)
        chiplet2 = random.randint(0, self.max_chiplets - 1)
        
        slot1 = random.randint(0, MAX_PES_PER_CHIPLET - 1)
        slot2 = random.randint(0, MAX_PES_PER_CHIPLET - 1)
        
        # Swap PE contents
        pe1 = neighbor.pe_slots[chiplet1][slot1]
        pe2 = neighbor.pe_slots[chiplet2][slot2]
        
        neighbor.pe_slots[chiplet1][slot1] = pe2
        neighbor.pe_slots[chiplet2][slot2] = pe1
        
        # Update assignments
        neighbor._update_pe_assignments_from_slots()
        neighbor._update_task_assignments_from_pe_slots()
        neighbor._create_feasible_schedule()
        neighbor._dirty = True
        
        return [neighbor]
    
    def _update_pe_assignments_from_slots(self):
        """Update PE assignments based on current slot configuration"""
        self.pe_assignments = {}
        for chiplet in range(self.max_chiplets):
            for pe in self.pe_slots[chiplet]:
                if pe != -1:
                    self.pe_assignments[pe] = chiplet
    
    def _update_task_assignments_from_pe_slots(self):
        """Update task assignments based on PE slot assignments"""
        for task in self.tasks:
            source_pe = self.problem.task_data[task]['source_pe']
            dest_pe = self.problem.task_data[task]['dest_pe']
            
            # Both source and dest PE must be in the same chiplet
            source_chiplet = self.pe_assignments.get(source_pe)
            dest_chiplet = self.pe_assignments.get(dest_pe)
            
            if source_chiplet == dest_chiplet and source_chiplet is not None:
                self.task_assignments[task] = source_chiplet
            else:
                # Handle edge case - assign to source PE's chiplet
                if source_chiplet is not None:
                    self.task_assignments[task] = source_chiplet
                elif dest_chiplet is not None:
                    self.task_assignments[task] = dest_chiplet
                else:
                    # Fallback
                    self.task_assignments[task] = 0
    
    def get_total_time(self):
        """Calculate total execution time (max finish time across all tasks)"""
        if not self.task_times:
            return 0
        return max(self.task_times[task] + self.task_durations.get(task, 1) 
                  for task in self.tasks if task in self.task_times)
    
    def to_dict(self):
        """Convert solution to dictionary format"""
        if not self.task_times:
            return {
                'status': 'infeasible',
                'total_time': 0,
                'num_chiplets': 0,
                'task_assignments': {},
                'task_times': {},
                'task_durations': {},
                'pe_assignments': {}
            }
            
        total_time = self.get_total_time()
        
        num_chiplets = len(set(self.task_assignments.values()))
        
        # Filter out -1 pseudo PEs from pe_assignments
        filtered_pe_assignments = {pe: chiplet for pe, chiplet in self.pe_assignments.items() if pe != -1}
        
        return {
            'status': 'feasible',
            'total_time': total_time,
            'num_chiplets': num_chiplets,
            'task_assignments': dict(self.task_assignments),
            'task_times': dict(self.task_times),
            'task_durations': dict(self.task_durations),
            'pe_assignments': filtered_pe_assignments
        }

# ================== SIMULATED ANNEALING SOLVER ==================

class SimulatedAnnealingSolver:
    """Simulated Annealing solver for the chiplet assignment problem"""
    
    def __init__(self, problem, constraints: List):
        self.problem = problem
        self.constraints = constraints
        
    def solve(self, max_chiplets: int = 10, 
              initial_temp: float = 1000.0,
              final_temp: float = 0.1,
              cooling_rate: float = 0.95,
              max_iterations: int = 10000,
              max_no_improvement: int = 1000,
              timeout: float = 300.0,
              save_solution_file: bool = True) -> Dict[str, Any]:
        """
        Solve using simulated annealing
        
        Args:
            max_chiplets: Maximum number of chiplets to use
            initial_temp: Initial temperature
            final_temp: Final temperature  
            cooling_rate: Temperature cooling rate
            max_iterations: Maximum iterations
            max_no_improvement: Stop if no improvement for this many iterations
            timeout: Timeout in seconds
            save_solution_file: Whether to save solution to file
        """
        
        print("=== SIMULATED ANNEALING SOLVER ===")
        print(f"Problem: {len(self.problem.tasks)} tasks, {len(self.problem.pes)} PEs")
        print(f"Parameters: temp {initial_temp}→{final_temp}, max_iter {max_iterations}, timeout {timeout}s")
        
        start_time = time_module.time()
        
        # Initialize random solution
        current_solution = SimulatedAnnealingSolution(self.problem, max_chiplets)
        current_solution.random_initialize()
        current_cost = current_solution.evaluate_cost()
        
        # Track best solution
        best_solution = copy.deepcopy(current_solution)
        best_cost = current_cost
        
        # SA parameters
        temperature = initial_temp
        iteration = 0
        no_improvement_count = 0
        
        print(f"Initial solution: cost={current_cost}")
        
        while (temperature > final_temp and 
               iteration < max_iterations and
               no_improvement_count < max_no_improvement and
               time_module.time() - start_time < timeout):
            
            # Performance timing
            iter_start = time_module.time()
            
            # Generate neighbors
            neighbor_start = time_module.time()
            neighbors = current_solution.get_neighbors()
            neighbor_time = time_module.time() - neighbor_start
            
            if not neighbors:
                break
                
            # Select a random neighbor
            neighbor = random.choice(neighbors)
            eval_start = time_module.time()
            neighbor_cost = neighbor.evaluate_cost()
            eval_time = time_module.time() - eval_start
            
            # Acceptance criteria
            if neighbor_cost < current_cost:
                # Always accept better solutions
                current_solution = neighbor
                current_cost = neighbor_cost
                no_improvement_count = 0
                
                # Update best solution
                if current_cost < best_cost:
                    best_solution = copy.deepcopy(current_solution)
                    best_cost = current_cost
                    
                    # Detailed improvement reporting
                    violations = best_solution.count_violations()
                    total_violations = sum(violations.values())
                    # Calculate actual max_time from task schedule using helper method
                    actual_max_time = best_solution.get_total_time()
                    chiplets_used = len(set(best_solution.task_assignments.values()))
                    elapsed = time_module.time() - start_time
                    
                    print(f"🔥 NEW BEST: iter={iteration}, cost={best_cost}, cycles={actual_max_time}, "
                          f"chiplets={chiplets_used}, violations={total_violations}, temp={temperature:.2f}, elapsed={elapsed:.1f}s")
                    
            else:
                # Accept worse solutions with probability
                delta = neighbor_cost - current_cost
                probability = math.exp(-delta / temperature)
                
                if random.random() < probability:
                    current_solution = neighbor
                    current_cost = neighbor_cost
                    no_improvement_count += 1
                else:
                    no_improvement_count += 1
            
            # Cool down
            temperature *= cooling_rate
            iteration += 1
            
            # Progress reporting - more frequent for large problems
            if iteration % 100 == 0:
                elapsed = time_module.time() - start_time
                iter_total = time_module.time() - iter_start
                violations = best_solution.count_violations()
                total_violations = sum(violations.values())
                # Calculate actual max_time from task schedule using helper method
                actual_max_time = best_solution.get_total_time()
                chiplets_used = len(set(best_solution.task_assignments.values()))
                
                print(f"Iter {iteration}: best_cost={best_cost}, temp={temperature:.2f}, "
                      f"cycles={actual_max_time}, chiplets={chiplets_used}, violations={total_violations}, elapsed={elapsed:.1f}s, "
                      f"iter_time={iter_total:.3f}s (neighbors={neighbor_time:.3f}s, eval={eval_time:.3f}s)")
        
        solve_time = time_module.time() - start_time
        
        print(f"SA completed: {iteration} iterations, {solve_time:.2f}s")
        print(f"Best cost: {best_cost}")
        
        # Convert to solution format
        solution_dict = best_solution.to_dict()
        solution_dict['solve_time'] = solve_time
        solution_dict['iterations'] = iteration
        solution_dict['algorithm'] = 'simulated_annealing'
        
        # Count violations for reporting
        violations = best_solution.count_violations()
        total_violations = sum(violations.values())
        
        if total_violations == 0:
            solution_dict['status'] = 'optimal'
        else:
            solution_dict['status'] = 'feasible_with_violations'
            solution_dict['violations'] = violations
            print(f"Solution has {total_violations} constraint violations: {violations}")
        
        # Generate solution file
        if save_solution_file and solution_dict['status'] in ['optimal', 'feasible_with_violations']:
            solution_file = generate_solution_file(
                solution_dict, 'SA', self.problem.traffic_file, self.problem.task_data
            )
            if solution_file:
                solution_dict['solution_file'] = solution_file
                print(f"Solution saved to: {solution_file}")
        
        return solution_dict

# ================== CHIPLET PROBLEM CLASS FOR SA ==================

class ChipletProblemSA:
    """Chiplet assignment problem setup for Simulated Annealing"""
    
    def __init__(self, traffic_file: str):
        """
        Initialize problem from traffic data file
        
        Args:
            traffic_file: Path to traffic data file
        """
        self.traffic_file = traffic_file
        self.constraints = []
        
        # Load and parse traffic data
        self.task_data, self.tasks, self.pes, self.dependencies = self._load_traffic_data(traffic_file)
        
    def _load_traffic_data(self, filename: str):
        """Load traffic data from file"""
        print(f"Loading traffic data from {filename}")
        
        # Read traffic data
        df = pd.read_csv(filename, sep='\t', comment='#', 
                        names=['task_id', 'source_pe', 'dest_pe', 'data_size', 'wait_ids'])
        
        task_data = {}
        tasks = set()
        pes = set()
        dependencies = defaultdict(list)
        
        for _, row in df.iterrows():
            task_id = row['task_id']
            source_pe = row['source_pe']
            dest_pe = row['dest_pe']
            data_size = row['data_size']
            wait_ids = row['wait_ids']
            
            task_data[task_id] = {
                'source_pe': source_pe,
                'dest_pe': dest_pe,
                'data_size': data_size
            }
            
            tasks.add(task_id)
            pes.add(source_pe)
            pes.add(dest_pe)
            
            # Parse dependencies
            if pd.notna(wait_ids) and str(wait_ids).strip() != 'None':
                deps = [int(float(x.strip())) for x in str(wait_ids).split(',')]
                dependencies[task_id] = deps
            else:
                dependencies[task_id] = []
        
        print(f"Loaded: {len(tasks)} tasks, {len(pes)} PEs, {sum(len(deps) for deps in dependencies.values())} dependencies")
        
        return task_data, tasks, pes, dict(dependencies)
    
    def add_constraint(self, constraint):
        """Add a constraint to the problem"""
        self.constraints.append(constraint)
        
    def solve(self, timeout: float = 300, max_chiplets: int = 10, 
              save_solution_file: bool = True, **sa_params) -> Dict[str, Any]:
        """
        Solve the problem using Simulated Annealing
        
        Args:
            timeout: Timeout in seconds
            max_chiplets: Maximum number of chiplets
            save_solution_file: Whether to save solution file
            **sa_params: Additional SA parameters
            
        Returns:
            Solution dictionary
        """
        solver = SimulatedAnnealingSolver(self, self.constraints)
        
        # Set SA parameters
        sa_config = {
            'max_chiplets': max_chiplets,
            'timeout': timeout,
            'save_solution_file': save_solution_file,
            'initial_temp': sa_params.get('initial_temp', 1000.0),
            'final_temp': sa_params.get('final_temp', 0.1),
            'cooling_rate': sa_params.get('cooling_rate', 0.95),
            'max_iterations': sa_params.get('max_iterations', 10000),
            'max_no_improvement': sa_params.get('max_no_improvement', 1000)
        }
        
        return solver.solve(**sa_config)

# ================== EXAMPLE USAGE ==================

if __name__ == "__main__":
    # Create problem
    problem = ChipletProblemSA('gpt2_transformer_small.txt')
    
    # Add all constraints
    problem.add_constraint(TaskAssignmentConstraint())
    problem.add_constraint(ChipletUsageConstraint())
    problem.add_constraint(TaskDependencyConstraint())
    problem.add_constraint(ChipletCapacityConstraint(max_pes=MAX_PES_PER_CHIPLET))
    problem.add_constraint(PEExclusivityConstraint())
    problem.add_constraint(InterChipletCommConstraint(bandwidth=INTER_CHIPLET_BANDWIDTH))
    problem.add_constraint(TimeBoundsConstraint())
    problem.add_constraint(NoMulticastingConstraint())
    
    print(f"Problem setup: {len(problem.tasks)} tasks, {len(problem.pes)} PEs")
    print(f"Constraints: {len(problem.constraints)}")
    
    # Solve with SA
    solution = problem.solve(
        timeout=60,
        max_chiplets=6,
        save_solution_file=True,
        initial_temp=1000.0,
        max_iterations=5000
    )
    
    print(f"\nSA Solution: {solution['status']}")
    if solution['status'] != 'infeasible':
        print(f"Total time: {solution['total_time']} cycles")
        print(f"Chiplets used: {solution['num_chiplets']}")
        print(f"Tasks assigned: {len(solution.get('task_assignments', {}))}")
        
        if 'violations' in solution:
            print(f"Constraint violations: {solution['violations']}")