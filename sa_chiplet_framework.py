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
        # Deterministic traversal
        self.tasks = sorted(problem.tasks)
        self.pes = sorted(problem.pes)

        # Solution representation
        self.task_assignments: Dict[int, int] = {}  # task -> chiplet_id
        self.task_times: Dict[int, int] = {}        # task -> start_time
        self.task_durations: Dict[int, int] = {}    # task -> duration (>=1 cycles)
        self.pe_assignments: Dict[int, int] = {}    # pe -> chiplet_id

        # Slots: chiplet -> list of PEs length MAX_PES_PER_CHIPLET (-1 = empty)
        self.pe_slots: Dict[int, List[int]] = {c: [-1] * MAX_PES_PER_CHIPLET for c in range(self.max_chiplets)}

        # Cached values for efficiency
        self._cached_cost: Optional[float] = None
        self._cached_violations: Optional[Dict[str, int]] = None
        self._dirty = True

        # Communication map (pair -> frequency). Built at initialization.
        self.pe_communication: Optional[Dict[Tuple[int, int], int]] = None
        # Outgoing edges: src_pe -> list[(dest_pe, data_size)]
        self.out_edges: Dict[int, List[Tuple[int, int]]] = defaultdict(list)

    # -------------------- Initialization --------------------

    def random_initialize(self):
        """Initialize with MINIMUM chiplets, expanding only if cycle improvement justifies cost."""
        print(f"Chiplet-minimizing initialization for {len(self.pes)} PEs")

        # STEP 1: Theoretical minimum chiplets by PE capacity
        min_chiplets_capacity = (len(self.pes) + MAX_PES_PER_CHIPLET - 1) // MAX_PES_PER_CHIPLET

        # STEP 2: Analyze communication patterns & build maps
        pe_communication: Dict[Tuple[int, int], int] = {}
        out_edges: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        for task in self.tasks:
            src = self.problem.task_data[task]['source_pe']
            dst = self.problem.task_data[task]['dest_pe']
            data = int(self.problem.task_data[task]['data_size'])
            pair = tuple(sorted((src, dst)))
            pe_communication[pair] = pe_communication.get(pair, 0) + 1
            out_edges[src].append((dst, data))
        self.pe_communication = pe_communication  # store for guided moves
        self.out_edges = out_edges

        print(f"Minimum chiplets by capacity: {min_chiplets_capacity}")
        print(f"Found {len(pe_communication)} PE communication pairs")

        # STEP 3: Try increasing chiplet counts, starting from minimum
        best_solution = None
        best_cost = float('inf')

        for target_chiplets in range(min_chiplets_capacity, self.max_chiplets + 1):
            print(f"Trying {target_chiplets} chiplets...")

            clusters = self._create_communication_clusters(pe_communication, target_chiplets)
            if not clusters:
                continue

            test_slots = self._assign_clusters_to_slots(clusters, target_chiplets)
            test_cost = self._evaluate_chiplet_configuration(test_slots, target_chiplets)
            print(f"  {target_chiplets} chiplets → cost={test_cost}")

            # EXPONENTIAL penalty for additional chiplets
            extra_chiplets = max(0, target_chiplets - min_chiplets_capacity)
            chiplet_penalty = 0 if extra_chiplets == 0 else COST_WEIGHTS['exponential_penalty_base'] * (2 ** extra_chiplets - 1)
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
            print(f"FALLBACK: Using minimum {min_chiplets_capacity} chiplets")
            clusters = self._create_communication_clusters(pe_communication, min_chiplets_capacity)
            self.pe_slots = self._assign_clusters_to_slots(clusters, min_chiplets_capacity)

        # Create PE -> chiplet map
        self._update_pe_assignments_from_slots()

        # Assign tasks based on PE assignments
        self._update_task_assignments_from_pe_slots()

        # Create a feasible schedule respecting dependencies
        self._create_feasible_schedule()

        # NEW: greedy post-init refinement (fast, targeted)
        self._post_init_refine(limit_sources=min(30, len(self.pes)))

        # Mark caches dirty
        self._dirty = True
        self._cached_cost = None
        self._cached_violations = None

    def _create_communication_clusters(self, pe_communication, target_chiplets):
        """Create communication-aware clusters for target chiplet count."""
        clusters: List[set] = []
        used_pes = set()

        # Sort PE pairs by descending frequency
        sorted_pairs = sorted(pe_communication.items(), key=lambda x: x[1], reverse=True)

        # Seed clusters with highest-communicating pairs
        for (pe1, pe2), _ in sorted_pairs:
            if pe1 in used_pes or pe2 in used_pes or len(clusters) >= target_chiplets:
                continue

            cluster = {pe1, pe2}
            used_pes.add(pe1)
            used_pes.add(pe2)

            # Greedy expansion by communication adjacency
            # Keep sizes balanced around len(PEs)/target + margin
            max_size = min(MAX_PES_PER_CHIPLET, len(self.pes) // max(1, target_chiplets) + 5)
            for other_pe in self.pes:
                if other_pe in used_pes or len(cluster) >= max_size:
                    continue
                comm_count = sum(pe_communication.get(tuple(sorted((other_pe, p))), 0) for p in cluster)
                if comm_count > 0:
                    cluster.add(other_pe)
                    used_pes.add(other_pe)
            clusters.append(cluster)

        # Handle remaining PEs
        remaining_pes = [pe for pe in self.pes if pe not in used_pes]
        for pe in remaining_pes:
            # Fit into the smallest cluster with room; else create new (if under target)
            candidate = None
            for cl in clusters:
                if len(cl) < MAX_PES_PER_CHIPLET:
                    if candidate is None or len(cl) < len(candidate):
                        candidate = cl
            if candidate is not None:
                candidate.add(pe)
            else:
                if len(clusters) < target_chiplets:
                    clusters.append({pe})
                else:
                    # Force into smallest cluster (can overflow here, we’ll fix in slot assignment)
                    min_cluster = min(clusters, key=len)
                    min_cluster.add(pe)

        return clusters

    def _assign_clusters_to_slots(self, clusters, target_chiplets):
        """Assign clusters to chiplet slots without dropping PEs."""
        # Initialize all chiplets with -1 slots; we fill only the first `target_chiplets`
        pe_slots = {i: [-1] * MAX_PES_PER_CHIPLET for i in range(self.max_chiplets)}

        # Flatten PEs by cluster order (priority = earlier cluster)
        all_pes: List[int] = []
        for cl in clusters:
            # ensure deterministic order inside a set cluster
            all_pes.extend(sorted(list(cl)))

        # We must have enough total slots if target_chiplets >= capacity minimum.
        write_slots = [(c, s) for c in range(target_chiplets) for s in range(MAX_PES_PER_CHIPLET)]
        if len(all_pes) > len(write_slots):
            print("WARNING: more PEs than available slots; truncating assignment.")
        for i, (c, s) in enumerate(write_slots):
            if i >= len(all_pes):
                break
            pe_slots[c][s] = all_pes[i]

        return pe_slots

    def _evaluate_chiplet_configuration(self, test_slots, target_chiplets):
        """Evaluate a chiplet configuration without modifying current state."""
        # Save current state
        old_pe_slots = copy.deepcopy(getattr(self, 'pe_slots', None))
        old_pe_assignments = copy.deepcopy(getattr(self, 'pe_assignments', {}))
        old_task_assignments = copy.deepcopy(getattr(self, 'task_assignments', {}))
        old_task_times = copy.deepcopy(getattr(self, 'task_times', {}))
        old_task_durations = copy.deepcopy(getattr(self, 'task_durations', {}))

        try:
            # Apply test config
            self.pe_slots = copy.deepcopy(test_slots)
            self._update_pe_assignments_from_slots()
            self._update_task_assignments_from_pe_slots()
            self._create_feasible_schedule()

            # Force recomputation of cost
            self._dirty = True
            self._cached_cost = None
            self._cached_violations = None

            cost = self.evaluate_cost()
            return cost

        finally:
            # Restore original state and invalidate caches safely
            self.pe_slots = old_pe_slots
            self.pe_assignments = old_pe_assignments
            self.task_assignments = old_task_assignments
            self.task_times = old_task_times
            self.task_durations = old_task_durations
            self._dirty = True
            self._cached_cost = None
            self._cached_violations = None

    # -------------------- Cost & schedule --------------------

    def _bandwidth_value(self) -> int:
        bw = INTER_CHIPLET_BANDWIDTH
        for constraint in self.problem.constraints:
            if hasattr(constraint, 'bandwidth'):
                bw = constraint.bandwidth
                break
        return bw

    def _calculate_task_duration(self, task):
        """Duration = 1 if intra-chiplet; else ceil(data_size / bandwidth)."""
        source_pe = self.problem.task_data[task]['source_pe']
        dest_pe = self.problem.task_data[task]['dest_pe']
        data_size = self.problem.task_data[task]['data_size']

        source_chiplet = self.pe_assignments.get(source_pe)
        dest_chiplet = self.pe_assignments.get(dest_pe)

        # Intra-chiplet tasks always take 1 cycle
        if (source_chiplet is None or dest_chiplet is None or source_chiplet == dest_chiplet):
            return 1

        # Inter-chiplet: based on bandwidth (take the first bandwidth constraint if present)
        inter_chiplet_bandwidth = self._bandwidth_value()
        duration = math.ceil(data_size / inter_chiplet_bandwidth)
        return max(1, duration)

    def _create_feasible_schedule(self):
        """Create a feasible schedule respecting dependencies and serialization."""
        dependencies = self.problem.dependencies

        # Pre-calculate all task durations
        self.task_durations = {}
        for task in self.tasks:
            self.task_durations[task] = self._calculate_task_duration(task)

        visited = set()
        in_stack = set()
        self.task_times = {}
        # Track last finish time per source PE to serialize sends (NoMulticasting)
        last_finish = defaultdict(int)

        def schedule_task(task: int) -> int:
            if task in visited:
                return self.task_times.get(task, 0)
            if task in in_stack:
                raise ValueError(f"Cycle detected in dependencies involving task {task}")
            in_stack.add(task)

            earliest_start = 0
            # Dependencies: successor cannot start before predecessor finishes
            for dep_task in dependencies.get(task, []):
                dep_start = schedule_task(dep_task)
                dep_duration = self.task_durations[dep_task]
                dep_required_end = dep_start + dep_duration
                earliest_start = max(earliest_start, dep_required_end)

            # NoMulticasting: serialize sends from same source PE
            source_pe = self.problem.task_data[task]['source_pe']
            earliest_start = max(earliest_start, last_finish[source_pe])

            self.task_times[task] = earliest_start
            finish = earliest_start + self.task_durations[task]
            last_finish[source_pe] = finish

            in_stack.remove(task)
            visited.add(task)
            return self.task_times[task]

        # Schedule all tasks in deterministic order
        for task in self.tasks:
            schedule_task(task)

    def evaluate_cost(self):
        """Evaluate the cost of the current solution."""
        if not self._dirty and self._cached_cost is not None:
            return self._cached_cost

        max_time = 0
        if self.task_times:
            max_time = max(self.task_times[t] + self.task_durations.get(t, 1)
                           for t in self.tasks if t in self.task_times)

        # Count constraint violations
        violations = self.count_violations()
        total_violations = sum(violations.values())

        # Used/provisioned chiplets = chiplets that currently host any PE
        used_chiplets = len({ch for ch in range(self.max_chiplets)
                             if any(pe != -1 for pe in self.pe_slots.get(ch, []))})

        # Exponential chiplet penalty around the theoretical minimum (capacity-based)
        min_chiplets_capacity = (len(self.pes) + MAX_PES_PER_CHIPLET - 1) // MAX_PES_PER_CHIPLET
        extra_chiplets = max(0, used_chiplets - min_chiplets_capacity)
        if extra_chiplets == 0:
            chiplet_penalty = 500 * used_chiplets  # base cost even at minimum
        else:
            base_cost = 500 * min_chiplets_capacity
            exponential_penalty = COST_WEIGHTS['exponential_penalty_base'] * (2 ** extra_chiplets - 1)
            chiplet_penalty = base_cost + exponential_penalty

        cost = (COST_WEIGHTS['cycle_weight'] * max_time
                + chiplet_penalty
                + COST_WEIGHTS['violation_penalty'] * total_violations)

        self._cached_cost = cost
        self._cached_violations = violations
        self._dirty = False
        return cost

    def count_violations(self):
        """Count violations of each constraint type."""
        violations = {
            'task_assignment': 0,
            'chiplet_capacity': 0,
            'pe_exclusivity': 0,
            'task_dependencies': 0,
            'no_multicasting': 0,
            'inter_chiplet_comm': 0  # placeholder (not computed explicitly)
        }

        # Task assignment violations (should be 0 by construction)
        unassigned_tasks = [t for t in self.tasks if t not in self.task_assignments]
        violations['task_assignment'] = len(unassigned_tasks)

        # Chiplet capacity violations:
        for chiplet in range(self.max_chiplets):
            used_slots = sum(1 for pe in self.pe_slots.get(chiplet, []) if pe != -1)
            if used_slots > MAX_PES_PER_CHIPLET:
                violations['chiplet_capacity'] += used_slots - MAX_PES_PER_CHIPLET

        # PE exclusivity: ensure each PE maps to at most one chiplet
        pe_chiplet_count = defaultdict(set)
        for pe, chiplet in self.pe_assignments.items():
            pe_chiplet_count[pe].add(chiplet)
        for pe, chiplets in pe_chiplet_count.items():
            if len(chiplets) > 1:
                violations['pe_exclusivity'] += len(chiplets) - 1

        # Task dependency violations: task must start after all predecessors finish
        dependencies = self.problem.dependencies
        for task in self.tasks:
            if task not in self.task_times:
                continue
            task_start = self.task_times[task]
            for dep_task in dependencies.get(task, []):
                if dep_task not in self.task_times:
                    continue
                dep_start = self.task_times[dep_task]
                dep_duration = self.task_durations.get(dep_task, 1)
                dep_end = dep_start + dep_duration
                required_start = dep_end
                if task_start < required_start:
                    violations['task_dependencies'] += 1

        # No multicasting violations (should be 0 if schedule serialization works)
        time_pe_tasks = defaultdict(lambda: defaultdict(list))
        for task in self.tasks:
            if task in self.task_times:
                start_time = self.task_times[task]
                duration = self.task_durations.get(task, 1)
                source_pe = self.problem.task_data[task]['source_pe']
                dest_pe = self.problem.task_data[task]['dest_pe']
                for cycle in range(start_time, start_time + duration):
                    time_pe_tasks[cycle][source_pe].append((task, dest_pe))

        for _, pe_tasks in time_pe_tasks.items():
            for source_pe, tasks_dests in pe_tasks.items():
                if len(tasks_dests) > 1:
                    unique_dests = set(dest for _, dest in tasks_dests)
                    if len(unique_dests) > 1:
                        violations['no_multicasting'] += len(unique_dests) - 1

        # Note: inter_chiplet_comm could check per-cycle link loads vs bandwidth (optional).
        return violations

    # -------------------- Greedy post-init refinement --------------------

    def _post_init_refine(self, limit_sources: int = 20):
        """Fast, deterministic improvements: relocate hot sources to best chiplets, then repack a few dests."""
        hot_sources = self._top_hot_sources(limit_sources)
        improved = False
        for src in hot_sources:
            if self._try_move_source_to_best_chiplet(src):
                improved = True
        # small repack for each hot source (uses available slack only)
        for src in hot_sources:
            if self._pack_top_dests_into_source_chiplet(src, k_max=3):
                improved = True
        if improved:
            self._update_pe_assignments_from_slots()
            self._update_task_assignments_from_pe_slots()
            self._create_feasible_schedule()
            self._dirty = True

    # -------------------- Guided neighbor helpers --------------------

    def _find_slot_of_pe(self, pe: int) -> Optional[Tuple[int, int]]:
        for c in range(self.max_chiplets):
            for s, val in enumerate(self.pe_slots[c]):
                if val == pe:
                    return (c, s)
        return None

    def _find_free_slot_in_chiplet(self, chiplet: int) -> Optional[int]:
        slots = self.pe_slots.get(chiplet, [])
        for idx, pe in enumerate(slots):
            if pe == -1:
                return idx
        return None

    def _top_hot_sources(self, k: int) -> List[int]:
        totals = defaultdict(int)
        for t in self.tasks:
            src = self.problem.task_data[t]['source_pe']
            totals[src] += self.task_durations.get(t, 1)
        return [pe for pe, _ in sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[:k]]

    def _intra_benefit(self, src_pe: int, dst_pe: int, data_size: int) -> int:
        """Estimated cycles saved if we co-locate src and dst."""
        a = self.pe_assignments.get(src_pe)
        b = self.pe_assignments.get(dst_pe)
        if a is None or b is None or a == b:
            return 0
        bw = self._bandwidth_value()
        inter = math.ceil(data_size / bw)
        return max(0, inter - 1)

    def _benefit_move_source_to_chiplet(self, src_pe: int, chiplet: int) -> int:
        """Total cycles saved by moving src_pe to `chiplet` (approx)."""
        cur = self.pe_assignments.get(src_pe)
        if cur is None or chiplet == cur:
            return 0
        total = 0
        for dst, data in self.out_edges.get(src_pe, []):
            dst_chip = self.pe_assignments.get(dst)
            if dst_chip == chiplet and dst_chip != cur:
                total += self._intra_benefit(src_pe, dst, data)
        return total

    def _choose_victim_low_synergy(self, chiplet: int, avoid: Optional[int] = None) -> Optional[Tuple[int, int]]:
        """Pick a PE in chiplet to evict with minimal damage (lowest comm with local set)."""
        best_score = None
        best = None
        local = [pe for pe in self.pe_slots[chiplet] if pe != -1 and pe != avoid]
        for s, pe in enumerate(self.pe_slots[chiplet]):
            if pe == -1 or pe == avoid:
                continue
            score = 0
            if self.pe_communication is not None:
                for other in local:
                    if other == pe:
                        continue
                    key = tuple(sorted((pe, other)))
                    score += self.pe_communication.get(key, 0)
            if best_score is None or score < best_score:
                best_score = score
                best = (pe, s)
        return best

    def _try_move_source_to_best_chiplet(self, src_pe: int) -> bool:
        """Move `src_pe` to the chiplet where most of its dests already are (benefit-weighted)."""
        src_loc = self._find_slot_of_pe(src_pe)
        if src_loc is None:
            return False
        src_chip, src_slot = src_loc
        # Evaluate benefits across chiplets
        best_chip = None
        best_gain = 0
        for c in range(self.max_chiplets):
            if c == src_chip:
                continue
            gain = self._benefit_move_source_to_chiplet(src_pe, c)
            if gain > best_gain:
                best_gain, best_chip = gain, c
        if best_chip is None or best_gain <= 0:
            return False
        # Try to move: prefer free slot, else swap with low-synergy victim
        free = self._find_free_slot_in_chiplet(best_chip)
        if free is not None:
            # move src to best_chip: free origin slot
            self.pe_slots[best_chip][free] = src_pe
            self.pe_slots[src_chip][src_slot] = -1
        else:
            victim = self._choose_victim_low_synergy(best_chip, avoid=None)
            if victim is None:
                return False
            vic_pe, vic_slot = victim
            # swap src <-> victim
            self.pe_slots[best_chip][vic_slot], self.pe_slots[src_chip][src_slot] = src_pe, vic_pe
        return True

    def _pack_top_dests_into_source_chiplet(self, src_pe: int, k_max: int = 3) -> bool:
        """Bring up to k_max best-benefit dests into src's chiplet using available slack only."""
        src_loc = self._find_slot_of_pe(src_pe)
        if src_loc is None:
            return False
        src_chip, _ = src_loc
        # Count slack
        slack = sum(1 for pe in self.pe_slots[src_chip] if pe == -1)
        if slack <= 0:
            return False
        # Rank candidate dests by benefit
        cand = []
        for dst, data in self.out_edges.get(src_pe, []):
            dst_chip = self.pe_assignments.get(dst)
            if dst_chip is None or dst_chip == src_chip:
                continue
            b = self._intra_benefit(src_pe, dst, data)
            if b > 0:
                cand.append((b, dst))
        if not cand:
            return False
        cand.sort(reverse=True)
        moved_any = False
        for _, dst in cand[:min(slack, k_max)]:
            dst_loc = self._find_slot_of_pe(dst)
            if dst_loc is None:
                continue
            dst_chip, dst_slot = dst_loc
            # move dst into a free slot on src_chip
            free = self._find_free_slot_in_chiplet(src_chip)
            if free is None:
                break
            self.pe_slots[src_chip][free] = dst
            self.pe_slots[dst_chip][dst_slot] = -1
            moved_any = True
        return moved_any

    # -------------------- Neighbor generator --------------------

    def get_neighbors(self):
        """Generate ONE neighbor using macro-moves; fallback to random swap."""
        neighbor = copy.deepcopy(self)
        r = random.random()
        # 0.45: move hottest source to best chiplet
        if r < 0.45:
            srcs = neighbor._top_hot_sources(1)
            src = srcs[0] if srcs else None
            if src is not None and neighbor._try_move_source_to_best_chiplet(src):
                neighbor._update_pe_assignments_from_slots()
                neighbor._update_task_assignments_from_pe_slots()
                neighbor._create_feasible_schedule()
                neighbor._dirty = True
                neighbor._cached_cost = None
                neighbor._cached_violations = None
                return [neighbor]
        # 0.45 - 0.90: pack a few best dests into source chiplet (uses slack only)
        if r < 0.90:
            srcs = neighbor._top_hot_sources(1)
            src = srcs[0] if srcs else None
            if src is not None and neighbor._pack_top_dests_into_source_chiplet(src, k_max=3):
                neighbor._update_pe_assignments_from_slots()
                neighbor._update_task_assignments_from_pe_slots()
                neighbor._create_feasible_schedule()
                neighbor._dirty = True
                neighbor._cached_cost = None
                neighbor._cached_violations = None
                return [neighbor]
        # Fallback: the original random non–no-op swap
        positions = [(c, s) for c in range(self.max_chiplets) for s in range(MAX_PES_PER_CHIPLET)]
        random.shuffle(positions)
        for i in range(len(positions) - 1):
            c1, s1 = positions[i]
            c2, s2 = positions[i + 1]
            pe1 = neighbor.pe_slots[c1][s1]
            pe2 = neighbor.pe_slots[c2][s2]
            if pe1 == -1 and pe2 == -1:
                continue
            if c1 == c2 and s1 == s2:
                continue
            neighbor.pe_slots[c1][s1], neighbor.pe_slots[c2][s2] = pe2, pe1
            neighbor._update_pe_assignments_from_slots()
            neighbor._update_task_assignments_from_pe_slots()
            neighbor._create_feasible_schedule()
            neighbor._dirty = True
            neighbor._cached_cost = None
            neighbor._cached_violations = None
            return [neighbor]
        return [neighbor]

    # -------------------- Mapping updates --------------------

    def _update_pe_assignments_from_slots(self):
        """Update PE assignments based on current slot configuration."""
        self.pe_assignments = {}
        seen = set()
        for chiplet in range(self.max_chiplets):
            for pe in self.pe_slots.get(chiplet, []):
                if pe != -1:
                    assert pe not in seen, f"PE {pe} placed twice across chiplets"
                    seen.add(pe)
                    self.pe_assignments[pe] = chiplet
        self._dirty = True
        self._cached_cost = None
        self._cached_violations = None

    def _update_task_assignments_from_pe_slots(self):
        """Update task assignments based on PE slot assignments."""
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
        self._dirty = True
        self._cached_cost = None
        self._cached_violations = None

    def get_total_time(self):
        """Calculate total execution time (max finish time across all tasks)."""
        if not self.task_times:
            return 0
        return max(self.task_times[t] + self.task_durations.get(t, 1)
                   for t in self.tasks if t in self.task_times)

    def to_dict(self):
        """Convert solution to dictionary format."""
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

        # Report provisioned chiplets (consistent with penalty)
        num_chiplets = len({ch for ch in range(self.max_chiplets)
                            if any(pe != -1 for pe in self.pe_slots.get(ch, []) )})

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
              save_solution_file: bool = True,
              seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Solve using simulated annealing.

        Args:
            max_chiplets: Maximum number of chiplets to use
            initial_temp: Initial temperature
            final_temp: Final temperature
            cooling_rate: Temperature cooling rate
            max_iterations: Maximum iterations
            max_no_improvement: Stop if no improvement for this many iterations
            timeout: Timeout in seconds
            save_solution_file: Whether to save solution to file
            seed: Optional RNG seed for reproducibility
        """

        print("=== SIMULATED ANNEALING SOLVER ===")
        print(f"Problem: {len(self.problem.tasks)} tasks, {len(self.problem.pes)} PEs")
        print(f"Parameters: temp {initial_temp}→{final_temp}, max_iter {max_iterations}, timeout {timeout}s")

        if seed is not None:
            random.seed(seed)

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
                current_solution = neighbor
                current_cost = neighbor_cost
                no_improvement_count = 0

                if current_cost < best_cost:
                    best_solution = copy.deepcopy(current_solution)
                    best_cost = current_cost

                    violations = best_solution.count_violations()
                    total_violations = sum(violations.values())
                    actual_max_time = best_solution.get_total_time()
                    chiplets_used = len({ch for ch in range(max_chiplets)
                                         if any(pe != -1 for pe in best_solution.pe_slots.get(ch, []))})
                    elapsed = time_module.time() - start_time

                    print(f"🔥 NEW BEST: iter={iteration}, cost={best_cost}, cycles={actual_max_time}, "
                          f"chiplets={chiplets_used}, violations={total_violations}, temp={temperature:.2f}, elapsed={elapsed:.1f}s")

            else:
                # Accept worse solutions with probability
                delta = neighbor_cost - current_cost
                probability = math.exp(-delta / temperature) if temperature > 0 else 0.0

                if random.random() < probability:
                    current_solution = neighbor
                    current_cost = neighbor_cost
                    no_improvement_count += 1
                else:
                    no_improvement_count += 1

            # Cool down
            temperature *= cooling_rate
            iteration += 1

            # Progress reporting
            if iteration % 100 == 0:
                elapsed = time_module.time() - start_time
                iter_total = time_module.time() - iter_start
                violations = best_solution.count_violations()
                total_violations = sum(violations.values())
                actual_max_time = best_solution.get_total_time()
                chiplets_used = len({ch for ch in range(max_chiplets)
                                     if any(pe != -1 for pe in best_solution.pe_slots.get(ch, []))})

                print(f"Iter {iteration}: best_cost={best_cost}, temp={temperature:.2f}, "
                      f"cycles={actual_max_time}, chiplets={chiplets_used}, violations={total_violations}, "
                      f"elapsed={elapsed:.1f}s, iter_time={iter_total:.3f}s (neighbors={neighbor_time:.3f}s, eval={eval_time:.3f}s)")

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
        self.constraints: List[Any] = []

        # Load and parse traffic data
        self.task_data, self.tasks, self.pes, self.dependencies = self._load_traffic_data(traffic_file)

    def _load_traffic_data(self, filename: str):
        """Load traffic data from file."""
        print(f"Loading traffic data from {filename}")

        # Read traffic data
        df = pd.read_csv(filename, sep='\t', comment='#',
                         names=['task_id', 'source_pe', 'dest_pe', 'data_size', 'wait_ids'])

        task_data: Dict[int, Dict[str, Any]] = {}
        tasks: set = set()
        pes: set = set()
        dependencies = defaultdict(list)

        for _, row in df.iterrows():
            task_id = int(row['task_id'])
            source_pe = int(row['source_pe'])
            dest_pe = int(row['dest_pe'])
            data_size = int(row['data_size'])
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
        """Add a constraint to the problem."""
        self.constraints.append(constraint)

    def solve(self, timeout: float = 300, max_chiplets: int = 10,
              save_solution_file: bool = True, **sa_params) -> Dict[str, Any]:
        """
        Solve the problem using Simulated Annealing.

        Args:
            timeout: Timeout in seconds
            max_chiplets: Maximum number of chiplets
            save_solution_file: Whether to save solution file
            **sa_params: Additional SA parameters

        Returns:
            Solution dictionary
        """
        solver = SimulatedAnnealingSolver(self, self.constraints)

        # SA parameters
        sa_config = {
            'max_chiplets': max_chiplets,
            'timeout': timeout,
            'save_solution_file': save_solution_file,
            'initial_temp': sa_params.get('initial_temp', 1000.0),
            'final_temp': sa_params.get('final_temp', 0.1),
            'cooling_rate': sa_params.get('cooling_rate', 0.95),
            'max_iterations': sa_params.get('max_iterations', 10000),
            'max_no_improvement': sa_params.get('max_no_improvement', 1000),
            'seed': sa_params.get('seed', None)
        }

        return solver.solve(**sa_config)

