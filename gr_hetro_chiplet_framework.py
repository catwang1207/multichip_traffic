#!/usr/bin/env python3

import pandas as pd
import time as time_module
import json
import os
import random
import math
import copy
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

# Utilities (generate_solution_file, etc.)
from config import MAX_PES_PER_CHIPLET, INTER_CHIPLET_BANDWIDTH, COST_WEIGHTS
from chiplet_utils import generate_solution_file


# ================== HEURISTIC SOLUTION (used by GRASP) ==================

class HeuristicSolution:
    """
    Solution holder + schedule/cost/violations + constructive & local moves
    """

    def __init__(self, problem, max_chiplets: int, rng: Optional[random.Random] = None, 
                 pe_type_strategy: str = 'mixed', imc_ratio: float = 0.75):
        self.problem = problem
        self.max_chiplets = max_chiplets
        self.rng = rng or random.Random()
        self.pe_type_strategy = pe_type_strategy  # 'separated', 'mixed', or 'ignore'
        self.imc_ratio = imc_ratio  # For mixed strategy: ratio of IMC PEs per chiplet

        # Deterministic traversal order of ids (data itself is fixed)
        self.tasks = sorted(problem.tasks)
        self.pes = sorted(problem.pes)
        
        # PE type information
        self.pe_types = getattr(problem, 'pe_types', {})

        # Representation
        self.task_assignments: Dict[int, int] = {}  # task -> chiplet_id (for reporting)
        self.task_times: Dict[int, int] = {}        # task -> start_time
        self.task_durations: Dict[int, int] = {}    # task -> duration (>=1 cycles)
        self.pe_assignments: Dict[int, int] = {}    # pe -> chiplet_id

        # Slots: chiplet -> list of PEs length MAX_PES_PER_CHIPLET (-1 = empty)
        self.pe_slots: Dict[int, List[int]] = {c: [-1] * MAX_PES_PER_CHIPLET for c in range(self.max_chiplets)}
        
        # Chiplet type designations (for separated strategy)
        self.chiplet_types: Dict[int, str] = {}  # chiplet -> 'IMC' or 'DIG'

        # Caches
        self._cached_cost: Optional[float] = None
        self._cached_violations: Optional[Dict[str, int]] = None
        self._dirty = True

        # Communication maps
        self.pe_communication: Optional[Dict[Tuple[int, int], int]] = None
        self.out_edges: Dict[int, List[Tuple[int, int]]] = defaultdict(list)

    # -------------------- GRASP constructive phase --------------------

    def grasp_construct(self, rcl_size: int = 6):
        """
        Build an initial solution using greedy randomized clustering with an RCL.
        Then schedule & do a fast greedy refine.
        """
        # Build comm maps (deterministic from data)
        pe_communication: Dict[Tuple[int, int], int] = {}
        out_edges: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        for task in self.tasks:
            src = self.problem.task_data[task]['source_pe']
            dst = self.problem.task_data[task]['dest_pe']
            data = int(self.problem.task_data[task]['data_size'])
            pair = tuple(sorted((src, dst)))
            pe_communication[pair] = pe_communication.get(pair, 0) + 1
            out_edges[src].append((dst, data))
        self.pe_communication = pe_communication
        self.out_edges = out_edges

        min_chiplets_capacity = (len(self.pes) + MAX_PES_PER_CHIPLET - 1) // MAX_PES_PER_CHIPLET
        print(f"Chiplet-minimizing constructive build for {len(self.pes)} PEs")
        print(f"Minimum chiplets by capacity: {min_chiplets_capacity}")
        print(f"Found {len(pe_communication)} PE communication pairs")

        best_slots = None
        best_adj_cost = float('inf')
        best_target = None

        # Try increasing chiplet counts; stop when adjusted cost worsens
        for target_chiplets in range(min_chiplets_capacity, self.max_chiplets + 1):
            print(f"Trying {target_chiplets} chiplets (GRASP build)...")
            clusters = self._create_comm_clusters_grasp(pe_communication, target_chiplets, rcl_size)
            if not clusters:
                continue
            test_slots = self._assign_clusters_to_slots(clusters, target_chiplets)
            test_cost = self._evaluate_chiplet_configuration(test_slots)

            # EXPONENTIAL penalty for > capacity-min chiplets
            extra = max(0, target_chiplets - min_chiplets_capacity)
            chiplet_penalty = 0 if extra == 0 else COST_WEIGHTS['exponential_penalty_base'] * (2 ** extra - 1)
            adjusted_cost = test_cost + chiplet_penalty

            print(f"  {target_chiplets} chiplets → base_cost={test_cost}, adjusted={adjusted_cost}")

            if adjusted_cost < best_adj_cost:
                best_adj_cost = adjusted_cost
                best_slots = test_slots
                best_target = target_chiplets
                print(f"  ✅ New best at {target_chiplets} chiplets (adjusted={adjusted_cost})")
            else:
                print("  ❌ Worse after penalty; stop expanding.")
                break

        # Apply best
        if best_slots is None:
            # Fallback minimal pack
            clusters = self._create_comm_clusters_grasp(pe_communication, min_chiplets_capacity, rcl_size)
            best_slots = self._assign_clusters_to_slots(clusters, min_chiplets_capacity)
            best_target = min_chiplets_capacity

        self.pe_slots = best_slots
        print(f"OPTIMAL (constructive): {best_target} chiplets selected")

        # Map & schedule
        self._update_pe_assignments_from_slots()
        self._update_task_assignments_from_pe_slots()
        self._create_feasible_schedule()

        # Quick greedy refinement right after construction
        self._post_init_refine(limit_sources=min(30, len(self.pes)))

        self._dirty = True
        self._cached_cost = None
        self._cached_violations = None

    def _create_comm_clusters_grasp(self, pe_communication, target_chiplets, rcl_size):
        """
        Communication-aware cluster build with randomized selection from an RCL.
        Now supports PE type-aware clustering strategies.
        """
        if self.pe_type_strategy == 'separated':
            return self._create_type_separated_clusters(pe_communication, target_chiplets, rcl_size)
        elif self.pe_type_strategy == 'mixed':
            return self._create_mixed_type_clusters(pe_communication, target_chiplets, rcl_size)
        else:
            # Original algorithm (ignore types)
            return self._create_original_clusters(pe_communication, target_chiplets, rcl_size)
    
    def _create_type_separated_clusters(self, pe_communication, target_chiplets, rcl_size):
        """
        Strategy 1: Create pure IMC and DIG chiplets (no mixing).
        """
        # Group PEs by type
        imc_pes = [pe for pe in self.pes if self.pe_types.get(pe) == 'IMC']
        dig_pes = [pe for pe in self.pes if self.pe_types.get(pe) == 'DIG']
        other_pes = [pe for pe in self.pes if self.pe_types.get(pe) not in ['IMC', 'DIG']]
        
        print(f"Type separation: {len(imc_pes)} IMC, {len(dig_pes)} DIG, {len(other_pes)} other PEs")
        
        clusters: List[set] = []
        
        # Determine chiplet allocation
        imc_chiplets_needed = (len(imc_pes) + MAX_PES_PER_CHIPLET - 1) // MAX_PES_PER_CHIPLET
        dig_chiplets_needed = (len(dig_pes) + MAX_PES_PER_CHIPLET - 1) // MAX_PES_PER_CHIPLET
        total_needed = imc_chiplets_needed + dig_chiplets_needed
        
        if total_needed > target_chiplets:
            print(f"Pure separation requires {total_needed} chiplets (IMC: {imc_chiplets_needed}, DIG: {dig_chiplets_needed})")
            print(f"Target was {target_chiplets} chiplets - using {total_needed} chiplets for pure separation")
            # Force the required number of chiplets for pure separation
            target_chiplets = total_needed
        
        # Create IMC clusters
        imc_clusters = self._cluster_pes_by_communication(imc_pes, pe_communication, imc_chiplets_needed, rcl_size)
        for i, cluster in enumerate(imc_clusters):
            clusters.append(cluster)
            self.chiplet_types[len(clusters) - 1] = 'IMC'
            
        # Create DIG clusters  
        dig_clusters = self._cluster_pes_by_communication(dig_pes, pe_communication, dig_chiplets_needed, rcl_size)
        for i, cluster in enumerate(dig_clusters):
            clusters.append(cluster)
            self.chiplet_types[len(clusters) - 1] = 'DIG'
            
        # Handle other PEs (add to smallest compatible cluster or create new)
        for pe in other_pes:
            if clusters:
                best_cluster = min(clusters, key=lambda c: len(c))
                best_cluster.add(pe)
            else:
                clusters.append({pe})
                
        return clusters
        
    def _create_mixed_type_clusters(self, pe_communication, target_chiplets, rcl_size):
        """
        Strategy 2: Create mixed chiplets with specified IMC/DIG ratios.
        """
        imc_pes = [pe for pe in self.pes if self.pe_types.get(pe) == 'IMC']
        dig_pes = [pe for pe in self.pes if self.pe_types.get(pe) == 'DIG']
        other_pes = [pe for pe in self.pes if self.pe_types.get(pe) not in ['IMC', 'DIG']]
        
        clusters: List[set] = []
        imc_per_chiplet = int(MAX_PES_PER_CHIPLET * self.imc_ratio)
        dig_per_chiplet = MAX_PES_PER_CHIPLET - imc_per_chiplet
        
        print(f"Mixed strategy: {imc_per_chiplet} IMC + {dig_per_chiplet} DIG per chiplet")
        
        imc_idx = 0
        dig_idx = 0
        
        for chiplet_idx in range(target_chiplets):
            cluster = set()
            
            # Add IMC PEs
            for _ in range(min(imc_per_chiplet, len(imc_pes) - imc_idx)):
                if imc_idx < len(imc_pes):
                    cluster.add(imc_pes[imc_idx])
                    imc_idx += 1
                    
            # Add DIG PEs  
            for _ in range(min(dig_per_chiplet, len(dig_pes) - dig_idx)):
                if dig_idx < len(dig_pes):
                    cluster.add(dig_pes[dig_idx])
                    dig_idx += 1
                    
            if cluster:
                clusters.append(cluster)
                self.chiplet_types[chiplet_idx] = 'MIXED'
                
        # Handle remaining PEs
        remaining_imc = imc_pes[imc_idx:]
        remaining_dig = dig_pes[dig_idx:]
        all_remaining = remaining_imc + remaining_dig + other_pes
        
        for pe in all_remaining:
            if clusters:
                best_cluster = min(clusters, key=lambda c: len(c))
                best_cluster.add(pe)
            else:
                clusters.append({pe})
                
        return clusters
        
    def _cluster_pes_by_communication(self, pe_list, pe_communication, num_clusters, rcl_size):
        """
        Helper to cluster a specific set of PEs based on communication patterns.
        """
        if not pe_list:
            return []
            
        clusters: List[set] = []
        used = set()
        
        # Find communication pairs within this PE set
        internal_pairs = []
        for (a, b), count in pe_communication.items():
            if a in pe_list and b in pe_list:
                internal_pairs.append(((a, b), count))
        internal_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Seed clusters with high-communication pairs
        pair_idx = 0
        while len(clusters) < num_clusters and pair_idx < len(internal_pairs):
            (a, b), _ = internal_pairs[pair_idx]
            if a not in used and b not in used:
                cluster = {a, b}
                used.add(a)
                used.add(b)
                clusters.append(cluster)
            pair_idx += 1
            
        # Create singleton clusters for remaining slots
        remaining = [pe for pe in pe_list if pe not in used]
        while len(clusters) < num_clusters and remaining:
            pe = remaining.pop(0)
            clusters.append({pe})
            used.add(pe)
            
        # Expand clusters
        for cluster in clusters:
            max_size = min(MAX_PES_PER_CHIPLET, len(pe_list) // max(1, num_clusters) + 3)
            while len(cluster) < max_size:
                # Find best candidate by communication with cluster
                best_pe = None
                best_score = 0
                for pe in pe_list:
                    if pe in used:
                        continue
                    score = 0
                    for cluster_pe in cluster:
                        key = tuple(sorted((pe, cluster_pe)))
                        score += pe_communication.get(key, 0)
                    if score > best_score:
                        best_score = score
                        best_pe = pe
                        
                if best_pe is None:
                    break
                    
                cluster.add(best_pe)
                used.add(best_pe)
                
        # Distribute remaining PEs
        remaining = [pe for pe in pe_list if pe not in used]
        for pe in remaining:
            if clusters:
                smallest = min(clusters, key=lambda c: len(c))
                smallest.add(pe)
                
        return clusters
        
    def _create_original_clusters(self, pe_communication, target_chiplets, rcl_size):
        """
        Original clustering algorithm (type-agnostic).
        """
        clusters: List[set] = []
        used = set()

        # Candidate seeds: PE pairs sorted by comm count (desc)
        pairs_sorted = sorted(pe_communication.items(), key=lambda x: x[1], reverse=True)
        pair_idx = 0

        # Seed up to target_chiplets clusters
        while len(clusters) < target_chiplets and pair_idx < len(pairs_sorted):
            # Build an RCL of high-comm pairs that don't overlap used PEs
            rcl_pairs = []
            scan = pair_idx
            while len(rcl_pairs) < rcl_size and scan < len(pairs_sorted):
                (a, b), cnt = pairs_sorted[scan]
                if a not in used and b not in used:
                    rcl_pairs.append((a, b, cnt))
                scan += 1
            if not rcl_pairs:
                break
            a, b, _ = self.rng.choice(rcl_pairs)
            cluster = {a, b}
            used.add(a); used.add(b)
            clusters.append(cluster)
            pair_idx = scan  # advance

        # If we still need clusters, seed singletons
        all_remaining = [pe for pe in self.pes if pe not in used]
        while len(clusters) < target_chiplets and all_remaining:
            pe = all_remaining.pop(0)
            clusters.append({pe})
            used.add(pe)

        # Expand each cluster greedily with RCL picks by adjacency score
        for cl in clusters:
            # Keep sizes balanced
            max_size = min(MAX_PES_PER_CHIPLET, len(self.pes) // max(1, target_chiplets) + 5)
            while len(cl) < max_size:
                # Score remaining PEs by comm sum to current cluster
                scores = []
                for pe in self.pes:
                    if pe in used:
                        continue
                    s = 0
                    for q in cl:
                        key = tuple(sorted((pe, q)))
                        s += pe_communication.get(key, 0)
                    if s > 0:
                        scores.append((pe, s))
                if not scores:
                    break
                scores.sort(key=lambda t: t[1], reverse=True)
                # RCL over top candidates
                take = scores[:min(len(scores), rcl_size)]
                choice_pe = self.rng.choice(take)[0]
                cl.add(choice_pe)
                used.add(choice_pe)

        # Distribute any leftover PEs into smallest clusters first
        remain = [pe for pe in self.pes if pe not in used]
        for pe in remain:
            best = min(clusters, key=lambda c: len(c))
            best.add(pe)

        return clusters

    def _assign_clusters_to_slots(self, clusters, target_chiplets):
        pe_slots = {i: [-1] * MAX_PES_PER_CHIPLET for i in range(self.max_chiplets)}
        flat: List[int] = []
        for cl in clusters:
            flat.extend(sorted(list(cl)))
        write_slots = [(c, s) for c in range(target_chiplets) for s in range(MAX_PES_PER_CHIPLET)]
        if len(flat) > len(write_slots):
            print("WARNING: more PEs than available slots; truncating assignment.")
        for i, (c, s) in enumerate(write_slots):
            if i >= len(flat):
                break
            pe_slots[c][s] = flat[i]
        return pe_slots

    def _evaluate_chiplet_configuration(self, test_slots):
        # Save current
        old_pe_slots = copy.deepcopy(self.pe_slots)
        old_pe_assign = copy.deepcopy(self.pe_assignments)
        old_task_assign = copy.deepcopy(self.task_assignments)
        old_task_times = copy.deepcopy(self.task_times)
        old_task_durs = copy.deepcopy(self.task_durations)
        try:
            self.pe_slots = copy.deepcopy(test_slots)
            self._update_pe_assignments_from_slots()
            self._update_task_assignments_from_pe_slots()
            self._create_feasible_schedule()
            self._dirty = True
            self._cached_cost = None
            self._cached_violations = None
            return self.evaluate_cost()
        finally:
            self.pe_slots = old_pe_slots
            self.pe_assignments = old_pe_assign
            self.task_assignments = old_task_assign
            self.task_times = old_task_times
            self.task_durations = old_task_durs
            self._dirty = True
            self._cached_cost = None
            self._cached_violations = None

    # -------------------- Local search (stronger) --------------------

    def local_search(self,
                     max_passes: int = 4,
                     pair_swap_samples: int = 900,
                     time_budget_s: Optional[float] = None,
                     start_time: Optional[float] = None):
        """
        Passes of improvement moves:
          A) co-locate top communication pairs
          B) move hot sources to best chiplets
          C) pack-or-swap best dests into source chiplet
          D) steepest sampled pair-swap (repeat while improving)
        """
        if start_time is None:
            start_time = time_module.time()

        def time_left_ok():
            if time_budget_s is None:
                return True
            return (time_module.time() - start_time) < time_budget_s * 0.98

        base_cost = self.evaluate_cost()
        passes = 0
        improved_any = True

        while improved_any and passes < max_passes and time_left_ok():
            improved_any = False
            passes += 1

            # A) co-locate top communication pairs (bounded)
            if not time_left_ok():
                break
            if self._co_locate_top_pairs(limit_pairs=200, accept_only_improving=True):
                improved_any = True

            # B) move hot sources
            hot_sources = self._top_hot_sources(min(30, len(self.pes)))
            for src in hot_sources:
                if not time_left_ok():
                    break
                before = self.evaluate_cost()
                snap = copy.deepcopy(self.pe_slots)
                if self._try_move_source_to_best_chiplet(src):
                    self._update_pe_assignments_from_slots()
                    self._update_task_assignments_from_pe_slots()
                    self._create_feasible_schedule()
                    after = self.evaluate_cost()
                    if after < before:
                        improved_any = True
                    else:
                        # revert
                        self.pe_slots = snap
                        self._update_pe_assignments_from_slots()
                        self._update_task_assignments_from_pe_slots()
                        self._create_feasible_schedule()

            # C) pack-or-swap best dests into source chiplet
            hot_sources = self._top_hot_sources(min(30, len(self.pes)))
            for src in hot_sources:
                if not time_left_ok():
                    break
                before = self.evaluate_cost()
                snap = copy.deepcopy(self.pe_slots)
                changed = self._pack_or_swap_top_dests_into_source_chiplet(src, k_max=3)
                if changed:
                    self._update_pe_assignments_from_slots()
                    self._update_task_assignments_from_pe_slots()
                    self._create_feasible_schedule()
                    after = self.evaluate_cost()
                    if after < before:
                        improved_any = True
                    else:
                        self.pe_slots = snap
                        self._update_pe_assignments_from_slots()
                        self._update_task_assignments_from_pe_slots()
                        self._create_feasible_schedule()

            # D) steepest sampled pair-swap (hill-climb with sampling)
            if not time_left_ok():
                break
            if self._steepest_pair_swap(pair_swap_samples, time_left_ok):
                improved_any = True

        final_cost = self.evaluate_cost()
        if final_cost < base_cost:
            print(f"Local search improved: {base_cost} → {final_cost}")
        return final_cost

    # ---- A) Co-locate top communication pairs ----
    def _co_locate_top_pairs(self, limit_pairs: int = 200, accept_only_improving: bool = True) -> bool:
        if not self.pe_communication:
            return False
        improved = False
        pairs = sorted(self.pe_communication.items(), key=lambda kv: kv[1], reverse=True)[:limit_pairs]
        for (a, b), _ in pairs:
            a_chip = self.pe_assignments.get(a)
            b_chip = self.pe_assignments.get(b)
            if a_chip is None or b_chip is None or a_chip == b_chip:
                continue

            # try moving a to b's chiplet or b to a's — pick better actual delta
            before = self.evaluate_cost()

            # Option 1: move a -> b_chip
            snap1 = copy.deepcopy(self.pe_slots)
            moved1 = self._try_move_pe_to_chiplet(a, b_chip)
            if moved1:
                self._update_pe_assignments_from_slots()
                self._update_task_assignments_from_pe_slots()
                self._create_feasible_schedule()
                cost1 = self.evaluate_cost()
            else:
                cost1 = float('inf')

            # revert for option 2
            if moved1:
                self.pe_slots = snap1
                self._update_pe_assignments_from_slots()
                self._update_task_assignments_from_pe_slots()
                self._create_feasible_schedule()

            # Option 2: move b -> a_chip
            snap2 = copy.deepcopy(self.pe_slots)
            moved2 = self._try_move_pe_to_chiplet(b, a_chip)
            if moved2:
                self._update_pe_assignments_from_slots()
                self._update_task_assignments_from_pe_slots()
                self._create_feasible_schedule()
                cost2 = self.evaluate_cost()
            else:
                cost2 = float('inf')

            # choose best option if any improves
            best_cost = min(cost1, cost2)
            if best_cost < before:
                improved = True
                # keep the better modification
                if cost1 <= cost2 and moved1:
                    # reapply option 1
                    self.pe_slots = snap1
                    self._try_move_pe_to_chiplet(a, b_chip)
                elif moved2:
                    # snap2 already has the move; keep
                    pass
                # finalize mapping/schedule
                self._update_pe_assignments_from_slots()
                self._update_task_assignments_from_pe_slots()
                self._create_feasible_schedule()
            else:
                # keep original (snap2 to revert if needed)
                if moved2:
                    self.pe_slots = snap2
                    self._update_pe_assignments_from_slots()
                    self._update_task_assignments_from_pe_slots()
                    self._create_feasible_schedule()

            if accept_only_improving and not improved:
                continue
        return improved

    def _try_move_pe_to_chiplet(self, pe: int, chiplet: int) -> bool:
        """Generalized 'move PE to chiplet' using free slot or low-synergy swap."""
        loc = self._find_slot_of_pe(pe)
        if loc is None:
            return False
        cur_chip, cur_slot = loc
        if cur_chip == chiplet:
            return False
        free = self._find_free_slot_in_chiplet(chiplet)
        if free is not None:
            self.pe_slots[chiplet][free] = pe
            self.pe_slots[cur_chip][cur_slot] = -1
            return True
        victim = self._choose_victim_low_synergy(chiplet, avoid=None)
        if victim is None:
            return False
        vic_pe, vic_slot = victim
        self.pe_slots[chiplet][vic_slot], self.pe_slots[cur_chip][cur_slot] = pe, vic_pe
        return True

    # ---- B) & C) helpers (existing + stronger pack-or-swap) ----

    def _pack_or_swap_top_dests_into_source_chiplet(self, src_pe: int, k_max: int = 3) -> bool:
        """
        Like _pack_top_dests... but if no slack, evict lowest-synergy local PE
        when the estimated benefit is higher than the eviction damage.
        """
        src_loc = self._find_slot_of_pe(src_pe)
        if src_loc is None:
            return False
        src_chip, _ = src_loc

        # build candidate dests with benefit
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
        to_take = cand[:k_max]
        for b_gain, dst in to_take:
            slack = sum(1 for pe in self.pe_slots[src_chip] if pe == -1)
            if slack > 0:
                # use slack
                dst_loc = self._find_slot_of_pe(dst)
                if dst_loc is None:
                    continue
                dst_chip, dst_slot = dst_loc
                free = self._find_free_slot_in_chiplet(src_chip)
                if free is None:
                    continue
                self.pe_slots[src_chip][free] = dst
                self.pe_slots[dst_chip][dst_slot] = -1
                moved_any = True
            else:
                # no slack: consider evicting lowest-synergy local
                victim = self._choose_victim_low_synergy(src_chip, avoid=src_pe)
                if not victim:
                    continue
                vic_pe, vic_slot = victim

                # cheap estimate: how bad is removing 'vic_pe' from src_chip?
                # sum of comm synergy with locals
                damage = 0
                local = [pe for pe in self.pe_slots[src_chip] if pe != -1 and pe != vic_pe]
                for other in local:
                    key = tuple(sorted((vic_pe, other)))
                    damage += self.pe_communication.get(key, 0) if self.pe_communication else 0

                # if estimated gain dominates damage, do swap-in
                if b_gain > damage:
                    dst_loc = self._find_slot_of_pe(dst)
                    if dst_loc is None:
                        continue
                    dst_chip, dst_slot = dst_loc
                    # swap dst into src_chip (victim out to dst_chip)
                    self.pe_slots[src_chip][vic_slot], self.pe_slots[dst_chip][dst_slot] = dst, vic_pe
                    moved_any = True

        return moved_any

    # ---- D) Steepest sampled pair-swap (while-improving) ----
    def _steepest_pair_swap(self, samples_per_round: int, time_ok_fn) -> bool:
        """
        Try many sampled cross-chiplet swaps; apply the best improving swap each round.
        Repeat while we keep improving and have time.
        """
        improved_once = False
        while time_ok_fn():
            positions = [(c, s) for c in range(self.max_chiplets) for s in range(MAX_PES_PER_CHIPLET)]
            occupied = [(c, s) for (c, s) in positions if self.pe_slots[c][s] != -1]
            if len(occupied) < 2:
                break

            before_cost = self.evaluate_cost()
            best_delta = 0.0
            best_pair = None

            for _ in range(samples_per_round):
                c1, s1 = self.rng.choice(occupied)
                c2, s2 = self.rng.choice(occupied)
                if (c1 == c2 and s1 == s2) or c1 == c2:
                    continue
                pe1 = self.pe_slots[c1][s1]
                pe2 = self.pe_slots[c2][s2]
                if pe1 == -1 or pe2 == -1:
                    continue

                # apply swap
                self.pe_slots[c1][s1], self.pe_slots[c2][s2] = pe2, pe1
                self._update_pe_assignments_from_slots()
                self._update_task_assignments_from_pe_slots()
                self._create_feasible_schedule()
                after_cost = self.evaluate_cost()

                delta = before_cost - after_cost
                # revert immediately (we only keep best)
                self.pe_slots[c1][s1], self.pe_slots[c2][s2] = pe1, pe2
                self._update_pe_assignments_from_slots()
                self._update_task_assignments_from_pe_slots()
                self._create_feasible_schedule()

                if delta > best_delta:
                    best_delta = delta
                    best_pair = (c1, s1, c2, s2)

            if best_pair and best_delta > 0:
                c1, s1, c2, s2 = best_pair
                pe1 = self.pe_slots[c1][s1]
                pe2 = self.pe_slots[c2][s2]
                self.pe_slots[c1][s1], self.pe_slots[c2][s2] = pe2, pe1
                self._update_pe_assignments_from_slots()
                self._update_task_assignments_from_pe_slots()
                self._create_feasible_schedule()
                improved_once = True
            else:
                break

        return improved_once

    # -------------------- Scheduling, cost, violations --------------------

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

        if (source_chiplet is None or dest_chiplet is None or source_chiplet == dest_chiplet):
            return 1

        inter_chiplet_bandwidth = self._bandwidth_value()
        duration = math.ceil(data_size / inter_chiplet_bandwidth)
        return max(1, duration)

    def _create_feasible_schedule(self):
        """Topological-like serial scheduling with NoMulticasting serialization."""
        dependencies = self.problem.dependencies

        # Pre-calc durations
        self.task_durations = {t: self._calculate_task_duration(t) for t in self.tasks}

        visited = set()
        in_stack = set()
        self.task_times = {}
        last_finish = defaultdict(int)  # serialize sends from same source PE

        def schedule_task(task: int) -> int:
            if task in visited:
                return self.task_times.get(task, 0)
            if task in in_stack:
                raise ValueError(f"Cycle detected in dependencies involving task {task}")
            in_stack.add(task)

            earliest = 0
            for dep_task in dependencies.get(task, []):
                dep_start = schedule_task(dep_task)
                dep_dur = self.task_durations[dep_task]
                earliest = max(earliest, dep_start + dep_dur)

            source_pe = self.problem.task_data[task]['source_pe']
            earliest = max(earliest, last_finish[source_pe])

            self.task_times[task] = earliest
            finish = earliest + self.task_durations[task]
            last_finish[source_pe] = finish

            in_stack.remove(task)
            visited.add(task)
            return self.task_times[task]

        for t in self.tasks:
            schedule_task(t)

    def evaluate_cost(self):
        if not self._dirty and self._cached_cost is not None:
            return self._cached_cost

        max_time = 0
        if self.task_times:
            max_time = max(self.task_times[t] + self.task_durations.get(t, 1)
                           for t in self.tasks if t in self.task_times)

        violations = self.count_violations()
        total_violations = sum(violations.values())

        used_chiplets = len({ch for ch in range(self.max_chiplets)
                             if any(pe != -1 for pe in self.pe_slots.get(ch, []))})

        min_chiplets_capacity = (len(self.pes) + MAX_PES_PER_CHIPLET - 1) // MAX_PES_PER_CHIPLET
        extra_chiplets = max(0, used_chiplets - min_chiplets_capacity)
        if extra_chiplets == 0:
            chiplet_penalty = 500 * used_chiplets
        else:
            exponential_penalty = COST_WEIGHTS['exponential_penalty_base'] * (2 ** extra_chiplets - 1)
            chiplet_penalty = exponential_penalty

        cost = (COST_WEIGHTS['cycle_weight'] * max_time
                + chiplet_penalty
                + COST_WEIGHTS['violation_penalty'] * total_violations)

        self._cached_cost = cost
        self._cached_violations = violations
        self._dirty = False
        return cost

    def count_violations(self):
        violations = {
            'task_assignment': 0,
            'chiplet_capacity': 0,
            'pe_exclusivity': 0,
            'task_dependencies': 0,
            'no_multicasting': 0,
            'inter_chiplet_comm': 0,  # placeholder
            'pe_type_separation': 0  # PE type mixing violations
        }

        # Task assignment
        unassigned_tasks = [t for t in self.tasks if t not in self.task_assignments]
        violations['task_assignment'] = len(unassigned_tasks)

        # Chiplet capacity
        for chiplet in range(self.max_chiplets):
            used_slots = sum(1 for pe in self.pe_slots.get(chiplet, []) if pe != -1)
            if used_slots > MAX_PES_PER_CHIPLET:
                violations['chiplet_capacity'] += used_slots - MAX_PES_PER_CHIPLET

        # PE exclusivity
        pe_chiplet = defaultdict(set)
        for pe, ch in self.pe_assignments.items():
            pe_chiplet[pe].add(ch)
        for pe, chs in pe_chiplet.items():
            if len(chs) > 1:
                violations['pe_exclusivity'] += len(chs) - 1

        # Dependencies
        deps = self.problem.dependencies
        for t in self.tasks:
            if t not in self.task_times:
                continue
            t_start = self.task_times[t]
            for d in deps.get(t, []):
                if d not in self.task_times:
                    continue
                d_end = self.task_times[d] + self.task_durations.get(d, 1)
                if t_start < d_end:
                    violations['task_dependencies'] += 1

        # NoMulticasting
        time_pe_tasks = defaultdict(lambda: defaultdict(list))
        for t in self.tasks:
            if t in self.task_times:
                start = self.task_times[t]
                dur = self.task_durations.get(t, 1)
                src = self.problem.task_data[t]['source_pe']
                dst = self.problem.task_data[t]['dest_pe']
                for cycle in range(start, start + dur):
                    time_pe_tasks[cycle][src].append((t, dst))

        for _, pe_tasks in time_pe_tasks.items():
            for source_pe, tasks_dests in pe_tasks.items():
                if len(tasks_dests) > 1:
                    unique_dests = set(dest for _, dest in tasks_dests)
                    if len(unique_dests) > 1:
                        violations['no_multicasting'] += len(unique_dests) - 1

        # PE Type separation violations (for separated strategy)
        if self.pe_type_strategy == 'separated':
            violations['pe_type_separation'] = self._count_type_mixing_violations()

        return violations

    def _count_type_mixing_violations(self) -> int:
        """
        Count violations when PEs of different types are placed on the same chiplet.
        Only applies when pe_type_strategy == 'separated'.
        """
        if not self.pe_types or self.pe_type_strategy != 'separated':
            return 0
            
        violations = 0
        for chiplet in range(self.max_chiplets):
            # Get PE types on this chiplet
            chiplet_types = set()
            for pe in self.pe_slots.get(chiplet, []):
                if pe != -1 and pe in self.pe_types:
                    chiplet_types.add(self.pe_types[pe])
            
            # If chiplet has more than one type, it's a violation
            if len(chiplet_types) > 1:
                # Count the number of minority type PEs as violations
                type_counts = {}
                for pe in self.pe_slots.get(chiplet, []):
                    if pe != -1 and pe in self.pe_types:
                        pe_type = self.pe_types[pe]
                        type_counts[pe_type] = type_counts.get(pe_type, 0) + 1
                
                if len(type_counts) > 1:
                    # Violation = number of PEs that are not of the majority type
                    max_count = max(type_counts.values())
                    total_pes = sum(type_counts.values())
                    violations += total_pes - max_count
                    
        return violations

    # -------------------- Greedy helpers & mapping updates --------------------

    def _top_hot_sources(self, k: int) -> List[int]:
        totals = defaultdict(int)
        for t in self.tasks:
            src = self.problem.task_data[t]['source_pe']
            totals[src] += self.task_durations.get(t, 1)
        return [pe for pe, _ in sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[:k]]

    def _intra_benefit(self, src_pe: int, dst_pe: int, data_size: int) -> int:
        a = self.pe_assignments.get(src_pe)
        b = self.pe_assignments.get(dst_pe)
        if a is None or b is None or a == b:
            return 0
        bw = self._bandwidth_value()
        inter = math.ceil(data_size / bw)
        return max(0, inter - 1)

    def _benefit_move_source_to_chiplet(self, src_pe: int, chiplet: int) -> int:
        cur = self.pe_assignments.get(src_pe)
        if cur is None or chiplet == cur:
            return 0
        total = 0
        for dst, data in self.out_edges.get(src_pe, []):
            dst_chip = self.pe_assignments.get(dst)
            if dst_chip == chiplet and dst_chip != cur:
                total += self._intra_benefit(src_pe, dst, data)
        return total

    def _find_slot_of_pe(self, pe: int) -> Optional[Tuple[int, int]]:
        for c in range(self.max_chiplets):
            for s, val in enumerate(self.pe_slots[c]):
                if val == pe:
                    return (c, s)
        return None

    def _find_free_slot_in_chiplet(self, chiplet: int) -> Optional[int]:
        for idx, pe in enumerate(self.pe_slots.get(chiplet, [])):
            if pe == -1:
                return idx
        return None

    def _choose_victim_low_synergy(self, chiplet: int, avoid: Optional[int] = None) -> Optional[Tuple[int, int]]:
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
        src_loc = self._find_slot_of_pe(src_pe)
        if src_loc is None:
            return False
        src_chip, src_slot = src_loc

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

        free = self._find_free_slot_in_chiplet(best_chip)
        if free is not None:
            self.pe_slots[best_chip][free] = src_pe
            self.pe_slots[src_chip][src_slot] = -1
        else:
            victim = self._choose_victim_low_synergy(best_chip, avoid=None)
            if victim is None:
                return False
            vic_pe, vic_slot = victim
            self.pe_slots[best_chip][vic_slot], self.pe_slots[src_chip][src_slot] = src_pe, vic_pe
        return True

    def _update_pe_assignments_from_slots(self):
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
        if not self.task_times:
            return 0
        return max(self.task_times[t] + self.task_durations.get(t, 1)
                   for t in self.tasks if t in self.task_times)

    def to_dict(self):
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
        num_chiplets = len({ch for ch in range(self.max_chiplets)
                            if any(pe != -1 for pe in self.pe_slots.get(ch, []))})
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

    # -------------------- Quick greedy polish used after construction --------------------

    def _post_init_refine(self, limit_sources: int = 20):
        hot_sources = self._top_hot_sources(limit_sources)
        improved = False
        for src in hot_sources:
            if self._try_move_source_to_best_chiplet(src):
                improved = True
        for src in hot_sources:
            if self._pack_or_swap_top_dests_into_source_chiplet(src, k_max=3):
                improved = True
        if improved:
            self._update_pe_assignments_from_slots()
            self._update_task_assignments_from_pe_slots()
            self._create_feasible_schedule()
            self._dirty = True


# ================== GRASP SOLVER ==================

class GRASPSolver:
    """
    GRASP + Local Search: repeat (construct via RCL, then local search),
    keep the best until time or starts exhausted.
    """

    def __init__(self, problem, constraints: List):
        self.problem = problem
        self.constraints = constraints
        self.pe_type_strategy = 'ignore'  # Default: ignore types
        self.imc_ratio = 0.75  # For mixed strategy

    def solve(self,
              max_chiplets: int = 10,
              timeout: float = 300.0,
              save_solution_file: bool = True,
              starts: Optional[int] = None,
              seed: Optional[int] = None,
              rcl_size: int = 3,
              ls_max_passes: int = 4,
              pair_swap_samples: int = 900) -> Dict[str, Any]:

        print("=== GRASP + Local Search ===")
        print(f"Problem: {len(self.problem.tasks)} tasks, {len(self.problem.pes)} PEs")
        if starts is not None:
            print(f"Parameters: starts={starts}, rcl_size={rcl_size}, ls_passes={ls_max_passes}, "
                  f"pair_swap_samples={pair_swap_samples}, timeout={timeout}s")
        else:
            print(f"Parameters: starts=auto-until-timeout, rcl_size={rcl_size}, ls_passes={ls_max_passes}, "
                  f"pair_swap_samples={pair_swap_samples}, timeout={timeout}s")

        start_time = time_module.time()
        best_solution: Optional[HeuristicSolution] = None
        best_cost = float('inf')
        starts_done = 0
        base_seed = seed

        def chiplets_used(sol: HeuristicSolution) -> int:
            return len({ch for ch in range(sol.max_chiplets)
                        if any(pe != -1 for pe in sol.pe_slots.get(ch, []))})


        while time_module.time() - start_time < timeout and (starts is None or starts_done < starts):
            run_seed = (base_seed + starts_done) if base_seed is not None else None
            rng = random.Random(run_seed) if run_seed is not None else random.Random()

            # Build & refine
            pe_type_strategy = getattr(self, 'pe_type_strategy', 'ignore')
            imc_ratio = getattr(self, 'imc_ratio', 0.75)
            sol = HeuristicSolution(self.problem, max_chiplets, rng=rng, 
                                   pe_type_strategy=pe_type_strategy, imc_ratio=imc_ratio)
            sol.grasp_construct(rcl_size=rcl_size)

            # Give local search the remaining slice conservatively
            if starts is not None:
                # Spread remaining time over remaining starts
                remaining_starts = max(1, starts - starts_done)
                per_start = max(1.0, (timeout - (time_module.time() - start_time)) / remaining_starts)
            else:
                # If starts=None, still cap each start to a fraction so we actually get multiple starts
                per_start = max(1.0, 0.25 * timeout)  # try 25% each; tune as you like

            # Let local_search use its own local start reference (do NOT pass start_time)
            sol.local_search(max_passes=ls_max_passes,
                            pair_swap_samples=pair_swap_samples,
                            time_budget_s=per_start)

            cost = sol.evaluate_cost()
            if cost < best_cost:
                best_cost = cost
                best_solution = copy.deepcopy(sol)
                violations = best_solution.count_violations()
                total_violations = sum(violations.values())
                cycles = best_solution.get_total_time()
                print(f"🔥 NEW BEST: start={starts_done}, cost={best_cost}, cycles={cycles}, "
                      f"chiplets={chiplets_used(best_solution)}, violations={total_violations}, "
                      f"elapsed={time_module.time() - start_time:.1f}s")

            starts_done += 1

        if best_solution is None:
            # Should not happen, but return infeasible structure
            return {
                'status': 'infeasible',
                'total_time': 0,
                'num_chiplets': 0,
                'task_assignments': {},
                'solve_time': time_module.time() - start_time,
                'iterations': starts_done,
                'algorithm': 'grasp'
            }

        solve_time = time_module.time() - start_time
        solution_dict = best_solution.to_dict()
        solution_dict['solve_time'] = solve_time
        solution_dict['iterations'] = starts_done
        solution_dict['algorithm'] = 'grasp'

        violations = best_solution.count_violations()
        total_violations = sum(violations.values())
        if total_violations == 0:
            solution_dict['status'] = 'optimal'
        else:
            solution_dict['status'] = 'feasible_with_violations'
            solution_dict['violations'] = violations
            print(f"Solution has {total_violations} constraint violations: {violations}")

        if save_solution_file and solution_dict['status'] in ['optimal', 'feasible_with_violations']:
            solution_file = generate_solution_file(
                solution_dict, 'GR', self.problem.traffic_file, self.problem.task_data
            )
            if solution_file:
                solution_dict['solution_file'] = solution_file
                print(f"Solution saved to: {solution_file}")

        return solution_dict


# ================== CHIPLET PROBLEM CLASS (same external API) ==================

class ChipletProblemGRASP:
    """
    Kept the class name for compatibility with your test file.
    Internally uses GRASP instead of GRASP.
    """

    def __init__(self, traffic_file: str):
        self.traffic_file = traffic_file
        self.constraints: List[Any] = []
        self.pe_types: Dict[int, str] = {}  # Will be populated by _load_traffic_data
        self.task_data, self.tasks, self.pes, self.dependencies = self._load_traffic_data(traffic_file)

    def _load_traffic_data(self, filename: str):
        print(f"Loading traffic data from {filename}")
        
        # Try CSV format first (new format with type columns)
        try:
            df = pd.read_csv(filename)
            # Check for both possible column name formats
            required_cols_v1 = ['task_id', 'src_pe', 'dest_pe', 'bytes', 'wait_ids', 'src_type', 'dst_type']
            required_cols_v2 = ['task_id', 'src_pe', 'dest_pe', 'bytes', 'wait_ids', 'src_type', 'dst_type']
            
            if all(col in df.columns for col in ['task_id', 'src_pe', 'dest_pe', 'bytes', 'wait_ids', 'src_type', 'dst_type']):
                print("Detected CSV format with PE types (v1 column names)")
                return self._load_csv_with_types(df)
            elif all(col in df.columns for col in ['task_id', 'src_pe', 'dest_pe', 'bytes', 'wait_ids', 'src_type', 'dst_type']):
                print("Detected CSV format with PE types (v2 column names)")
                return self._load_csv_with_types(df)
            else:
                print(f"CSV columns found: {list(df.columns)}")
                print("Required columns not found, trying legacy format...")
        except Exception as e:
            print(f"Failed to load as CSV: {e}")
            
        # Fallback to tab-separated format (legacy)
        try:
            df = pd.read_csv(filename, sep='\t', comment='#',
                             names=['task_id', 'source_pe', 'dest_pe', 'data_size', 'wait_ids'])
            print("Detected tab-separated format (legacy)")
            return self._load_tab_separated(df)
        except Exception as e:
            raise ValueError(f"Could not load traffic data from {filename}: {e}")
            
    def _parse_pe_id(self, pe_str):
        """Parse PE ID from string format like 'P0', 'P-1', etc."""
        pe_str = str(pe_str).strip()
        if pe_str.startswith('P'):
            return int(pe_str[1:])
        else:
            return int(pe_str)
            
    def _load_csv_with_types(self, df):
        """Load CSV format with PE type information"""
        task_data: Dict[int, Dict[str, Any]] = {}
        tasks: set = set()
        pes: set = set()
        pe_types: Dict[int, str] = {}  # PE -> type mapping
        dependencies = defaultdict(list)

        for _, row in df.iterrows():
            task_id = int(row['task_id'])
            source_pe = self._parse_pe_id(row['src_pe'])
            dest_pe = self._parse_pe_id(row['dest_pe'])
            data_size = int(row['bytes'])
            wait_ids = row['wait_ids']
            src_type = str(row['src_type'])
            dst_type = str(row['dst_type'])

            task_data[task_id] = {
                'source_pe': source_pe,
                'dest_pe': dest_pe,
                'data_size': data_size,
                'src_type': src_type,
                'dst_type': dst_type
            }

            tasks.add(task_id)
            pes.add(source_pe)
            pes.add(dest_pe)
            
            # Track PE types (use source type for now, could be refined)
            pe_types[source_pe] = src_type
            pe_types[dest_pe] = dst_type

            if pd.notna(wait_ids) and str(wait_ids).strip() not in ['None', 'none', '']:
                try:
                    deps = [int(float(x.strip())) for x in str(wait_ids).split(',')]
                    dependencies[task_id] = deps
                except ValueError:
                    dependencies[task_id] = []
            else:
                dependencies[task_id] = []

        # Store PE types in the problem instance
        self.pe_types = pe_types
        
        print(f"Loaded: {len(tasks)} tasks, {len(pes)} PEs, {sum(len(deps) for deps in dependencies.values())} dependencies")
        type_counts = {}
        for pe_type in pe_types.values():
            type_counts[pe_type] = type_counts.get(pe_type, 0) + 1
        print(f"PE types: {type_counts}")
        
        return task_data, tasks, pes, dict(dependencies)
        
    def _load_tab_separated(self, df):
        """Load legacy tab-separated format"""
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

            if pd.notna(wait_ids) and str(wait_ids).strip() != 'None':
                deps = [int(float(x.strip())) for x in str(wait_ids).split(',')]
                dependencies[task_id] = deps
            else:
                dependencies[task_id] = []

        # No PE types for legacy format
        self.pe_types = {}
        
        print(f"Loaded: {len(tasks)} tasks, {len(pes)} PEs, {sum(len(deps) for deps in dependencies.values())} dependencies")
        return task_data, tasks, pes, dict(dependencies)

    def add_constraint(self, constraint):
        self.constraints.append(constraint)

    def solve(self,
              timeout: float = 300,
              max_chiplets: int = 10,
              save_solution_file: bool = True,
              pe_type_strategy: str = 'ignore',
              imc_ratio: float = 0.75,
              **params) -> Dict[str, Any]:
        """
        Solve using GRASP + Local Search.

        Accepted params (others are ignored safely):
          - starts: Optional[int]
          - seed: Optional[int]
          - rcl_size: int (default 8)
          - ls_max_passes: int (default 4)
          - pair_swap_samples: int (default 900)
          - pe_type_strategy: str ('ignore', 'separated', 'mixed')
          - imc_ratio: float (ratio of IMC PEs for mixed strategy)
        """
        solver = GRASPSolver(self, self.constraints)
        solver.pe_type_strategy = pe_type_strategy
        solver.imc_ratio = imc_ratio

        grasp_config = {
            'max_chiplets': max_chiplets,
            'timeout': timeout,
            'save_solution_file': save_solution_file,
            'starts': params.get('starts', None),
            'seed': params.get('seed', None),
            'rcl_size': params.get('rcl_size', 1),
            'ls_max_passes': params.get('ls_max_passes', 4),
            'pair_swap_samples': params.get('pair_swap_samples', 900),
        }

        # Ignore GRASP-only params silently
        return solver.solve(**grasp_config)
