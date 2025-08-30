#!/usr/bin/env python3

import pandas as pd
import time as time_module
import random
import math
import copy
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, deque

from config import (
    MAX_PES_PER_CHIPLET,
    INTER_CHIPLET_BANDWIDTH,
    COST_WEIGHTS,                # use only cycle_weight, violation_penalty
    DEFAULT_IMC_RATIO,
    DEFAULT_PE_TYPE_STRATEGY,
    get_imc_dig_counts,
)
from chiplet_utils import generate_solution_file


# ---------- Public helper for sweep scripts ----------

def compute_min_required_chiplets(problem,
                                  pe_type_strategy: str = DEFAULT_PE_TYPE_STRATEGY,
                                  imc_ratio: float = DEFAULT_IMC_RATIO) -> int:
    """
    Strategy-aware theoretical minimum chiplets for a given problem instance.
    """
    K = MAX_PES_PER_CHIPLET
    pes = sorted(problem.pes)
    pe_types = getattr(problem, 'pe_types', {})
    imc_count = sum(1 for pe in pes if pe_types.get(pe) == 'IMC')
    dig_count = sum(1 for pe in pes if pe_types.get(pe) == 'DIG')
    other_count = len(pes) - imc_count - dig_count

    if pe_type_strategy == 'separated':
        total = 0
        total += (imc_count + K - 1) // K
        total += (dig_count + K - 1) // K
        total += (other_count + K - 1) // K
        return max(1, total)

    if pe_type_strategy == 'mixed':
        imc_per, dig_per = get_imc_dig_counts(imc_ratio, K)
        imc_per = max(0, min(imc_per, K))
        dig_per = max(0, min(dig_per, K))
        need_imc = (imc_count + max(1, imc_per) - 1) // max(1, imc_per) if imc_count > 0 else 0
        need_dig = (dig_count + max(1, dig_per) - 1) // max(1, dig_per) if dig_count > 0 else 0
        need_other = (other_count + K - 1) // K
        return max(1, need_imc, need_dig, need_other)

    return max(1, (len(pes) + K - 1) // K)


# ================== HEURISTIC SOLUTION (GRASP core) ==================

class HeuristicSolution:
    """
    Solution holder + schedule/cost/violations + constructive & local moves.
    Focused on speed and monotonicity when increasing chiplet budget.
    """

    # Speed knobs (tune if needed)
    NEIGHBOR_CAP = 32            # keep only top-K neighbors per PE in clustering/scoring
    COLOCATE_PAIR_LIMIT = 120    # how many heavy pairs to try co-locating per pass

    def __init__(self, problem, max_chiplets: int, rng: Optional[random.Random] = None,
                 pe_type_strategy: str = DEFAULT_PE_TYPE_STRATEGY, imc_ratio: float = DEFAULT_IMC_RATIO,
                 # sweep controls:
                 force_target_chiplets: Optional[int] = None,
                 exhaustive_construct: bool = False,
                 preferred_extra_type_for_separated: Optional[str] = None,
                 # warm-start (slots layout) to expand from
                 warm_start_slots: Optional[Dict[int, List[int]]] = None):
        self.problem = problem
        self.max_chiplets = max_chiplets
        self.rng = rng or random.Random()
        self.pe_type_strategy = pe_type_strategy
        self.imc_ratio = imc_ratio

        # Sweep controls
        self.force_target_chiplets = force_target_chiplets
        self.exhaustive_construct = exhaustive_construct
        self.preferred_extra_type_for_separated = preferred_extra_type_for_separated

        # When forcing, keep exactly that many used chiplets (don’t allow going below).
        # We still allow local moves as long as we don't drop below T used chiplets.
        self.enforced_min_chiplets = force_target_chiplets

        # Deterministic traversal order
        self.tasks = sorted(problem.tasks)
        self.pes = sorted(problem.pes)

        # PE types
        self.pe_types = getattr(problem, 'pe_types', {})

        # Representation
        self.task_assignments: Dict[int, int] = {}
        self.task_times: Dict[int, int] = {}
        self.task_durations: Dict[int, int] = {}
        self.pe_assignments: Dict[int, int] = {}

        # Slots
        self.pe_slots: Dict[int, List[int]] = {c: [-1] * MAX_PES_PER_CHIPLET for c in range(self.max_chiplets)}

        # Chiplet types
        self.chiplet_types: Dict[int, str] = {}

        # Strategy-aware minimum chiplets
        self.min_chiplets_required = 0

        # Caches
        self._cached_cost: Optional[float] = None
        self._cached_violations: Optional[Dict[str, int]] = None
        self._dirty = True

        # Communication maps
        self.pe_communication: Optional[Dict[Tuple[int, int], int]] = None
        self.out_edges: Dict[int, List[Tuple[int, int]]] = defaultdict(list)

        # Sparse adjacency (for fast scoring)
        self.adj: Optional[Dict[int, Dict[int, int]]] = None
        self.top_neighbors: Optional[Dict[int, List[Tuple[int, int]]]] = None

        # Optional warm-start
        self.warm_start_slots = copy.deepcopy(warm_start_slots) if warm_start_slots else None

    # -------------------- GRASP constructive phase --------------------

    def grasp_construct(self, rcl_size: int = 6):
        """
        Build solution using communication-aware clustering + schedule + quick polish.
        If force_target_chiplets is set, we build for that T (never below).
        """
        # Build comm maps (PE<->PE weighted by total bytes)
        pe_communication: Dict[Tuple[int, int], int] = {}
        out_edges: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        for task in self.tasks:
            src = self.problem.task_data[task]['source_pe']
            dst = self.problem.task_data[task]['dest_pe']
            data = int(self.problem.task_data[task]['data_size'])
            pair = tuple(sorted((src, dst)))
            pe_communication[pair] = pe_communication.get(pair, 0) + data
            out_edges[src].append((dst, data))
        self.pe_communication = pe_communication
        self.out_edges = out_edges

        # Build sparse adjacency and top-K neighbors (speed)
        self._build_sparse_adjacency()

        # Strategy min (just for info/feasibility caps)
        min_by_strategy = self._calculate_min_chiplets_by_strategy()
        self.min_chiplets_required = min_by_strategy
        print(f"Chiplet-constructive build for {len(self.pes)} PEs")
        print(f"Minimum chiplets by strategy '{self.pe_type_strategy}': {min_by_strategy}")
        print(f"Found {len(pe_communication)} PE communication pairs (weighted by bytes)")

        # Forced build: respect available chiplets exactly (>= strategy minimum), not below it
        if self.force_target_chiplets is not None:
            T_req = self.force_target_chiplets
            # Feasibility caps (don’t exceed max_chiplets; don’t go below strategy min; also don’t exceed #PEs)
            T = max(T_req, min_by_strategy)
            T = min(T, self.max_chiplets, len(self.pes))
            print(f"[FORCE] Target chiplets requested T={T_req} → using T={T} after feasibility caps")

            # Warm-start? Reuse and expand (this preserves co-located heavy pairs)
            if self.warm_start_slots:
                slots = copy.deepcopy(self.warm_start_slots)
                slots = self._expand_to_target_chiplets(slots, T)
                used = self._used_chiplets(slots)
                self.pe_slots = slots
                self._update_pe_assignments_from_slots()
                self._update_task_assignments_from_pe_slots()
                self._create_feasible_schedule()
                # Skip post-init refine when forcing chiplets - preserve forced layout for testing
                if self.force_target_chiplets is None:
                    self._post_init_refine(limit_sources=min(30, len(self.pes)))
                print(f"[FORCE][WARM] Expanded warm start → used={used} (target={T})")
                self._print_cross_stats()
                self._dirty = True
                self._cached_cost = None
                self._cached_violations = None
                return

            # Build *for T* (not from minimum)
            clusters, required = self._create_comm_clusters_grasp(self.pe_communication, T, rcl_size)
            target = max(T, required)  # never below required, but cap to T via expansion
            test_slots = self._assign_clusters_to_slots(clusters, target)
            test_slots = self._expand_to_target_chiplets(test_slots, T)  # hard-expand to exactly T if feasible
            used = self._used_chiplets(test_slots)
            print(f"[FORCE] Built with {used} chiplets (target={T})")

            self.pe_slots = test_slots
            self._update_pe_assignments_from_slots()
            self._update_task_assignments_from_pe_slots()
            self._create_feasible_schedule()
            # Skip post-init refine when forcing chiplets - it undoes our expansion
            if self.force_target_chiplets is None:
                self._post_init_refine(limit_sources=min(30, len(self.pes)))
            self._print_cross_stats()
            self._dirty = True
            self._cached_cost = None
            self._cached_violations = None
            return

        # Legacy exploratory sweep (if you ever call without force)
        best_slots = None
        best_cost = float('inf')
        best_target = None
        tried_any = False

        for target_chiplets in range(min_by_strategy, self.max_chiplets + 1):
            tried_any = True
            print(f"Trying {target_chiplets} chiplets (GRASP build)...")
            clusters, required = self._create_comm_clusters_grasp(self.pe_communication, target_chiplets, rcl_size)
            if not clusters:
                continue
            target = max(target_chiplets, required)
            test_slots = self._assign_clusters_to_slots(clusters, target)
            test_slots = self._expand_to_target_chiplets(test_slots, target)
            test_cost = self._evaluate_chiplet_configuration(test_slots)
            print(f"  {target} chiplets → cost={test_cost}")
            if test_cost < best_cost:
                best_cost = test_cost
                best_slots = test_slots
                best_target = target
                print(f"  ✅ New best at {target} chiplets (cost={test_cost})")
            elif not self.exhaustive_construct:
                print("  ❌ Worse; stop expanding.")
                break

        if best_slots is None and tried_any:
            clusters, required = self._create_comm_clusters_grasp(self.pe_communication, min_by_strategy, rcl_size)
            target = max(min_by_strategy, required)
            best_slots = self._assign_clusters_to_slots(clusters, target)
            best_target = target

        self.pe_slots = best_slots
        print(f"OPTIMAL (constructive): {best_target} chiplets selected")
        self._update_pe_assignments_from_slots()
        self._update_task_assignments_from_pe_slots()
        self._create_feasible_schedule()
        self._post_init_refine(limit_sources=min(30, len(self.pes)))
        self._print_cross_stats()
        self._dirty = True
        self._cached_cost = None
        self._cached_violations = None

    # -------------------- Helpers for construction --------------------

    def _build_sparse_adjacency(self):
        """Create adjacency dict and keep only top-K neighbors per PE to speed up scoring."""
        adj = defaultdict(dict)
        for (a, b), w in self.pe_communication.items():
            adj[a][b] = w
            adj[b][a] = w
        self.adj = adj

        K = self.NEIGHBOR_CAP
        top_neighbors = {}
        for pe, neighs in adj.items():
            # Keep top-K by weight
            if len(neighs) <= K:
                top_neighbors[pe] = sorted(neighs.items(), key=lambda kv: kv[1], reverse=True)
            else:
                # partial select faster than sort-all
                # but Python doesn't have nth_element; sort is acceptable here
                top_neighbors[pe] = sorted(neighs.items(), key=lambda kv: kv[1], reverse=True)[:K]
        self.top_neighbors = top_neighbors

    def _score_into_cluster(self, pe: int, cluster: set) -> int:
        if not cluster or self.top_neighbors is None:
            return 0
        score = 0
        neigh = self.top_neighbors.get(pe, [])
        if not neigh:
            return 0
        cluster_set = cluster
        for q, w in neigh:
            if q in cluster_set:
                score += w
        return score

    # count used chiplets in a slots mapping
    def _used_chiplets(self, slots: Optional[Dict[int, List[int]]] = None) -> int:
        slots = slots if slots is not None else self.pe_slots
        return len([ch for ch in range(self.max_chiplets) if any(pe != -1 for pe in slots[ch])])

    # Expand/split donors to get exactly 'target' used chiplets (if feasible)
    def _expand_to_target_chiplets(self, slots: Dict[int, List[int]], target: int) -> Dict[int, List[int]]:
        """
        Ensure used chiplets == target by splitting donors.
        For 'separated', preserve homogeneity; for 'mixed', respect IMC/DIG quotas.
        Only uses chiplet indices [0..target-1] to avoid exceeding target.
        """
        K = MAX_PES_PER_CHIPLET

        def used_list():
            return [ch for ch in range(target) if any(pe != -1 for pe in slots[ch])]

        def empties_list():
            return [ch for ch in range(target) if all(pe == -1 for pe in slots[ch])]

        used = used_list()
        if len(used) >= target:
            return slots

        empties = empties_list()
        if not empties:
            return slots

        if self.pe_type_strategy == 'separated':
            pref_label = (self.preferred_extra_type_for_separated or 'IMC').upper()
            order = [pref_label, 'IMC', 'DIG', 'OTHER']
            seen = set()
            ordered_labels = [x for x in order if not (x in seen or seen.add(x))]

            while len(used) < target and empties:
                new_ch = empties.pop(0)
                moved = False

                # Pass 1: preferred label first, then others
                for label in ordered_labels:
                    donors = [
                        ch for ch in used
                        if self.chiplet_types.get(ch) == label
                        and sum(1 for pe in slots[ch] if pe != -1) >= 2
                    ]
                    donors.sort(key=lambda ch: sum(1 for q in slots[ch] if q != -1), reverse=True)
                    for donor in donors:
                        vic = self._choose_victim_low_synergy(donor, avoid=None)
                        if not vic:
                            continue
                        pe, sidx = vic
                        slots[new_ch][0] = pe
                        slots[donor][sidx] = -1
                        self.chiplet_types[new_ch] = label
                        used = used_list()
                        moved = True
                        break
                    if moved:
                        break

                # Pass 2: any label if still stuck
                if not moved:
                    donors = [ch for ch in used if sum(1 for pe in slots[ch] if pe != -1) >= 2]
                    donors.sort(key=lambda ch: sum(1 for q in slots[ch] if q != -1), reverse=True)
                    for donor in donors:
                        label = self.chiplet_types.get(donor, 'OTHER')
                        vic = self._choose_victim_low_synergy(donor, avoid=None)
                        if not vic:
                            continue
                        pe, sidx = vic
                        slots[new_ch][0] = pe
                        slots[donor][sidx] = -1
                        self.chiplet_types[new_ch] = label
                        used = used_list()
                        moved = True
                        break

                if not moved:
                    break
            return slots

        # Mixed strategy expansion (respect IMC/DIG quotas)
        imc_max, dig_max = get_imc_dig_counts(self.imc_ratio, K)
        imc_max = max(0, min(imc_max, K))
        dig_max = max(0, min(dig_max, K))

        def can_accept(pe: int, chiplet: int) -> bool:
            t = self.pe_types.get(pe)
            if t == 'IMC':
                return sum(1 for q in slots[chiplet] if q != -1 and self.pe_types.get(q) == 'IMC') + 1 <= imc_max
            if t == 'DIG':
                return sum(1 for q in slots[chiplet] if q != -1 and self.pe_types.get(q) == 'DIG') + 1 <= dig_max
            return True

        while len(used) < target and empties:
            new_ch = empties.pop(0)
            self.chiplet_types[new_ch] = 'MIXED'
            donors = [ch for ch in used if sum(1 for pe in slots[ch] if pe != -1) >= 2]
            donors.sort(key=lambda ch: sum(1 for q in slots[ch] if q != -1), reverse=True)

            moved = False
            for preferred in ('IMC', 'DIG', None):
                if moved:
                    break
                for donor in donors:
                    for s, pe in enumerate(slots[donor]):
                        if pe == -1:
                            continue
                        t = self.pe_types.get(pe)
                        if preferred is not None and t != preferred:
                            continue
                        if not can_accept(pe, new_ch):
                            continue
                        slots[new_ch][0] = pe
                        slots[donor][s] = -1
                        moved = True
                        break
                    if moved:
                        break
            if not moved:
                break
            used = used_list()
        return slots

    def _calculate_min_chiplets_by_strategy(self) -> int:
        return compute_min_required_chiplets(self.problem, self.pe_type_strategy, self.imc_ratio)

    def _create_comm_clusters_grasp(self, pe_communication, target_chiplets, rcl_size) -> Tuple[List[set], int]:
        if self.pe_type_strategy == 'separated':
            return self._create_type_separated_clusters(pe_communication, rcl_size)
        if self.pe_type_strategy == 'mixed':
            return self._create_mixed_type_clusters(pe_communication, target_chiplets, rcl_size)
        # default: mixed-like clustering
        clusters = self._cluster_pes_by_communication(self.pes, pe_communication,
                                                      (len(self.pes) + MAX_PES_PER_CHIPLET - 1) // MAX_PES_PER_CHIPLET,
                                                      rcl_size)
        for i in range(len(clusters)):
            self.chiplet_types[i] = 'MIXED'
        return clusters, len(clusters)

    def _create_type_separated_clusters(self, pe_communication, rcl_size) -> Tuple[List[set], int]:
        K = MAX_PES_PER_CHIPLET
        imc_pes = [pe for pe in self.pes if self.pe_types.get(pe) == 'IMC']
        dig_pes = [pe for pe in self.pes if self.pe_types.get(pe) == 'DIG']
        other_pes = [pe for pe in self.pes if self.pe_types.get(pe) not in ['IMC', 'DIG']]

        clusters: List[set] = []

        def add_typed_groups(pe_list: List[int], label: str):
            nonlocal clusters
            if not pe_list:
                return
            need = (len(pe_list) + K - 1) // K
            typed_clusters = self._cluster_pes_by_communication(pe_list, pe_communication, need, rcl_size)
            for cl in typed_clusters:
                clusters.append(set(cl))
                self.chiplet_types[len(clusters) - 1] = label

        add_typed_groups(imc_pes, 'IMC')
        add_typed_groups(dig_pes, 'DIG')
        add_typed_groups(other_pes, 'OTHER')

        required = len(clusters)
        return clusters, required

    def _create_mixed_type_clusters(self, pe_communication, target_chiplets, rcl_size) -> Tuple[List[set], int]:
        """
        Start with exactly target_chiplets mixed clusters and fill with IMC/DIG/OTHER
        under per-cluster quotas to encourage low cross-chip traffic.
        """
        K = MAX_PES_PER_CHIPLET
        imc_quota_base, dig_quota_base = get_imc_dig_counts(self.imc_ratio, K)
        imc_quota_base = max(0, min(imc_quota_base, K))
        dig_quota_base = max(0, min(dig_quota_base, K))

        imc_pool = [pe for pe in self.pes if self.pe_types.get(pe) == 'IMC']
        dig_pool = [pe for pe in self.pes if self.pe_types.get(pe) == 'DIG']
        other_pool = [pe for pe in self.pes if self.pe_types.get(pe) not in ['IMC', 'DIG']]

        remaining_imc = set(imc_pool)
        remaining_dig = set(dig_pool)
        remaining_other = set(other_pool)

        def score(pe, cluster):
            return self._score_into_cluster(pe, cluster)

        clusters: List[set] = []
        quotas: List[List[int]] = []  # [IMC_left, DIG_left, ANY_left]

        def add_one_cluster():
            clusters.append(set())
            any_left = K - (imc_quota_base + dig_quota_base)
            quotas.append([imc_quota_base, dig_quota_base, max(0, any_left)])
            self.chiplet_types[len(clusters) - 1] = 'MIXED'

        for _ in range(max(1, target_chiplets)):
            add_one_cluster()

        while remaining_imc or remaining_dig or remaining_other:
            progress = 0

            # IMC
            if remaining_imc:
                for c_idx in range(len(clusters)):
                    if not remaining_imc:
                        break
                    IMC_left, DIG_left, ANY_left = quotas[c_idx]
                    if IMC_left <= 0 and ANY_left <= 0:
                        continue
                    cand = [(score(pe, clusters[c_idx]), pe) for pe in list(remaining_imc)]
                    if not cand:
                        continue
                    cand.sort(reverse=True)
                    _, chosen = self.rng.choice(cand[:min(len(cand), rcl_size)])
                    clusters[c_idx].add(chosen)
                    if IMC_left > 0:
                        quotas[c_idx][0] -= 1
                    else:
                        quotas[c_idx][2] -= 1
                    remaining_imc.remove(chosen)
                    progress += 1

            # DIG
            if remaining_dig:
                for c_idx in range(len(clusters)):
                    if not remaining_dig:
                        break
                    IMC_left, DIG_left, ANY_left = quotas[c_idx]
                    if DIG_left <= 0 and ANY_left <= 0:
                        continue
                    cand = [(score(pe, clusters[c_idx]), pe) for pe in list(remaining_dig)]
                    if not cand:
                        continue
                    cand.sort(reverse=True)
                    _, chosen = self.rng.choice(cand[:min(len(cand), rcl_size)])
                    clusters[c_idx].add(chosen)
                    if DIG_left > 0:
                        quotas[c_idx][1] -= 1
                    else:
                        quotas[c_idx][2] -= 1
                    remaining_dig.remove(chosen)
                    progress += 1

            # OTHER
            if remaining_other:
                for c_idx in range(len(clusters)):
                    if not remaining_other:
                        break
                    IMC_left, DIG_left, ANY_left = quotas[c_idx]
                    if IMC_left + DIG_left + ANY_left <= 0:
                        continue
                    cand = [(score(pe, clusters[c_idx]), pe) for pe in list(remaining_other)]
                    if not cand:
                        continue
                    cand.sort(reverse=True)
                    _, chosen = self.rng.choice(cand[:min(len(cand), rcl_size)])
                    clusters[c_idx].add(chosen)
                    if ANY_left > 0:
                        quotas[c_idx][2] -= 1
                    elif IMC_left >= DIG_left and IMC_left > 0:
                        quotas[c_idx][0] -= 1
                    elif DIG_left > 0:
                        quotas[c_idx][1] -= 1
                    remaining_other.remove(chosen)
                    progress += 1

            if progress == 0:
                add_one_cluster()

        # trim empties
        nonempty = []
        for i, cl in enumerate(clusters):
            if len(cl) > 0:
                nonempty.append(cl)
            else:
                self.chiplet_types.pop(i, None)
        clusters = nonempty
        required = len(clusters)
        return clusters, required

    def _cluster_pes_by_communication(self, pe_list, pe_communication, num_clusters, rcl_size):
        """
        Communication-aware clustering with sparse scoring and RCL (GRASP).
        """
        if not pe_list:
            return []

        K = MAX_PES_PER_CHIPLET
        clusters: List[set] = []
        used = set()

        # Seed by heavy internal pairs among the given list
        internal_pairs = []
        for (a, b), weight in pe_communication.items():
            if a in pe_list and b in pe_list:
                internal_pairs.append(((a, b), weight))
        internal_pairs.sort(key=lambda x: x[1], reverse=True)

        pair_idx = 0
        while len(clusters) < num_clusters and pair_idx < len(internal_pairs):
            (a, b), _ = internal_pairs[pair_idx]
            if a not in used and b not in used:
                clusters.append({a, b})
                used.add(a); used.add(b)
            pair_idx += 1

        # Singletons if needed
        remaining = [pe for pe in pe_list if pe not in used]
        while len(clusters) < num_clusters and remaining:
            pe = remaining.pop(0)
            clusters.append({pe})
            used.add(pe)

        # Greedy fill each cluster to capacity using sparse scores
        for cluster in clusters:
            while len(cluster) < K:
                best = []
                for pe in pe_list:
                    if pe in used:
                        continue
                    s = self._score_into_cluster(pe, cluster)
                    if s > 0:
                        best.append((s, pe))
                if not best:
                    break
                best.sort(reverse=True)
                # pick randomly within top RCL
                _, chosen = self.rng.choice(best[:min(len(best), rcl_size)])
                cluster.add(chosen)
                used.add(chosen)

        # Distribute any leftover PEs to least-filled clusters
        remaining = [pe for pe in pe_list if pe not in used]
        for pe in remaining:
            candidates = [(len(c), idx) for idx, c in enumerate(clusters) if len(c) < K]
            if not candidates:
                clusters.append({pe})
            else:
                _, idx = min(candidates)
                clusters[idx].add(pe)

        return clusters

    def _assign_clusters_to_slots(self, clusters: List[set], target_chiplets: int):
        K = MAX_PES_PER_CHIPLET
        fixed_clusters: List[List[int]] = []
        fixed_types: List[str] = []
        for idx, cl in enumerate(clusters):
            cl_list = sorted(list(cl))
            label = self.chiplet_types.get(idx, 'MIXED')
            for start in range(0, len(cl_list), K):
                chunk = cl_list[start:start + K]
                fixed_clusters.append(chunk)
                fixed_types.append(label)

        required = len(fixed_clusters)
        target = min(max(target_chiplets, required, self.min_chiplets_required), self.max_chiplets)

        if required > target:
            fixed_clusters = fixed_clusters[:target]
            fixed_types = fixed_types[:target]
            required = target

        pe_slots = {i: [-1] * K for i in range(self.max_chiplets)}
        self.chiplet_types = {}

        for chiplet_idx in range(required):
            self.chiplet_types[chiplet_idx] = fixed_types[chiplet_idx]
            for s, pe in enumerate(fixed_clusters[chiplet_idx]):
                pe_slots[chiplet_idx][s] = pe

        return pe_slots

    def _evaluate_chiplet_configuration(self, test_slots):
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

    # -------------------- Local search --------------------

    def local_search(self,
                     max_passes: int = 4,
                     pair_swap_samples: int = 700,
                     time_budget_s: Optional[float] = None,
                     start_time: Optional[float] = None):
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

            if not time_left_ok():
                break
            if self._co_locate_top_pairs(limit_pairs=self.COLOCATE_PAIR_LIMIT, accept_only_improving=True):
                improved_any = True

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
                        self.pe_slots = snap
                        self._update_pe_assignments_from_slots()
                        self._update_task_assignments_from_pe_slots()
                        self._create_feasible_schedule()

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

            if not time_left_ok():
                break
            if self._steepest_pair_swap(pair_swap_samples, time_left_ok):
                improved_any = True

        final_cost = self.evaluate_cost()
        if final_cost < base_cost:
            print(f"Local search improved: {base_cost} → {final_cost}")
        return final_cost

    def _co_locate_top_pairs(self, limit_pairs: int = 120, accept_only_improving: bool = True) -> bool:
        if not self.pe_communication:
            return False
        
            
        improved = False
        pairs = sorted(self.pe_communication.items(), key=lambda kv: kv[1], reverse=True)[:limit_pairs]
        for (a, b), _ in pairs:
            a_chip = self.pe_assignments.get(a)
            b_chip = self.pe_assignments.get(b)
            if a_chip is None or b_chip is None or a_chip == b_chip:
                continue

            before = self.evaluate_cost()

            snap1 = copy.deepcopy(self.pe_slots)
            moved1 = self._try_move_pe_to_chiplet(a, b_chip)
            if moved1:
                self._update_pe_assignments_from_slots()
                self._update_task_assignments_from_pe_slots()
                self._create_feasible_schedule()
                cost1 = self.evaluate_cost()
            else:
                cost1 = float('inf')

            if moved1:
                self.pe_slots = snap1
                self._update_pe_assignments_from_slots()
                self._update_task_assignments_from_pe_slots()
                self._create_feasible_schedule()

            snap2 = copy.deepcopy(self.pe_slots)
            moved2 = self._try_move_pe_to_chiplet(b, a_chip)
            if moved2:
                self._update_pe_assignments_from_slots()
                self._update_task_assignments_from_pe_slots()
                self._create_feasible_schedule()
                cost2 = self.evaluate_cost()
            else:
                cost2 = float('inf')

            best_cost = min(cost1, cost2)
            if best_cost < before:
                improved = True
                if cost1 <= cost2 and moved1:
                    self.pe_slots = snap1
                    self._try_move_pe_to_chiplet(a, b_chip)
                elif moved2:
                    pass
                self._update_pe_assignments_from_slots()
                self._update_task_assignments_from_pe_slots()
                self._create_feasible_schedule()
            else:
                if moved2:
                    self.pe_slots = snap2
                    self._update_pe_assignments_from_slots()
                    self._update_task_assignments_from_pe_slots()
                    self._create_feasible_schedule()

            if accept_only_improving and not improved:
                continue
        return improved

    # -------------------- Scheduling, cost, violations --------------------

    def _bandwidth_value(self) -> int:
        bw = INTER_CHIPLET_BANDWIDTH
        for constraint in self.problem.constraints:
            if hasattr(constraint, 'bandwidth'):
                bw = constraint.bandwidth
                break
        return bw

    def _calculate_task_duration(self, task):
        source_pe = self.problem.task_data[task]['source_pe']
        dest_pe = self.problem.task_data[task]['dest_pe']
        data_size = self.problem.task_data[task]['data_size']

        source_chiplet = self.pe_assignments.get(source_pe)
        dest_chiplet = self.pe_assignments.get(dest_pe)

        if (source_chiplet is not None) and (dest_chiplet is not None) and (source_chiplet == dest_chiplet):
            return 1

        inter_chiplet_bandwidth = self._bandwidth_value()
        duration = math.ceil(data_size / inter_chiplet_bandwidth)
        return max(1, duration)

    def _create_feasible_schedule(self):
        """
        Iterative Kahn topo schedule:
        - Enforces dependencies
        - Serializes by source_pe via last_finish[source_pe]
        """
        self.task_durations = {t: self._calculate_task_duration(t) for t in self.tasks}

        deps = self.problem.dependencies
        indeg = {t: 0 for t in self.tasks}
        rev = defaultdict(list)
        for t in self.tasks:
            for d in deps.get(t, []):
                indeg[t] += 1
                rev[d].append(t)

        q = deque([t for t in self.tasks if indeg[t] == 0])
        self.task_times = {}
        last_finish = defaultdict(int)
        seen_count = 0

        while q:
            t = q.popleft()
            seen_count += 1
            # earliest by deps
            earliest = 0
            for d in deps.get(t, []):
                earliest = max(earliest, self.task_times[d] + self.task_durations[d])
            # serialize by source
            src = self.problem.task_data[t]['source_pe']
            earliest = max(earliest, last_finish[src])

            self.task_times[t] = earliest
            last_finish[src] = earliest + self.task_durations[t]

            for u in rev[t]:
                indeg[u] -= 1
                if indeg[u] == 0:
                    q.append(u)

        if seen_count != len(self.tasks):
            # Graph has cycles; we still keep partial schedule for the seen tasks.
            # (Matches old behavior of raising on cycles; here we warn instead.)
            print("⚠️  Warning: dependency graph may contain cycles; partial schedule created.")

    def evaluate_cost(self):
        """
        For forced chiplet testing: Cost = total_cycles only.
        Dependencies and task durations are respected in scheduling.
        Violations get heavy penalty to ensure feasibility.
        """
        if not self._dirty and self._cached_cost is not None:
            return self._cached_cost

        total_cycles = self.get_total_time()
        violations = self.count_violations()
        total_violations = sum(violations.values())

        # For forced chiplet testing: Pure cycle optimization (no weights)
        # Heavy violation penalty to ensure feasibility
        if total_violations > 0:
            cost = total_cycles + (1000000 * total_violations)  # Fixed penalty, no config dependency
        else:
            cost = total_cycles  # Pure cycle minimization when feasible

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
            'pe_type_separation': 0,
        }

        unassigned_tasks = [t for t in self.tasks if t not in self.task_assignments]
        violations['task_assignment'] = len(unassigned_tasks)

        for chiplet in range(self.max_chiplets):
            used_slots = sum(1 for pe in self.pe_slots.get(chiplet, []) if pe != -1)
            if used_slots > MAX_PES_PER_CHIPLET:
                violations['chiplet_capacity'] += used_slots - MAX_PES_PER_CHIPLET

        pe_chiplet = defaultdict(set)
        for pe, ch in self.pe_assignments.items():
            pe_chiplet[pe].add(ch)
        for pe, chs in pe_chiplet.items():
            if len(chs) > 1:
                violations['pe_exclusivity'] += len(chs) - 1

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

        time_pe_tasks = defaultdict(lambda: defaultdict(list))
        for t in self.tasks:
            if t in self.task_times:
                start = self.task_times[t]
                dur = self.task_durations.get(t, 1)
                src = self.problem.task_data[t]['source_pe']
                dst = self.problem.task_data[t]['dest_pe']
                for cycle in range(start, start + dur):
                    time_pe_tasks[cycle][src].append((t, dst))

        # multicast violation (same cycle, same source, multiple distinct dests)
        for _, pe_tasks in time_pe_tasks.items():
            for source_pe, tasks_dests in pe_tasks.items():
                if len(tasks_dests) > 1:
                    unique_dests = set(dest for _, dest in tasks_dests)
                    if len(unique_dests) > 1:
                        violations['no_multicasting'] += len(unique_dests) - 1

        violations['pe_type_separation'] = self._count_type_mixing_violations()
        return violations

    def _count_type_mixing_violations(self) -> int:
        if not self.pe_types:
            return 0

        violations = 0

        if self.pe_type_strategy == 'separated':
            for chiplet in range(self.max_chiplets):
                types_here = set()
                count_here = 0
                for pe in self.pe_slots.get(chiplet, []):
                    if pe != -1:
                        t = self.pe_types.get(pe, 'OTHER')
                        types_here.add(t)
                        count_here += 1
                if len(types_here) > 1:
                    violations += count_here
        elif self.pe_type_strategy == 'mixed':
            max_imc, max_dig = get_imc_dig_counts(self.imc_ratio, MAX_PES_PER_CHIPLET)
            for chiplet in range(self.max_chiplets):
                used = [pe for pe in self.pe_slots.get(chiplet, []) if pe != -1]
                if not used:
                    continue
                imc_count = sum(1 for pe in used if self.pe_types.get(pe) == 'IMC')
                dig_count = sum(1 for pe in used if self.pe_types.get(pe) == 'DIG')
                if imc_count > max_imc:
                    violations += imc_count - max_imc
                if dig_count > max_dig:
                    violations += dig_count - max_dig

        return violations

    # -------------------- Greedy helpers & mapping updates --------------------

    def _print_cross_stats(self):
        ec, bc = self._cross_chip_stats()
        print(f"[STATS] cross_edges={ec}, cross_bytes={bc}, total_cycles={self.get_total_time()}")

    def _cross_chip_stats(self) -> Tuple[int, int]:
        bytes_cross = 0
        edges_cross = 0
        for t in self.tasks:
            s = self.problem.task_data[t]['source_pe']
            d = self.problem.task_data[t]['dest_pe']
            if self.pe_assignments.get(s) != self.pe_assignments.get(d):
                edges_cross += 1
                bytes_cross += self.problem.task_data[t]['data_size']
        return edges_cross, bytes_cross

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

    def _try_move_pe_to_chiplet(self, pe: int, chiplet: int) -> bool:
        loc = self._find_slot_of_pe(pe)
        if loc is None:
            return False
        cur_chip, cur_slot = loc
        if cur_chip == chiplet:
            return False

        if self._would_violate_pe_type_constraints(pe, chiplet):
            return False
        if self._would_violate_min_chiplet_constraint(pe, cur_chip):
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

        if self._would_violate_pe_type_constraints(vic_pe, cur_chip):
            return False

        self.pe_slots[chiplet][vic_slot], self.pe_slots[cur_chip][cur_slot] = pe, vic_pe
        return True

    def _would_violate_pe_type_constraints(self, pe: int, target_chiplet: int) -> bool:
        pe_type = self.pe_types.get(pe)
        if pe_type is None or pe_type not in ['IMC', 'DIG']:
            return False

        if self.pe_type_strategy == 'separated':
            chiplet_label = self.chiplet_types.get(target_chiplet)
            if chiplet_label is not None and chiplet_label != pe_type:
                return True
            current_types = set()
            for slot_pe in self.pe_slots.get(target_chiplet, []):
                if slot_pe != -1:
                    t = self.pe_types.get(slot_pe)
                    if t in ['IMC', 'DIG', 'OTHER']:
                        current_types.add(t)
            if current_types and pe_type not in current_types:
                return True

        elif self.pe_type_strategy == 'mixed':
            K = MAX_PES_PER_CHIPLET
            imc_max, dig_max = get_imc_dig_counts(self.imc_ratio, K)
            imc_max = max(0, min(imc_max, K))
            dig_max = max(0, min(dig_max, K))
            imc_count = 0
            dig_count = 0
            for slot_pe in self.pe_slots.get(target_chiplet, []):
                if slot_pe != -1 and slot_pe != pe:
                    t = self.pe_types.get(slot_pe)
                    if t == 'IMC':
                        imc_count += 1
                    elif t == 'DIG':
                        dig_count += 1
            if pe_type == 'IMC' and imc_count + 1 > imc_max:
                return True
            if pe_type == 'DIG' and dig_count + 1 > dig_max:
                return True

        return False

    def _would_violate_min_chiplet_constraint(self, pe: int, source_chiplet: int) -> bool:
        """
        Enforce EXACT chiplet count when forcing:
        - Disallow moves that would leave 'source_chiplet' empty and thus reduce the
            number of used chiplets below the forced target.
        When not forcing, preserve strategy minimum.
        """
        remaining_pes = sum(1 for slot_pe in self.pe_slots.get(source_chiplet, [])
                            if slot_pe != -1 and slot_pe != pe)
        if remaining_pes == 0:
            current_used = self._used_chiplets()
            if self.force_target_chiplets is not None:
                min_keep = self.force_target_chiplets   # <-- no slack
            else:
                min_keep = (self.min_chiplets_required or 1)
            if current_used - 1 < min_keep:
                return True
        return False


    def _pack_or_swap_top_dests_into_source_chiplet(self, src_pe: int, k_max: int = 3) -> bool:
        src_loc = self._find_slot_of_pe(src_pe)
        if src_loc is None:
            return False
        src_chip, _ = src_loc

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
                dst_loc = self._find_slot_of_pe(dst)
                if dst_loc is None:
                    continue
                dst_chip, dst_slot = dst_loc
                free = self._find_free_slot_in_chiplet(src_chip)
                if free is None:
                    continue
                if (self._would_violate_pe_type_constraints(dst, src_chip) or
                        self._would_violate_min_chiplet_constraint(dst, dst_chip)):
                    continue
                self.pe_slots[src_chip][free] = dst
                self.pe_slots[dst_chip][dst_slot] = -1
                moved_any = True
            else:
                victim = self._choose_victim_low_synergy(src_chip, avoid=src_pe)
                if not victim:
                    continue
                vic_pe, vic_slot = victim
                damage = 0
                local = [pe for pe in self.pe_slots[src_chip] if pe != -1 and pe != vic_pe]
                for other in local:
                    key = tuple(sorted((vic_pe, other)))
                    damage += self.pe_communication.get(key, 0) if self.pe_communication else 0
                if b_gain > damage:
                    dst_loc = self._find_slot_of_pe(dst)
                    if dst_loc is None:
                        continue
                    dst_chip, dst_slot = dst_loc
                    if (self._would_violate_pe_type_constraints(dst, src_chip) or
                            self._would_violate_pe_type_constraints(vic_pe, dst_chip)):
                        continue
                    self.pe_slots[src_chip][vic_slot], self.pe_slots[dst_chip][dst_slot] = dst, vic_pe
                    moved_any = True

        return moved_any

    def _steepest_pair_swap(self, samples_per_round: int, time_ok_fn) -> bool:
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

                if (self._would_violate_pe_type_constraints(pe1, c2) or
                        self._would_violate_pe_type_constraints(pe2, c1)):
                    continue

                self.pe_slots[c1][s1], self.pe_slots[c2][s2] = pe2, pe1
                self._update_pe_assignments_from_slots()
                self._update_task_assignments_from_pe_slots()
                self._create_feasible_schedule()
                after_cost = self.evaluate_cost()

                delta = before_cost - after_cost
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

    # -------------------- Mapping updates --------------------

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

        # If forced, report the *target* chiplets; else count used
        if self.force_target_chiplets is not None:
            num_chiplets = self.force_target_chiplets
        else:
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
        """
        After the initial constructive build and schedule, do a quick greedy polish:
        - Try moving hot sources to their best chiplets
        - Then try packing or swapping top destinations into each source chiplet
        """
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

    # -------------------- Helper bundle used by _post_init_refine --------------------

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
                best_chip, best_gain = c, gain
        if best_chip is None or best_gain <= 0:
            return False

        if self._would_violate_pe_type_constraints(src_pe, best_chip):
            return False

        free = self._find_free_slot_in_chiplet(best_chip)
        if free is not None:
            if self._would_violate_min_chiplet_constraint(src_pe, src_chip):
                return False
            self.pe_slots[best_chip][free] = src_pe
            self.pe_slots[src_chip][src_slot] = -1
            return True

        victim = self._choose_victim_low_synergy(best_chip, avoid=None)
        if victim is None:
            return False
        vic_pe, vic_slot = victim
        if self._would_violate_pe_type_constraints(vic_pe, src_chip):
            return False
        self.pe_slots[best_chip][vic_slot], self.pe_slots[src_chip][src_slot] = src_pe, vic_pe
        return True


# ================== GRASP SOLVER ==================

class GRASPSolver:
    def __init__(self, problem, constraints: List):
        self.problem = problem
        self.constraints = constraints
        self.pe_type_strategy = DEFAULT_PE_TYPE_STRATEGY
        self.imc_ratio = DEFAULT_IMC_RATIO

    def solve(self,
              max_chiplets: int = 10,
              timeout: float = 300.0,
              save_solution_file: bool = True,
              starts: Optional[int] = None,
              seed: Optional[int] = None,
              rcl_size: int = 4,
              ls_max_passes: int = 4,
              pair_swap_samples: int = 700,
              solution_file_suffix: str = "",
              # sweep controls forwarded
              force_target_chiplets: Optional[int] = None,
              exhaustive_construct: bool = False,
              preferred_extra_type_for_separated: Optional[str] = None,
              # warm start from previous T (optional)
              warm_start_slots: Optional[Dict[int, List[int]]] = None) -> Dict[str, Any]:

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

            sol = HeuristicSolution(
                self.problem, max_chiplets, rng=rng,
                pe_type_strategy=self.pe_type_strategy,
                imc_ratio=self.imc_ratio,
                force_target_chiplets=force_target_chiplets,
                exhaustive_construct=exhaustive_construct,
                preferred_extra_type_for_separated=preferred_extra_type_for_separated,
                warm_start_slots=warm_start_slots,
            )
            sol.grasp_construct(rcl_size=rcl_size)

            # Allocate per-start time fairly if starts specified
            if starts is not None:
                remaining_starts = max(1, starts - starts_done)
                per_start = max(1.0, (timeout - (time_module.time() - start_time)) / remaining_starts)
            else:
                per_start = max(1.0, 0.30 * timeout)

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
                solution_dict, 'GR', self.problem.traffic_file, self.problem.task_data, solution_file_suffix
            )
            if solution_file:
                solution_dict['solution_file'] = solution_file
                print(f"Solution saved to: {solution_file}")

        return solution_dict


# ================== CHIPLET PROBLEM CLASS ==================

class ChipletProblemGRASP:
    """
    Uses GRASP for construction + local search.
    """

    def __init__(self, traffic_file: str):
        self.traffic_file = traffic_file
        self.constraints: List[Any] = []
        self.pe_types: Dict[int, str] = {}
        self.task_data, self.tasks, self.pes, self.dependencies = self._load_traffic_data(traffic_file)

    def _load_traffic_data(self, filename: str):
        print(f"Loading traffic data from {filename}")
        # Try CSV first
        try:
            df = pd.read_csv(filename)
            if all(col in df.columns for col in ['task_id', 'src_pe', 'dest_pe', 'bytes', 'wait_ids', 'src_type', 'dst_type']):
                print("Detected CSV format with PE types")
                return self._load_csv_with_types(df)
            else:
                print(f"CSV columns found: {list(df.columns)}")
                print("Required columns not found, trying legacy format...")
        except Exception as e:
            print(f"Failed to load as CSV: {e}")

        # Fallback TSV legacy
        try:
            df = pd.read_csv(filename, sep='\t', comment='#',
                             names=['task_id', 'source_pe', 'dest_pe', 'data_size', 'wait_ids'])
            print("Detected tab-separated format (legacy)")
            return self._load_tab_separated(df)
        except Exception as e:
            raise ValueError(f"Could not load traffic data from {filename}: {e}")

    def _parse_pe_id(self, pe_str):
        pe_str = str(pe_str).strip()
        if pe_str.startswith('P'):
            return int(pe_str[1:])
        else:
            return int(pe_str)

    def _load_csv_with_types(self, df):
        task_data: Dict[int, Dict[str, Any]] = {}
        tasks: set = set()
        pes: set = set()
        pe_types: Dict[int, str] = {}
        dependencies = defaultdict(list)

        def _norm_type(s):
            u = str(s).strip().upper()
            if u in ('DIGITAL', 'DIG'):
                return 'DIG'
            if u == 'IMC':
                return 'IMC'
            return u

        for _, row in df.iterrows():
            task_id = int(row['task_id'])
            source_pe = self._parse_pe_id(row['src_pe'])
            dest_pe = self._parse_pe_id(row['dest_pe'])
            data_size = int(row['bytes'])
            wait_ids = row['wait_ids']
            src_type = _norm_type(row['src_type'])
            dst_type = _norm_type(row['dst_type'])

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

        self.pe_types = pe_types

        print(f"Loaded: {len(tasks)} tasks, {len(pes)} PEs, {sum(len(deps) for deps in dependencies.values())} dependencies")
        type_counts = {}
        for pe_type in pe_types.values():
            type_counts[pe_type] = type_counts.get(pe_type, 0) + 1
        print(f"PE types: {type_counts}")

        return task_data, tasks, pes, dict(dependencies)

    def _load_tab_separated(self, df):
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

        self.pe_types = {}

        print(f"Loaded: {len(tasks)} tasks, {len(pes)} PEs, {sum(len(deps) for deps in dependencies.values())} dependencies")
        return task_data, tasks, pes, dict(dependencies)

    def add_constraint(self, constraint):
        self.constraints.append(constraint)

    def solve(self,
              timeout: float = 300,
              max_chiplets: int = 10,
              save_solution_file: bool = True,
              pe_type_strategy: str = DEFAULT_PE_TYPE_STRATEGY,
              imc_ratio: float = DEFAULT_IMC_RATIO,
              solution_file_suffix: str = "",
              **params) -> Dict[str, Any]:
        """
        Solve using GRASP + Local Search.

        Accepted params:
          - starts, seed, rcl_size, ls_max_passes, pair_swap_samples
          - pe_type_strategy: 'separated' | 'mixed'
          - imc_ratio: used when strategy='mixed'
          - force_target_chiplets: int | None
          - exhaustive_construct: bool
          - preferred_extra_type_for_separated: 'IMC'|'DIG'|'OTHER'|None
          - warm_start_slots: Dict[int, List[int]] | None   (optional sweep speedup)
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
            'rcl_size': params.get('rcl_size', 4),
            'ls_max_passes': params.get('ls_max_passes', 4),
            'pair_swap_samples': params.get('pair_swap_samples', 700),
            'solution_file_suffix': solution_file_suffix,
            # sweep controls
            'force_target_chiplets': params.get('force_target_chiplets', None),
            'exhaustive_construct': params.get('exhaustive_construct', False),
            'preferred_extra_type_for_separated': params.get('preferred_extra_type_for_separated', None),
            'warm_start_slots': params.get('warm_start_slots', None),
        }

        return solver.solve(**grasp_config)
