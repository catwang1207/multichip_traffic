#!/usr/bin/env python3
"""
Utility functions for chiplet optimization frameworks.
Summary-only solution file writer.
"""

import json
import os
import time as time_module
from math import gcd


def _norm_type(s):
    if s is None:
        return None
    u = str(s).strip().upper()
    if u in ("DIGITAL", "DIG"):
        return "DIG"
    if u == "IMC":
        return "IMC"
    return "OTHER"


def _build_pe_types_from_task_data(task_data):
    """
    Best-effort PE->type map using src_type/dst_type in task_data (CSV mode).
    Falls back to 'OTHER' where unknown.
    Returns (pe_type_map, has_any_type_info)
    """
    pe_types = {}
    saw_any_tag = False
    for t, info in task_data.items():
        src = info.get("source_pe")
        dst = info.get("dest_pe")

        st = _norm_type(info.get("src_type")) if "src_type" in info else None
        dt = _norm_type(info.get("dst_type")) if "dst_type" in info else None

        if st:
            saw_any_tag = saw_any_tag or (st in ("IMC", "DIG"))
            pe_types[src] = st
        if dt:
            saw_any_tag = saw_any_tag or (dt in ("IMC", "DIG"))
            pe_types[dst] = dt

        # If no typed columns existed, nothing to do; solver still passes source/dest IDs.
    return pe_types, saw_any_tag


def generate_solution_file(solution, solver_name, data_file, task_data, suffix=""):
    """
    Write a compact JSON summary:
      - total_cycles
      - violations (total count)
      - chiplets_used (actually occupied)
      - avg_pes_per_used_chiplet
      - type_mode: 'separated' | 'mixed' | 'unknown'
        * separated -> {'imc_chiplets', 'dig_chiplets'}
        * mixed     -> {'imc_pes', 'dig_pes', 'imc_to_dig', 'imc_share'}
    Returns path to the JSON file, or None if infeasible/error.
    """
    if solution.get("status") == "infeasible":
        print(f"Cannot generate solution file - {solver_name} solution is infeasible")
        return None

    # Core fields
    total_time = int(solution.get("total_time", 0))
    pe_assignments = solution.get("pe_assignments", {})  # {pe_id: chiplet_id}
    violations_breakdown = solution.get("violations", {}) or {}
    violations_total = int(sum(violations_breakdown.values()))
    solve_time = float(solution.get("solve_time", 0.0))

    # Chiplets actually used (from pe_assignments)
    used_chiplets_set = set(pe_assignments.values()) if pe_assignments else set()
    chiplets_used = len(used_chiplets_set)

    # Average used PEs per used chiplet
    if chiplets_used > 0:
        # count PEs per chiplet
        pes_per_chiplet = {}
        for pe, ch in pe_assignments.items():
            pes_per_chiplet[ch] = pes_per_chiplet.get(ch, 0) + 1
        total_pes_used = sum(pes_per_chiplet.values())
        avg_pes_per_used_chiplet = total_pes_used / chiplets_used
    else:
        total_pes_used = 0
        avg_pes_per_used_chiplet = 0.0

    # Try to infer types
    pe_types, has_type_info = _build_pe_types_from_task_data(task_data)

    type_section = {"type_mode": "unknown"}
    if has_type_info and chiplets_used > 0:
        # Build type sets per chiplet
        chiplet_type_sets = {ch: set() for ch in used_chiplets_set}
        for pe, ch in pe_assignments.items():
            t = pe_types.get(pe, "OTHER")
            chiplet_type_sets[ch].add(t if t in ("IMC", "DIG") else "OTHER")

        any_mixed = any(len(cs & {"IMC", "DIG"}) > 1 for cs in chiplet_type_sets.values())
        only_single = all(len(cs & {"IMC", "DIG"}) <= 1 for cs in chiplet_type_sets.values())

        if any_mixed:
            # Observed MIXED: show global IMC:DIG PE ratio
            imc_pes = sum(1 for pe in pe_assignments if pe_types.get(pe) == "IMC")
            dig_pes = sum(1 for pe in pe_assignments if pe_types.get(pe) == "DIG")
            if imc_pes == 0 and dig_pes == 0:
                ratio_str = "0:0"
                imc_share = 0.0
            else:
                g = gcd(imc_pes, dig_pes) if (imc_pes and dig_pes) else 1
                ratio_str = f"{imc_pes // g}:{dig_pes // g}"
                denom = imc_pes + dig_pes
                imc_share = imc_pes / denom if denom else 0.0
            type_section = {
                "type_mode": "mixed",
                "mixed_ratio": {
                    "imc_pes": imc_pes,
                    "dig_pes": dig_pes,
                    "imc_to_dig": ratio_str,
                    "imc_share": imc_share,
                },
            }
        elif only_single:
            # Observed SEPARATED: count chiplets labeled purely IMC vs purely DIG
            imc_chiplets = sum(1 for cs in chiplet_type_sets.values() if cs & {"IMC"} and not (cs & {"DIG"}))
            dig_chiplets = sum(1 for cs in chiplet_type_sets.values() if cs & {"DIG"} and not (cs & {"IMC"}))
            type_section = {
                "type_mode": "separated",
                "separated_counts": {
                    "imc_chiplets": imc_chiplets,
                    "dig_chiplets": dig_chiplets,
                },
            }
        else:
            # No IMC/DIG found (e.g., all OTHER)
            type_section = {"type_mode": "unknown"}

    # Compose minimal summary payload
    base_name = os.path.splitext(os.path.basename(data_file))[0]
    out_path = f"{base_name}_{solver_name.lower()}{suffix}_summary.json"

    summary = {
        "solver": solver_name,
        "data_file": data_file,
        "timestamp": time_module.strftime("%Y-%m-%d %H:%M:%S"),
        "status": solution.get("status", "feasible"),
        "total_cycles": total_time,
        "chiplets_used": chiplets_used,
        "violations": violations_total,
        "avg_pes_per_used_chiplet": avg_pes_per_used_chiplet,
        **type_section,
        # keep a few helpful extras without being verbose:
        "solve_time_seconds": solve_time,
    }

    try:
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        return out_path
    except Exception as e:
        print(f"❌ Error generating solution file: {e}")
        return None
