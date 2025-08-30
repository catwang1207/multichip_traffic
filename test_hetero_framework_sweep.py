#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sweep experiment for the heterogeneous chiplet framework.

Writes:
- Per-run CSVs:
    results_separated.csv
    results_mixed.csv
- Per-target averages of utilization:
    avg_utilization_by_target_separated.csv
    avg_utilization_by_target_mixed.csv
- Figures:
    plot_separated.png
    plot_mixed.png
"""

import os
from typing import Dict, Any, List

import pandas as pd
import matplotlib.pyplot as plt

from config import LARGE_DATASET, DEFAULT_IMC_RATIO, MAX_PES_PER_CHIPLET
from gr_hetro_chiplet_framework_sweep import (
    ChipletProblemGRASP,
    compute_min_required_chiplets,
)

# --------------------------- CONFIGURE HERE ---------------------------

TRAFFIC_FILE = "traffic_table_large.csv"

# Ratios to test for mixed strategy
MIXED_RATIOS = [
    5/6,      # ≈0.833
    11/12,    # ≈0.917
    17/18    # ≈0.944
]

# Sweep width: run min_required .. min_required + SWEEP_PLUS (inclusive)
SWEEP_PLUS = 4

# Starts/time per run — adjust for your dataset size
STARTS = 1
TIMEOUT_PER_RUN = LARGE_DATASET.get('timeout_seconds', 60.0)
RCL_SIZE = LARGE_DATASET.get('rcl_size', 1)

# ---------------------------------------------------------------------


def ensure_traffic_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Traffic file not found: {path}")


def _utilization_fields(result: Dict[str, Any], target_chiplets: int) -> Dict[str, Any]:
    used_pes = len(result.get('pe_assignments', {}))
    total_slots = target_chiplets * MAX_PES_PER_CHIPLET
    util_pct = 100.0 * used_pes / total_slots if total_slots > 0 else 0.0
    return {
        'used_pes': used_pes,
        'total_pes_slots': total_slots,
        'utilization_pct': util_pct,
    }


def run_separated_sweep(problem: ChipletProblemGRASP) -> pd.DataFrame:
    """
    Strategy 1: 'separated'. Extra capacity is IMC-only.
    """
    print("=" * 80)
    print("SWEEP: STRATEGY 1 (separated) — extra chiplets are IMC-only")
    print("=" * 80)

    min_required = compute_min_required_chiplets(problem, pe_type_strategy='separated',
                                                 imc_ratio=DEFAULT_IMC_RATIO)
    print(f"[Separated] Strategy-aware minimum chiplets = {min_required}")

    rows: List[Dict[str, Any]] = []

    for plus in range(0, SWEEP_PLUS + 1):
        target = min_required + plus
        print(f"\n[Separated] Target chiplets: {target}")

        result = problem.solve(
            timeout=TIMEOUT_PER_RUN,
            max_chiplets=target,                  # cap at T
            pe_type_strategy='separated',
            rcl_size=RCL_SIZE,
            starts=STARTS,
            seed=12345 + plus,                    # reproducible-ish
            solution_file_suffix=f"_sep_T{target}",
            # New knobs:
            force_target_chiplets=target,         # force constructive target and enforce as min used
            exhaustive_construct=True,            # don’t early-stop
            preferred_extra_type_for_separated='IMC',  # extra clusters go to IMC
        )

        util = _utilization_fields(result, target)

        rows.append({
            'strategy': 'separated',
            'imc_ratio': None,
            'target_chiplets': target,
            'used_chiplets': result.get('num_chiplets', 0),
            'total_cycles': result.get('total_time', 0),
            'status': result.get('status', 'unknown'),
            'violations_total': sum(result.get('violations', {}).values())
                                if 'violations' in result else 0,
            'solve_time_s': result.get('solve_time', 0.0),
            **util,
        })

        print(f"[Separated] → {rows[-1]['used_chiplets']} used chiplets, "
              f"{rows[-1]['total_cycles']} cycles, "
              f"util={rows[-1]['utilization_pct']:.2f}%, "
              f"violations={rows[-1]['violations_total']}, "
              f"solve_time={rows[-1]['solve_time_s']:.2f}s")

    df = pd.DataFrame(rows)
    return df


def run_mixed_sweep(problem: ChipletProblemGRASP,
                    ratios: List[float]) -> pd.DataFrame:
    """
    Strategy 2: 'mixed'. For each IMC ratio, sweep min..min+SWEEP_PLUS.
    """
    print("=" * 80)
    print("SWEEP: STRATEGY 2 (mixed) — multiple IMC ratios")
    print("=" * 80)

    rows: List[Dict[str, Any]] = []

    for ratio_idx, ratio in enumerate(ratios):
        print(f"\n[Mixed] Ratio={ratio:.6f}")
        min_required = compute_min_required_chiplets(problem, pe_type_strategy='mixed',
                                                     imc_ratio=ratio)
        print(f"[Mixed ratio={ratio:.6f}] Strategy-aware minimum chiplets = {min_required}")

        for plus in range(0, SWEEP_PLUS + 1):
            target = min_required + plus
            print(f"[Mixed ratio={ratio:.6f}] Target chiplets: {target}")

            result = problem.solve(
                timeout=TIMEOUT_PER_RUN,
                max_chiplets=target,
                pe_type_strategy='mixed',
                imc_ratio=ratio,
                rcl_size=RCL_SIZE,
                starts=STARTS,
                seed=54321 + ratio_idx * 100 + plus,
                solution_file_suffix=f"_mix_r{ratio:.4f}_T{target}",
                # New knobs:
                force_target_chiplets=target,
                exhaustive_construct=True,
            )

            util = _utilization_fields(result, target)

            rows.append({
                'strategy': 'mixed',
                'imc_ratio': ratio,
                'target_chiplets': target,
                'used_chiplets': result.get('num_chiplets', 0),
                'total_cycles': result.get('total_time', 0),
                'status': result.get('status', 'unknown'),
                'violations_total': sum(result.get('violations', {}).values())
                                    if 'violations' in result else 0,
                'solve_time_s': result.get('solve_time', 0.0),
                **util,
            })

            print(f"[Mixed r={ratio:.6f}] → {rows[-1]['used_chiplets']} used chiplets, "
                  f"{rows[-1]['total_cycles']} cycles, "
                  f"util={rows[-1]['utilization_pct']:.2f}%, "
                  f"violations={rows[-1]['violations_total']}, "
                  f"solve_time={rows[-1]['solve_time_s']:.2f}s")

    df = pd.DataFrame(rows)
    return df


def plot_separated(df_sep: pd.DataFrame, out_png: str = "plot_separated.png") -> None:
    """Plot used chiplets vs total cycles for separated strategy."""
    if df_sep.empty:
        print("No separated data to plot.")
        return

    df_sep = df_sep.sort_values('used_chiplets')
    plt.figure(figsize=(7, 5))
    plt.plot(df_sep['used_chiplets'], df_sep['total_cycles'], marker='o')
    for _, r in df_sep.iterrows():
        plt.annotate(f"T{int(r['target_chiplets'])}",
                     (r['used_chiplets'], r['total_cycles']),
                     textcoords="offset points", xytext=(5, 5), fontsize=8)
    plt.xlabel("Used chiplets")
    plt.ylabel("Total cycles (makespan)")
    plt.title("Separated strategy: chiplets vs cycles")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    print(f"Saved: {out_png}")


def plot_mixed(df_mixed: pd.DataFrame, out_png: str = "plot_mixed.png") -> None:
    """Plot used chiplets vs total cycles for mixed strategy, one curve per ratio."""
    if df_mixed.empty:
        print("No mixed data to plot.")
        return

    plt.figure(figsize=(8, 6))
    for ratio, sub in df_mixed.groupby('imc_ratio'):
        sub = sub.sort_values('used_chiplets')
        plt.plot(sub['used_chiplets'], sub['total_cycles'], marker='o', label=f"ratio={ratio:.3f}")
        for _, r in sub.iterrows():
            plt.annotate(f"T{int(r['target_chiplets'])}",
                         (r['used_chiplets'], r['total_cycles']),
                         textcoords="offset points", xytext=(5, 5), fontsize=7)

    plt.xlabel("Used chiplets")
    plt.ylabel("Total cycles (makespan)")
    plt.title("Mixed strategy: chiplets vs cycles (per IMC ratio)")
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    print(f"Saved: {out_png}")


def main():
    print("Running chiplet sweep experiments")
    print("=" * 80)

    ensure_traffic_exists(TRAFFIC_FILE)

    # Build the problem once (the solver makes copies internally)
    problem = ChipletProblemGRASP(TRAFFIC_FILE)

    # -------- Mixed sweep --------
    df_mix = run_mixed_sweep(problem, ratios=MIXED_RATIOS)
    df_mix.to_csv("results_mixed.csv", index=False)
    print("\nMixed sweep summary:\n", df_mix)
    plot_mixed(df_mix, "plot_mixed.png")

    # Per-target average utilization (mixed, averaged across ratios)
    mix_avg = (df_mix.groupby('target_chiplets', as_index=False)['utilization_pct']
                     .mean()
                     .rename(columns={'utilization_pct': 'avg_utilization_pct'}))
    mix_avg.to_csv("avg_utilization_by_target_mixed.csv", index=False)
    print("Saved: avg_utilization_by_target_mixed.csv")

    # -------- Separated sweep --------
    df_sep = run_separated_sweep(problem)
    df_sep.to_csv("results_separated.csv", index=False)
    print("\nSeparated sweep summary:\n", df_sep)
    plot_separated(df_sep, "plot_separated.png")

    # Per-target average utilization (separated)
    sep_avg = (df_sep.groupby('target_chiplets', as_index=False)['utilization_pct']
                     .mean()
                     .rename(columns={'utilization_pct': 'avg_utilization_pct'}))
    sep_avg.to_csv("avg_utilization_by_target_separated.csv", index=False)
    print("Saved: avg_utilization_by_target_separated.csv")

    print("\nDone. CSVs: results_separated.csv, results_mixed.csv,"
          " avg_utilization_by_target_separated.csv, avg_utilization_by_target_mixed.csv |"
          " Figures: plot_separated.png, plot_mixed.png")


if __name__ == "__main__":
    main()
