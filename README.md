# Multichip Traffic Optimization

A combinatorial optimization framework for chiplet assignment and task scheduling in multi-chiplet systems, specifically designed for neural network workloads like GPT2 transformers. The framework enforces **unicast communication** (no multicasting) to ensure realistic hardware constraints.

## Overview

This system solves the problem of optimally assigning computational tasks to chiplets while respecting hardware constraints like PE (Processing Element) capacity, inter-chiplet communication bandwidth, task dependencies, and **preventing source PEs from multicasting** (sending to multiple destinations simultaneously).

## Key Components

### Core Framework
- **`chiplet_framework.py`** - Main optimization framework using PuLP ILP solver
- **`validate_solution.py`** - Standalone solution validator for constraint checking

### Test Scripts
- **`test_small.py`** - Tests small GPT2 transformer workload (399 tasks, 60s timeout)
- **`test_large.py`** - Tests large GPT2 transformer workload (1338 tasks, 300s timeout)

### Data Files
- **`gpt2_transformer_small.txt`** - Small dataset (399 tasks, 20x20 NoC)
- **`gpt2_transformer.txt`** - Large dataset (1338 tasks, 20x20 NoC)
- **`demo_traffic.csv`** - Sample traffic data in CSV format

### Solution Files
- **`gpt2_transformer_small_ilp_solution.json`** - Unicast solution for small dataset (134 cycles, 6 chiplets)

## Problem Constraints

The optimization considers these constraints:

1. **Task Assignment** - Each task assigned to exactly one chiplet
2. **Chiplet Usage** - Chiplets activated only if tasks assigned to them
3. **Task Dependencies** - Dependent tasks execute in correct order with proper timing
4. **Chiplet Capacity** - Max 32 unique PEs per chiplet
5. **PE Exclusivity** - Each PE belongs to at most one chiplet
6. **Inter-chiplet Communication** - Large transfers (>8192B) take 2 cycles, small transfers take 1 cycle
7. **Time Bounds** - Total time covers all task completion times
8. **🚫 No Multicasting** - Source PEs cannot send to multiple destinations simultaneously (unicast only)

## Usage

### Running Optimization

```bash
# Small dataset with NoMulticastingConstraint (60s timeout, max 6 chiplets)
python test_small.py

# Large dataset with NoMulticastingConstraint (300s timeout, max 12 chiplets)  
python test_large.py
```

### Validating Solutions

```bash
# Validate a solution file
python validate_solution.py <solution_file.json> <data_file.txt>

# Examples:
python validate_solution.py gpt2_transformer_small_ilp_solution.json gpt2_transformer_small.txt
python validate_solution.py gpt2_transformer_ilp_solution.json gpt2_transformer.txt
```

### Direct Framework Usage

```python
from chiplet_framework import *

# Load problem
problem = ChipletProblem('gpt2_transformer_small.txt')

# Add constraints
problem.add_constraint(TaskAssignmentConstraint())
problem.add_constraint(ChipletUsageConstraint()) 
problem.add_constraint(TaskDependencyConstraint())
problem.add_constraint(ChipletCapacityConstraint(max_pes=32))
problem.add_constraint(PEExclusivityConstraint())
problem.add_constraint(InterChipletCommConstraint(bandwidth=8192))
problem.add_constraint(TimeBoundsConstraint())
problem.add_constraint(NoMulticastingConstraint())  # Enforce unicast only

# Solve with timeout
solution = problem.solve(timeout=60, max_chiplets=6, save_solution_file=True)
```

## Data Format

Traffic data files use tab-separated format:
```
task_id	source_pe	dest_pe	data_size	wait_ids
1	0	1	16384	None
2	0	2	16384	None  
3	1	2	8192	1,2
```

- **task_id** - Unique task identifier
- **source_pe** - Source processing element ID
- **dest_pe** - Destination processing element ID  
- **data_size** - Data transfer size in bytes
- **wait_ids** - Comma-separated list of prerequisite task IDs (or None)

## Solution Format

Solutions are saved as JSON files containing:
- **metadata** - Solver info, timing, status
- **pe_assignments** - PE-to-chiplet mapping
- **schedule** - Task execution per cycle and chiplet

## Dependencies

- Python 3.x
- pandas
- PuLP (for ILP solving)

## Current Status

### Unicast Implementation (NoMulticastingConstraint)
- **Small dataset (399 tasks)**: ✅ 6 chiplets, 134 cycles (timeout_feasible) - All constraints satisfied
- **Large dataset (1338 tasks)**: ❌ Infeasible within 300s timeout due to complexity (11,769 ordering constraints)

### Branch Information
- **`main` branch**: Current unicast implementation with NoMulticastingConstraint
- **`multicast` branch**: Legacy implementation allowing multicasting (faster but unrealistic)

The unicast constraint dramatically increases problem complexity but provides realistic hardware modeling.