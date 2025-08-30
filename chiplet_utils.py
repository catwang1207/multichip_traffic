#!/usr/bin/env python3
"""
Utility functions for chiplet optimization frameworks.
Contains only the necessary functions extracted from the old ILP framework.
"""

import json
import os
import time as time_module


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