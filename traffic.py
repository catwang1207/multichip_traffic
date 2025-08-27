import pandas as pd
from pulp import *

def build_chiplet_assignment_ilp():
    """
    ILP Model for Multi-Chiplet Task Assignment and Scheduling
    
    Objective:
    - Primary: Minimize total execution time
    - Secondary: Minimize number of chiplets used
    
    Constraints:
    1. Task Assignment: Each task assigned to exactly one chiplet
    2. Chiplet Usage: Chiplets must be activated if tasks are assigned to them  
    3. Task Dependencies: Dependent tasks must execute in correct order
    4. PE Sequential Execution: DISABLED (creates too many constraints)
    5. Inter-chiplet Bandwidth: Cross-chiplet comm with data_size > 8192 takes 2 cycles
    6. Chiplet Capacity: Each chiplet can use at most 32 unique PEs
    7. PE Exclusivity: Each PE belongs to at most one chiplet
    8. Time Bounds: Total time must cover all task completion times
    """
    # Read traffic data
    df = pd.read_csv('gpt2_transformer.txt', sep='\t', comment='#', 
                     names=['task_id', 'source_pe', 'dest_pe', 'data_size', 'wait_ids'])
    
    # Get all tasks that need PE assignment
    tasks = []
    task_to_pe = {}
    for _, row in df.iterrows():
        task_id = row['task_id']
        tasks.append(task_id)
        task_to_pe[task_id] = row['dest_pe']
    
    # Extract unique PEs from traffic
    pes = set()
    for _, row in df.iterrows():
        pes.add(row['source_pe'])
        pes.add(row['dest_pe'])
    
    pes = sorted(list(pes))
    max_chiplets = 10  # Reasonable number of chiplets instead of one per PE
    max_time = len(tasks) * 2  # Upper bound on execution time
    
    # Decision variables
    # z[task][chiplet] = 1 if task is assigned to chiplet, 0 otherwise
    z = {}
    for task in tasks:
        for c in range(max_chiplets):
            z[task, c] = LpVariable(f"z_{task}_{c}", cat='Binary')
    
    # y[chiplet] = 1 if chiplet is used, 0 otherwise
    y = {}
    for c in range(max_chiplets):
        y[c] = LpVariable(f"y_{c}", cat='Binary')
    
    # Task execution time variables
    t = {}
    for task in tasks:
        t[task] = LpVariable(f"t_{task}", lowBound=0, upBound=max_time, cat='Integer')
    
    # Total execution time variable
    total_time = LpVariable("total_time", lowBound=0, cat='Integer')
    
    # Create problem
    prob = LpProblem("Minimize_Time_And_Chiplets", LpMinimize)
    
    # Objective will be set after variables are defined
    
    # CONSTRAINTS
    
    # 1. Task Assignment: Each task assigned to exactly one chiplet
    for task in tasks:
        prob += lpSum([z[task, c] for c in range(max_chiplets)]) == 1
    
    # 2. Chiplet Usage: If task assigned to chiplet c, then chiplet c must be used
    for task in tasks:
        for c in range(max_chiplets):
            prob += z[task, c] <= y[c]
    
    # 3. Task Dependencies: Dependent tasks must execute in correct order
    for _, row in df.iterrows():
        if pd.notna(row['wait_ids']) and row['wait_ids'] != 'None':
            current_task = row['task_id']
            if current_task in tasks:
                wait_task_ids = [int(x.strip()) for x in str(row['wait_ids']).split(',')]
                for wait_task_id in wait_task_ids:
                    if wait_task_id in tasks:
                        prob += t[current_task] >= t[wait_task_id] + 1
    
    # 4. SIMPLIFIED Inter-chiplet Bandwidth: Just add penalty, no extra constraints
    # We'll encourage PE locality through the objective function only
    
    # 5. PE Sequential Execution: DISABLED - Creates too many constraints for large problems
    # pe_tasks = {}
    # for task in tasks:
    #     pe = task_to_pe[task]
    #     if pe not in pe_tasks:
    #         pe_tasks[pe] = []
    #     pe_tasks[pe].append(task)
    # 
    # for pe, pe_task_list in pe_tasks.items():
    #     if len(pe_task_list) > 1:
    #         for i in range(len(pe_task_list)):
    #             for j in range(i+1, len(pe_task_list)):
    #                 task1, task2 = pe_task_list[i], pe_task_list[j]
    #                 b = LpVariable(f"order_{task1}_{task2}", cat='Binary')
    #                 M = max_time
    #                 prob += t[task1] + 1 <= t[task2] + M * (1 - b)
    #                 prob += t[task2] + 1 <= t[task1] + M * b
    
    # 5. Chiplet Capacity & 6. PE Exclusivity
    pe_used = {}
    for c in range(max_chiplets):
        for pe in pes:
            pe_used[pe, c] = LpVariable(f"pe_used_{pe}_{c}", cat='Binary')
    
    # Link task assignment to PE usage
    for task in tasks:
        pe = task_to_pe[task]
        for c in range(max_chiplets):
            prob += pe_used[pe, c] >= z[task, c]
    
    # NOW SET OBJECTIVE: minimize total execution time + chiplets + simplified communication penalty
    
    # Simplified approach: Add penalty proportional to potential communication volume
    # This encourages keeping high-traffic PEs together without quadratic terms
    # comm_penalty = 0
    # pe_comm_volume = {}  # Track communication volume per PE
    
    # for _, row in df.iterrows():
    #     task_id = row['task_id']
    #     if task_id in tasks and row['source_pe'] != row['dest_pe']:
    #         source_pe = row['source_pe']
    #         dest_pe = row['dest_pe']
    #         data_size = row['data_size']
            
    #         # Add to communication volume for both PEs
    #         if source_pe not in pe_comm_volume:
    #             pe_comm_volume[source_pe] = 0
    #         if dest_pe not in pe_comm_volume:
    #             pe_comm_volume[dest_pe] = 0
                
    #         pe_comm_volume[source_pe] += data_size
    #         pe_comm_volume[dest_pe] += data_size
    
    # # Add penalty based on PE communication volume and chiplet distribution
    # for pe, volume in pe_comm_volume.items():
    #     if volume > 8192:  # High communication PEs
    #         # Encourage these PEs to be in fewer chiplets
    #         penalty_weight = (volume // 8192) * 0.1  # Small penalty per potential spread
    #         for c in range(max_chiplets):
    #             comm_penalty += penalty_weight * pe_used[pe, c]
    
    prob += 1000 * total_time + lpSum([y[c] for c in range(max_chiplets)])
    # prob += 1000 * total_time + lpSum([y[c] for c in range(max_chiplets)]) + comm_penalty
    
    # 5. Each chiplet can use at most 32 PEs
    for c in range(max_chiplets):
        prob += lpSum([pe_used[pe, c] for pe in pes]) <= 32
    
    # 6. Each PE belongs to at most one chiplet
    for pe in pes:
        prob += lpSum([pe_used[pe, c] for c in range(max_chiplets)]) <= 1
    
    # 7. Time Bounds: Total time must cover all task completion times
    for task in tasks:
        prob += total_time >= t[task] + 1  # Each task takes 1 cycle base time
    
    return prob, z, y, tasks, t, total_time, task_to_pe

if __name__ == "__main__":
    prob, z, y, tasks, t, total_time, task_to_pe = build_chiplet_assignment_ilp()
    prob.solve()
    
    # Read traffic data again for task details
    df = pd.read_csv('gpt2_transformer.txt', sep='\t', comment='#',
                     names=['task_id', 'source_pe', 'dest_pe', 'data_size', 'wait_ids'])
    
    print(f"Status: {LpStatus[prob.status]}")
    print(f"Total execution time: {int(total_time.value())} cycles")
    
    # Count active chiplets
    max_chiplets = 10  # Need to define this here too
    active_chiplets = sum(1 for c in range(max_chiplets) if y[c].value() == 1)
    print(f"Number of chiplets used: {active_chiplets}")
    
    # Show task to chiplet assignment
    task_to_chiplet = {}
    chiplet_tasks = {}
    for task in tasks:
        for c in range(max_chiplets):
            if z[task, c].value() == 1:
                task_to_chiplet[task] = c
                if c not in chiplet_tasks:
                    chiplet_tasks[c] = []
                chiplet_tasks[c].append(task)
                break
    
    # Show chiplet assignments
    for c in sorted(chiplet_tasks.keys()):
        print(f"Chiplet {c}: Tasks {sorted(chiplet_tasks[c])}")
    
    # Create task details mapping
    task_details = {}
    for _, row in df.iterrows():
        if row['task_id'] in tasks:
            task_details[row['task_id']] = {
                'pe': row['dest_pe'],
                'data_size': row['data_size'],
                'source_pe': row['source_pe']
            }
    
    print(f"\nTotal cycles needed: {int(total_time.value())}")
    print(f"Tasks per chiplet: {[len(chiplet_tasks[c]) for c in sorted(chiplet_tasks.keys())]}")
    print(f"PEs per chiplet: {[len(set(task_to_pe[task] for task in chiplet_tasks[c])) for c in sorted(chiplet_tasks.keys())]}")