#!/usr/bin/env python3

import pandas as pd
from collections import defaultdict, deque

def analyze_traffic_patterns():
    print("=== Analyzing Traffic Table Patterns ===")
    
    # Load the CSV
    df = pd.read_csv('traffic_table_large.csv')
    print(f"Total tasks: {len(df)}")
    
    # PE usage analysis
    src_counts = df['src_pe'].value_counts()
    dst_counts = df['dest_pe'].value_counts()
    
    print(f"\n=== Top 10 Most Active Source PEs ===")
    for pe, count in src_counts.head(10).items():
        print(f"{pe}: {count} tasks")
    
    print(f"\n=== Top 10 Most Active Destination PEs ===") 
    for pe, count in dst_counts.head(10).items():
        print(f"{pe}: {count} tasks")
    
    # Dependency chain analysis
    print(f"\n=== Dependency Chain Analysis ===")
    
    # Build dependency graph
    dependencies = defaultdict(list)
    task_to_id = {}
    
    for _, row in df.iterrows():
        task_id = row['task_id']
        wait_ids = row['wait_ids']
        task_to_id[task_id] = task_id
        
        if pd.notna(wait_ids) and wait_ids != 'None':
            # Handle multiple dependencies
            if isinstance(wait_ids, str) and ',' in wait_ids:
                deps = [int(x.strip()) for x in wait_ids.split(',')]
            else:
                deps = [int(wait_ids)]
            
            for dep in deps:
                dependencies[dep].append(task_id)
    
    print(f"Total dependency edges: {sum(len(deps) for deps in dependencies.values())}")
    
    # Find longest dependency chains using DFS
    def find_longest_chain_from(start_task):
        visited = set()
        max_depth = 0
        max_path = []
        
        def dfs(task, depth, path):
            nonlocal max_depth, max_path
            if task in visited:
                return
            visited.add(task)
            
            if depth > max_depth:
                max_depth = depth
                max_path = path.copy()
            
            for next_task in dependencies.get(task, []):
                if next_task not in visited:
                    path.append(next_task)
                    dfs(next_task, depth + 1, path)
                    path.pop()
        
        dfs(start_task, 0, [start_task])
        return max_depth, max_path
    
    # Find chains starting from tasks with no dependencies (roots)
    all_tasks = set(df['task_id'])
    tasks_with_deps = set()
    for deps_list in dependencies.values():
        tasks_with_deps.update(deps_list)
    
    root_tasks = all_tasks - tasks_with_deps
    print(f"Root tasks (no dependencies): {len(root_tasks)}")
    
    longest_chain = 0
    longest_path = []
    
    # Sample a few root tasks to find long chains
    for root in list(root_tasks)[:20]:  # Check first 20 roots
        depth, path = find_longest_chain_from(root)
        if depth > longest_chain:
            longest_chain = depth
            longest_path = path
    
    print(f"Longest dependency chain found: {longest_chain} tasks")
    print(f"Chain: {longest_path[:10]}{'...' if len(longest_path) > 10 else ''}")
    
    # Critical path analysis
    critical_pes = set()
    for task_id in longest_path[:20]:  # Check first 20 tasks in longest chain
        task_row = df[df['task_id'] == task_id]
        if not task_row.empty:
            src_pe = task_row.iloc[0]['src_pe']
            critical_pes.add(src_pe)
    
    print(f"\nPEs on critical path: {list(critical_pes)[:10]}")
    
    # Data size analysis
    print(f"\n=== Data Transfer Analysis ===")
    print(f"Average bytes per task: {df['bytes'].mean():.1f}")
    print(f"Max bytes per task: {df['bytes'].max()}")
    print(f"Tasks with bytes = 16384: {len(df[df['bytes'] == 16384])}")
    
    # Communication intensity per PE
    pe_communication = defaultdict(int)
    for _, row in df.iterrows():
        pe_communication[row['src_pe']] += row['bytes']
    
    print(f"\n=== Top Communication-Heavy PEs ===")
    sorted_comm = sorted(pe_communication.items(), key=lambda x: x[1], reverse=True)[:10]
    for pe, total_bytes in sorted_comm:
        print(f"{pe}: {total_bytes:,} bytes total")

if __name__ == "__main__":
    analyze_traffic_patterns()