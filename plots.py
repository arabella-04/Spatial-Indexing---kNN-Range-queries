import matplotlib.pyplot as plt
import time
import random
import pandas as pd
import numpy as np
from hw1 import knn_linear_search, range_query_linear_search
from hw1 import build_grid_index, knn_grid_index, range_query_grid_index
from hw1 import build_kd_tree, knn_kd_tree, range_query_kd_tree

def compare_results(a, b):
    if ((a is None) and (b is None)) or ((len(a)==0) and (len(b)==0)):
        return True
    if (a is None) or (b is None) or (len(a) == 0) or (len(b) == 0):
        return False
    ids_a = set(a[:, 0])
    ids_b = set(b[:, 0])
    dist_a = set(np.round(a[:, 4].astype(float), 6))
    dist_b = set(np.round(b[:, 4].astype(float), 6))
    return (ids_a == ids_b) and (dist_a == dist_b)

def plot_knn_linear(df, N, K, num_trials=10):
    execution_times = {k: [] for k in K}
    for n in N:        
        for k in K:
            trial_times = []
            sampled_df = df.sample(n=n, random_state=7).reset_index(drop=True)
            for trial in range(num_trials):
                target_id = random.choice(sampled_df['@id'].tolist())
                data = sampled_df.to_numpy()
                start_time = time.perf_counter()
                neighbors = knn_linear_search(data, target_id, k)
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                trial_times.append(elapsed_time)
            
            avg_time = np.nanmean(trial_times) if any(~np.isnan(t) for t in trial_times) else None
            execution_times[k].append(avg_time)
    
    plot_data = pd.DataFrame(execution_times, index=N)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    for k in K:
        if plot_data[k].notnull().any():
            plt.plot(plot_data.index, plot_data[k], marker='o', label=f'k={k}')
    
    plt.xlabel('Dataset Size (N)')
    plt.ylabel('Average Execution Time (seconds)')
    plt.title('KNN Linear Search Execution Time')
    plt.xscale('linear')
    plt.yscale('linear')
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()

def plot_range_query_linear(df, N, R, num_trials=10):
    execution_times = {r: [] for r in R}
    for n in N:        
        for r in R:
            trial_times = []
            sampled_df = df.sample(n=n, random_state=7).reset_index(drop=True)
            for trial in range(num_trials):
                target_id = random.choice(sampled_df['@id'].tolist())
                data = sampled_df.to_numpy()
                start_time = time.perf_counter()
                neighbors = range_query_linear_search(data, target_id, r)
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                trial_times.append(elapsed_time)
            
            avg_time = np.nanmean(trial_times) if any(~np.isnan(t) for t in trial_times) else None
            execution_times[r].append(avg_time)
    
    plot_data = pd.DataFrame(execution_times, index=N)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    for r in R:
        if plot_data[r].notnull().any():
            plt.plot(plot_data.index, plot_data[r], marker='o', label=f'r={r}')
    
    plt.xlabel('Dataset Size (N)')
    plt.ylabel('Average Execution Time (seconds)')
    plt.title('Range query Linear Search Execution Time')
    plt.xscale('linear')
    plt.yscale('linear')
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()

def plot_grid_knn(df, N, K, num_trials=10, cell_size = 0.05):
    execution_times = {k: [] for k in K}
    for n in N:        
        for k in K:
            trial_times = []
            sampled_df = df.sample(n=n, random_state=7).reset_index(drop=True)
            for trial in range(num_trials):
                target_poi = sampled_df.sample(n=1).iloc[0]
                data = sampled_df.to_numpy()
                grid = build_grid_index(data, cell_size)
                start_time = time.perf_counter()
                neighbors_grid = knn_grid_index(grid, target_poi, k, cell_size)
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                trial_times.append(elapsed_time)
            avg_time = np.nanmean(trial_times) if any(~np.isnan(t) for t in trial_times) else None
            execution_times[k].append(avg_time)
    
    plot_data = pd.DataFrame(execution_times, index=N)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    for k in K:
        if plot_data[k].notnull().any():
            plt.plot(plot_data.index, plot_data[k], marker='o', label=f'k={k}')
    
    plt.xlabel('Dataset Size (N)')
    plt.ylabel('Average Execution Time (seconds)')
    plt.title(f'KNN Grid Index Execution Time for cell size = {cell_size}')
    plt.xscale('linear')
    plt.yscale('linear')
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()

def plot_grid_range_query(df, N, R, num_trials=10, cell_size = 0.05):
    execution_times = {r: [] for r in R}
    for n in N:        
        for r in R:
            trial_times = []
            sampled_df = df.sample(n=n, random_state=7).reset_index(drop=True)
            for trial in range(num_trials):
                target_poi = sampled_df.sample(n=1).iloc[0]
                data = sampled_df.to_numpy()
                grid = build_grid_index(data, cell_size)
                start_time = time.perf_counter()
                neighbors = range_query_grid_index(grid, target_poi, r, cell_size)
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                trial_times.append(elapsed_time)
            avg_time = np.nanmean(trial_times) if any(~np.isnan(t) for t in trial_times) else None
            execution_times[r].append(avg_time)
    
    plot_data = pd.DataFrame(execution_times, index=N)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    for r in R:
        if plot_data[r].notnull().any():
            plt.plot(plot_data.index, plot_data[r], marker='o', label=f'r={r}')
    
    plt.xlabel('Dataset Size (N)')
    plt.ylabel('Average Execution Time (seconds)')
    plt.title(f'Range query Grid Index Execution Time for cell size = {cell_size}')
    plt.xscale('linear')
    plt.yscale('linear')
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()

def plot_grid_knn_cell_size(df, N, cell_sizes, num_trials=10, k = 5):
    execution_times = {cell_size: [] for cell_size in cell_sizes}
    for n in N:        
        for cell_size in cell_sizes:
            trial_times = []
            sampled_df = df.sample(n=n, random_state=7).reset_index(drop=True)
            for trial in range(num_trials):
                target_poi = sampled_df.sample(n=1).iloc[0]
                data = sampled_df.to_numpy()
                grid = build_grid_index(data, cell_size)
                start_time = time.perf_counter()
                neighbors_grid = knn_grid_index(grid, target_poi, k, cell_size)
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                trial_times.append(elapsed_time)
            avg_time = np.nanmean(trial_times) if any(~np.isnan(t) for t in trial_times) else None
            execution_times[cell_size].append(avg_time)
    
    plot_data = pd.DataFrame(execution_times, index=N)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    for cell_size in cell_sizes:
        if plot_data[cell_size].notnull().any():
            plt.plot(plot_data.index, plot_data[cell_size], marker='o', label=f'cell size={cell_size}')
    
    plt.xlabel('Dataset Size (N)')
    plt.ylabel('Average Execution Time (seconds)')
    plt.title(f'KNN Grid Index Execution Time for k = {k}')
    plt.xscale('linear')
    plt.yscale('linear')
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()

def plot_grid_range_query_cell_size(df, N, cell_sizes, num_trials = 10, r = 0.1):
    execution_times = {cell_size: [] for cell_size in cell_sizes}
    for n in N:        
        for cell_size in cell_sizes:
            trial_times = []
            sampled_df = df.sample(n=n, random_state=7).reset_index(drop=True)
            for trial in range(num_trials):
                target_poi = sampled_df.sample(n=1).iloc[0]
                data = sampled_df.to_numpy()
                grid = build_grid_index(data, cell_size)
                start_time = time.perf_counter()
                neighbors_grid = range_query_grid_index(grid, target_poi, r, cell_size)
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                trial_times.append(elapsed_time)
            avg_time = np.nanmean(trial_times) if any(~np.isnan(t) for t in trial_times) else None
            execution_times[cell_size].append(avg_time)
    
    plot_data = pd.DataFrame(execution_times, index=N)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    for cell_size in cell_sizes:
        if plot_data[cell_size].notnull().any():
            plt.plot(plot_data.index, plot_data[cell_size], marker='o', label=f'cell size={cell_size}')
    
    plt.xlabel('Dataset Size (N)')
    plt.ylabel('Average Execution Time (seconds)')
    plt.title(f'Range Query Grid Index Execution Time for r = {r}' )
    plt.xscale('linear')
    plt.yscale('linear')
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()

def plot_grid_vs_linear_knn(df, N, num_trials = 10, k = 5, cell_size = 0.05):
    execution_times = {'grid':[], 'linear':[]}
    res = []
    for n in N:        
        trial_times_grid = []
        trial_times_linear = []
        sampled_df = df.sample(n=n, random_state=7).reset_index(drop=True)
        for trial in range(num_trials):
            target_poi = sampled_df.sample(n=1).iloc[0]
            data = sampled_df.to_numpy()
            grid = build_grid_index(data, cell_size)
            start_time = time.perf_counter()
            neighbors_grid = knn_grid_index(grid, target_poi, k, cell_size)
            end_time = time.perf_counter()
            elapsed_time_grid = end_time - start_time
            trial_times_grid.append(elapsed_time_grid)
            target_id = target_poi[0]
            start_time = time.perf_counter()
            neighbors_linear = knn_linear_search(data, target_id, k)
            end_time = time.perf_counter()
            elapsed_time_linear = end_time - start_time
            trial_times_linear.append(elapsed_time_linear)
            res.append(compare_results(neighbors_linear, neighbors_grid))
        avg_time_grid = np.nanmean(trial_times_grid) if any(~np.isnan(t) for t in trial_times_grid) else None
        execution_times['grid'].append(avg_time_grid)
        avg_time_linear = np.nanmean(trial_times_linear) if any(~np.isnan(t) for t in trial_times_linear) else None
        execution_times['linear'].append(avg_time_linear)

    if all(res):
        print('Grid Index KNN returns the exact same POIs as Linear Search')
    else:
        print('Grid Index KNN does not return the exact same POIs as Linear Search')
    plot_data = pd.DataFrame(execution_times, index=N)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    for key in execution_times.keys():
        if plot_data[key].notnull().any():
            plt.plot(plot_data.index, plot_data[key], marker='o', label=f'index = {key}')
    
    plt.xlabel('Dataset Size (N)')
    plt.ylabel('Average Execution Time (seconds)')
    plt.title('KNN Linear vs. Grid Index Execution Time for cell size = '+ str(cell_size)+' , k = '+str(k))
    plt.xscale('linear')
    plt.yscale('linear')
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()

def plot_grid_vs_linear_range_query(df, N, num_trials = 10, r = 0.1, cell_size = 0.05):
    execution_times = {'grid':[], 'linear':[]}
    res = []
    for n in N:        
        trial_times_grid = []
        trial_times_linear = []
        sampled_df = df.sample(n=n, random_state=7).reset_index(drop=True)
        for trial in range(num_trials):
            target_poi = sampled_df.sample(n=1).iloc[0]
            data = sampled_df.to_numpy()
            grid = build_grid_index(data, cell_size)
            start_time = time.perf_counter()
            neighbors_grid = range_query_grid_index(grid, target_poi, r, cell_size)
            end_time = time.perf_counter()
            elapsed_time_grid = end_time - start_time
            trial_times_grid.append(elapsed_time_grid)
            target_id = target_poi[0]
            start_time = time.perf_counter()
            neighbors_linear = range_query_linear_search(data, target_id, r)
            end_time = time.perf_counter()
            elapsed_time_linear = end_time - start_time
            trial_times_linear.append(elapsed_time_linear)
            res.append(compare_results(neighbors_linear, neighbors_grid))
        avg_time_grid = np.nanmean(trial_times_grid) if any(~np.isnan(t) for t in trial_times_grid) else None
        execution_times['grid'].append(avg_time_grid)
        avg_time_linear = np.nanmean(trial_times_linear) if any(~np.isnan(t) for t in trial_times_linear) else None
        execution_times['linear'].append(avg_time_linear)

    if all(res):
        print('Grid Index Range Query returns the exact same POIs as Linear Search')
    else:
        print('Grid Index Range Query does not return the exact same POIs as Linear Search')
    plot_data = pd.DataFrame(execution_times, index=N)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    for key in execution_times.keys():
        if plot_data[key].notnull().any():
            plt.plot(plot_data.index, plot_data[key], marker='o', label=f'index = {key}')
    
    plt.xlabel('Dataset Size (N)')
    plt.ylabel('Average Execution Time (seconds)')
    plt.title('Range Query Linear vs. Grid Index Execution Time for cell size = '+ str(cell_size)+' , r = '+str(r))
    plt.xscale('linear')
    plt.yscale('linear')
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()

def plot_build_time(df, N, cell_sizes, num_trials=10):
    execution_times = {cell_size: [] for cell_size in cell_sizes}  
    execution_times['kd'] = []  

    for n in N:
        kd_trial_times = []
        grid_trial_times = {cell_size: [] for cell_size in cell_sizes}

        for _ in range(num_trials):
            sampled_df = df.sample(n=n, random_state=7).reset_index(drop=True)
            data = sampled_df.to_numpy()

            start_time = time.perf_counter()
            root = build_kd_tree(data)
            end_time = time.perf_counter()
            kd_trial_times.append(end_time - start_time)

            for cell_size in cell_sizes:
                start_time = time.perf_counter()
                grid = build_grid_index(data, cell_size)
                end_time = time.perf_counter()
                grid_trial_times[cell_size].append(end_time - start_time)

        execution_times['kd'].append(np.mean(kd_trial_times))
        for cell_size in cell_sizes:
            execution_times[cell_size].append(np.mean(grid_trial_times[cell_size]))

    plot_data = pd.DataFrame(execution_times, index=N)

    # Plotting
    plt.figure(figsize=(12, 8))

    plt.plot(plot_data.index, plot_data['kd'], marker='o', linestyle='-', label='KD Tree', color='black')

    for cell_size in cell_sizes:
        plt.plot(plot_data.index, plot_data[cell_size], marker='o', linestyle='--', label=f'Grid (cell_size={cell_size})')

    plt.xlabel('Dataset Size (N)')
    plt.ylabel('Average Build Time (seconds)')
    plt.title('KD Tree vs. Grid Index Build Time (Averaged over Trials)')
    plt.xscale('linear')
    plt.yscale('linear')
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()

def plot_kd_knn(df, N, K, num_trials=10):
    execution_times = {k: [] for k in K}
    for n in N:        
        for k in K:
            trial_times = []
            sampled_df = df.sample(n=n, random_state=7).reset_index(drop=True)
            for trial in range(num_trials):
                target_poi = sampled_df.sample(n=1).iloc[0].to_numpy()
                data = sampled_df.to_numpy()
                root = build_kd_tree(data)
                start_time = time.perf_counter()
                neighbors_kd = knn_kd_tree(root, target_poi, k)
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                trial_times.append(elapsed_time)
            avg_time = np.nanmean(trial_times) if any(~np.isnan(t) for t in trial_times) else None
            execution_times[k].append(avg_time)
    
    plot_data = pd.DataFrame(execution_times, index=N)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    for k in K:
        if plot_data[k].notnull().any():
            plt.plot(plot_data.index, plot_data[k], marker='o', label=f'k={k}')
    
    plt.xlabel('Dataset Size (N)')
    plt.ylabel('Average Execution Time (seconds)')
    plt.title('KNN KD Tree Execution Time')
    plt.xscale('linear')
    plt.yscale('linear')
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()

def plot_kd_range_query(df, N, R, num_trials = 10):
    execution_times = {r: [] for r in R}
    for n in N:        
        for r in R:
            trial_times = []
            sampled_df = df.sample(n=n, random_state=7).reset_index(drop=True)
            for trial in range(num_trials):
                target_poi = sampled_df.sample(n=1).iloc[0].to_numpy()
                data = sampled_df.to_numpy()
                root = build_kd_tree(data)
                start_time = time.perf_counter()
                neighbors_kd = range_query_kd_tree(root, target_poi, r)
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                trial_times.append(elapsed_time)
            avg_time = np.nanmean(trial_times) if any(~np.isnan(t) for t in trial_times) else None
            execution_times[r].append(avg_time)
    
    plot_data = pd.DataFrame(execution_times, index=N)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    for r in R:
        if plot_data[r].notnull().any():
            plt.plot(plot_data.index, plot_data[r], marker='o', label=f'r={r}')
    
    plt.xlabel('Dataset Size (N)')
    plt.ylabel('Average Execution Time (seconds)')
    plt.title('Range Query KD Tree Execution Time')
    plt.xscale('linear')
    plt.yscale('linear')
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()


def plot_kd_vs_linear_knn(df, N, num_trials = 10, k = 5):
    execution_times = {'kd':[], 'linear':[]}
    res = []
    for n in N:        
        trial_times_kd = []
        trial_times_linear = []
        sampled_df = df.sample(n=n, random_state=7).reset_index(drop=True)
        for trial in range(num_trials):
            target_poi = sampled_df.sample(n=1).iloc[0]
            data = sampled_df.to_numpy()
            root = build_kd_tree(data)
            start_time = time.perf_counter()
            neighbors_kd = knn_kd_tree(root, target_poi, k)
            end_time = time.perf_counter()
            elapsed_time_kd = end_time - start_time
            trial_times_kd.append(elapsed_time_kd)
            target_id = target_poi[0]
            start_time = time.perf_counter()
            neighbors_linear = knn_linear_search(data, target_id, k)
            end_time = time.perf_counter()
            elapsed_time_linear = end_time - start_time
            trial_times_linear.append(elapsed_time_linear)
            res.append(compare_results(neighbors_linear, neighbors_kd))
        avg_time_kd = np.nanmean(trial_times_kd) if any(~np.isnan(t) for t in trial_times_kd) else None
        execution_times['kd'].append(avg_time_kd)
        avg_time_linear = np.nanmean(trial_times_linear) if any(~np.isnan(t) for t in trial_times_linear) else None
        execution_times['linear'].append(avg_time_linear)
    if all(res):
        print('KD Tree KNN returns the exact same POIs as Linear Search')
    else:
        print('KD Tree KNN does not return the exact same POIs as Linear Search')
    plot_data = pd.DataFrame(execution_times, index=N)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    for key in execution_times.keys():
        if plot_data[key].notnull().any():
            plt.plot(plot_data.index, plot_data[key], marker='o', label=f'index = {key}')
    
    plt.xlabel('Dataset Size (N)')
    plt.ylabel('Average Execution Time (seconds)')
    plt.title('KNN Linear vs. KD Tree Execution Time for k = '+str(k))
    plt.xscale('linear')
    plt.yscale('linear')
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()

def plot_kd_vs_linear_range_query(df, N, num_trials = 10, r = 0.1):
    execution_times = {'kd':[], 'linear':[]}
    res = []
    for n in N:        
        trial_times_kd = []
        trial_times_linear = []
        sampled_df = df.sample(n=n, random_state=7).reset_index(drop=True)
        for trial in range(num_trials):
            target_poi = sampled_df.sample(n=1).iloc[0]
            data = sampled_df.to_numpy()
            root = build_kd_tree(data)
            start_time = time.perf_counter()
            neighbors_kd = range_query_kd_tree(root, target_poi, r)
            end_time = time.perf_counter()
            elapsed_time_kd = end_time - start_time
            trial_times_kd.append(elapsed_time_kd)
            target_id = target_poi[0]
            start_time = time.perf_counter()
            neighbors_linear = range_query_linear_search(data, target_id, r)
            end_time = time.perf_counter()
            elapsed_time_linear = end_time - start_time
            trial_times_linear.append(elapsed_time_linear)
            res.append(compare_results(neighbors_linear, neighbors_kd))
        avg_time_kd = np.nanmean(trial_times_kd) if any(~np.isnan(t) for t in trial_times_kd) else None
        execution_times['kd'].append(avg_time_kd)
        avg_time_linear = np.nanmean(trial_times_linear) if any(~np.isnan(t) for t in trial_times_linear) else None
        execution_times['linear'].append(avg_time_linear)

    if all(res):
        print('KD Tree Range Query returns the exact same POIs as Linear Search')
    else:
        print('KD Tree Range Query does not return the exact same POIs as Linear Search')

    plot_data = pd.DataFrame(execution_times, index=N)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    for key in execution_times.keys():
        if plot_data[key].notnull().any():
            plt.plot(plot_data.index, plot_data[key], marker='o', label=f'index = {key}')
    
    plt.xlabel('Dataset Size (N)')
    plt.ylabel('Average Execution Time (seconds)')
    plt.title('Range Query Linear vs. KD Tree Execution Time for r = '+str(r))
    plt.xscale('linear')
    plt.yscale('linear')
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()

def plot_kd_vs_grid_knn(df, N, num_trials = 10, k = 5, cell_size = 0.05):
    execution_times = {'kd':[], 'grid':[]}
    res = []
    for n in N:        
        trial_times_kd = []
        trial_times_grid = []
        sampled_df = df.sample(n=n, random_state=7).reset_index(drop=True)
        for trial in range(num_trials):
            target_poi = sampled_df.sample(n=1).iloc[0]
            data = sampled_df.to_numpy()
            root = build_kd_tree(data)
            grid = build_grid_index(data, cell_size)
            start_time = time.perf_counter()
            neighbors_kd = knn_kd_tree(root, target_poi, k)
            end_time = time.perf_counter()
            elapsed_time_kd = end_time - start_time
            trial_times_kd.append(elapsed_time_kd)
            start_time = time.perf_counter()
            neighbors_grid = knn_grid_index(grid, target_poi, k, cell_size)
            end_time = time.perf_counter()
            elapsed_time_grid = end_time - start_time
            trial_times_grid.append(elapsed_time_grid)
            res.append(compare_results(neighbors_grid, neighbors_kd))
        avg_time_kd = np.nanmean(trial_times_kd) if any(~np.isnan(t) for t in trial_times_kd) else None
        execution_times['kd'].append(avg_time_kd)
        avg_time_grid = np.nanmean(trial_times_grid) if any(~np.isnan(t) for t in trial_times_grid) else None
        execution_times['grid'].append(avg_time_grid)

    if all(res):
        print('KD Tree KNN returns the exact same POIs as Grid Index')
    else:
        print('KD Tree KNN does not return the exact same POIs as Grid Index')

    plot_data = pd.DataFrame(execution_times, index=N)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    for key in execution_times.keys():
        if plot_data[key].notnull().any():
            plt.plot(plot_data.index, plot_data[key], marker='o', label=f'index = {key}')
    
    plt.xlabel('Dataset Size (N)')
    plt.ylabel('Average Execution Time (seconds)')
    plt.title('KNN Grid Index vs. KD Tree Execution Time for k = '+str(k) + ' , cell_size = '+str(cell_size))
    plt.xscale('linear')
    plt.yscale('linear')
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()

def plot_kd_vs_grid_range_query(df, N, num_trials = 10, r = 0.1, cell_size = 0.05):
    execution_times = {'kd':[], 'grid':[]}
    res = []
    for n in N:        
        trial_times_kd = []
        trial_times_grid = []
        sampled_df = df.sample(n=n, random_state=7).reset_index(drop=True)
        for trial in range(num_trials):
            target_poi = sampled_df.sample(n=1).iloc[0]
            data = sampled_df.to_numpy()
            root = build_kd_tree(data)
            grid = build_grid_index(data, cell_size)
            start_time = time.perf_counter()
            neighbors_kd = range_query_kd_tree(root, target_poi, r)
            end_time = time.perf_counter()
            elapsed_time_kd = end_time - start_time
            trial_times_kd.append(elapsed_time_kd)
            start_time = time.perf_counter()
            neighbors_grid = range_query_grid_index(grid, target_poi, r, cell_size)
            end_time = time.perf_counter()
            elapsed_time_grid = end_time - start_time
            trial_times_grid.append(elapsed_time_grid)
            res.append(compare_results(neighbors_grid, neighbors_kd))
        avg_time_kd = np.nanmean(trial_times_kd) if any(~np.isnan(t) for t in trial_times_kd) else None
        execution_times['kd'].append(avg_time_kd)
        avg_time_grid = np.nanmean(trial_times_grid) if any(~np.isnan(t) for t in trial_times_grid) else None
        execution_times['grid'].append(avg_time_grid)
    if all(res):
        print('KD Tree Range Query returns the exact same POIs as Grid Index')
    else:
        print('KD Tree Range Query does not return the exact same POIs as Grid Index')

    plot_data = pd.DataFrame(execution_times, index=N)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    for key in execution_times.keys():
        if plot_data[key].notnull().any():
            plt.plot(plot_data.index, plot_data[key], marker='o', label=f'index = {key}')
    
    plt.xlabel('Dataset Size (N)')
    plt.ylabel('Average Execution Time (seconds)')
    plt.title('Range Query Grid Index vs. KD Tree Execution Time for r = '+str(r) + ' , cell_size = '+str(cell_size))
    plt.xscale('linear')
    plt.yscale('linear')
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()
