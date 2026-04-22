from os.path import join
import sys
import time
import concurrent.futures
import numpy as np
import os


def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


def jacobi(u, interior_mask, max_iter, atol=1e-6):
    u = np.copy(u)

    for i in range(max_iter):
        # Compute average of left, right, up and down neighbors, see eq. (1)
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior

        if delta < atol:
            break
    return u


def simulate_chunk(chunk_tasks):
    """
    Simulates a chunk of multiple building tasks in a single worker process.
    chunk_tasks is a list of tuples: (i, u0, mask, max_iter, atol)
    Returns a list of tuples: (i, u_result)
    """
    results = []
    for args in chunk_tasks:
        i, u0, mask, max_iter, atol = args
        u_result = jacobi(u0, mask, max_iter, atol)
        results.append((i, u_result))
    return results


def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = np.sum(u_interior > 18) / u_interior.size * 100
    pct_below_15 = np.sum(u_interior < 15) / u_interior.size * 100
    return {
        'mean_temp': mean_temp,
        'std_temp': std_temp,
        'pct_above_18': pct_above_18,
        'pct_below_15': pct_below_15,
    }


if __name__ == '__main__':
    # Start the timer!
    start_time = time.time()

    # Load data
    N = 10
    WORKERS = 4
    LOAD_DIR = 'modified_swiss_dwellings/'
    
    if len(sys.argv) > 1:
        N = int(sys.argv[1])
    if len(sys.argv) > 2:
        WORKERS = int(sys.argv[2])
    if len(sys.argv) > 3:
        LOAD_DIR = sys.argv[3]

    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()
        
    building_ids = building_ids[:N]

    # Load floor plans
    all_u0 = np.empty((N, 514, 514))
    all_interior_mask = np.empty((N, 512, 512), dtype='bool')
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    # Run jacobi iterations for each floor plan
    MAX_ITER = 10_000
    ABS_TOL = 1e-4
    all_u = np.empty_like(all_u0)

    # Package tasks and group them statically for workers
    tasks = [
        (i, all_u0[i], all_interior_mask[i], MAX_ITER, ABS_TOL) 
        for i in range(N)
    ]
    
    # Static allocation: split the task list into WORKERS number of equal chunks (or roughly equal)
    chunk_size = int(np.ceil(N / WORKERS))
    task_chunks = [tasks[x:x+chunk_size] for x in range(0, N, chunk_size)]

    # Execute calculations in parallel across multiple CPU cores
    with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as executor:
        for chunk_results in executor.map(simulate_chunk, task_chunks):
            for i, u_result in chunk_results:
                all_u[i] = u_result

    # Print summary statistics in CSV format
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        
    # Print the final execution time
    print(f"--- Process finished in {time.time() - start_time:.2f} seconds ---")