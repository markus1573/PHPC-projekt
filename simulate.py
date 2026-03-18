from os.path import join
import sys
import time
import concurrent.futures
import numpy as np


def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


def jacobi(u, interior_mask, max_iter, atol=1e-6):
    u = np.copy(u)
    u_inner = u[1:-1, 1:-1]
    
    for i in range(max_iter):
        # 1. Compute the new grid
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        
        # 2. Periodic convergence check (only calculate delta every 50 iterations)
        if i % 50 == 0:
            u_current_interior = u_inner[interior_mask]
            u_new_interior = u_new[interior_mask]
            delta = np.abs(u_current_interior - u_new_interior).max()
            
            np.copyto(u_inner, u_new, where=interior_mask)
            
            if delta < atol:
                break
        else:
            # 3. Fast update for the other 49 iterations
            np.copyto(u_inner, u_new, where=interior_mask)
            
    return u


def simulate_building(args):
    """Helper function to unpack arguments for the ProcessPoolExecutor"""
    i, u0, mask, max_iter, atol = args
    u_result = jacobi(u0, mask, max_iter, atol)
    return i, u_result


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
    LOAD_DIR = 'modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    if len(sys.argv) < 2:
        N = 5
    else:
        N = int(sys.argv[1])
    building_ids = building_ids[:N]

    # Load floor plans
    all_u0 = np.empty((N, 514, 514))
    all_interior_mask = np.empty((N, 512, 512), dtype='bool')
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    # Run jacobi iterations for each floor plan
    MAX_ITER = 20_000
    ABS_TOL = 1e-4
    all_u = np.empty_like(all_u0)

    # Package tasks for multiprocessing
    tasks = [
        (i, all_u0[i], all_interior_mask[i], MAX_ITER, ABS_TOL) 
        for i in range(N)
    ]

    # Execute calculations in parallel across multiple CPU cores
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i, u_result in executor.map(simulate_building, tasks):
            all_u[i] = u_result

    # Print summary statistics in CSV format
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))  # CSV header
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))
        
    # Print the final execution time
    print(f"\n--- Process finished in {time.time() - start_time:.2f} seconds ---")