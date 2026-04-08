from os.path import join
import sys
import time
import math
from line_profiler import profile
from numba import cuda

import numpy as np


def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


@cuda.jit
def jacobi_kernel(u, u_new, interior_mask):
    r, c = cuda.grid(2)
    
    if r > 0 and r < u.shape[0] - 1 and c > 0 and c < u.shape[1] - 1:
        if interior_mask[r-1, c-1]:
            u_new[r, c] = 0.25 * (u[r-1, c] + u[r+1, c] + u[r, c-1] + u[r, c+1])
        else:
            u_new[r, c] = u[r, c]
    elif r < u.shape[0] and c < u.shape[1]:
        u_new[r, c] = u[r, c]

def jacobi(u, interior_mask, max_iter):
    # Initialize data on device
    d_u = cuda.to_device(u)
    d_u_new = cuda.to_device(u)
    d_int_mask = cuda.to_device(interior_mask)
    
    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(u.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(u.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    # Run iterations
    for i in range(max_iter):
        jacobi_kernel[blockspergrid, threadsperblock](d_u, d_u_new, d_int_mask)
        cuda.synchronize()
        d_u, d_u_new = d_u_new, d_u
        
    return d_u.copy_to_host()


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
    start_time = time.time()
    # Load data
    LOAD_DIR = 'modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    if len(sys.argv) < 2:
        N = 10
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

    all_u = np.empty_like(all_u0)
    for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        u = jacobi(u0, interior_mask, MAX_ITER)
        all_u[i] = u

    # Print summary statistics in CSV format
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))  # CSV header
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))

    print(f"\n--- Process finished in {time.time() - start_time:.2f} seconds ---")