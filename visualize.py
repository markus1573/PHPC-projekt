"""
Visualize a single building before and after Jacobi solving.

Usage:
    python visualize.py [building_id]

If no building_id is given, the first one from building_ids.txt is used.
"""

from os.path import join
import sys
import numpy as np
import matplotlib.pyplot as plt


LOAD_DIR = 'modified_swiss_dwellings/'
MAX_ITER = 20_000
ABS_TOL  = 1e-4


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
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])

        if i % 50 == 0:
            delta = np.abs(u_inner[interior_mask] - u_new[interior_mask]).max()
            np.copyto(u_inner, u_new, where=interior_mask)
            if delta < atol:
                print(f"Converged after {i} iterations (delta={delta:.2e})")
                break
        else:
            np.copyto(u_inner, u_new, where=interior_mask)

    return u


def visualize(bid):
    print(f"Loading building {bid} ...")
    u_before, mask = load_data(LOAD_DIR, bid)

    print("Running Jacobi ...")
    u_after = jacobi(u_before, mask, MAX_ITER, ABS_TOL)

    domain_before = u_before[1:-1, 1:-1]
    wall_mask    = (~mask) & (domain_before != 0)
    visible_mask = mask | wall_mask

    plot_before = np.where(visible_mask, domain_before,          np.nan)
    plot_after  = np.where(visible_mask, u_after[1:-1, 1:-1],   np.nan)

    vmin = 0
    vmax = np.nanmax(plot_before)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color='lightgrey')

    for ax, data, title in zip(
        axes,
        [plot_before, plot_after],
        [f"Building {bid} – Before (initial state)",
         f"Building {bid} – After (Jacobi solved)"],
    ):
        im = ax.imshow(data, origin='upper', cmap=cmap,
                       vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_title(title, fontsize=12)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Temperature (°C)')

    plt.tight_layout()
    out = f"visualizations/building_{bid}_before_after.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved to {out}")
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) >= 2:
        building_id = sys.argv[1]
    else:
        with open(join(LOAD_DIR, 'building_ids.txt')) as f:
            building_id = f.readline().strip()

    visualize(building_id)
