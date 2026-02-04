import numpy as np
import numpy.random as rnd
from numba import njit, prange
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
import time

"""
A file to generate the histogram and other statistics of distances between all the points in two spheres
"""
t0 = time.perf_counter()
# problem set up: variables
d = 10 # diameter of the spheres
r = d/2 # radius of the spheres
D = 3 # dimension of the spheres
L = 1*d # distance between the centers of the spheres
n = 1000 # number of points in each sphere

# generate coordinates of points in each sphere
def generate_coords_in_sphere_1(center, radius, n, D):
    """
    center: list or array of length D
    """
    coords = np.zeros((n, D))
    successes = 0
    while successes < n:
        point = rnd.uniform(-radius, radius, D)
        if np.linalg.norm(point) <= radius:
            coords[successes] = point
            successes += 1
    coords += np.asarray(center) # adds center to every row
    return coords
# alternate rng
def generate_coords_in_sphere_2(center, radius, n, D, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    center = np.asarray(center, dtype=np.float64)

    pts = np.empty((n, D), dtype=np.float64)
    filled = 0
    r2 = radius * radius

    while filled < n:
        m = (n - filled) * 2 + 64  # oversample a bit
        cand = rng.uniform(-radius, radius, size=(m, D))
        keep = np.sum(cand * cand, axis=1) <= r2
        acc = cand[keep]
        take = min(acc.shape[0], n - filled)
        pts[filled:filled + take] = acc[:take]
        filled += take

    return pts + center

# sphere 1 centered at (r, r, ..., r)
center_1 = [r]*D
coords_1 = generate_coords_in_sphere_2(center_1, r, n, D)
# sphere 2 centered at (L+r, r, ..., r)
center_2 = [L + r] + [r]*(D-1)
coords_2 = generate_coords_in_sphere_2(center_2, r, n, D)

# find distance between all points in coords_1 and 2 and store in a numpy array. us njit to run the for loops
@njit
def distances(coords_1, coords_2):
    # extract D and n
    n, D = coords_1.shape
    # create distances array 
    distances = np.zeros(n * n)
    for i in range(n):
        # temp_i = coords_1[i]
        for j in range(n):
            d_2 = 0
            for k in range(D):
                diff = coords_1[i, k] - coords_2[j, k]
                d_2 += diff * diff
            distances[i * n + j] = math.sqrt(d_2)
    return distances
def L_hat_from_distances(dis, r, D):
    mean_d2 = np.mean(dis * dis)
    offset = (2.0 * D / (D + 2.0)) * (r * r)
    L2_hat = mean_d2 - offset
    return math.sqrt(L2_hat) if L2_hat > 0 else 0.0, mean_d2, offset

# plot as histogram the distances

if __name__ == "__main__":
    dis = distances(coords_1, coords_2)
    print(f"time taken = {time.perf_counter() - t0:.2f}s")
    print(f"size of distance array = {dis.size}")

    L_hat = L_hat_from_distances(dis, r, D)
    print(f"Estimated L_hat = {L_hat[0]:.4f}")
    
    # plt.hist(dis, bins=1000, density=True)
    # plt.title(f"Histogram of distances between points in two spheres of radius {r} in {D}D, separated by {L}, with {n} points each, overlaid with normal distribution.")
    # plt.xlabel("Distance")
    # plt.ylabel("Density")
    
    # xx = np.linspace(dis.min(), dis.max(), 200)
    # mu, sigma = np.mean(dis), np.std(dis)
    # plt.plot(xx, norm.pdf(xx, mu, sigma), color='red', lw=2)
    # plt.savefig(f"histogram_distances_spheres_r{r}_D{D}_L{L}_n{n}.png", dpi=300)
    # plt.show()