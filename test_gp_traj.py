import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist
import time

def squared_exp_kernel(t1, t2, sigma=0.05, length_scale=0.1):
    dists = cdist(t1[:, None], t2[:, None], 'sqeuclidean')
    return sigma**2 * np.exp(-0.5 * dists / length_scale**2)

def generate_gp_around_reference_with_fixed_start(ref_traj, N=10, sigma=0.5, length_scale=0.75, seed=42):
    """
    ref_traj: ndarray of shape (M, 2) – reference 2D trajectory over M time steps
    N: number of samples
    """
    np.random.seed(seed)
    M = ref_traj.shape[0]
    t_query = np.linspace(0, 1, M)
    
    t_obs = np.array([0.0])  # Fix only the start point
    fixed_idx = 0  # Index for fixed start

    trajectories = np.zeros((N, M, 2))
    log_probs = np.zeros(N)

    for d in range(2):
        y_obs = np.array([ref_traj[fixed_idx, d]])  # fixed start value

        # Build GP conditioned on fixed start
        K_oo = squared_exp_kernel(t_obs, t_obs, sigma, length_scale) + 1e-6 * np.eye(1)
        K_oq = squared_exp_kernel(t_obs, t_query, sigma, length_scale)
        K_qq = squared_exp_kernel(t_query, t_query, sigma, length_scale) + 1e-6 * np.eye(M)

        K_oo_inv = np.linalg.inv(K_oo)
        mu_post = K_oq.T @ K_oo_inv @ y_obs
        cov_post = K_qq - K_oq.T @ K_oo_inv @ K_oq

        # Add the deviation from the reference (excluding first point)
        delta = np.random.multivariate_normal(mu_post, cov_post, N)
        for i in range(N):
            delta[i, fixed_idx] = 0.0  # enforce zero deviation at t=0
        trajectories[:, :, d] = ref_traj[:, d] + delta

    # Log probs
    for i in range(N):
        prob_x = multivariate_normal(mean=ref_traj[:, 0], cov=cov_post, allow_singular=True).logpdf(trajectories[i, :, 0])
        prob_y = multivariate_normal(mean=ref_traj[:, 1], cov=cov_post, allow_singular=True).logpdf(trajectories[i, :, 1])
        log_probs[i] = prob_x + prob_y

    probs = np.exp(log_probs - np.max(log_probs))
    probs /= np.sum(probs)

    return t_query, trajectories, probs

# Create a curved reference trajectory (e.g., sine wave)
# Define start and end points
A = np.array([0.0, 0.0])
B = np.array([1.0, 1.0])
M = 20  # Number of time steps
t_query = np.linspace(0, 1, M)

# Linear interpolation for reference trajectory
ref_traj = np.outer(1 - t_query, A) + np.outer(t_query, B)

# Sample
# start = time.time()
t_query, trajs, probs = generate_gp_around_reference_with_fixed_start(ref_traj, N=20)
# print(time.time() - start)

# Plot
plt.figure(figsize=(8,6))
for i in range(trajs.shape[0]):
    plt.plot(trajs[i,:,0], trajs[i,:,1], alpha=0.6, label=f"Traj {i+1}, p={probs[i]:.2f}")
plt.plot(ref_traj[:,0], ref_traj[:,1], 'k--', label="Reference Trajectory", linewidth=2)
plt.title("GP Samples Around a Reference Trajectory")
plt.xlabel("X"); plt.ylabel("Y"); plt.grid(); plt.axis("equal")
plt.legend()
plt.show()
