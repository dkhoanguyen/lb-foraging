import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from utility_simulator import UtilitySimulator

np.random.seed(42)

# Parameters
actions_per_agent = [
    [0, 1],  # Agent 0
    [0, 1],   # Agent 1
    [0, 1, 2],  # Agent 2
    [0, 1, 2],   # Agent 3
    [0, 1, 2, 3],  # Agent 4
]
num_agents = len(actions_per_agent)
T = 0.1
iterations = 10
samples_per_joint_action = 10
alpha = 0.1
num_mc_samples = 10

# Initialize utility simulator
sim = UtilitySimulator(actions_per_agent, seed=42)

# Initialize distributions q[i] is a probability vector over actions_per_agent[i]
q = [np.ones(len(a)) / len(a) for a in actions_per_agent]

def estimate_G(samples_per_joint_action):
    joint_actions = list(product(*actions_per_agent))
    G_est = {}
    for joint_action in joint_actions:
        samples = [sim.rollout(joint_action) for _ in range(samples_per_joint_action)]
        G_est[joint_action] = np.mean(samples)
    return G_est

def compute_entropy(p):
    return -np.sum(p * np.log(p + 1e-12))

def monte_carlo_expected_util(q_all, i_agent, action_idx, num_samples):
    total = 0.0
    for _ in range(num_samples):
        joint_action = [
            np.random.choice(actions_per_agent[j], p=q_all[j]) for j in range(num_agents)
        ]
        joint_action[i_agent] = actions_per_agent[i_agent][action_idx]
        total += sim.rollout(joint_action)
    return total / num_samples

def monte_carlo_total_utility(q_all, num_samples):
    total = 0.0
    for _ in range(num_samples):
        joint_action = [
            np.random.choice(actions_per_agent[j], p=q_all[j]) for j in range(num_agents)
        ]
        total += sim.rollout(joint_action)
    return total / num_samples

def nearest_newton_update(q_all, i_agent, T, alpha, num_samples):
    q_self = q_all[i_agent]
    new_q = np.copy(q_self)
    entropy_q = compute_entropy(q_self)
    E_G = monte_carlo_total_utility(q_all, num_samples)

    for i in range(len(q_self)):
        E_G_given_i = monte_carlo_expected_util(q_all, i_agent, i, num_samples)
        delta_E = E_G_given_i - E_G
        grad = entropy_q + np.log(q_self[i] + 1e-12) + (delta_E) / T
        new_q[i] += alpha * q_self[i] * grad

    new_q = np.maximum(new_q, 1e-8)
    new_q /= np.sum(new_q)
    return new_q

# Data collection
expected_utilities = []
total_entropies = []
objectives = []

# Main optimization loop
for it in range(iterations):
    G = estimate_G(samples_per_joint_action)

    for i_agent in range(num_agents):
        q[i_agent] = nearest_newton_update(q, i_agent, T, alpha, num_mc_samples)

    joint_actions = list(product(*actions_per_agent))
    joint_probs = []
    for joint_action in joint_actions:
        prob = 1.0
        for i, a in enumerate(joint_action):
            a_idx = actions_per_agent[i].index(a)
            prob *= q[i][a_idx]
        joint_probs.append(prob)

    expected_utility = sum(prob * G[joint_action] for prob, joint_action in zip(joint_probs, joint_actions))
    entropy_total = sum(compute_entropy(qi) for qi in q)
    objective = expected_utility + T * entropy_total

    expected_utilities.append(expected_utility)
    total_entropies.append(entropy_total)
    objectives.append(objective)

    print(f"Iteration {it}:")
    for i, qi in enumerate(q):
        print(f"  q{i+1} = {qi.round(3)}")
    print(f"  Expected Utility = {expected_utility:.4f}")
    print(f"  Total Entropy = {entropy_total:.4f}")
    print(f"  Objective = {objective:.4f}\n")

# Final output
print("Final optimized distributions:")
for i, qi in enumerate(q):
    print(f"q{i+1}: {qi}")

# Optional: Plotting
# plt.figure()
# plt.plot(expected_utilities, label='Expected Utility')
# plt.plot(total_entropies, label='Total Entropy')
# plt.plot(objectives, label='Objective')
# plt.xlabel('Iteration')
# plt.ylabel('Value')
# plt.title('Optimization Progress')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
