import numpy as np
import matplotlib.pyplot as plt
from itertools import product

np.random.seed(42)

# Parameters
num_agents = 2
num_actions_per_agent = [2, 2]
T = 0.01
iterations = 10
samples_per_joint_action = 10
alpha = 0.01
num_mc_samples = 10

# Initialize distributions
q = [np.ones(n) / n for n in num_actions_per_agent]

def simulate_utility(actions):
    base_utility = {
        (0, 0): 11,
        (0, 1): 10,
        (1, 0): 10,
        (1, 1): 10
    }
    noise = np.random.normal(0, 0.001)
    value = base_utility.get(tuple(actions), 0)
    return value

def estimate_G(samples_per_joint_action):
    action_spaces = [range(n) for n in num_actions_per_agent]
    joint_actions = list(product(*action_spaces))
    G_est = {}
    for joint_action in joint_actions:
        samples = [simulate_utility(joint_action) for _ in range(samples_per_joint_action)]
        G_est[joint_action] = np.mean(samples)
    return G_est

def compute_entropy(p):
    return -np.sum(p * np.log(p + 1e-12))

def monte_carlo_expected_util(q_all, i_agent, action_idx, num_samples):
    total = 0.0
    for _ in range(num_samples):
        joint_action = [np.random.choice(len(qj), p=qj) for qj in q_all]
        joint_action[i_agent] = action_idx
        total += simulate_utility(joint_action)
    return total / num_samples

def monte_carlo_total_utility(q_all, num_samples):
    total = 0.0
    for _ in range(num_samples):
        joint_action = [np.random.choice(len(qj), p=qj) for qj in q_all]
        total += simulate_utility(joint_action)
    return total / num_samples

def nearest_newton_update(q_all, i_agent, T, alpha, num_samples):
    q_self = q_all[i_agent]
    new_q = np.copy(q_self)
    entropy_q = compute_entropy(q_self)
    E_G = monte_carlo_total_utility(q_all, num_samples)

    for i in range(len(q_self)):
        print(f"Agent {i_agent}, Action {i}:")
        E_G_given_i = monte_carlo_expected_util(q_all, i_agent, i, num_samples)
        delta_E = E_G_given_i - E_G

        print(f"Total E: {E_G}")
        print(f"E given action {i}: {E_G_given_i}")
        print(f"Delta E: {delta_E}")
        print()
        
        grad = entropy_q + np.log(q_self[i] + 1e-12) + (E_G_given_i - E_G) / T
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
        # print(q)

    # joint_actions = list(product(*[range(n) for n in num_actions_per_agent]))
    # joint_probs = []
    # for joint_action in joint_actions:
    #     prob = 1.0
    #     for i, a in enumerate(joint_action):
    #         prob *= q[i][a]
    #     joint_probs.append(prob)

    # expected_utility = sum(prob * G[joint_action] for prob, joint_action in zip(joint_probs, joint_actions))
    # entropy_total = sum(compute_entropy(qi) for qi in q)
    # objective = expected_utility + T * entropy_total

    # expected_utilities.append(expected_utility)
    # total_entropies.append(entropy_total)
    # objectives.append(objective)

    # print(f"Iteration {it}:")
    # for i, qi in enumerate(q):
    #     print(f"  q{i+1} = {qi.round(3)}")
    # print(f"  Expected Utility = {expected_utility:.4f}")
    # print(f"  Total Entropy = {entropy_total:.4f}")
    # print(f"  Objective = {objective:.4f}\n")

# Final output
print("Final optimized distributions:")
for i, qi in enumerate(q):
    print(f"q{i+1}: {qi}")

# # Plotting
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
