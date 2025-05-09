import numpy as np
from itertools import product

np.random.seed(42)

# Parameters
num_agents = 3
num_actions_per_agent = [4, 4, 4]
T = 0.25
iterations = 100
samples_per_joint_action = 20
alpha = 1.0
num_mc_samples = 100

# Initialize distributions
q = [np.ones(n) / n for n in num_actions_per_agent]

def simulate_utility(actions):
    """
    Simulate utility for a joint action tuple (a1, a2, ..., aN).
    Replace this with your domain-specific reward function.
    """
    base_utility = {
        (0, 0, 0): 5,
        (0, 1, 1): 3,
        (1, 0, 1): 2,
        (2, 1, 0): 4
    }
    noise = np.random.normal(0, 0.2)
    return base_utility.get(tuple(actions), 0) + noise

def estimate_G(samples_per_joint_action):
    """
    Estimate G[joint_actions] via Monte Carlo.
    Returns a dict mapping joint actions to expected rewards.
    """
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

def softmax_update(q_all, G, i_agent, T):
    q_self = q_all[i_agent]
    new_q = np.zeros_like(q_self)
    action_spaces = [range(n) for n in num_actions_per_agent]
    for i in action_spaces[i_agent]:
        expected_util = 0.0
        for joint_action in product(*action_spaces):
            if joint_action[i_agent] != i:
                continue
            prob = 1.0
            for j, a in enumerate(joint_action):
                if j == i_agent:
                    continue
                prob *= q_all[j][a]
            expected_util += prob * G[joint_action]
        new_q[i] = expected_util / T
    new_q = np.exp(new_q - np.max(new_q))
    new_q /= np.sum(new_q)
    return new_q

def nearest_newton_update(q_all, i_agent, T, alpha, num_samples):
    q_self = q_all[i_agent]
    new_q = np.copy(q_self)
    entropy_q = compute_entropy(q_self)
    E_G = monte_carlo_total_utility(q_all, num_samples)
    for i in range(len(q_self)):
        E_G_given_i = monte_carlo_expected_util(q_all, i_agent, i, num_samples)
        grad = entropy_q + np.log(q_self[i] + 1e-12) + (E_G_given_i - E_G) / T
        new_q[i] -= alpha * q_self[i] * grad
    new_q = np.maximum(new_q, 1e-8)
    new_q /= np.sum(new_q)
    return new_q

# Main optimization loop
for it in range(iterations):
    G = estimate_G(samples_per_joint_action)

    for i_agent in range(num_agents):
        # Uncomment softmax if needed
        # q[i_agent] = softmax_update(q, G, i_agent, T)
        q[i_agent] = nearest_newton_update(q, i_agent, T, alpha, num_mc_samples)

    # Compute joint expected utility
    joint_actions = list(product(*[range(n) for n in num_actions_per_agent]))
    joint_probs = []
    for joint_action in joint_actions:
        prob = 1.0
        for i, a in enumerate(joint_action):
            prob *= q[i][a]
        joint_probs.append(prob)

    expected_utility = sum(prob * G[joint_action] for prob, joint_action in zip(joint_probs, joint_actions))
    entropy_total = sum(compute_entropy(qi) for qi in q)
    objective = expected_utility + T * entropy_total

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
