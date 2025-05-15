import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from utility_simulator import UtilitySimulator
from typing import List, Dict, Tuple

from optimizer.pc_optimizer import PCOptimizer
from optimizer.softmax_optimizer import SoftmaxOptimizer

actions_per_agent = [
    [0, 1],        # Agent 0
    [0, 1],        # Agent 1
]

num_agents = len(actions_per_agent)
T = 0.05
iterations = 10
samples_per_joint_action = 10
alpha = 0.0015
num_mc_samples = 10

q = [np.ones(len(a)) / len(a) for a in actions_per_agent]
sim = UtilitySimulator(actions_per_agent, seed=42)

# Instantiate optimizers
nn_optimizer = PCOptimizer(
    sim,
    actions_per_agent,
    T=T,
    alpha=alpha,
    samples_per_joint_action=samples_per_joint_action,
    num_mc_samples=num_mc_samples,
    seed=42
)

softmax_optimizer = SoftmaxOptimizer(
    sim,
    actions_per_agent,
    T=T,
    samples_per_joint_action=samples_per_joint_action,
    num_mc_samples=num_mc_samples,
    seed=42
)

# --- Run Optimization ---
nn_utilities = []
softmax_utilities = []

for it in range(iterations):
    # print(f"Iteration {it}")
    q_next = []
    for i_agent in range(num_agents):
        # Aggregate q of the other agents
        other_q = []
        for j_agent in range(num_agents):
            if j_agent != i_agent:
                other_q.append(q[j_agent].copy())
        # Plan actions give other agents' actions and probabilities
        q_i = q[i_agent].copy()
        # Run 1 step optimization
        output = softmax_optimizer.optimize(q, i_agent, iterations=1)
        # output = nn_optimizer.optimize(q, i_agent, iterations=1)
        # print(output[i_agent])
        q_next.append(output[i_agent])
    q = q_next.copy()
for p in q:
    print(p)
