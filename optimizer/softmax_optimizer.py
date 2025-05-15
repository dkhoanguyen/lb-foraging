import numpy as np
from itertools import product
from typing import List, Dict, Tuple
from optimizer.optimizer import Optimizer  # Make sure this exists in your project

class SoftmaxOptimizer(Optimizer):
    def __init__(
            self,
            sim,
            actions_per_agent: List[List[int]],
            T: float = 0.1,
            alpha: float = 0.001,
            samples_per_joint_action: int = 10,
            num_mc_samples: int = 10,
            seed: int = 42):
        np.random.seed(seed)
        self._actions_per_agent = actions_per_agent
        self._num_agents = len(actions_per_agent)
        self._T = T
        self._alpha = alpha
        self._samples_per_joint_action = samples_per_joint_action
        self._num_mc_samples = num_mc_samples
        self._sim = sim

        self._expected_utilities = []
        self._total_entropies = []
        self._objectives = []

    def _compute_entropy(self, p: np.ndarray) -> float:
        return -np.sum(p * np.log(p + 1e-12))

    def _monte_carlo_expected_util(self, q_all: List[np.ndarray], i_agent: int, action_idx: int) -> float:
        total = 0.0
        for _ in range(self._num_mc_samples):
            joint_action = [
                np.random.choice(self._actions_per_agent[j], p=q_all[j]) for j in range(self._num_agents)
            ]
            joint_action[i_agent] = self._actions_per_agent[i_agent][action_idx]
            total += self._sim.rollout(joint_action)
        return total / self._num_mc_samples

    def _monte_carlo_total_utility(self, q_all: List[np.ndarray]) -> float:
        total = 0.0
        for _ in range(self._num_mc_samples):
            joint_action = [
                np.random.choice(self._actions_per_agent[j], p=q_all[j]) for j in range(self._num_agents)
            ]
            total += self._sim.rollout(joint_action)
        return total / self._num_mc_samples

    def _softmax_update(self, q_all: List[np.ndarray], i_agent: int) -> np.ndarray:
        num_actions = len(self._actions_per_agent[i_agent])
        logits = np.zeros(num_actions)
        E_G = self._monte_carlo_total_utility(q_all)
        for a in range(num_actions):
            E_G_given_i = self._monte_carlo_expected_util(q_all, i_agent, a)
            delta_E = -E_G_given_i
            logits[a] = delta_E / self._T
        logits -= np.max(logits)  # for numerical stability
        new_q = np.exp(logits)
        new_q /= np.sum(new_q)
        return new_q

    def optimize(self, q: List[np.ndarray], i_agent: int, iterations: int = 10):
        local_q = q.copy()
        for _ in range(iterations):
            local_q[i_agent] = self._softmax_update(local_q, i_agent)
        return local_q
