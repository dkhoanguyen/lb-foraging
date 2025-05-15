import numpy as np
from utility_simulator import UtilitySimulator
from typing import List


class SoftmaxOptimizer:
    def __init__(
        self,
        sim: UtilitySimulator,
        actions_per_agent: List[List[int]],
        T: float = 0.1,
        iterations: int = 10,
        samples_per_joint_action: int = 10,
        num_mc_samples: int = 10,
        seed: int = 42
    ):
        """Initialize the optimizer with agent action sets and hyperparameters."""

        # Set random seed for reproducibility
        np.random.seed(seed)

        # Action space for each agent
        self._actions_per_agent = actions_per_agent
        self._num_agents = len(actions_per_agent)

        # Temperature parameter for entropy-utility tradeoff
        self._T = T

        # Number of iterations for the optimization loop
        self._iterations = iterations

        # Number of samples used to estimate utility for each joint action
        self._samples_per_joint_action = samples_per_joint_action

        # Number of Monte Carlo samples per utility estimation
        self._num_mc_samples = num_mc_samples

        # Utility simulator for evaluating joint actions
        self._sim = sim

        # Initialize q as uniform distributions over actions for each agent
        self._q = [np.ones(len(a)) / len(a) for a in actions_per_agent]

    def _estimate_utilities(self, q_all: List[np.ndarray], i_agent: int) -> List[float]:
        """
        Estimate the expected utility for each action of a specific agent.

        Args:
            q_all: List of current action distributions for all agents.
            i_agent: Index of the agent to update.

        Returns:
            A list of expected utilities for each action.
        """
        q_util = []
        for i in range(len(self._actions_per_agent[i_agent])):
            total = 0.0
            for _ in range(self._num_mc_samples):
                joint_action = [
                    np.random.choice(self._actions_per_agent[j], p=q_all[j]) for j in range(self._num_agents)
                ]
                joint_action[i_agent] = self._actions_per_agent[i_agent][i]
                total += self._sim.rollout(joint_action)
            q_util.append(total / self._num_mc_samples)
        return q_util

    def optimize(self, i_agent: int) -> None:
        """
        Run softmax optimization for a single agent across several iterations.

        Args:
            i_agent: Index of the agent to optimize.
        """
        for it in range(self._iterations):
            # Estimate action utilities for the current agent
            utilities = self._estimate_utilities(self._q, i_agent)

            # Softmax update rule
            logits = np.array(utilities) / self._T
            max_logit = np.max(logits)  # For numerical stability
            exp_logits = np.exp(logits - max_logit)
            new_q = exp_logits / np.sum(exp_logits)

            # Update agent's distribution
            self._q[i_agent] = new_q