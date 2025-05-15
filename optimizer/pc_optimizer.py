

import numpy as np
from itertools import product
from optimizer.optimizer import Optimizer
from typing import List, Dict, Tuple


class PCOptimizer(Optimizer):
    def __init__(
            self,
            sim,
            actions_per_agent: List[List[int]],
            T: float = 0.1,
            alpha: float = 0.1,
            samples_per_joint_action: int = 10,
            num_mc_samples: int = 10,
            seed: int = 42):
        """Initialize the optimizer with agent action sets and hyperparameters."""

        # Set random seed for reproducibility
        np.random.seed(seed)

        # Action space for each agent
        self._actions_per_agent = actions_per_agent
        self._num_agents = len(actions_per_agent)

        # Temperature parameter for entropy-utility tradeoff
        self._T = T

        # Learning rate for update step
        self._alpha = alpha

        # Number of samples used to estimate utility for each joint action
        self._samples_per_joint_action = samples_per_joint_action

        # Number of Monte Carlo samples per utility estimation
        self._num_mc_samples = num_mc_samples

        # Utility simulator for evaluating joint actions
        self._sim = sim

        # Data tracking across iterations
        self._expected_utilities = []
        self._total_entropies = []
        self._objectives = []

    def _compute_entropy(self, p: np.ndarray) -> float:
        """
        Compute the entropy of a probability distribution.

        Args:
            p (np.ndarray): A numpy array representing a probability distribution.

        Returns:
            float: Entropy of the distribution.
        """
        return -np.sum(p * np.log(p + 1e-12))

    def _estimate_G(self) -> Dict[Tuple[int, ...], float]:
        """
        Estimate the expected utility of all possible joint actions.

        Returns:
            A dictionary mapping each joint action tuple to its estimated utility.
        """
        joint_actions = list(product(*self._actions_per_agent))
        G_est = {}
        for joint_action in joint_actions:
            # Run multiple rollouts for each joint action
            samples = [self._sim.rollout(joint_action)
                       for _ in range(self._samples_per_joint_action)]
            G_est[joint_action] = np.mean(samples)
        return G_est

    def _monte_carlo_expected_util(self, q_all: List[np.ndarray], i_agent: int, action_idx: int) -> float:
        """
        Estimate expected utility for a given agent fixing one action, while others sample from their distributions.

        Args:
            q_all: List of current action distributions for all agents.
            i_agent: Index of the agent to fix an action for.
            action_idx: Index of the fixed action for agent i_agent.

        Returns:
            Estimated expected utility.
        """
        total = 0.0
        for _ in range(self._num_mc_samples):
            # Sample joint action from current distributions
            joint_action = [
                np.random.choice(self._actions_per_agent[j], p=q_all[j]) for j in range(self._num_agents)
            ]
            # Override i_agent's action with a fixed action
            joint_action[i_agent] = self._actions_per_agent[i_agent][action_idx]
            total += self._sim.rollout(joint_action)
        return total / self._num_mc_samples

    def _monte_carlo_total_utility(self, q_all: List[np.ndarray]) -> float:
        """
        Estimate the expected utility when all agents sample from their current distributions.

        Args:
            q_all: List of current action distributions for all agents.

        Returns:
            Estimated total expected utility.
        """
        total = 0.0
        for _ in range(self._num_mc_samples):
            # Sample joint action from each agent's distribution
            joint_action = [
                np.random.choice(self._actions_per_agent[j], p=q_all[j]) for j in range(self._num_agents)
            ]
            total += self._sim.rollout(joint_action)
        return total / self._num_mc_samples

    def _nearest_newton_update(self, q_all: List[np.ndarray], i_agent: int) -> np.ndarray:
        """
        Perform a Nearest Newton update for a specific agent.

        Args:
            q_all: List of current action distributions for all agents.
            i_agent: Index of the agent to update.

        Returns:
            Updated probability distribution for the agent.
        """
        q_self = q_all[i_agent]  # Current distribution of i_agent
        # print(f"q_self: {q_self}")
        new_q = np.copy(q_self)  # Placeholder for updated distribution
        # Entropy of current distribution
        entropy_q = self._compute_entropy(q_self)
        # Total expected utility of current joint distribution
        E_G = self._monte_carlo_total_utility(q_all)

        for plan_i in range(len(q_self)):
            # Expected utility if i_agent chooses action i deterministically
            E_G_given_i = self._monte_carlo_expected_util(q_all, i_agent, plan_i)
            delta_E = E_G_given_i - E_G
            # print(f"delta_E: {delta_E}")
            grad = entropy_q + np.log(q_self[plan_i] + 1e-12) + delta_E / self._T
            # print(f"grad: {grad}")
            new_q[plan_i] -= self._alpha * q_self[plan_i] * grad
            # print(f"new_q[{plan_i}]: {new_q[plan_i]}")
            # print()
        
        # Normalize to ensure new_q is a valid distribution
        # print(f"new_q before normalization: {new_q}")
        new_q = np.maximum(new_q, 1e-8)
        
        new_q /= np.sum(new_q)
        # print(f"new_q: {new_q}")
        # print()
        return new_q

    def optimize(self, q: List[np.ndarray], i_agent: int, iterations: int = 10):
        """
        Run optimization for a single agent across several iterations, assuming others maintain their current distributions.

        Args:
            i_agent: Index of the agent to optimize.
        """
        local_q = q.copy()  # Copy the current distribution for the specified agent
        for it in range(iterations):
            # Update the distribution of the specified agent
            local_q[i_agent] = self._nearest_newton_update(local_q, i_agent)

            # # Compute joint action probabilities and metrics
            # joint_actions = list(product(*self._actions_per_agent))
            # joint_probs = []
            # for joint_action in joint_actions:
            #     prob = 1.0
            #     for i, a in enumerate(joint_action):
            #         a_idx = self._actions_per_agent[i].index(a)
            #         prob *= local_q[i][a_idx]
            #     joint_probs.append(prob)

            # # Calculate expected utility, entropy, and the objective
            # expected_utility = sum(prob * G[joint_action] for prob, joint_action in zip(joint_probs, joint_actions))
            # entropy_total = sum(self._compute_entropy(qi) for qi in local_q)
            # objective = expected_utility + self._T * entropy_total

            # # Log metrics for this iteration
            # self._expected_utilities.append(expected_utility)
            # self._total_entropies.append(entropy_total)
            # self._objectives.append(objective)
        return local_q
