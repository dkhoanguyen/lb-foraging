import numpy as np
import itertools

class UtilitySimulator:
    def __init__(self, actions_per_agent, noise_std=2, seed=None):
        """
        :param actions_per_agent: List of lists. Each sublist contains possible actions for one agent.
                                  Example: [[0, 1], [0, 1, 2], [1, 2]]
        :param noise_std: Standard deviation for the noise added to the utility
        :param seed: Random seed for reproducibility
        """
        self.actions_per_agent = actions_per_agent
        self.n_agents = len(actions_per_agent)
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)
        self.base_utility = self._generate_random_utilities()

    def _generate_random_utilities(self):
        base_utility = {
            (0, 0): 0,
            (0, 1): 20,
            (1, 0): 20,
            (1, 1): 20,
        }
        return base_utility

    def rollout(self, actions):
        """
        :param actions: List of actions, one per agent
        :return: Utility value with noise
        """
        actions = tuple(actions)
        base = self.base_utility.get(actions, 0)
        noise = self.rng.normal(0, self.noise_std)
        return base + noise
