from abc import ABC, abstractmethod

class Optimizer(ABC):
    @abstractmethod
    def optimize(self, i_agent: int, iterations: int = 10):
        pass