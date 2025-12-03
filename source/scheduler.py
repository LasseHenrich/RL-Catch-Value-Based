import math
from dataclasses import dataclass


@dataclass
class EpsilonDecay():
    epsilon_start: float
    epsilon_end: float
    decay_rate: float

    def __call__(self, t): # todo: change this to be linear
        factor: float
        linear = False # otherwise exponential
        if linear:
            factor = max(0.0, 1.0 - t / self.decay_rate)
        else:
            factor = math.exp(-t / self.decay_rate)

        epsilon = self.epsilon_end + \
            (self.epsilon_start - self.epsilon_end) * \
            factor
        return epsilon
