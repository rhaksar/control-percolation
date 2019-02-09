from collections import defaultdict
import numpy as np


class Policy(object):

    def __init__(self, capacity):
        self.capacity = capacity

    def probability(self, *args):
        raise NotImplementedError

    def control(self, *args):
        raise NotImplementedError


class UBTfires(Policy):
    """
    Choose boundary nodes to apply action with uniform probability.
    """

    def __init__(self, capacity):
        Policy.__init__(self, capacity)

    def probability(self, boundary_size):
        return np.amin([float(self.capacity)/boundary_size, 1])

    def control(self, boundary):
        control = defaultdict(lambda: (0, 0))
        pass
