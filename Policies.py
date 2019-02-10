from collections import defaultdict
import numpy as np


class Policy(object):

    def __init__(self, capacity, delta_p):
        self.capacity = capacity
        self.delta_p = delta_p

    def effect(self, *args):
        raise NotImplementedError

    def control(self, *args):
        raise NotImplementedError


class UBTfires(Policy):
    """
    Choose boundary nodes to apply action with uniform probability.
    """

    def __init__(self, capacity, delta_p):
        Policy.__init__(self, capacity, delta_p)

    def effect(self, branchmodel, gw):
        boundary_size = branchmodel.statistics[branchmodel.generations]['mean']
        if boundary_size == 0:
            return self.delta_p

        return np.amin([float(self.capacity)/boundary_size, 1])*self.delta_p

    def control(self, branchmodel):
        control = defaultdict(lambda: (0, 0))
        pass
