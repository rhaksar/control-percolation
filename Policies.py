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

    def control(self, simulation_object, branchmodel, control_map):
        control = defaultdict(lambda: (0, 0))
        boundary_size = len(branchmodel.boundary)
        if boundary_size == 0:
            return control

        if boundary_size < self.capacity:
            idx = range(boundary_size)
        else:
            idx = np.random.choice(boundary_size, size=self.capacity, replace=False)

        for i in idx:
            control[branchmodel.boundary[i]] = (0, control_map[branchmodel.boundary[i]])

        return control
