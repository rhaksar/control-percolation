from collections import defaultdict
import numpy as np


class Policy(object):

    def __init__(self, capacity, control_map_percolation, control_map_gmdp):
        self.capacity = capacity
        self.control_map_percolation = control_map_percolation
        self.control_map_gmdp = control_map_gmdp

    def effect(self, branchmodel, gw, parent_child):
        raise NotImplementedError

    def control(self, simulation_object, branchmodel):
        raise NotImplementedError


class UBTfires(Policy):
    """
    Choose boundary nodes to apply action with uniform probability.
    """

    def __init__(self, capacity, control_map_percolation, control_map_gmdp):
        Policy.__init__(self, capacity, control_map_percolation, control_map_gmdp)

    def effect(self, branchmodel, gw, parent_child):
        boundary_size = branchmodel.statistics[branchmodel.generations]['mean']
        if boundary_size == 0:
            return self.control_map_percolation[parent_child]

        return np.amin([float(self.capacity)/boundary_size, 1])*self.control_map_percolation[parent_child]

    def control(self, simulation_object, branchmodel):
        control = defaultdict(lambda: (0, 0))
        boundary = branchmodel.boundary
        boundary_size = len(boundary)
        if boundary_size == 0:
            return control

        if boundary_size <= self.capacity:
            idx = range(boundary_size)
        else:
            idx = np.random.choice(boundary_size, size=self.capacity, replace=False)

        for i in idx:
            control[boundary[i]] = self.control_map_gmdp[boundary[i]]

        return control
