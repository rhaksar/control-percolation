from collections import defaultdict
import numpy as np


class Policy(object):

    def __init__(self, capacity, control_map_percolation, control_map_gmdp):
        self.capacity = capacity
        self.control_map_percolation = control_map_percolation
        self.control_map_gmdp = control_map_gmdp
        self.map = None

    def cache_map(self, branchmodel):
        raise NotImplementedError

    def effect(self, parent_child):
        raise NotImplementedError

    def control(self, simulation_object, branchmodel):
        raise NotImplementedError


class UBTfires(Policy):
    """
    Choose boundary nodes to apply action with uniform probability.
    """

    def __init__(self, capacity, control_map_percolation, control_map_gmdp):
        Policy.__init__(self, capacity, control_map_percolation, control_map_gmdp)

    def effect(self, parent_child):
        return self.map(parent_child)

    def cache_map(self, branchmodel):
        boundary_size = branchmodel.statistics[branchmodel.generations]['mean']
        if boundary_size == 0:
            self.map = lambda parent_child: self.control_map_percolation[parent_child]

        else:
            coefficient = np.amin([float(self.capacity)/boundary_size, 1])
            self.map = lambda parent_child: coefficient*self.control_map_percolation[parent_child]

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


class DWTfires(Policy):
    """
    Choose boundary nodes to apply action with probability proportional to their out-degree.
    """

    def __init__(self, capacity, control_map_percolation, control_map_gmdp):
        Policy.__init__(self, capacity, control_map_percolation, control_map_gmdp)

    def effect(self, parent_child):
        return self.map(parent_child)

    def cache_map(self, branchmodel):
        boundary_size = branchmodel.statistics[branchmodel.generations]['mean']
        if boundary_size == 0:
            self.map = self.map = lambda parent_child: self.control_map_percolation[parent_child]

        else:
            coefficient = {}
            total_out_degree = 0
            for process in branchmodel.GWprocesses.values():
                for parent in process.current_parents:
                    total_out_degree += len(branchmodel.lattice_children[parent])

                    coefficient[parent] = len(branchmodel.lattice_children[parent])

            def compute_coefficient(di, sum_di):
                if sum_di == 0:
                    return 1
                return np.amin([(di*self.capacity)/sum_di, 1])
            coefficient = {parent: compute_coefficient(coefficient[parent], total_out_degree)
                           for parent in coefficient.keys()}

            self.map = lambda parent_child: coefficient[parent_child[0]]*self.control_map_percolation[parent_child]

    def control(self, simulation_object, branchmodel):
        control = defaultdict(lambda: (0, 0))
        boundary = branchmodel.boundary
        boundary_size = len(boundary)
        if boundary_size == 0:
            return control

        if boundary_size <= self.capacity:
            idx = range(boundary_size)

        else:
            total_out_degree = 0
            for node in boundary:
                total_out_degree += len(branchmodel.lattice_children[node])

            probabilities = [len(branchmodel.lattice_children[node])/total_out_degree for node in boundary]
            probabilities /= np.sum(probabilities)

            idx = np.random.choice(boundary_size, size=self.capacity, replace=False, p=probabilities)

        for i in idx:
            control[boundary[i]] = self.control_map_gmdp[boundary[i]]

        return control
