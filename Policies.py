from collections import defaultdict
import numpy as np


class Policy(object):

    def __init__(self, capacity, control_map_percolation, control_map_gmdp):
        self.capacity = capacity
        self.control_map_percolation = control_map_percolation
        self.control_map_gmdp = control_map_gmdp
        self.map = None

    def generate_map(self, branchmodel):
        raise NotImplementedError

    def control(self, simulation_object, branchmodel):
        raise NotImplementedError


class UBTfires(Policy):
    """
    Uniform Boundary Treatment.
    Choose boundary nodes to apply action with uniform probability.
    """

    def __init__(self, capacity, control_map_percolation, control_map_gmdp):
        Policy.__init__(self, capacity, control_map_percolation, control_map_gmdp)

    def generate_map(self, branchmodel):
        boundary_size = branchmodel.statistics[branchmodel.generations]['mean']
        if boundary_size <= self.capacity:
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
    Degree Weighted Treatment.
    Choose boundary nodes to apply action with probability proportional to their out-degree.
    """

    def __init__(self, capacity, control_map_percolation, control_map_gmdp):
        Policy.__init__(self, capacity, control_map_percolation, control_map_gmdp)

    def generate_map(self, branchmodel):
        boundary_size = branchmodel.statistics[branchmodel.generations]['mean']
        if boundary_size <= self.capacity:
            self.map = lambda parent_child: self.control_map_percolation[parent_child]

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


class BFTfires(Policy):
    """
    Best First Treatment.
    Deterministic policy that chooses the most dangerous fires to treat.
    """

    def __init__(self, capacity, control_map_percolation, control_map_gmdp):
        Policy.__init__(self, capacity, control_map_percolation, control_map_gmdp)

    def generate_map(self, branchmodel):
        boundary_size = branchmodel.statistics[branchmodel.generations]['mean']
        if boundary_size <= self.capacity:
            self.map = lambda parent_child: self.control_map_percolation[parent_child]

        else:
            coefficient = defaultdict(lambda: 0)

            elements = []
            for process in branchmodel.GWprocesses.values():
                for parent in process.current_parents:
                    # score = len(branchmodel.lattice_children[parent])

                    children = branchmodel.lattice_children[parent]
                    children = [child for child in children if child != parent and child not in process.history]
                    score = np.mean([branchmodel.lattice_parameters[(parent, child)] for child in children])
                    score *= len(children)

                    elements.append((score, parent))

            elements = sorted(elements, key=lambda x: x[0], reverse=True)[:self.capacity]
            elements = [e[1] for e in elements]
            for e in elements:
                coefficient[e] = 1

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
            idx = []
            for i, node in enumerate(boundary):
                # score = len(branchmodel.lattice_children[node])

                children = branchmodel.lattice_children[node]
                children = [child for child in children if child != node]
                score = np.mean([branchmodel.lattice_parameters[(node, child)] for child in children])
                score *= len(children)

                idx.append((score, i))

            idx = sorted(idx, key=lambda x: x[0], reverse=True)[:self.capacity]
            idx = [e[1] for e in idx]

        for i in idx:
            control[boundary[i]] = self.control_map_gmdp[boundary[i]]

        return control
