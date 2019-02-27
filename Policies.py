from collections import defaultdict
import numpy as np

from Utilities import *


class Policy(object):

    def __init__(self, capacity, alpha_set, beta_set, control_map_gmdp):
        self.capacity = capacity
        self.alpha_set = alpha_set
        self.beta_set = beta_set
        self.control_map_gmdp = control_map_gmdp

        # self.map = None
        # self.urbanboundary = []

    # def generate_map(self, branchmodel):
    #     raise NotImplementedError
    #
    # def control(self, simulation_object, branchmodel):
    #     raise NotImplementedError


class UBTfires(Policy):
    """
    Uniform Boundary Treatment.
    Choose boundary nodes to apply action with uniform probability.
    """

    def __init__(self, capacity, alpha_set, beta_set, control_map_gmdp):
        Policy.__init__(self, capacity, alpha_set, beta_set, control_map_gmdp)
        self.coefficient = None

    def map(self, parent, child):
        if self.coefficient == 0:
            return 0

        alpha = self.alpha_set[child]
        beta = self.beta_set[parent]

        delta_alpha = 0
        _, delta_beta = self.control_map_gmdp[parent]['on_fire']

        delta_p = percolation_parameter(alpha, beta) - percolation_parameter(alpha-delta_alpha, beta-delta_beta)
        return self.coefficient*delta_p

    def generate_map(self, branchmodel):
        boundary_size = branchmodel.statistics[branchmodel.generations]['mean']
        if boundary_size == 0:
            self.coefficient = 0
        else:
            self.coefficient = np.amin([float(self.capacity)/boundary_size, 1])

    def control(self, branchmodel):
        control = defaultdict(lambda: (0, 0))
        boundary = branchmodel.boundary
        boundary_size = len(boundary)

        if boundary_size == 0:
            return control
        elif boundary_size <= self.capacity:
            idx = range(boundary_size)
        else:
            idx = np.random.choice(boundary_size, size=self.capacity, replace=False)

        for i in idx:
            control[boundary[i]] = self.control_map_gmdp[boundary[i]]['on_fire']

        self.coefficient = None
        return control


class DWTfires(Policy):
    """
    Degree Weighted Treatment.
    Choose boundary nodes to apply action with probability proportional to their number of healthy neighbors.
    """

    def __init__(self, capacity, alpha_set, beta_set, control_map_gmdp):
        Policy.__init__(self, capacity, alpha_set, beta_set, control_map_gmdp)

        self.data = dict()

    def map(self, parent, child):
        alpha = self.alpha_set[child]
        beta = self.beta_set[parent]

        delta_alpha = 0
        _, delta_beta = self.control_map_gmdp[parent]['on_fire']

        delta_p = percolation_parameter(alpha, beta) - percolation_parameter(alpha-delta_alpha, beta-delta_beta)
        return self.data[parent]*delta_p

    def generate_map(self, branchmodel):
        boundary_size = branchmodel.statistics[branchmodel.generations]['mean']

        if boundary_size == 0:
            self.data = defaultdict(lambda: 0)
        elif boundary_size <= self.capacity:
            self.data = defaultdict(lambda: 1)
        else:
            total_out_degree = 0
            for process in branchmodel.GWprocesses.values():
                for parent in process.current_parents:
                    children = branchmodel.lattice_children[parent]
                    children = [child for child in children if child != parent and child not in process.history]

                    total_out_degree += len(children)
                    self.data[parent] = len(children)

            self.data = {parent: np.amin([(self.data[parent]*self.capacity)/total_out_degree, 1])
                         for parent in self.data.keys()}

    def control(self, branchmodel):
        control = defaultdict(lambda: (0, 0))
        boundary = branchmodel.boundary
        boundary_size = len(boundary)

        if boundary_size == 0:
            return control
        elif boundary_size <= self.capacity:
            idx = range(boundary_size)
        else:
            total_out_degree = 0
            probabilities = np.zeros(boundary_size)
            for i, node in enumerate(boundary):
                children = branchmodel.lattice_children[node]
                children = [child for child in children if child != node]
                total_out_degree += len(children)
                probabilities[i] = len(children)

            probabilities /= np.sum(probabilities)
            idx = np.random.choice(boundary_size, size=self.capacity, replace=False, p=probabilities)

        for i in idx:
            control[boundary[i]] = self.control_map_gmdp[boundary[i]]['on_fire']

        self.data = dict()
        return control


# class BFTfires(Policy):
#     """
#     Best First Treatment.
#     Deterministic policy that chooses the most dangerous fires to treat.
#     """
#
#     def __init__(self, capacity, control_map_percolation, control_map_gmdp):
#         Policy.__init__(self, capacity, control_map_percolation, control_map_gmdp)
#
#     def generate_map(self, branchmodel):
#         boundary_size = branchmodel.statistics[branchmodel.generations]['mean']
#         if boundary_size <= self.capacity:
#             self.map = lambda parent_child: self.control_map_percolation[parent_child]
#
#         else:
#             coefficient = defaultdict(lambda: 0)
#
#             elements = []
#             for process in branchmodel.GWprocesses.values():
#                 for parent in process.current_parents:
#                     # score = len(branchmodel.lattice_children[parent])
#
#                     children = branchmodel.lattice_children[parent]
#                     children = [child for child in children if child != parent and child not in process.history]
#                     children_parameters = [branchmodel.lattice_parameters[(parent, child)] for child in children]
#                     score = 0 if not children_parameters else len(children)*np.mean(children_parameters)
#
#                     elements.append((score, parent))
#
#             elements = sorted(elements, key=lambda x: x[0], reverse=True)[:self.capacity]
#             elements = [e[1] for e in elements]
#             for e in elements:
#                 coefficient[e] = 1
#
#             self.map = lambda parent_child: coefficient[parent_child[0]]*self.control_map_percolation[parent_child]
#
#     def control(self, simulation_object, branchmodel):
#         control = defaultdict(lambda: (0, 0))
#         boundary = branchmodel.boundary
#         boundary_size = len(boundary)
#
#         if boundary_size == 0:
#             return control
#
#         elif boundary_size <= self.capacity:
#             idx = range(boundary_size)
#
#         else:
#             idx = []
#             for i, node in enumerate(boundary):
#                 # score = len(branchmodel.lattice_children[node])
#
#                 children = branchmodel.lattice_children[node]
#                 children = [child for child in children if child != node]
#                 score = np.mean([branchmodel.lattice_parameters[(node, child)] for child in children])
#                 score *= len(children)
#
#                 idx.append((score, i))
#
#             idx = sorted(idx, key=lambda x: x[0], reverse=True)[:self.capacity]
#             idx = [e[1] for e in idx]
#
#         for i in idx:
#             control[boundary[i]] = self.control_map_gmdp[boundary[i]]
#
#         return control


class RHTfires(Policy):
    """
    Receding Horizon Treatment.
    """

    def __init__(self, capacity, horizon, alpha_set, beta_set, control_map_gmdp):
        Policy.__init__(self, capacity, alpha_set, beta_set, control_map_gmdp)
        self.horizon = horizon
        self.control_decisions = []
        # self.rollout_policy = UBTfires(self.capacity, alpha_set, beta_set, self.control_map_gmdp)

    def get_score(self, p, hist, branchmodel, counter=1):
        if p not in branchmodel.lattice_children:
            branchmodel.lattice_children[p] = branchmodel.children_function(p)

        cs = branchmodel.lattice_children[p]
        cs = [ci for ci in cs if ci != p and ci not in hist]

        if counter == self.horizon:
            c_parameters = [branchmodel.lattice_parameters[(p, ci)] for ci in cs]
            s = 0 if not c_parameters else len(cs)*np.mean(c_parameters)
            return s

        else:
            hist.extend(cs)
            ss = [self.get_score(ci, hist, branchmodel, counter+1) for ci in cs]
            return np.sum(ss)

    def map(self, parent, child):
        alpha = self.alpha_set[child]
        beta = self.beta_set[parent]

        delta_alpha, delta_beta = 0, 0
        if parent in self.control_decisions:
            _, delta_beta = self.control_map_gmdp[parent]['on_fire']
        if child in self.control_decisions:
            delta_alpha, _ = self.control_map_gmdp[child]['healthy']

        return percolation_parameter(alpha, beta) - percolation_parameter(alpha-delta_alpha, beta-delta_beta)

    def generate_map(self, branchmodel):
        boundary_size = branchmodel.statistics[branchmodel.generations]['mean']
        if boundary_size == 0:
            return
        else:
            coefficient = defaultdict(lambda: 0)

            elements = []
            for process in branchmodel.GWprocesses.values():
                for parent in process.current_parents:
                    score = self.get_score(parent, [], branchmodel)
                    elements.append((score, parent))

            elements = sorted(elements, key=lambda x: x[0], reverse=True)[:self.capacity]
            elements = [e[1] for e in elements]
            for e in elements:
                self.control_decisions.append(e)

    def control(self, branchmodel):
        control = defaultdict(lambda: (0, 0))

        for fire in branchmodel.boundary:
            if fire in self.control_decisions:
                control[fire] = self.control_map_gmdp[fire]['on_fire']

        self.control_decisions = []
        return control


class UPTfires(Policy):
    """
    Urban Preservation Treatment.
    Treat fires and remove urban areas to preserve an urban region.
    """

    def __init__(self, horizon, capacity, alpha_set, beta_set, control_map_gmdp):
        Policy.__init__(self, capacity, alpha_set, beta_set, control_map_gmdp)
        self.horizon = horizon

        self.control_decisions = []
        self.alpha_set = alpha_set
        self.beta_set = beta_set

    def get_score(self, p, hist, branchmodel, counter=1):
        if p not in branchmodel.lattice_children:
            branchmodel.lattice_children[p] = branchmodel.children_function(p)

        cs = branchmodel.lattice_children[p]
        cs = [ci for ci in cs if ci != p and ci not in hist]

        if counter == self.horizon:
            c_parameters = [branchmodel.lattice_parameters[(p, ci)] for ci in cs]
            s = 0 if not c_parameters else len(cs)*np.mean(c_parameters)
            return s, hist

        else:
            hist.extend(cs)
            ss = [self.get_score(ci, hist, branchmodel, counter+1)[0] for ci in cs]
            return np.sum(ss), hist

    def map(self, parent, child):
        alpha = self.alpha_set[child]
        beta = self.beta_set[parent]

        delta_alpha, delta_beta = 0, 0
        if parent in self.control_decisions:
            _, delta_beta = self.control_map_gmdp[parent]['on_fire']
        if child in self.control_decisions:
            delta_alpha, _ = self.control_map_gmdp[child]['healthy']

        return percolation_parameter(alpha, beta) - percolation_parameter(alpha-delta_alpha, beta-delta_beta)

    def generate_map(self, branchmodel, urbanmodel):
        fires = []
        histories = {}
        for process in branchmodel.GWprocesses.values():
            for parent in process.current_parents:
                score_position = parent[1]
                score_spreadingrate, parent_history = self.get_score(parent, [], branchmodel)
                # if score_spreadingrate <= self.horizon:
                #     continue
                fires.append((score_position, score_spreadingrate, parent))
                histories[parent] = parent_history

        urban = []
        for ub in urbanmodel.current_boundary:
            score = 0
            for parent in histories.keys():
                if ub in histories[parent]:
                    score += 1

            if score > 0:
                urban.append((score, ub))

        if urban:
            fires = [f for f in fires if f[1] >= self.horizon]

        if len(urban) > 0:
            if len(urban) <= self.capacity:
                ordered = [ub[1] for ub in urban]
            else:
                ordered = sorted(urban, key=lambda x: x[0], reverse=True)[:self.capacity]
                ordered = [s[1] for s in ordered]

            for e in ordered:
                self.control_decisions.append(e)

            extra_control = self.capacity - len(urban)
            if extra_control > 0:
                ordered = sorted(fires, key=lambda x: (x[0], x[1]), reverse=True)[:extra_control]
                ordered = [f[2] for f in ordered]
                for e in ordered:
                    self.control_decisions.append(e)

        elif len(fires) > 0:
            if len(fires) <= self.capacity:
                ordered = [f[2] for f in fires]
            else:
                ordered = sorted(fires, key=lambda x: (x[0], x[1]), reverse=True)[:self.capacity]
                ordered = [f[2] for f in ordered]

            for e in ordered:
                self.control_decisions.append(e)

        return

    def control(self, branchmodel, urbanmodel):
        control = defaultdict(lambda: (0, 0))

        for fire in branchmodel.boundary:
            if fire in self.control_decisions:
                control[fire] = self.control_map_gmdp[fire]['on_fire']
                # print('treating fire at', fire)

        for urban in urbanmodel.boundary:
            if urban in self.control_decisions:
                control[urban] = self.control_map_gmdp[urban]['healthy']
                # print('treating boundary at', urban)

        self.control_decisions = []
        return control

    # def control(self, simulation_object, branchmodel):
    #     control = defaultdict(lambda: (0, 0))
    #
    #     fires_order = []
    #     histories = {}
    #     for parent in branchmodel.boundary:
    #         score1 = parent[1]
    #         score2, parent_history = self.get_score(parent, [], branchmodel)
    #         fires_order.append((score1, score2, parent))
    #         histories[parent] = parent_history
    #
    #     urban_order = []
    #     for ub in self.urbanboundary:
    #         score = 0
    #         for parent in histories.keys():
    #             if ub in histories[parent]:
    #                 score += 1
    #
    #         if score > 0:
    #             urban_order.append((score, ub))
    #
    #     control_elements = []
    #     if len(urban_order) > 0:
    #         if len(urban_order) <= self.capacity:
    #             sorted_order = [ub[1] for ub in urban_order]
    #         else:
    #             sorted_order = sorted(urban_order, key=lambda x: x[0], reverse=True)[:self.capacity]
    #             sorted_order = [e[1] for e in sorted_order]
    #
    #         for e in sorted_order:
    #             control_elements.append(e)
    #
    #         extra_control = self.capacity - len(urban_order)
    #         if extra_control > 0:
    #             sorted_order = sorted(fires_order, key=lambda x: (x[0], x[1]), reverse=True)[:extra_control]
    #             sorted_order = [f[2] for f in sorted_order]
    #             for e in sorted_order:
    #                 control_elements.append(e)
    #
    #     elif len(fires_order) > 0:
    #         if len(fires_order) <= self.capacity:
    #             sorted_order = [f[2] for f in fires_order]
    #         else:
    #             sorted_order = sorted(fires_order, key=lambda x: (x[0], x[1]), reverse=True)[:self.capacity]
    #             sorted_order = [s[2] for s in sorted_order]
    #
    #         for e in sorted_order:
    #             control_elements.append(e)
    #
    #     else:
    #         return control
    #
    #     for e in control_elements:
    #         control[e] = self.control_map_gmdp[e]
    #
    #     return control


# class RHTfiresNeedsWork(Policy):
#     """
#     Receding Horizon Treatment.
#     Deterministic policy that treats nodes with the largest future generation size.
#     """
#
#     def __init__(self, capacity, control_map_percolation, control_map_gmdp, horizon=3):
#         Policy.__init__(self, capacity, control_map_percolation, control_map_gmdp)
#         self.horizon = horizon
#         self.rollout_policy = UBTfires(capacity=self.capacity,
#                                        control_map_percolation=self.control_map_percolation,
#                                        control_map_gmdp=self.control_map_gmdp)
#
#     def generate_map(self, branchmodel):
#         boundary_size = branchmodel.statistics[branchmodel.generations]['mean']
#         if boundary_size <= self.capacity:
#             self.map = lambda parent_child: self.control_map_percolation[parent_child]
#
#         else:
#             coefficient = defaultdict(lambda: 0)
#
#             new_model = copy.deepcopy(branchmodel)
#             new_model.generations = 0
#             new_model.lattice_children = branchmodel.lattice_children
#
#             new_boundary = []
#             for process in branchmodel.GWprocesses.values():
#                 new_boundary.extend(process.current_parents)
#             new_model.set_boundary(new_boundary)
#
#             for _ in range(self.horizon):
#                 new_model.next_generation(self.rollout_policy)
#
#             elements = []
#             for node in new_model.boundary:
#                 process = new_model.GWprocesses[node]
#                 generation_means = [process.generation_data[i]['mean'] for i in range(1, new_model.generations+1)]
#                 mean = np.prod(generation_means)
#                 elements.append((mean, node))
#
#             elements = sorted(elements, key=lambda x: x[0], reverse=True)[:self.capacity]
#             elements = [e[1] for e in elements]
#             for e in elements:
#                 coefficient[e] = 1
#
#             self.map = lambda parent_child: coefficient[parent_child[0]]*self.control_map_percolation[parent_child]
#
#     def control(self, simulation_object, branchmodel):
#         control = defaultdict(lambda: (0, 0))
#         boundary = branchmodel.boundary
#         boundary_size = len(boundary)
#
#         if boundary_size == 0:
#             return control
#
#         elif boundary_size <= self.capacity:
#             idx = range(boundary_size)
#
#             for i in idx:
#                 control[boundary[i]] = self.control_map_gmdp[boundary[i]]
#
#         else:
#             elements = []
#             for node in branchmodel.boundary:
#                 process = branchmodel.GWprocesses[node]
#                 generation_means = [process.generation_data[i]['mean'] for i in range(1, branchmodel.generations+1)]
#                 process_mean = np.prod(generation_means)
#                 elements.append((process_mean, node))
#
#             elements = sorted(elements, key=lambda x: x[0], reverse=True)[:self.capacity]
#             elements = [e[1] for e in elements]
#
#             for e in elements:
#                 control[e] = self.control_map_gmdp[e]
#
#         return control
