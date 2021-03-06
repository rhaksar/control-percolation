from collections import defaultdict
import numpy as np

from Utilities import percolation_parameter


class Policy(object):
    """
    Policy template class.
    """

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


class NCTfires(Policy):
    """
    No Control Treatment.
    Do nothing.
    """

    def __init__(self, capacity, alpha_set, beta_set, control_map_gmdp):
        Policy.__init__(self, capacity, alpha_set, beta_set, control_map_gmdp)
        self.name = 'NCT'

    def map(self, parent, child):
        return 0

    def generate_map(self, branchmodel):
        pass

    def control(self, branchmodel):
        return defaultdict(lambda: (0, 0))


class UBTfires(Policy):
    """
    Uniform Boundary Treatment.
    Choose boundary nodes to apply action with uniform probability.
    """

    def __init__(self, capacity, alpha_set, beta_set, control_map_gmdp):
        Policy.__init__(self, capacity, alpha_set, beta_set, control_map_gmdp)
        self.coefficient = None
        self.name = 'UBT_C' + str(capacity)

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
        self.name = 'DWT_C' + str(capacity)

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


class RHTfires(Policy):
    """
    Receding Horizon Treatment.
    """

    def __init__(self, capacity, horizon, alpha_set, beta_set, control_map_gmdp):
        Policy.__init__(self, capacity, alpha_set, beta_set, control_map_gmdp)
        self.horizon = horizon
        self.control_decisions = []
        self.name = 'RHT' + str(self.horizon) + '_C' + str(capacity)

    def get_score(self, p, branchmodel, counter=1):
        if p not in branchmodel.lattice_children:
            branchmodel.lattice_children[p] = branchmodel.children_function(p)

        cs = branchmodel.lattice_children[p]
        cs = [ci for ci in cs if ci != p]

        if counter == self.horizon:
            c_parameters = [branchmodel.lattice_parameters[(p, ci)] for ci in cs]
            s = 0 if not c_parameters else len(cs)*np.mean(c_parameters)
            return s

        else:
            ss = [self.get_score(ci, branchmodel, counter+1) for ci in cs]
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
            elements = []
            for process in branchmodel.GWprocesses.values():
                for parent in process.current_parents:
                    score = self.get_score(parent, branchmodel)
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


class USTfires(Policy):
    """
    Urban Preservation Treatment.
    Treat fires and remove urban areas to preserve an urban region.
    """

    def __init__(self, horizon, capacity, alpha_set, beta_set, control_map_gmdp):
        Policy.__init__(self, capacity, alpha_set, beta_set, control_map_gmdp)
        self.horizon = horizon
        self.control_decisions = []
        self.name = 'UST' + str(self.horizon) + '_C' + str(capacity)

    def get_score(self, parent, hist, branchmodel, counter=1):
        if parent not in branchmodel.lattice_children:
            branchmodel.lattice_children[parent] = branchmodel.children_function(parent)

        cs = branchmodel.lattice_children[parent]
        cs = [ci for ci in cs if ci != parent]

        if counter == self.horizon:
            c_parameters = [branchmodel.lattice_parameters[(parent, ci)] for ci in cs]
            s = 0 if not c_parameters else len(cs)*np.mean(c_parameters)
            hist.extend(cs)
            return s, hist

        else:
            ss = 0
            for ci in cs:
                score, hist = self.get_score(ci, hist, branchmodel, counter+1)
                ss += score
                hist.extend(cs)
                hist = list(set(hist))
            # ss = [self.get_score(ci, hist, branchmodel, counter+1)[0] for ci in cs]
            # hist.extend(cs)
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

        for urban in urbanmodel.boundary:
            if urban in self.control_decisions:
                control[urban] = self.control_map_gmdp[urban]['healthy']
                # print('treating urban at', urban)

        for fire in branchmodel.boundary:
            if fire in self.control_decisions:
                control[fire] = self.control_map_gmdp[fire]['on_fire']
                # print('treating fire at', fire)

        self.control_decisions = []
        return control
