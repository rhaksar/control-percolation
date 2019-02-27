import numpy as np


# helper function to carefully multiply probabilities
def multiply_probabilities(values):
    tolerance = 1e-100
    if any([v < 1e-100 for v in values]):
        return 0
    else:
        sum_log = sum([np.log(v) for v in values])
        if sum_log <= np.log(tolerance):
            return 0
        else:
            return np.exp(sum_log)


def binomial_pgf(gw):

    if gw.generations == 1:
        p_stop_previous = 0
    else:
        p_stop_previous = gw.generation_data[gw.generations-1]['p_stop']

    p_children = gw.generation_data[gw.generations]['p_children']
    branch_factor = gw.generation_data[gw.generations]['branch_factor']

    return p_children*branch_factor, np.power(1 + (p_stop_previous-1)*p_children, branch_factor)


class GW(object):

    def __init__(self, root, pgf):
        self.root = root
        self.current_parents = [self.root]
        self.history = [self.root]

        self.pgf = pgf

        self.generations = 0
        self.generation_data = {}

    def add_generation(self, lattice_parameters, lattice_children, policy):
        all_children = []
        p_children = 0

        for parent in self.current_parents:
            children = lattice_children[parent]
            children = [child for child in children if child != parent and child not in self.history]

            child_parameters = [np.amax([lattice_parameters[(parent, child)] - policy(parent, child), 0])
                                for child in children]
            p_children += 0 if not child_parameters else len(children)*np.mean(child_parameters)

            all_children.extend(children)

        total_children = len(all_children)
        if total_children > 0:
            p_children /= total_children

        branch_factor = total_children
        if self.generations >= 1 and self.generation_data[self.generations]['total_children'] > 0:
            branch_factor /= self.generation_data[self.generations]['total_children']

        self.generations += 1
        self.generation_data[self.generations] = {'p_children': p_children,
                                                  'total_children': total_children,
                                                  'branch_factor': branch_factor,
                                                  'parents': self.current_parents}

        mean, p_stop = self.pgf(self)
        self.generation_data[self.generations]['mean'] = mean
        self.generation_data[self.generations]['p_stop'] = p_stop

        all_children = list(set(all_children))
        self.history.extend(all_children)
        self.current_parents = all_children


class BranchModel(object):

    def __init__(self, lattice_parameters, pgf):
        self.boundary = None
        self.children_function = None

        self.lattice_parameters = lattice_parameters
        self.pgf = pgf

        self.lattice_children = {}
        self.GWprocesses = {}
        self.statistics = {}

        self.generations = 0

    def reset(self):
        self.__init__(self.lattice_parameters, self.pgf)

    def set_boundary(self, boundary):
        self.boundary = boundary
        self.GWprocesses = {element: GW(element, self.pgf) for element in self.boundary}
        self.statistics[0] = {'mean': len(self.boundary), 'p_stop': 0}

    def set_children_function(self, children_function):
        self.children_function = children_function

    def next_generation(self, policy):

        for process in self.GWprocesses.values():
            for parent in process.current_parents:
                if parent not in self.lattice_children:
                    self.lattice_children[parent] = self.children_function(parent)

        # policy.generate_map(self)
        for process in self.GWprocesses.values():
            process.add_generation(self.lattice_parameters,
                                   self.lattice_children,
                                   policy.map)

        self.generations += 1

        mean, p_stop = self.model_statistics()
        self.statistics[self.generations] = {'mean': mean, 'p_stop': p_stop}

    def model_statistics(self):
        p_values = []
        mean = 0

        for process in self.GWprocesses.values():
            p_values.append(process.generation_data[self.generations]['p_stop'])
            generation_means = [process.generation_data[i]['mean'] for i in range(1, self.generations+1)]
            mean += np.prod(generation_means)

        p_stop = multiply_probabilities(p_values)
        return mean, p_stop

    def prediction(self):
        return self.statistics[self.generations]['mean'], self.statistics[self.generations]['p_stop']


class StaticModel(object):

    def __init__(self):
        self.boundary = None
        self.current_boundary = None

    def set_boundary(self, boundary):
        self.boundary = boundary
        self.current_boundary = self.boundary

    def next_boundary(self, policy_control):
        self.current_boundary = [b for b in self.current_boundary if b not in policy_control]
