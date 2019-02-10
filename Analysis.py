import numpy as np


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

    def add_generation(self, lattice_parameters, lattice_children, children_function, policy):
        all_children = []
        p_children = 0
        self.generations += 1

        for parent in self.current_parents:

            if parent not in lattice_children:
                lattice_children[parent] = children_function(parent)

            children = lattice_children[parent]
            children = [child for child in children if child != parent and child not in self.history]

            child_parameters = [lattice_parameters[(parent, child)] for child in children]
            p_children += 0 if not child_parameters else len(children)*np.mean(child_parameters)

            all_children.extend(children)

        total_children = len(all_children)
        if total_children > 0:
            p_children /= total_children

        branch_factor = total_children
        if self.generations > 1 and self.generation_data[self.generations-1]['total_children'] > 0:
            branch_factor /= self.generation_data[self.generations-1]['total_children']

        # effect of control policy
        delta_p_children = policy(self)
        p_children = np.amax([p_children - delta_p_children, 0])

        self.generation_data[self.generations] = {'p_children': p_children,
                                                  'total_children': total_children,
                                                  'branch_factor': branch_factor}

        mean, p_stop = self.pgf(self)
        self.generation_data[self.generations]['mean'] = mean
        self.generation_data[self.generations]['p_stop'] = p_stop

        self.history.extend(all_children)
        self.current_parents = all_children


class BranchModel(object):

    def __init__(self, boundary_function, lattice_parameters, pgf):
        self.boundary = None

        self.boundary_function = boundary_function
        self.lattice_parameters = lattice_parameters
        self.pgf = pgf

        self.lattice_children = {}
        self.GWprocesses = {}
        self.statistics = {}

        self.generations = 0

    def reset(self):
        self.__init__(self.boundary_function, self.lattice_parameters, self.pgf)

    def set_boundary(self, simulation_object):
        self.boundary = self.boundary_function(simulation_object)
        self.GWprocesses = {i: GW(element, self.pgf)
                            for i, element in enumerate(self.boundary)}
        self.statistics[0] = {'mean': len(self.boundary), 'p_stop': 0}

    def next_generation(self, children_function, policy):

        for process in self.GWprocesses.values():
            def process_policy(gw):
                return policy.effect(self, gw)

            process.add_generation(self.lattice_parameters,
                                   self.lattice_children, children_function,
                                   process_policy)

        self.generations += 1

        mean, p_stop = self.model_statistics()
        self.statistics[self.generations] = {'mean': mean, 'p_stop': p_stop}

    def model_statistics(self):
        p_stop = 1
        mean = 0

        for process in self.GWprocesses.values():
            p_stop *= process.generation_data[self.generations]['p_stop']
            generation_means = [process.generation_data[i]['mean'] for i in range(1, self.generations+1)]
            mean += np.prod(generation_means)

        return mean, p_stop

    def prediction(self):
        return self.statistics[self.generations]['mean'], self.statistics[self.generations]['p_stop']
