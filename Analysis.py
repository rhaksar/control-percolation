import numpy as np


def binomial_pgf(x, p, s):
    return np.power(1 + (x-1)*p, s)


class GW(object):

    def __init__(self, root, pgf):
        self.root = root
        self.current_parents = [self.root]
        self.history = [self.root]

        self.pgf = pgf

        self.generations = 0
        self.generation_data = {}

    def add_generation(self, lattice_parameters, lattice_children, children_function):
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

        sigma = len(all_children)
        if sigma > 0:
            p_children /= sigma

        if self.generations == 1:
            p_stop = self.pgf(0, p_children, sigma)

        else:
            p_stop_previous = self.generation_data[self.generations-1]['p_stop']
            p_stop = self.pgf(p_stop_previous, p_children, sigma)

        self.generation_data[self.generations] = {'p_children': p_children,
                                                  'sigma': sigma,
                                                  'mean': p_children*sigma,
                                                  'p_stop': p_stop}

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

        self.generations = 0

    def reset(self):
        self.__init__(self.boundary_function, self.lattice_parameters, self.pgf)

    def set_boundary(self, simulation_object):
        self.boundary = self.boundary_function(simulation_object)
        self.GWprocesses = {i: GW(element, self.pgf)
                            for i, element in enumerate(self.boundary)}

    def next_generation(self, children_function, policy):

        for process in self.GWprocesses.values():
            process.add_generation(self.lattice_parameters, self.lattice_children, children_function)

        self.generations += 1

    def statistics(self):
        p_stop = 1
        mean = 0

        for process in self.GWprocesses.values():
            p_stop *= process.generation_data[self.generations]['p_stop']
            generation_means = [process.generation_data[i]['mean'] for i in range(1, self.generations+1)]
            # print(generation_means, np.prod(generation_means))
            mean += np.prod(generation_means)

        return mean, p_stop
