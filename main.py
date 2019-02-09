from collections import defaultdict
import os
import sys
sys.path.insert(0, os.getcwd() + '/simulators')

from Analysis import BranchModel, binomial_pgf
from fires.LatticeForest import LatticeForest
from Policies import UBTfires


def latticeforest_boundary(latticeforest):
    boundary = []
    for fire in latticeforest.fires:
        element = latticeforest.group[fire]
        neighbors = [latticeforest.group[j].is_healthy(latticeforest.group[j].state) for j in element.neighbors
                     if any(isinstance(latticeforest.group[j], t) for t in element.neighbors_types)]
        if neighbors.count(True) > 0:
            boundary.append(fire)

    return boundary


def latticeforest_children(latticeforest, tree):
    neighbors = latticeforest.group[tree].neighbors
    children = [j for j in neighbors if latticeforest.group[j].is_healthy(latticeforest.group[j].state)]

    return children


if __name__ == '__main__':
    dimension = 3

    # given alpha and beta, compute lattice probabilities for every (parent, child) pair
    lattice_parameters = defaultdict(lambda: 0.5)

    sim = LatticeForest(dimension, alpha=None, beta=None)

    policy = None

    bm = BranchModel(boundary_function=latticeforest_boundary,
                     lattice_parameters=lattice_parameters,
                     pgf=binomial_pgf)

    bm.reset()
    bm.set_boundary(sim)

    print('sim iteration %d' % sim.iter)
    print('number of boundary nodes: %d' % len(bm.boundary))

    def children_function(parent):
        return latticeforest_children(sim, parent)

    for n in range(10):
        bm.next_generation(children_function, policy)
        mean, p_stop = bm.statistics()
        print('generation %d: mean %d, p_stop %0.4e' % (n+1, mean, p_stop))

    print()
