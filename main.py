from collections import defaultdict
import numpy as np
import os
import sys
sys.path.insert(0, os.getcwd() + '/simulators')

from Analysis import BranchModel, binomial_pgf
from fires.LatticeForest import LatticeForest
from Policies import UBTfires, DWTfires, BFTfires


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


def percolation_parameter(a, b):
    return a/(1 - (1-a)*b)


def equivalent_gmdp_control(a, b, dp):
    p = percolation_parameter(a, b)
    return 0, b - (p - dp - a)/((1-a)*(p-dp))


def equivalent_percolation_control(a, b, delta_a, delta_b):
    return percolation_parameter(a, b) - percolation_parameter(a - delta_a, b - delta_b)


def uniform_lattice():
    # given alpha and beta, compute lattice probabilities for every (parent, child) pair
    a = 0.2763
    b = np.exp(-1/10)
    p = percolation_parameter(a, b)
    if p <= 0.5:
        raise Exception('Percolation paramter {0:0.2f} is not supercritical'.format(p))
    lattice_p = defaultdict(lambda: p)

    # given (delta_alpha, delta_beta), construct the equivalent delta_p
    delta_a = 0
    delta_b = 0.4
    dp = equivalent_percolation_control(a, b, delta_a, delta_b)
    if p - dp >= 0.5:
        raise Exception('Control is insufficient: p - dp = {0:0.2f} - {1:0.2f} = {2:0.2f}'.format(p, dp, p-dp))
    control_p = defaultdict(lambda: dp)
    control_ab = defaultdict(lambda: (delta_a, delta_b))

    # or given delta_p, construct the equivalent (delta_alpha, delta_beta)
    # delta_p = 0.4
    # control_percolation = defaultdict(lambda: delta_p)
    # control_gmdp = defaultdict(lambda: equivalent_gmdp_control(a, b, delta_p))

    a = defaultdict(lambda: a)
    b = defaultdict(lambda: b)
    return a, b, lattice_p, control_p, control_ab


def nonuniform_lattice(simulation):
    a = dict()
    b = defaultdict(lambda: np.exp(-1/10))
    lattice_p = dict()

    alpha_start = 0.2
    alpha_end = 0.3
    for r in range(simulation.dims[0]):
        for c in range(simulation.dims[1]):
            a[(r, c)] = alpha_start + (c/(simulation.dims[1]-1))*(alpha_end-alpha_start)

    delta_a = 0
    delta_b = 0.35
    control_p = dict()
    control_ab = defaultdict(lambda: (delta_a, delta_b))

    for tree_rc in simulation.group.keys():
        for j in simulation.group[tree_rc].neighbors:
            p = percolation_parameter(a[j], b[tree_rc])
            if p <= 0.5:
                raise Exception('Percolation parameter {0:0.2f} is not supercritical'.format(p))

            lattice_p[(tree_rc, j)] = p

            dp = equivalent_percolation_control(a[j], b[tree_rc], delta_a, delta_b)
            if p - dp >= 0.5:
                raise Exception('Control is insufficient: p - dp = {0:0.2f} - {1:0.2f} = {2:0.2f}'.
                                format(p, dp, p - dp))

            control_p[(tree_rc, j)] = dp

    return a, b, lattice_p, control_p, control_ab


def benchmark(simulation, branchmodel, policy, num_generations=1, num_simulations=1):

    results = []

    for seed in range(num_simulations):
        np.random.seed(seed)
        sim.reset()
        sim.rng = seed

        while not sim.early_end:
            branchmodel.reset()
            branchmodel.set_boundary(simulation)

            def children_function(parent):
                return latticeforest_children(sim, parent)

            for _ in range(num_generations):
                bm.next_generation(children_function, policy)

            # apply control and update simulator
            control = policy.control(sim, bm)
            sim.update(control)

        results.append(simulation.stats[0]/np.sum(simulation.stats))

    print('average remaining trees: {0:0.2f}%'.format(100*np.mean(results)))


# def generic_boundary(foo):
#     return [1]
#
#
# def generic_children(foo, bar):
#     return [2, 3]

if __name__ == '__main__':
    # bm = BranchModel(boundary_function=generic_boundary,
    #                  lattice_parameters=defaultdict(lambda: 0.2),
    #                  pgf=binomial_pgf)
    #
    # bm.reset()
    # bm.set_boundary(None)
    #
    # def children_function(parent):
    #     return generic_children(None, parent)
    #
    # for n in range(5):
    #     bm.next_generation(children_function, None)
    #     mean, p_stop = bm.statistics()
    #     # print(bm.GWprocesses[0].generation_data[n+1])
    #     print('generation %d: mean %0.4f, p_stop %0.4e' % (n+1, mean, p_stop))

    # forest size
    dimension = 50

    # generate information for uniform or non-uniform case
    # alpha, beta, lattice_parameters, control_percolation, control_gmdp = uniform_lattice()
    alpha, beta, lattice_parameters, control_percolation, control_gmdp = nonuniform_lattice(LatticeForest(dimension))
    sim = LatticeForest(dimension, alpha=alpha, beta=beta)

    # define policy
    # policy = UBTfires(capacity=10, control_map_percolation=control_percolation, control_map_gmdp=control_gmdp)
    # policy = DWTfires(capacity=10, control_map_percolation=control_percolation, control_map_gmdp=control_gmdp)
    policy = BFTfires(capacity=10, control_map_percolation=control_percolation, control_map_gmdp=control_gmdp)

    # instantiate branching process model approximation
    bm = BranchModel(boundary_function=latticeforest_boundary,
                     lattice_parameters=lattice_parameters,
                     pgf=binomial_pgf)

    # benchmark(sim, bm, policy, num_simulations=1)

    np.random.seed(3)
    for _ in range(10):
    # while not sim.early_end:

        bm.reset()
        bm.set_boundary(sim)

        print('sim iteration %d' % sim.iter)
        mean, p_stop = bm.prediction()
        print('generation {0:2d}: size {1:6.2f}'.format(0, mean))

        def children_function(parent):
            return latticeforest_children(sim, parent)

        for n in range(5):
            bm.next_generation(children_function, policy)
            mean, p_stop = bm.prediction()
            print('generation {0:2d}: mean {1:6.2f} | stop {2:5.2f}%'.format(n+1, mean, 100*p_stop))

        # apply control and update simulator
        control = policy.control(sim, bm)
        sim.update(control)
        print()

    print('remaining trees: {0:0.2f}%'.format(100*sim.stats[0]/np.sum(sim.stats)))
    print()
