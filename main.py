from collections import defaultdict
import numpy as np
import os
import sys
import time
sys.path.insert(0, os.getcwd() + '/simulators')

from Analysis import BranchModel, binomial_pgf
from fires.ForestElements import Tree
from fires.LatticeForest import LatticeForest
from fires.UrbanForest import UrbanForest
from Policies import *

np.seterr(all='raise')


def latticeforest_fireboundary(latticeforest):
    boundary = []
    for fire in latticeforest.fires:
        element = latticeforest.group[fire]
        neighbors = [latticeforest.group[j].is_healthy(latticeforest.group[j].state) for j in element.neighbors
                     if any(isinstance(latticeforest.group[j], t) for t in element.neighbors_types)]
        if neighbors.count(True) > 0:
            boundary.append(fire)

    return boundary


def latticeforest_urbanboundary(urbanforest):
    boundary = []

    for urban in urbanforest.urban_elements:
        element = urbanforest.group[urban]

        if element.state == 0:
            count = 0
            for j in element.neighbors:
                neighbor_element = urbanforest.group[j]
                if isinstance(neighbor_element, Tree):
                    if neighbor_element.is_healthy(neighbor_element.state) \
                            or neighbor_element.is_on_fire(neighbor_element.state):
                        count += 1

        if count > 0:
            boundary.append(urban)

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
        raise Warning('Percolation paramter {0:0.2f} is not supercritical'.format(p))
    lattice_p = defaultdict(lambda: p)

    # given (delta_alpha, delta_beta), construct the equivalent delta_p
    delta_a = 0
    delta_b = 0.4
    dp = equivalent_percolation_control(a, b, delta_a, delta_b)
    if p - dp >= 0.5:
        raise Warning('Control is insufficient: p - dp = {0:0.2f} - {1:0.2f} = {2:0.2f}'.format(p, dp, p-dp))
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
    b = defaultdict(lambda: np.exp(-1/9))
    lattice_p = dict()

    delta_b = 0.35
    control_ab = dict()

    alpha_start = 0.2
    alpha_end = 0.3
    for r in range(simulation.dims[0]):
        for c in range(simulation.dims[1]):
            a[(r, c)] = alpha_start + (c/(simulation.dims[1]-1))*(alpha_end-alpha_start)

            control_ab[(r, c)] = (a[(r, c)], delta_b)

    control_p = dict()
    for tree_rc in simulation.group.keys():
        for j in simulation.group[tree_rc].neighbors:
            p = percolation_parameter(a[j], b[tree_rc])
            if p <= 0.5:
                raise Warning('Percolation parameter {0:0.2f} is not supercritical'.format(p))

            lattice_p[(tree_rc, j)] = p

            da, db = control_ab[j]
            dp = equivalent_percolation_control(a[j], b[tree_rc], da, db)
            if p - dp >= 0.5:
                raise Warning('Control is insufficient: p - dp = {0:0.2f} - {1:0.2f} = {2:0.2f}'.
                                format(p, dp, p - dp))

            control_p[(tree_rc, j)] = dp

    return a, b, lattice_p, control_p, control_ab


def benchmark(simulation, branchmodel, policy, num_generations=1, num_simulations=1):

    print('started at {0:s}'.format(time.strftime('%d-%b-%Y %H:%M')))
    tic = time.clock()
    results = []

    for seed in range(num_simulations):
        np.random.seed(seed)
        sim.reset()
        sim.rng = seed

        while not sim.early_end:
            branchmodel.reset()
            branchmodel.set_boundary(latticeforest_fireboundary(simulation))

            def children_function(parent):
                return latticeforest_children(sim, parent)
            branchmodel.set_children_function(children_function)

            for _ in range(num_generations):
                branchmodel.next_generation(policy)

            # apply control and update simulator
            sim.update(policy.control(sim, branchmodel))

        if (seed+1) % 10 == 0:
            print('completed {0:d} simulations'.format((seed+1)))

        results.append(simulation.stats[0]/np.sum(simulation.stats))

    toc = time.clock()
    dt = toc - tic
    print('finished at {0:s}'.format(time.strftime('%d-%b-%Y %H:%M')))
    print('{0:0.2f}s = {1:0.2f}m = {2:0.2f}h elapsed'.format(dt, dt/60, dt/3600))

    print('median remaining trees: {0:0.2f}%'.format(100*np.median(results)))
    print('mean remaining trees: {0:0.2f}%'.format(100*np.mean(results)))
    print('minimum {0:0.2f}, maximum {1:0.2f}'.format(100*np.amin(results), 100*np.amax(results)))
    first, third = np.percentile(results, [25, 75])
    print('1st quartile {0:0.2f}, 3rd quartile {1:0.2f}'.format(100*first, 100*third))
    return


if __name__ == '__main__':

    # forest size
    dimension = 50

    # generate information for uniform or non-uniform case
    # alpha, beta, lattice_parameters, control_percolation, control_gmdp = uniform_lattice()
    alpha, beta, lattice_parameters, map_percolation, map_gmdp = nonuniform_lattice(LatticeForest(dimension))
    # sim = LatticeForest(dimension, alpha=alpha, beta=beta)
    sim = UrbanForest(dimension, alpha=alpha, beta=beta)

    # define policy
    cap = 7
    # pi = UBTfires(capacity=cap, control_map_percolation=map_percolation, control_map_gmdp=map_gmdp)
    # pi = DWTfires(capacity=cap, control_map_percolation=map_percolation, control_map_gmdp=map_gmdp)
    # pi = BFTfires(capacity=cap, control_map_percolation=map_percolation, control_map_gmdp=map_gmdp)
    # pi = RHTfires(capacity=cap, horizon=3, control_map_percolation=map_percolation, control_map_gmdp=map_gmdp)
    pi = UPTfires(capacity=cap, horizon=5, control_map_percolation=map_percolation, control_map_gmdp=map_gmdp)

    # create branching process model approximation
    bm = BranchModel(lattice_parameters=lattice_parameters, pgf=binomial_pgf)

    # benchmark(sim, bm, pi, num_generations=1, num_simulations=500)

    np.random.seed(7)
    for _ in range(50):
    # while not sim.early_end:

        bm.reset()
        bm.set_boundary(latticeforest_fireboundary(sim))

        pi.urbanboundary = latticeforest_urbanboundary(sim)
        print(len(pi.urbanboundary))

        print('sim iteration %d' % sim.iter)
        mean, p_stop = bm.prediction()
        print('generation {0:2d}: size {1:6.2f}'.format(0, mean))

        def children_function(parent):
            return latticeforest_children(sim, parent)
        bm.set_children_function(children_function)

        for n in range(2):
            bm.next_generation(pi)
            mean, p_stop = bm.prediction()
            print('generation {0:2d}: mean {1:6.2f} | stop {2:5.2f}%'.format(n+1, mean, 100*p_stop))

        # apply control and update simulator
        control, urban_control = pi.control(sim, bm)
        for ub in urban_control:
            print(ub, sim.group[ub].state, control[ub])

        sim.update(control)

        for ub in urban_control:
            print(ub, sim.group[ub].state, control[ub])

        print()

    print('remaining trees: {0:0.2f}%'.format(100*sim.stats[0]/np.sum(sim.stats)))

    ub_states = []
    for ub in sim.urban_elements:
        ub_states.append(sim.group[ub].state)

    print([True if ub == 3 else False for ub in ub_states].count(True))
    print([True if ub == 0 else False for ub in ub_states].count(True))

    print()
