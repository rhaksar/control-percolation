from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import warnings
sys.path.insert(0, os.getcwd() + '/simulators')

from Analysis import StaticModel, BranchModel, binomial_pgf
from fires.LatticeForest import LatticeForest
from fires.UrbanForest import UrbanForest
from Utilities import *
from Policies import *

np.seterr(all='raise')


def uniform():
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


def nonuniform(simulation):
    alpha_set = dict()
    # beta_set = defaultdict(lambda: np.exp(-1/9))
    beta_set = dict()
    p_set = dict()

    delta_beta = 0.35
    control_gmdp = dict()

    alpha_start = 0.2
    alpha_end = 0.4
    for r in range(simulation.dims[0]):
        for c in range(simulation.dims[1]):
            alpha_set[(r, c)] = alpha_start + (c/(simulation.dims[1]-1))*(alpha_end-alpha_start)

            control_gmdp[(r, c)] = {'healthy': (alpha_set[(r, c)], 0), 'on_fire': (0, delta_beta)}

    beta1 = np.exp(-1/5)
    beta2 = np.exp(-1/10)
    for r in range(dimension):
        for c in range(dimension):
            if c < dimension-simulation.urban_width:
                beta_set[(r, c)] = beta1
            else:
                beta_set[(r, c)] = beta2

    # control_p = dict()
    for tree_rc in simulation.group.keys():
        for neighbor in simulation.group[tree_rc].neighbors:
            p = percolation_parameter(alpha_set[neighbor], beta_set[tree_rc])
            if p <= 0.5:
                warnings.warn('p = {0:0.2f} <= 0.5'.format(p))

            p_set[(tree_rc, neighbor)] = p
            # control_p[(tree_rc, neighbor)] = dict()
            #
            # for k in control_gmdp[neighbor].keys():
            #     da, db = control_gmdp[neighbor][k]
            #     dp = equivalent_percolation_control(alpha_set[neighbor], beta_set[tree_rc], da, db)
            #     if p - dp >= 0.5:
            #         warnings.warn('p - dp = {0:0.2f} - {1:0.2f} = {2:0.2f} >= 0.5'.format(p, dp, p - dp))
            #
            #     control_p[(tree_rc, neighbor)][k] = dp

    return alpha_set, beta_set, control_gmdp, p_set


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
            branchmodel.set_boundary(fire_boundary(simulation))

            def children_function(parent):
                return forest_children(sim, parent)
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

    # forest parameters
    dimension = 50
    urban_width = 10

    # generate information for uniform or non-uniform case
    # alpha, beta, lattice_parameters, control_percolation, control_gmdp = uniform(LatticeForest(dimension))
    # alpha, beta, p_parameters, map_percolation, map_gmdp = nonuniform(LatticeForest(dimension))
    alpha, beta, map_gmdp, p_parameters = nonuniform(UrbanForest(dimension, urban_width))

    # sim = LatticeForest(dimension, alpha=alpha, beta=beta)
    sim = UrbanForest(dimension, urban_width, alpha=alpha, beta=beta)

    # define policy
    cap = 6
    # pi = UBTfires(capacity=cap, alpha_set=alpha, beta_set=beta, control_map_gmdp=map_gmdp)
    # pi = DWTfires(capacity=cap, alpha_set=alpha, beta_set=beta, control_map_gmdp=map_gmdp)
    # pi = BFTfires(capacity=cap, control_map_percolation=map_percolation, control_map_gmdp=map_gmdp)
    pi = RHTfires(capacity=cap, horizon=3, alpha_set=alpha, beta_set=beta, control_map_gmdp=map_gmdp)

    # urban preservation policy requires a different setup
    # pi = UPTfires(capacity=cap, horizon=5, control_map_gmdp=map_gmdp, alpha_set=alpha, beta_set=beta)

    # create branching process model approximation
    bm = BranchModel(lattice_parameters=p_parameters, pgf=binomial_pgf)
    # sm = StaticModel()

    # benchmark(sim, bm, pi, num_generations=1, num_simulations=500)

    np.random.seed(10)
    # for _ in range(45):  # 50
    while not sim.early_end:

        bm.reset()
    
        bm.set_boundary(fire_boundary(sim))
        # sm.set_boundary(urban_boundary(sim))
        # pi.urbanboundary = urban_boundary(sim)
        print('fire boundary size:', len(bm.boundary))
        # print('urban boundary size:', len(sm.boundary))

        print('sim iteration %d' % sim.iter)
        mean, p_stop = bm.prediction()
        print('generation {0:2d}: size {1:6.2f}'.format(0, mean))

        def children_function(parent):
            return forest_children(sim, parent)
        bm.set_children_function(children_function)

        for n in range(2):
            for process in bm.GWprocesses.values():
                for parent in process.current_parents:
                    if parent not in bm.lattice_children:
                        bm.lattice_children[parent] = bm.children_function(parent)

            pi.generate_map(bm)
            # pi.generate_map(bm, sm)

            bm.next_generation(pi)
            mean, p_stop = bm.prediction()
            print('generation {0:2d}: mean {1:6.2f} | stop {2:5.2f}%'.format(n+1, mean, 100*p_stop))

            # sm.next_boundary(pi.control_decisions)

        # apply control and update simulator
        control = pi.control(bm)
        # control = pi.control(bm, sm)
        dense_state = sim.dense_state()

        sim.update(control)

        dense_state = sim.dense_state()
        print()

    print('remaining trees: {0:0.2f}%'.format(100*sim.stats[0]/np.sum(sim.stats)))

    ub_states = []
    for ub in sim.urban:
        ub_states.append(sim.group[ub].state)

    print('removed urban areas:', [True if ub == 3 else False for ub in ub_states].count(True))
    print('remaining urban areas:', [True if ub == 0 else False for ub in ub_states].count(True))

    dense_state = sim.dense_state()

    print()
