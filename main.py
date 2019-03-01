from collections import defaultdict
import itertools
import numpy as np
import os
import pickle
import sys
import time
import warnings
sys.path.insert(0, os.getcwd() + '/simulators')

from Analysis import *
from fires.LatticeForest import LatticeForest
from fires.UrbanForest import UrbanForest
from Policies import *
from Utilities import *

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

    beta1 = np.exp(-1/5)
    beta2 = np.exp(-1/10)
    for r in range(dimension):
        for c in range(dimension):
            if c < dimension-simulation.urban_width:
                beta_set[(r, c)] = beta1
            else:
                beta_set[(r, c)] = beta2

            control_gmdp[(r, c)] = {'healthy': (alpha_set[(r, c)], 0),
                                    'on_fire': (0, np.amin([delta_beta, beta_set[(r, c)]]))}

    # control_p = dict()
    for tree_rc in simulation.group.keys():
        for neighbor in simulation.group[tree_rc].neighbors:
            p = percolation_parameter(alpha_set[neighbor], beta_set[tree_rc])
            if p <= 0.5:
                warnings.warn('p({0:0.2f}, {1:0.2f}) = {2:0.2f} <= 0.5'.format(alpha_set[neighbor],
                                                                               beta_set[tree_rc], p))

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

    print('Running policy {0:s} with capacity {1:d} for {2:d} simulations'.format(policy.name,
                                                                                  policy.capacity,
                                                                                  num_simulations))

    print('started at {0:s}'.format(time.strftime('%d-%b-%Y %H:%M')))
    tic = time.clock()
    results = dict()

    staticmodel = StaticModel()

    for seed in range(num_simulations):
        np.random.seed(seed)
        simulation.reset()
        simulation.rng = seed

        while not simulation.early_end:
            branchmodel.reset()
            branchmodel.set_boundary(fire_boundary(simulation))

            if isinstance(policy, USTfires):
                staticmodel.set_boundary(urban_boundary(simulation))
                policy.urbanboundary = urban_boundary(simulation)

            def children_function(p):
                return forest_children(simulation, p)
            branchmodel.set_children_function(children_function)

            for _ in range(num_generations):
                for process in branchmodel.GWprocesses.values():
                    for parent in process.current_parents:
                        if parent not in branchmodel.lattice_children:
                            branchmodel.lattice_children[parent] = branchmodel.children_function(parent)

                if not isinstance(policy, USTfires):
                    policy.generate_map(branchmodel)
                else:
                    policy.generate_map(branchmodel, staticmodel)

                branchmodel.next_generation(policy)

                if isinstance(policy, USTfires):
                    staticmodel.next_boundary(policy.control_decisions)

            # apply control and update simulator
            if not isinstance(policy, USTfires):
                control = policy.control(branchmodel)
            else:
                control = policy.control(branchmodel, staticmodel)
            simulation.update(control)

        if (seed+1) % 10 == 0:
            print('completed {0:d} simulations'.format((seed+1)))

        results[seed] = {'healthy_trees': simulation.stats_trees[0]/np.sum(simulation.stats_trees),
                         'healthy_urban': simulation.stats_urban[0]/np.sum(simulation.stats_urban),
                         'razed_urban': simulation.stats_urban[3]/np.sum(simulation.stats_urban)}

    toc = time.clock()
    dt = toc - tic
    print('finished at {0:s}'.format(time.strftime('%d-%b-%Y %H:%M')))
    print('{0:0.2f}s = {1:0.2f}m = {2:0.2f}h elapsed'.format(dt, dt/60, dt/3600))

    filename = policy.name + '_C' + str(policy.capacity) + '_s' + str(num_simulations) + '.pkl'
    output = open('results/' + filename, 'wb')
    pickle.dump(results, output)
    output.close()

    print('median healthy trees: {0:0.2f}%'.format(100*np.median([results[s]['healthy_trees']
                                                                  for s in results.keys()])))
    print('median healthy urban developments: {0:0.2f}%'.format(100*np.median([results[s]['healthy_urban']
                                                                               for s in results.keys()])))
    print('median removed urban developments: {0:0.2f}%'.format(100*np.median([results[s]['razed_urban']
                                                                             for s in results.keys()])))
    # print('mean remaining trees: {0:0.2f}%'.format(100*np.mean(results)))
    # print('minimum {0:0.2f}, maximum {1:0.2f}'.format(100*np.amin(results), 100*np.amax(results)))
    # first, third = np.percentile(results, [25, 75])
    # print('1st quartile {0:0.2f}, 3rd quartile {1:0.2f}'.format(100*first, 100*third))
    return


if __name__ == '__main__':

    # forest parameters
    dimension = 50
    urban_width = 10

    # set initial condition
    initial_fire = []
    r_center = np.floor((dimension - 1) / 2).astype(np.uint8)
    c_center = np.floor((dimension - 1) / 2).astype(np.uint8)

    delta_r = [k for k in range(-2, 3)]
    delta_c = [k for k in range(-2, 3)]
    deltas = itertools.product(delta_r, delta_c)

    for (dr, dc) in deltas:
        if dr == 0 and dc == 0:
            continue
        elif (dr == -2 or dr == 2) and (dc == -2 or dc == 2):
            continue
        elif dc == dr or dc == -dr:
            continue

        r, c = r_center + dr, c_center + dc
        initial_fire.append((r, c))

    # generate information for uniform or non-uniform case
    # alpha, beta, lattice_parameters, control_percolation, control_gmdp = uniform(LatticeForest(dimension))
    # alpha, beta, p_parameters, map_percolation, map_gmdp = nonuniform(LatticeForest(dimension))
    alpha, beta, map_gmdp, p_parameters = nonuniform(UrbanForest(dimension, urban_width))

    # sim = LatticeForest(dimension, alpha=alpha, beta=beta)
    sim = UrbanForest(dimension, urban_width, initial_fire=initial_fire, alpha=alpha, beta=beta)

    # define policy
    cap = 6
    # pi = NCTfires(capacity=cap, alpha_set=alpha, beta_set=beta, control_map_gmdp=map_gmdp)
    # pi = UBTfires(capacity=cap, alpha_set=alpha, beta_set=beta, control_map_gmdp=map_gmdp)
    # pi = DWTfires(capacity=cap, alpha_set=alpha, beta_set=beta, control_map_gmdp=map_gmdp)
    # pi = RHTfires(capacity=cap, horizon=1, alpha_set=alpha, beta_set=beta, control_map_gmdp=map_gmdp)
    # pi = USTfires(capacity=cap, horizon=5, control_map_gmdp=map_gmdp, alpha_set=alpha, beta_set=beta)

    # create branching process model approximation
    bm = BranchModel(lattice_parameters=p_parameters, pgf=binomial_pgf)
    sm = StaticModel()

    benchmark(sim, bm, pi, num_generations=1, num_simulations=10)

    # np.random.seed(4)
    # # for _ in range(45):  # 50
    # while not sim.early_end:
    #
    #     bm.reset()
    #
    #     bm.set_boundary(fire_boundary(sim))
    #     print('fire boundary size:', len(bm.boundary))
    #
    #     if isinstance(pi, USTfires):
    #         sm.set_boundary(urban_boundary(sim))
    #         pi.urbanboundary = urban_boundary(sim)
    #         print('urban boundary size:', len(sm.boundary))
    #
    #     print('sim iteration %d' % sim.iter)
    #     mean, p_stop = bm.prediction()
    #     print('generation {0:2d}: size {1:6.2f}'.format(0, mean))
    #
    #     def children_function(parent):
    #         return forest_children(sim, parent)
    #     bm.set_children_function(children_function)
    #
    #     for n in range(1):
    #         for process in bm.GWprocesses.values():
    #             for parent in process.current_parents:
    #                 if parent not in bm.lattice_children:
    #                     bm.lattice_children[parent] = bm.children_function(parent)
    #
    #         if not isinstance(pi, USTfires):
    #             pi.generate_map(bm)
    #         else:
    #             pi.generate_map(bm, sm)
    #
    #         bm.next_generation(pi)
    #         mean, p_stop = bm.prediction()
    #         print('generation {0:2d}: mean {1:6.2f} | stop {2:5.2f}%'.format(n+1, mean, 100*p_stop))
    #
    #         if isinstance(pi, USTfires):
    #             sm.next_boundary(pi.control_decisions)
    #
    #     # apply control and update simulator
    #     if not isinstance(pi, USTfires):
    #         control = pi.control(bm)
    #     else:
    #         control = pi.control(bm, sm)
    #     dense_state = sim.dense_state()
    #
    #     sim.update(control)
    #
    #     dense_state = sim.dense_state()
    #     print()

    # print('remaining trees: {0:0.2f}%'.format(100*sim.stats[0]/np.sum(sim.stats)))
    #
    # ub_states = []
    # for ub in sim.urban:
    #     ub_states.append(sim.group[ub].state)
    #
    # print('removed urban areas:', [True if ub == 3 else False for ub in ub_states].count(True))
    # print('remaining urban areas:', [True if ub == 0 else False for ub in ub_states].count(True))

    # dense_state = sim.dense_state()
    # print(sim.stats_trees)
    # print(sim.stats_urban)
    # print(len(sim.fires))
    #
    # print('percent healthy trees: {0:0.2f}%'.format(100*sim.stats_trees[0]/np.sum(sim.stats_trees)))
    # print('percent healthy urban developments: {0:0.2f}%'.format(100*sim.stats_urban[0]/np.sum(sim.stats_urban)))
    # print('percent razed urban developments: {0:0.2f}%'.format(100*sim.stats_urban[3]/np.sum(sim.stats_urban)))
    #
    print()
