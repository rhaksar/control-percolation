import os
import sys
sys.path.insert(0, os.getcwd() + '/simulators')

from fires.ForestElements import Tree
    

def fire_boundary(forest_sim):
    boundary = []

    for fire in forest_sim.fires:
        element = forest_sim.group[fire]
        neighbors = [forest_sim.group[j].is_healthy(forest_sim.group[j].state) for j in element.neighbors
                     if any(isinstance(forest_sim.group[j], t) for t in element.neighbors_types)]
        if neighbors.count(True) > 0:
            boundary.append(fire)

    return boundary


def urban_boundary(forest_sim):
    boundary = []

    for urban in forest_sim.urban:
        element = forest_sim.group[urban]
        if not element.is_healthy(element.state):
            continue

        count = 0
        for j in element.neighbors:
            neighbor = forest_sim.group[j]
            if isinstance(neighbor, Tree) and neighbor.is_healthy(neighbor.state):
                count += 1
            elif neighbor.is_on_fire(neighbor.state):
                count += 1

        if count > 0:
            boundary.append(urban)

    return boundary


def forest_children(forest_sim, i):
    neighbors = forest_sim.group[i].neighbors
    children = [j for j in neighbors if forest_sim.group[j].is_healthy(forest_sim.group[j].state)]

    return children


def percolation_parameter(a, b):
    return a/(1 - (1-a)*b)


# def equivalent_gmdp_control(a, b, dp):
#     p = percolation_parameter(a, b)
#     return 0, b - (p - dp - a)/((1-a)*(p-dp))


def equivalent_percolation_control(a, b, delta_a, delta_b):
    return percolation_parameter(a, b) - percolation_parameter(a - delta_a, b - delta_b)