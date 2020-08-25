import os
import sys
sys.path.insert(0, os.getcwd() + '/simulators')

from fires.ForestElements import Tree
    

def fire_boundary(forest_sim):
    """
    Generate the fire boundary for the current time step of the simulation.

    :param forest_sim: simulation object for the lattice-based forest fire simulator.
    :return: a list of (row, column) tuples which are the lattice locations for each boundary element.
    """
    boundary = []

    for fire in forest_sim.fires:
        element = forest_sim.group[fire]
        neighbors = [forest_sim.group[j].is_healthy(forest_sim.group[j].state) for j in element.neighbors
                     if any(isinstance(forest_sim.group[j], t) for t in element.neighbors_types)]
        if neighbors.count(True) > 0:
            boundary.append(fire)

    return boundary


def urban_boundary(forest_sim):
    """
    Generate the urban boundary for the current time step of the simulation.

    :param forest_sim: simulation object for the lattice-based urban forest fire simulator.
    :return: a list of (row, column) tuples which are the lattice locations for each boundary element.
    """
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
    """
    Generate the children for each boundary element.

    :param forest_sim: simulator object for the lattice-based forest fire simulator.
    :param i: a boundary element which is the parent of the generated children.
    :return: a list of (row, column) tuples which are the lattice locations for each child.
    """
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