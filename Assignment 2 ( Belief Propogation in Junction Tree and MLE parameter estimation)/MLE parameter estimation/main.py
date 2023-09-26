""" CS5340 Lab 2 Part 2: Parameter Learning
See accompanying PDF for instructions.

Name: Parashara Ramesh
Email: e1216292@u.nus.edu
Student ID: A0285647M
"""
import copy
import os
from collections import OrderedDict

import numpy as np
import json
import networkx as nx
from argparse import ArgumentParser

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')  # we will store the input data files here!
OBSERVATION_DIR = os.path.join(DATA_DIR, 'observations')
PREDICTION_DIR = os.path.join(DATA_DIR, 'predictions')

""" ADD HELPER FUNCTIONS HERE """


def create_networkx_graph(nodes, edges):
    '''
    creates a directed graphical model given the nodes and the edges
    @param nodes:
    @param edges:
    @return:
    '''
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph


def get_parents_of_node(node, graph):
    '''

    @param node:
    @param graph: networkx graph
    @return:
    '''
    return list(graph.predecessors(node))


""" D ADD HELPER FUNCTIONS HERE """


def _learn_node_parameter_w(outputs, inputs=None):
    """
    Returns the weight parameters of the linear Gaussian [w0, w1, ..., wI], where I is the number of inputs. Students
    are encouraged to use numpy.linalg.solve() to get the weights. Learns weights for one node only.
    Call once for each node.

    NOTE to self: 'I' refers to the parent here!

    Args:
        outputs: numpy array of N output observations of the node
        inputs: N x I numpy array of input observations to the linear Gaussian model

    Returns:
        numpy array of (I + 1) weights [w0, w1, ..., wI]
    """
    # NOTE to TA: Commenting this out as I dont use this anywhere!
    # num_inputs = 0 if inputs is None else inputs.shape[1]
    # weights = np.zeros(shape=num_inputs + 1)

    """ YOUR CODE HERE """
    '''
    A = [
        equation 0,
        equation 1,
        equation 2,
        .
        .
        equation I (assuming there are I parents)
        ]
    
    #only for solving bias
    equation_0 =[N (for bias), ...sum(of all observations for parent i){for ith parent}..... ]
    b0 = sum(xu all observations)
    
    #for the ith parent
    equation_i = [sum(x_pi_i_observations){bias}, .. sum([x_pi_i_n* x_pi_{1->c}_n,...]) ..... ]
    bi = sum([x_u_n * x_pi_c_n for x_u_n, x_pi_c_n in zip(x_u_obs, x_pi_c_observations)])
      
    '''
    A = []
    b = []
    I = len(inputs[0]) if len(inputs) > 0 else 0
    N = len(outputs)

    # give better names
    node_observations = outputs
    observations_of_parents_of_node = inputs

    # initialize A & b with the equation 0 (for bias)
    equation0_bias_weight = [N]

    # now add based on parents across all N trials (if parents exist)!
    equation0_weights = list(np.sum(observations_of_parents_of_node, axis=0)) if len(
        observations_of_parents_of_node) > 0 else []

    # equations
    equations = [equation0_bias_weight + equation0_weights]

    A.append(equations[0])
    b.append(sum(node_observations))

    # we need to find I equations apart from one already A has
    for i in range(1, I + 1):
        parent_index = i-1

        # sum of all observations of the ith parent (just the ith index of equation0_weights basically)
        equation_i_bias_weight = [equation0_weights[parent_index]]

        # sum of the (nth observation of node_observations * nth observation of ith parent ) {basically dot product}...
        bi = np.dot(node_observations, observations_of_parents_of_node[:, parent_index])

        # array of [sum of the (nth observation of the ith parent * nth observation of cth parent){for all c parents} ,..]
        equation_i_weights = [
            np.dot(
                observations_of_parents_of_node[:, parent_index],
                observations_of_parents_of_node[:, c]
            )
            for c in range(I)  # basically go through all parents again.
        ]

        equations.append(equation_i_bias_weight + equation_i_weights)

        # add equation_i to A
        A.append(equations[-1])

        # add the ith bias
        b.append(bi)

    # using linear algebra solver
    weights = np.linalg.solve(np.array(A), np.array(b))

    """ END YOUR CODE HERE """

    return weights


def _learn_node_parameter_var(outputs, weights, inputs):
    """
    Returns the variance i.e. sigma^2 for the node. Learns variance for one node only. Call once for each node.

    Args:
        outputs: numpy array of N output observations of the node
        weights: numpy array of (I + 1) weights of the linear Gaussian model
        inputs:  N x I numpy array of input observations to the linear Gaussian model.

    Returns:
        variance of the node's Linear Gaussian model
    """
    var = 0.

    """ YOUR CODE HERE """
    node_bias = weights[0]
    parent_weights = np.array(weights[1:])
    do_parents_exist = len(parent_weights) > 0

    for n in range(len(outputs)):
        node_observation_for_nth_trial = outputs[n]
        nth_trail_node_mu = node_bias
        if do_parents_exist:
            parent_observations_for_nth_trial = inputs[n]
            nth_trail_node_mu += np.dot(parent_weights, parent_observations_for_nth_trial)

        var += (node_observation_for_nth_trial - nth_trail_node_mu) ** 2

    # divide by no of trials
    var /= len(outputs)

    """ END YOUR CODE HERE """

    return var


def _get_learned_parameters(nodes, edges, observations):
    """
    Learns the parameters for each node in nodes and returns the parameters as a dictionary. The nodes are given in
    ascending numerical order e.g. [1, 2, ..., V]

    Args:
        nodes: numpy array V nodes in the graph e.g. [1, 2, 3, ..., V]
        edges: numpy array of edges in the graph e.g. [i, j] implies i -> j where i is the parent of j
        observations: dictionary of node: observations pair where observations[1] returns a list of
                    observations for node 1.

    Returns:
        dictionary of parameters e.g.
        parameters = {
            "1": {  // first node
                "bias": w0 weight for node "1",
                "variance": variance for node "1"

                "2": weight for node "2", who is the parent of "1"
                ...
                // weights for other parents of "1"
            },
            ...
            // parameters of other nodes.
        }
    """
    # NOTE to TA: changing to OrderedDict in order for the json output to come correctly
    parameters = OrderedDict()

    """ YOUR CODE HERE """
    # Firstly create a networkx graph
    graph = create_networkx_graph(nodes, edges)

    # set up parameters in the expected manner (using dictionary comprehension)
    for node in nodes:
        parameters[node] = OrderedDict({
                "variance": None,
                "bias": None
                # other weights can also come
            })

    # for each node learn the parameters
    for node in nodes:
        node_observations = observations[node]
        parents_of_node = get_parents_of_node(node, graph)
        all_observations_for_parent_nodes = [observations[parent] for parent in parents_of_node]

        # get all parent's nth trial observations values
        parent_observations_per_trial = np.array(list(zip(*all_observations_for_parent_nodes)))

        # learn w0->wC for node u
        all_weights_of_a_node = _learn_node_parameter_w(node_observations, parent_observations_per_trial)

        # save bias
        node_bias = all_weights_of_a_node[0]
        parameters[node]["bias"] = node_bias

        # save the parent's weights
        weights_of_parents_of_node = all_weights_of_a_node[1:]
        for parent, parent_weight in zip(parents_of_node, weights_of_parents_of_node):
            parameters[node][parent] = parent_weight

        # now to find out the variance
        node_variance = _learn_node_parameter_var(node_observations, all_weights_of_a_node,
                                                  parent_observations_per_trial)

        # save variance
        parameters[node]["variance"] = node_variance

    """ END YOUR CODE HERE """

    return parameters


def main():
    """
    Helper function to load the observations, call your parameter learning function and save your results.
    DO NOT EDIT THIS FUNCTION.
    """
    argparser = ArgumentParser()
    argparser.add_argument('--case', type=int, required=True,
                           help='case number to create observations e.g. 1 if 1.json')
    args = argparser.parse_args()

    case = args.case
    observation_file = os.path.join(OBSERVATION_DIR, '{}.json'.format(case))
    with open(observation_file, 'r') as f:
        observation_config = json.load(f)

    nodes = observation_config['nodes']
    edges = observation_config['edges']
    observations = observation_config['observations']

    # solution part
    parameters = _get_learned_parameters(nodes=nodes, edges=edges, observations=observations)
    # end solution part

    # json only recognises floats, not np.float, so we need to cast the values into floats.
    for node, node_params in parameters.items():
        for param, val in node_params.items():
            node_params[param] = float(val)
        parameters[node] = node_params

    if not os.path.exists(PREDICTION_DIR):
        os.makedirs(PREDICTION_DIR)
    prediction_file = os.path.join(PREDICTION_DIR, '{}.json'.format(case))

    with open(prediction_file, 'w') as f:
        json.dump(parameters, f, indent=1)
    print('INFO: Results for test case {} are stored in {}'.format(case, prediction_file))


if __name__ == '__main__':
    main()
