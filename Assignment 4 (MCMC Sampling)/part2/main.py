""" CS5340 Lab 4 Part 2: Gibbs Sampling
See accompanying PDF for instructions.

Name: Parashara Ramesh
Email: e1216292@u.nus.edu
Student ID: A0285647M
"""

import copy
import os
import json
import numpy as np
import networkx as nx
from tqdm import tqdm
from collections import Counter, defaultdict
from argparse import ArgumentParser
from factor_utils import factor_evidence, factor_marginalize, assignment_to_index, index_to_assignment
from factor import Factor
import matplotlib.pyplot as plt

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
INPUT_DIR = os.path.join(DATA_DIR, 'inputs')
PREDICTION_DIR = os.path.join(DATA_DIR, 'predictions')
GROUND_TRUTH_DIR = os.path.join(DATA_DIR, 'ground-truth')

""" HELPER FUNCTIONS HERE """


def visualize_graph(graph, weighted=False):
    pos = nx.spring_layout(graph)
    nx.draw_networkx(graph, pos=pos, with_labels=True, font_weight='bold',
                     node_size=1000, arrowsize=20)
    # only for weighted
    if weighted:
        edge_labels = {(u, v): d['weight'] for u, v, d in graph.edges(data=True)}
        nx.draw_networkx_edge_labels(graph, pos=pos, edge_labels=edge_labels)
    plt.axis('off')
    plt.show()


def construct_factor_graph(nodes, edges):
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph


def markov_blanket(graph, node):
    parents = list(graph.predecessors(node))
    children = list(graph.successors(node))
    co_parents = []

    for child in children:
        co_parents.extend(list(graph.predecessors(child)))

    mb = parents + children + co_parents + [node]

    # Remove duplicates and ensures uniqueness
    mb = list(set(mb))
    return mb


def calculate_conditional_prob(actual_sampled_states, var_and_card, factor):
    #count no of times each state occurs
    state_counts = defaultdict(int)
    for sampled_state in actual_sampled_states:
        state_key = tuple(sorted(sampled_state.items()))
        if state_key not in state_counts:
            state_counts[state_key] = 0
        else:
            state_counts[state_key] += 1

    #calculate state_probs
    normalizer = sum(state_counts.values())
    state_probs = {state: count/normalizer for state,count in state_counts.items()}

    #from state_probs create a factor
    factor.var = np.array(sorted(var_and_card.keys()))
    factor.card = np.array([var_and_card[var] for var in factor.var])

    #now come up with the vals based on each of the assignment
    all_var_state_configurations = factor.get_all_assignments()  # to ensure that we fill the val in the correct order
    factor.val = []
    for var_state_configuration in all_var_state_configurations:
        # Construct the key in order to use it to get the value from state_probs since there the key is in the format ((var, state),..)
        key_to_search_with = []
        for var, var_state in zip(factor.var, var_state_configuration):
            key_to_search_with.append((var, var_state))
        key_to_search_with = tuple(key_to_search_with)

        # Get the probability of this state from the state_probs dict
        prob_of_state_configuration = state_probs[key_to_search_with]
        factor.val.append(prob_of_state_configuration)

    #make it into a np array for consistency sake
    factor.val = np.array(factor.val)

    return factor


""" END HELPER FUNCTIONS HERE"""


def _sample_step(nodes, factors, in_samples):
    """
    Performs gibbs sampling for a single iteration. Returns a sample for each node

    Args:
        nodes: numpy array of nodes
        factors: dictionary of factors e.g. factors[x1] returns the local factor for x1
        in_samples: dictionary of input samples (from previous iteration)

    Returns:
        dictionary of output samples where samples[x1] returns the sample for x1.
    """
    samples = copy.deepcopy(in_samples)

    """ YOUR CODE HERE """
    for node in nodes:
        '''
        For current node ,all of the nodes before would have the updated value, but the nodes which comes after needs to be updated
        '''
        factor = factors[node]
        other_nodes = list(set(nodes).difference(set([node])))
        evidence_based_on_other_nodes = {node: samples[node] for node in other_nodes}
        node_row_factor = factor_evidence(factor, evidence_based_on_other_nodes)
        node_row_probs = node_row_factor.val

        # Sample a state based on the probabilities
        states = list(range(len(node_row_probs)))
        sampled_state = np.random.choice(states, p=node_row_probs)
        samples[node] = sampled_state

    """ END YOUR CODE HERE """

    return samples


def _get_conditional_probability(nodes, edges, factors, evidence, initial_samples, num_iterations, num_burn_in):
    """
    Returns the conditional probability p(Xf | Xe) where Xe is the set of observed nodes and Xf are the query nodes
    i.e. the unobserved nodes. The conditional probability is approximated using Gibbs sampling.

    Additionally reduce the proposal distributions for each node to its
    Markov Blanket and the graph structure must be updated with
    the evidence variables.

    Args:
        nodes: numpy array of nodes e.g. [x1, x2, ...].
        edges: numpy array of edges e.g. [i, j] implies that nodes[i] is the parent of nodes[j].
        factors: dictionary of Factors e.g. factors[x1] returns the conditional probability of x1 given all other nodes.
        evidence: dictionary of evidence e.g. evidence[x4] returns the provided evidence for x4.
        initial_samples: dictionary of initial samples to initialize Gibbs sampling.
        num_iterations: number of sampling iterations
        num_burn_in: number of burn-in iterations

    Returns:
        returns Factor of conditional probability.
    """
    assert num_iterations > num_burn_in
    conditional_prob = Factor()

    """ YOUR CODE HERE """
    # 1.a Construct the graph
    print("Constructing factor graph & observing evidence")
    factor_graph = construct_factor_graph(nodes, edges)

    # 1.b visualize graphs
    # visualize_graph(factor_graph)

    var_and_card = {node: None for node in nodes}

    # 2. Observe the evidence for all the factors & marginalize away everything not in markov blanket
    for node, factor in factors.items():
        # find the variable and its cardinality
        idx_in_var = np.where(factor.var == node)[0][0]
        var_and_card[node] = factor.card[idx_in_var]

        # observe evidence
        observed_factor = factor_evidence(factor, evidence)

        # find markov blanket
        markov_blanket_nodes = markov_blanket(factor_graph, node)

        # marginalize away everything not in the markov blanket nodes
        other_nodes = list(set(nodes).difference(set(markov_blanket_nodes)))
        marginalized_factor = factor_marginalize(observed_factor, other_nodes)

        # normalize
        marginalized_factor.val = marginalized_factor.val / np.sum(marginalized_factor.val)

        # update factors
        factors[node] = marginalized_factor

    # 3. Do gibbs sampling
    burn_in_sampled_states = []
    actual_sampled_states = []

    in_samples = {node: 0 for node in nodes}

    for t in tqdm(range(num_burn_in + num_iterations)):  # for time t from [0,burn_in+num_iterations-1]
        if t == num_burn_in:
            print("Just finished the burn_in period! Now starting actual sampling")

        # Sample it
        in_samples = _sample_step(nodes, factors, in_samples)

        if t < num_burn_in:
            # keep collecting in burn_insamples in another array
            burn_in_sampled_states.append(in_samples)
        else:
            # keep collecting in burn_insamples in another array
            actual_sampled_states.append(in_samples)

    #4. based on the actual_sampled_states just count the no of times that state occurs and create the output factor
    conditional_prob = calculate_conditional_prob(actual_sampled_states, var_and_card, conditional_prob)

    """ END YOUR CODE HERE """

    return conditional_prob


def load_input_file(input_file: str) -> (Factor, dict, dict, int):
    """
    Returns the target factor, proposal factors for each node and evidence. DO NOT EDIT THIS FUNCTION

    Args:
        input_file: input file to open

    Returns:
        Factor of the target factor which is the target joint distribution of all nodes in the Bayesian network
        dictionary of node:Factor pair where Factor is the proposal distribution to sample node observations. Other
                    nodes in the Factor are parent nodes of the node
        dictionary of node:val pair where node is an evidence node while val is the evidence for the node.
    """
    with open(input_file, 'r') as f:
        input_config = json.load(f)
    proposal_factors_dict = input_config['proposal-factors']

    def parse_factor_dict(factor_dict):
        var = np.array(factor_dict['var'])
        card = np.array(factor_dict['card'])
        val = np.array(factor_dict['val'])
        return Factor(var=var, card=card, val=val)

    nodes = np.array(input_config['nodes'], dtype=int)
    edges = np.array(input_config['edges'], dtype=int)
    node_factors = {int(node): parse_factor_dict(factor_dict=proposal_factor_dict) for
                    node, proposal_factor_dict in proposal_factors_dict.items()}

    evidence = {int(node): ev for node, ev in input_config['evidence'].items()}
    initial_samples = {int(node): initial for node, initial in input_config['initial-samples'].items()}

    num_iterations = input_config['num-iterations']
    num_burn_in = input_config['num-burn-in']
    return nodes, edges, node_factors, evidence, initial_samples, num_iterations, num_burn_in


def main():
    """
    Helper function to load the observations, call your parameter learning function and save your results.
    DO NOT EDIT THIS FUNCTION.
    """
    argparser = ArgumentParser()
    argparser.add_argument('--case', type=int, required=True,
                           help='case number to create observations e.g. 1 if 1.json')
    args = argparser.parse_args()
    # np.random.seed(0)

    case = args.case
    input_file = os.path.join(INPUT_DIR, '{}.json'.format(case))
    nodes, edges, node_factors, evidence, initial_samples, num_iterations, num_burn_in = \
        load_input_file(input_file=input_file)

    # solution part
    conditional_probability = _get_conditional_probability(nodes=nodes, edges=edges, factors=node_factors,
                                                           evidence=evidence, initial_samples=initial_samples,
                                                           num_iterations=num_iterations, num_burn_in=num_burn_in)
    print(conditional_probability)
    # end solution part

    # json only recognises floats, not np.float, so we need to cast the values into floats.
    save__dict = {
        'var': np.array(conditional_probability.var).astype(int).tolist(),
        'card': np.array(conditional_probability.card).astype(int).tolist(),
        'val': np.array(conditional_probability.val).astype(float).tolist()
    }

    if not os.path.exists(PREDICTION_DIR):
        os.makedirs(PREDICTION_DIR)
    prediction_file = os.path.join(PREDICTION_DIR, '{}.json'.format(case))

    with open(prediction_file, 'w') as f:
        json.dump(save__dict, f, indent=1)
    print('INFO: Results for test case {} are stored in {}'.format(case, prediction_file))


if __name__ == '__main__':
    main()
