""" CS5340 Lab 4 Part 1: Importance Sampling
See accompanying PDF for instructions.

Name: Parashara Ramesh
Email: e1216292@u.nus.edu
Student ID: A0285647M
"""

import os
import json
import numpy as np
import networkx as nx
from factor_utils import factor_evidence, factor_product, assignment_to_index, index_to_assignment
from factor import Factor
from argparse import ArgumentParser
from tqdm import tqdm
import matplotlib.pyplot as plt

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
INPUT_DIR = os.path.join(DATA_DIR, 'inputs')
PREDICTION_DIR = os.path.join(DATA_DIR, 'ta_predictions')
# PREDICTION_DIR = os.path.join(DATA_DIR, 'ta_predictions')

""" ADD HELPER FUNCTIONS HERE """
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

def construct_graph_from_factors(factors, evidence):
    '''

    @param factors:
    @return: nx graph & topological order
    '''

    graph = nx.DiGraph()

    nodes_to_delete = []
    for node, factor in factors.items():
        if factor.is_empty():
            print("Skipping factor")
            continue

        is_current_node_observed = node in evidence
        does_factor_have_evidence_var = len(set(evidence.keys()).intersection(set(factor.var))) > 0

        if is_current_node_observed:
            #no need for this factor anymore
            nodes_to_delete.append(node)
        else:
            if does_factor_have_evidence_var:
                #if this is the case call factor_evidence and update the factor
                factor_after_observing_evidence = factor_evidence(factor, evidence)

                #update
                factors[node] = factor_after_observing_evidence

            #potentially new factor
            potentially_updated_factor = factors[node]
            is_factor_univariate = len(potentially_updated_factor.var) == 1

            # Add the factors as nodes
            graph.add_nodes_from(potentially_updated_factor.var)

            #if the factor does not have any evidence
            if not is_factor_univariate:
                # add the edges from every var to the last one (e.g. [var1,var2,var3] => (var1,var3), (var2,var3)
                last_var = potentially_updated_factor.var[-1]
                for var in potentially_updated_factor.var[:-1]:
                    graph.add_edge(var, last_var)

    #remove unnecessary factors
    for node in nodes_to_delete:
        del factors[node]

    topological_order = list(nx.topological_sort(graph))
    return graph, topological_order, factors


""" END HELPER FUNCTIONS HERE """


def _sample_step(nodes, proposal_factors):
    """
    Performs one iteration of importance sampling where it should sample a sample for each node. The sampling should
    be done in topological order.

    Args:
        nodes: numpy array of nodes. nodes are sampled in the order specified in nodes
        proposal_factors: dictionary of proposal factors where proposal_factors[1] returns the
                sample distribution for node 1

    Returns:
        dictionary of node samples where samples[1] return the scalar sample for node 1.
    """
    samples = {}

    """ YOUR CODE HERE: Use np.random.choice """

    #going through the topological order
    for node in nodes:
        factor = proposal_factors[node]
        # Sample a state based on the probabilities
        states = list(range(len(factor.val)))
        sampled_state = np.random.choice(states, p=factor.val)
        samples[node] = sampled_state

    """ END YOUR CODE HERE """

    assert len(samples.keys()) == len(nodes)
    return samples


def _get_conditional_probability(target_factors, proposal_factors, evidence, num_iterations):
    """
    Performs multiple iterations of importance sampling and returns the conditional distribution p(Xf | Xe) where
    Xe are the evidence nodes and Xf are the query nodes (unobserved).

    The graph structure must be updated with the evidence variables.

    Args:
        target_factors: dictionary of node:Factor pair where Factor is the target distribution of the node.
                        Other nodes in the Factor are parent nodes of the node. The product of the target
                        distribution gives our joint target distribution.
        proposal_factors: dictionary of node:Factor pair where Factor is the proposal distribution to sample node
                        observations. Other nodes in the Factor are parent nodes of the node
        evidence: dictionary of node:val pair where node is an evidence node while val is the evidence for the node.
        num_iterations: number of importance sampling iterations

    Returns:
        Approximate conditional distribution of p(Xf | Xe) where Xf is the set of query nodes (not observed) and
        Xe is the set of evidence nodes. Return result as a Factor
    """
    out = Factor()

    """ YOUR CODE HERE """
    #1.a Construct the graph from the proposal factors
    print("Constructing a proposal graph")
    proposal_graph, proposal_factor_topological_order, proposal_factors = construct_graph_from_factors(proposal_factors, evidence)

    #1.b visualize graphs
    # visualize_graph(proposal_graph)

    #2. Get all the samples from proposal distribution
    print("Going to sample from the proposal distribution")
    all_sampled_states = []

    for iteration in tqdm(range(num_iterations)):
        state_of_variables_in_proposal_distribution = _sample_step(proposal_factor_topological_order, proposal_factors)
        all_sampled_states.append(state_of_variables_in_proposal_distribution)

    #3. Calculate the p , q, r, w values

    #TODO.2.a for this write some function where given a factor and some particular slice we return only that row ( might have to use assignmentToIndex here)

    #TODO.3 for all the Xf nodes sample it in a topological order ( either throw dice only once or for each node) and using that information compute it recursively
    #TODO.3a sample only from the proposal distribution and in one such sample step you get the states for each of the nodes from the proposal distribution
    #TODO.3b get a list of samples after all the iterations

    #TODO.4 from this list of samples(from each iteration) use both the target and the proposal factors to find out the probabilities and create another dict

    #TODO.5 normalize the dict


    """ END YOUR CODE HERE """

    return out


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
    target_factors_dict = input_config['target-factors']
    proposal_factors_dict = input_config['proposal-factors']
    assert isinstance(target_factors_dict, dict) and isinstance(proposal_factors_dict, dict)

    def parse_factor_dict(factor_dict):
        var = np.array(factor_dict['var'])
        card = np.array(factor_dict['card'])
        val = np.array(factor_dict['val'])
        return Factor(var=var, card=card, val=val)

    target_factors = {int(node): parse_factor_dict(factor_dict=target_factor) for
                      node, target_factor in target_factors_dict.items()}
    proposal_factors = {int(node): parse_factor_dict(factor_dict=proposal_factor_dict) for
                        node, proposal_factor_dict in proposal_factors_dict.items()}
    evidence = input_config['evidence']
    evidence = {int(node): ev for node, ev in evidence.items()}
    num_iterations = input_config['num-iterations']
    return target_factors, proposal_factors, evidence, num_iterations


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
    target_factors, proposal_factors, evidence, num_iterations = load_input_file(input_file=input_file)

    # solution part
    conditional_probability = _get_conditional_probability(target_factors=target_factors,
                                                           proposal_factors=proposal_factors,
                                                           evidence=evidence, num_iterations=num_iterations)
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
