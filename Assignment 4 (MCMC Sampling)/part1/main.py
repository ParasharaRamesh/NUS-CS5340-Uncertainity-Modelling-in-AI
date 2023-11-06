""" CS5340 Lab 4 Part 1: Importance Sampling
See accompanying PDF for instructions.

Name: Parashara Ramesh
Email: e1216292@u.nus.edu
Student ID: A0285647M
"""

import os
import json
from collections import defaultdict

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
PREDICTION_DIR = os.path.join(DATA_DIR, 'predictions')
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
    var_and_card = dict()

    nodes_to_delete = []
    for node, factor in factors.items():
        if factor.is_empty():
            print("Skipping factor")
            continue

        # find the variable and its cardinality
        idx_in_var = np.where(factor.var == node)[0][0]
        var_and_card[node] = factor.card[idx_in_var]

        is_current_node_observed = node in evidence
        does_factor_have_evidence_var = len(set(evidence.keys()).intersection(set(factor.var))) > 0

        if is_current_node_observed:
            # no need for this factor anymore
            nodes_to_delete.append(node)
        else:
            if does_factor_have_evidence_var:
                # if this is the case call factor_evidence and update the factor
                factor_after_observing_evidence = factor_evidence(factor, evidence)

                # update
                factors[node] = factor_after_observing_evidence

            # potentially new factor
            potentially_updated_factor = factors[node]
            is_factor_univariate = len(potentially_updated_factor.var) == 1

            # Add the factors as nodes
            graph.add_nodes_from(potentially_updated_factor.var)

            # if the factor does not have any evidence
            if not is_factor_univariate:
                # add the edges from every var to the last one (e.g. [var1,var2,var3] => (var1,var3), (var2,var3)
                last_var = potentially_updated_factor.var[-1]
                for var in potentially_updated_factor.var[:-1]:
                    graph.add_edge(var, last_var)

    # remove unnecessary factors
    for node in nodes_to_delete:
        del factors[node]

    topological_order = list(nx.topological_sort(graph))

    return graph, topological_order, factors, var_and_card


def calculate_p_values_from_target(target_factors, all_sampled_target_states):
    '''

    @param target_factors: dictionary of target factors
    @param all_sampled_target_states: list of proposal variable states [{var1: state1,..},..]
    @return:
    '''
    p_values = []

    for sampled_state in tqdm(all_sampled_target_states):
        p_value = 1
        for factor in target_factors.values():
            vars = factor.var
            var_states = [sampled_state[var] for var in vars]
            row_idx = assignment_to_index(var_states,factor.card)
            p_value *= factor.val[row_idx] # keep multiplying for every factor
        p_values.append(p_value)

    return p_values


def calculate_q_values_from_proposal(proposal_factors, all_sampled_proposal_states):
    '''


    @param proposal_factors: dictionary of proposal factors
    @param all_sampled_proposal_states: list of proposal variable states [{var1: state1,..},..]
    @return:
    '''
    q_values = []

    for proposal_sampled_state in tqdm(all_sampled_proposal_states):
        q_value = 1
        for factor in proposal_factors.values():
            vars = factor.var
            var_states = [proposal_sampled_state[var] for var in vars]
            row_idx = assignment_to_index(var_states,factor.card)
            q_value *= factor.val[row_idx] # keep multiplying for every factor
        q_values.append(q_value)

    return q_values


def calculate_w_values_for_all_samples(all_sampled_proposal_states, proposal_factors, target_factors, evidence):
    all_sampled_target_states = []

    #Update with evidence
    for proposal_state in all_sampled_proposal_states:
        proposal_state_copy = dict(proposal_state)
        proposal_state_copy.update(evidence)
        all_sampled_target_states.append(proposal_state_copy)

    print("Going to find all the p values")
    all_p_values = calculate_p_values_from_target(target_factors, all_sampled_target_states)
    print("Going to find all the q values")
    all_q_values = calculate_q_values_from_proposal(proposal_factors, all_sampled_proposal_states)
    print("Found all the p & q values, converting both to numpy arrays for easier processing")
    all_p_values = np.array(all_p_values)
    all_q_values = np.array(all_q_values)
    # 4. Calculate r values
    print("Going to find all the r values")
    all_r_values = all_p_values / all_q_values
    # 5. Calculate w values
    print("Going to find all the w values")
    all_w_values = all_r_values / np.sum(all_r_values)
    return all_w_values


def get_probs_for_each_query_node_state_configuration(all_sampled_proposal_states, all_w_values):
    print("Going to find the total probability for every state configuration of the query nodes")
    all_sampled_states_and_its_corresponding_w_values = zip(all_sampled_proposal_states, all_w_values)
    state_probs = defaultdict(int)  # this will return 0 in case a particular configuration is missing
    for sampled_proposal_states, w_value in all_sampled_states_and_its_corresponding_w_values:
        state_key = tuple(sorted(sampled_proposal_states.items()))
        if state_key not in state_probs:
            state_probs[state_key] = w_value
        else:
            state_probs[state_key] += w_value
    print(
        "Finished calculating the probability for each state configuration of the query nodes, now constructing the out factor")
    return state_probs


def create_out_factor_from_state_probs(out, state_probs, var_and_card):
    state_configuration_key = list(state_probs.keys())[0]  # this is a tuple of tuples ((var_i, var_state_i),..)
    out.var = np.array([item[0] for item in state_configuration_key])
    out.card = np.array([var_and_card[var] for var in out.var])
    all_var_state_configurations = out.get_all_assignments()  # to ensure that we fill the val in the correct order
    out.val = []
    for var_state_configuration in all_var_state_configurations:
        # Construct the key in order to use it to get the value from state_probs since there the key is in the format ((var, state),..)
        key_to_search_with = []
        for var, var_state in zip(out.var, var_state_configuration):
            key_to_search_with.append((var, var_state))
        key_to_search_with = tuple(key_to_search_with)

        # Get the probability of this state from the state_probs dict
        prob_of_state_configuration = state_probs[key_to_search_with]
        out.val.append(prob_of_state_configuration)
    out.val = np.array(out.val)
    return out


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

    # going through the topological order
    for node in nodes:
        factor = proposal_factors[node]

        '''
        In case the current factor has multiple variables,
        then we need to get that particular row which has the slice containing the previously sampled states for its dependent vars.
        
        Since we are going in topological order this should be quite straightforward
        '''
        curr_factor_vars_and_cards = zip(factor.var, factor.card)

        previously_sampled_var_state = []
        previously_sampled_card = []

        for var, card in curr_factor_vars_and_cards:
            # it was already previously sampled
            if var in samples:
                previously_sampled_var_state.append(samples[var])
                previously_sampled_card.append(card)

        '''
        If at all the dependent vars were sampled earlier get all slices based on curr var's changing cardinality, else just use the factor val directly
        '''
        if len(previously_sampled_var_state) > 0 and len(previously_sampled_card) > 0:
            var_idx = np.where(factor.var == node)[0][0]
            node_card = factor.card[var_idx]

            row_probs = []
            for var_card_state in range(node_card):
                # Get the row index based on this index (basically previously_sampled_var_state would all have fixed states, but the current node state is still fluid so we need to consider all the rows based on its changing cardinality)
                row_idx_where_curr_var_has_card_state = assignment_to_index(previously_sampled_var_state + [node],
                                                                            previously_sampled_card + [node_card])
                # Get the probability from that row
                prob_value_from_that_row = factor.val[row_idx_where_curr_var_has_card_state]

                # Add it to the list of probabilities
                row_probs.append(prob_value_from_that_row)
        else:
            # in case there was no need to get the row slice (this will be the case when there is only one node)
            row_probs = factor.val

        # Sample a state based on the probabilities
        states = list(range(len(row_probs)))
        sampled_state = np.random.choice(states, p=row_probs)
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
    # 1.a Construct the graph from the proposal factors
    print("Constructing a proposal graph")
    proposal_graph, proposal_factor_topological_order, proposal_factors, var_and_card = construct_graph_from_factors(
        proposal_factors, evidence)

    # 1.b visualize graphs
    # visualize_graph(proposal_graph)

    # 2. Get all the samples from proposal distribution
    print("Going to sample from the proposal distribution")
    all_sampled_proposal_states = []

    for iteration in tqdm(range(num_iterations)):
        state_of_variables_in_proposal_distribution = _sample_step(proposal_factor_topological_order, proposal_factors)
        all_sampled_proposal_states.append(state_of_variables_in_proposal_distribution)

    # 3. Calculate the p, q values from #num_iterations
    all_w_values = calculate_w_values_for_all_samples(all_sampled_proposal_states, proposal_factors, target_factors,
                                                      evidence)

    # 6. Calculate the marginal probability
    state_probs = get_probs_for_each_query_node_state_configuration(all_sampled_proposal_states, all_w_values)

    # 7. Create the out factor
    out = create_out_factor_from_state_probs(out, state_probs, var_and_card)

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
