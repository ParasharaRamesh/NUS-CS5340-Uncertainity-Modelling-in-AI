""" CS5340 Lab 2 Part 1: Junction Tree Algorithm
See accompanying PDF for instructions.

Name: Parashara Ramesh
Email: e1216292@u.nus.edu
Student ID: A0285647M
"""
import copy
import os
import numpy as np
import json
import networkx as nx
from argparse import ArgumentParser

from factor import Factor
from jt_construction import *
from factor_utils import factor_product, factor_evidence, factor_marginalize

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
INPUT_DIR = os.path.join(DATA_DIR, 'inputs')  # we will store the input data files here!
PREDICTION_DIR = os.path.join(DATA_DIR, 'predictions')  # we will store the prediction files here!

""" ADD HELPER FUNCTIONS HERE """


def find_parent_and_children(graph, root):
    '''
    childToParent (dict) => {childNode: parentNode}
    parentToChildren (dict) => {parentNode: [childNode,..]}

    @param graph: networkX graph
    @param root: root node name
    @return: parents dictionary, children dictionary
    '''
    childToParent = OrderedDict()
    parentToChildren = OrderedDict()

    frontier = [root]
    seenParents = set()

    while len(frontier) > 0:
        node = frontier.pop()
        allNeighbours = graph.neighbors(node)

        # Assign parent -> child
        neighbours = list(
            set(allNeighbours).difference(seenParents))  # we dont want to account for parents already seen!
        parentToChildren[node] = neighbours
        seenParents.add(node)

        # for each neighbour store reverse relationship
        for neighbour in neighbours:
            childToParent[neighbour] = node

        # add to frontier
        frontier.extend(neighbours)

    # add root
    childToParent[root] = None

    return childToParent, parentToChildren


# upward pass related functions
def compute_mij_upward(i, j, messages, jt_cliques, jt_clique_factors, parentToChildren):
    '''
    Compute the message between ith node to jth node when going upwards.

    mij = marginalize_i (binary potential(i,j) * product of all k children(mki))
    @param i: child node
    @param j: parent node
    @param messages: stored messages
    @param jt_cliques: cliques
    @param jt_clique_factors: clique factors
    @param parentToChildren: dict of parent -> children
    @return:
    '''
    mij = jt_clique_factors[i]
    for child in parentToChildren[i]:
        mChildToI = messages[child][i]
        mij = factor_product(mij, mChildToI)

    # finding the sepset between cliques i & j
    clique_i = set(jt_cliques[i])
    clique_j = set(jt_cliques[j])
    sepset = clique_i.intersection(clique_j)

    # finding out the variables to be marginalized away and marginalizing it
    to_marg_away = clique_i.difference(sepset)
    mij_marginalized_i = factor_marginalize(mij, np.array(list(to_marg_away)))

    return mij_marginalized_i


def do_upward_pass(messages, childToParent, parentToChildren, junction_tree, jt_cliques, jt_clique_factors):
    '''

    @param messages:
    @param childToParent:
    @param parentToChildren:
    @param junction_tree:
    @param jt_cliques:
    @param jt_clique_factors:
    @return:
    '''
    # 1.1 find out the nodes for which the messages need to be computed in upward pass (initially the leaves)
    nodesToComputeMsgFor = list(
        filter(lambda node: len(parentToChildren[node]) == 0, parentToChildren)
    )  # initially it's just the leaves!

    allNodes = [node for node in parentToChildren]
    msgComputationStatusForNodesChildren = {}
    for node in allNodes:
        children = parentToChildren[node]
        msgComputationStatusForNodesChildren[node] = {}
        for child in children:
            msgComputationStatusForNodesChildren[node][child] = False  # has node received message from its child?

    while len(
            msgComputationStatusForNodesChildren) > 1:  # we stop when we are left only with the root but the upward pass is computed!
        # doing it stage by stage from bottom -> up
        nextNodesToComputeMsgFor = set()

        # for each node compute the message with its parent
        for nodeToComputeMsgFor in nodesToComputeMsgFor:
            # message from i -> j
            i = nodeToComputeMsgFor
            j = childToParent[nodeToComputeMsgFor]

            # checking size as all([]) is True so that takes care of the leaf case!
            areMsgsFromAllChildrenOfIComputed = all(
                [isChildComputed for isChildComputed in msgComputationStatusForNodesChildren[i].values()]
            )

            # check if it's children have already computed their messages
            if areMsgsFromAllChildrenOfIComputed and j != None:
                mij = compute_mij_upward(i, j, messages, jt_cliques, jt_clique_factors, parentToChildren)

                # store the computed message
                messages[i][j] = mij

                # add the parents of the current active nodes
                nextNodesToComputeMsgFor.add(j)

                # mark status that the mij has been computed
                msgComputationStatusForNodesChildren[j][i] = True

                # remove i from msgComputationStatusForNodesChildren
                del msgComputationStatusForNodesChildren[i]

        # find the next set of nodesToComputeMsg for
        nodesToComputeMsgFor = list(nextNodesToComputeMsgFor)

    # At this point msgComputationStatusForNodesChildren will contain the root as the only key along with its children values set to True!
    return messages


# downward pass related functions
def compute_mij_downward(i, j, messages, graph, root, jt_cliques, jt_clique_factors):
    '''

    @param i:
    @param j:
    @param messages:
    @param graph:
    @param root:
    @param jt_cliques:
    @param jt_clique_factors:
    @return:
    '''

    mij = jt_clique_factors[i]

    # get set of all the neighbours for i (apart from j)
    allNeighbours = graph.neighbors(i)
    neighboursApartFromJ = list(set(allNeighbours).difference(set([j])))

    # use already computed messages here and consider the binary potential also
    for neighbour in neighboursApartFromJ:
        messageFromNeighbourToI = messages[neighbour][i]
        mij = factor_product(mij, messageFromNeighbourToI)

    # finding the sepset between cliques i & j
    clique_i = set(jt_cliques[i])
    clique_j = set(jt_cliques[j])
    sepset = clique_i.intersection(clique_j)

    # finding out the variables to be marginalized away and marginalizing it
    to_marg_away = clique_i.difference(sepset)
    mij_marginalized_i = factor_marginalize(mij, np.array(list(to_marg_away)))

    return mij_marginalized_i


def distribute_downward(i, j, messages, graph, parentToChildren, root, jt_cliques, jt_clique_factors):
    '''
    Do the downward pass

    @param i:
    @param j:
    @param messages:
    @param graph:
    @param parentToChildren:
    @param root:
    @param jt_cliques:
    @param jt_clique_factors:
    @return:
    '''
    # compute normalized message
    mij = compute_mij_downward(i, j, messages, graph, root, jt_cliques, jt_clique_factors)

    # store the computed message
    messages[i][j] = mij

    # further propagate that downward message
    for child in parentToChildren[j]:
        # returning messages here
        messages = distribute_downward(j, child, messages, graph, parentToChildren, root, jt_cliques, jt_clique_factors)

    return messages


def do_downward_pass(messages, root, junction_tree, parentToChildren, jt_cliques, jt_clique_factors):
    '''

    @param messages:
    @param root:
    @param junction_tree:
    @param parentToChildren:
    @param jt_cliques:
    @param jt_clique_factors:
    @return:
    '''
    for child in parentToChildren[root]:
        # returning messages here
        messages = distribute_downward(root, child, messages, junction_tree, parentToChildren, root, jt_cliques,
                                       jt_clique_factors)
    return messages


# upward and downward pass
def compute_all_messages_in_two_phases(root, junction_tree, messages, jt_cliques, jt_clique_factors):
    '''
    Given a particular root and the junction tree of the component the root belongs to, find the messages in both upward and downward direction
    modify those messages and return it.

    @param root:
    @param junction_tree:
    @param messages:
    @param jt_cliques:
    @param jt_clique_factors:
    @return:
    '''
    # Phase 0. Preprocessing:
    # make a custom linkage of parents and children dictionary given a particular root ( assuming it is directed )
    childToParent, parentToChildren = find_parent_and_children(junction_tree, root)

    # -------------------------------------------------------------------------------------------------------------------------------------------
    # Phase 1. Upward pass
    messages = do_upward_pass(messages, childToParent, parentToChildren, junction_tree, jt_cliques, jt_clique_factors)

    # -------------------------------------------------------------------------------------------------------------------------------------------
    # Phase 2. Downward pass
    messages = do_downward_pass(messages, root, junction_tree, parentToChildren, jt_cliques, jt_clique_factors)

    # -------------------------------------------------------------------------------------------------------------------------------------------
    return messages


def compute_normalized_clique_probabilities_from_messages(jt_cliques, messages, jt_clique_factors, junction_tree):
    '''
    For each clique, given the messages and the clique factors find out the normalized clique level probabilities

    @param jt_cliques:
    @param messages:
    @param jt_clique_factors:
    @param junction_tree
    @return:
    '''
    clique_normalized_probabilities = []
    for i, jt_clique in enumerate(jt_cliques):
        # NOTE: i represents the node number in the junction tree

        # initialize with the clique potential of this clique
        clique_unnormalized_probability = jt_clique_factors[i]

        # find all the cliques which are the neigbours
        neighbours = junction_tree.neighbors(i)

        # do factor product with the messages from all of its neighbours
        for neighbour in neighbours:
            neighbour_clique_potential = messages[neighbour][i]
            clique_unnormalized_probability = factor_product(clique_unnormalized_probability,
                                                             neighbour_clique_potential)

        # for that clique marginalize away evverything call it X and divide with X we get p(Clique_i) -> p(x1,x2...xn)
        normalizing_factor = factor_marginalize(clique_unnormalized_probability, clique_unnormalized_probability.var)
        normalizing_value = normalizing_factor.val[0]

        # by dividing it is now normalized
        clique_unnormalized_probability.val = clique_unnormalized_probability.val / normalizing_value
        clique_normalized_probability = copy.deepcopy(clique_unnormalized_probability)

        # add it to the list of normalized probabilities
        clique_normalized_probabilities.append(clique_normalized_probability)

    return clique_normalized_probabilities


""" END HELPER FUNCTIONS HERE """


# MAIN FUNCTIONS
def _update_mrf_w_evidence(all_nodes, evidence, edges, factors):
    """
    Update the MRF graph structure from observing the evidence

    Args:
        all_nodes: numpy array of nodes in the MRF
        evidence: dictionary of node:observation pairs where evidence[x1] returns the observed value of x1
        edges: numpy array of edges in the MRF
        factors: list of Factors in teh MRF

    Returns:
        numpy array of query nodes
        numpy array of updated edges (after observing evidence)
        list of Factors (after observing evidence; empty factors should be removed)
    """
    # query_nodes = all_nodes
    # updated_factors = factors

    # My implementation
    # remove all nodes from the evidence as only those will be left anyways
    query_nodes = np.array(
        list(
            set(all_nodes).difference(set(evidence.keys()))
        )
    )
    updated_edges = copy.deepcopy(edges.tolist())
    updated_factors = []

    """ YOUR CODE HERE """
    for factor in factors:
        updated_factor = factor_evidence(factor, evidence)

        # add it to updated factors if it is not empty
        if not updated_factor.is_empty():
            updated_factors.append(updated_factor)

        # Find out which RV's were removed
        has_any_var_been_observed = len(updated_factor.var) < len(factor.var)
        # (2 cases, both can be removed or one can be removed but you just find that edge and remove it in either case)
        if has_any_var_been_observed:
            # as there will always only be 2 variables inside an edge factor
            x, y = list(factor.var)
            edge = [x, y]
            reverse_edge = [y, x]
            is_edge_present = edge in updated_edges
            is_reverse_edge_present = reverse_edge in updated_edges
            is_some_edge_present = is_edge_present or is_reverse_edge_present
            if is_some_edge_present:
                if is_edge_present:
                    updated_edges.remove(edge)
                elif is_reverse_edge_present:
                    updated_edges.remove(reverse_edge)
            else:
                print("There is some problem!")
                exit(1)

    # type cast it again
    updated_edges = np.array(updated_edges)
    """ END YOUR CODE HERE """

    return query_nodes, updated_edges, updated_factors


def _get_clique_potentials(jt_cliques, jt_edges, jt_clique_factors):
    """
    Returns the list of clique potentials after performing the sum-product algorithm on the junction tree

    Args:
        jt_cliques: list of junction tree nodes e.g. [[x1, x2], ...]
        jt_edges: numpy array of junction tree edges e.g. [i,j] implies that jt_cliques[i] and jt_cliques[j] are
                neighbors
        jt_clique_factors: list of clique factors where jt_clique_factors[i] is the factor for cliques[i]

    Returns:
        list of clique potentials computed from the sum-product algorithm
    """
    # clique_potentials = jt_clique_factors

    """ YOUR CODE HERE """
    # construct maximal spanning tree
    junction_tree = create_junction_tree_nx_graph(jt_cliques, jt_edges)

    # visualize junction tree
    # print(visualize_graph(junction_tree))

    # since there is a chance that there can be multiple disjoint connected components we need to do sum product on each such disjoint graph
    connected_components = nx.connected_components(junction_tree)

    # here we are dealing with clique numbers and not the RV numbers
    root_clique_index_of_each_component = [list(component)[0] for component in connected_components]

    # Create structure to hold messages
    num_nodes = junction_tree.number_of_nodes()

    # This is the place where store each mij
    messages = [[None] * num_nodes for _ in range(num_nodes)]

    # for each root do the sum product and store the messages
    for root in root_clique_index_of_each_component:
        messages = compute_all_messages_in_two_phases(root, junction_tree, messages, jt_cliques, jt_clique_factors)

    # find the clique potentials now by using the messages from each direction
    clique_potentials = compute_normalized_clique_probabilities_from_messages(jt_cliques, messages, jt_clique_factors,
                                                                              junction_tree)

    """ END YOUR CODE HERE """

    assert len(clique_potentials) == len(jt_cliques)
    return clique_potentials


def _get_node_marginal_probabilities(query_nodes, cliques, clique_potentials):
    """
    Returns the marginal probability for each query node from the clique potentials.

    Args:
        query_nodes: numpy array of query nodes e.g. [x1, x2, ..., xN]
        cliques: list of cliques e.g. [[x1, x2], ... [x2, x3, .., xN]]
        clique_potentials: list of clique potentials (Factor class)

    Returns:
        list of node marginal probabilities (Factor class)2

    """
    query_marginal_probabilities = []

    """ YOUR CODE HERE """
    # for each query node find out the (first) clique it belongs to and use that.
    for query_node in query_nodes:
        i = 0

        # find out the first clique where this query_node belongs to
        while i < len(cliques):
            clique = cliques[i]
            is_querynode_in_clique = query_node in clique

            if is_querynode_in_clique:
                break

            i += 1

        # using the ith clique , marginalize away every other variable from this particular clique potential apart from the query_node
        to_marginalize_away = np.array(
            list(
                set(
                    list(cliques[i])
                ).difference(
                    {query_node}
                )
            )
        )

        # for each rv to be marginalized away perform the operation and store the result. e.g.in this p(x1,x2...xn) if we want only p(xi) -> we marg away everything else apart from the ith query node
        query_marginal_probability = factor_marginalize(clique_potentials[i], to_marginalize_away)
        query_marginal_probabilities.append(query_marginal_probability)

    """ END YOUR CODE HERE """

    return query_marginal_probabilities


def get_conditional_probabilities(all_nodes, evidence, edges, factors):
    """
    Returns query nodes and query Factors representing the conditional probability of each query node
    given the evidence e.g. p(xf|Xe) where xf is a single query node and Xe is the set of evidence nodes.

    Args:
        all_nodes: numpy array of all nodes (random variables) in the graph
        evidence: dictionary of node:evidence pairs e.g. evidence[x1] returns the observed value for x1
        edges: numpy array of all edges in the graph e.g. [[x1, x2],...] implies that x1 is a neighbor of x2
        factors: list of factors in the MRF.

    Returns:
        numpy array of query nodes
        list of Factor
    """
    query_nodes, updated_edges, updated_node_factors = _update_mrf_w_evidence(all_nodes=all_nodes, evidence=evidence,
                                                                              edges=edges, factors=factors)

    jt_cliques, jt_edges, jt_factors = construct_junction_tree(nodes=query_nodes, edges=updated_edges,
                                                               factors=updated_node_factors)

    clique_potentials = _get_clique_potentials(jt_cliques=jt_cliques, jt_edges=jt_edges, jt_clique_factors=jt_factors)

    query_node_marginals = _get_node_marginal_probabilities(query_nodes=query_nodes, cliques=jt_cliques,
                                                            clique_potentials=clique_potentials)

    return query_nodes, query_node_marginals


# NOT TO BE EDITED
def parse_input_file(input_file: str):
    """ Reads the input file and parses it. DO NOT EDIT THIS FUNCTION. """
    with open(input_file, 'r') as f:
        input_config = json.load(f)

    nodes = np.array(input_config['nodes'])
    edges = np.array(input_config['edges'])

    # parse evidence
    raw_evidence = input_config['evidence']
    evidence = {}
    for k, v in raw_evidence.items():
        evidence[int(k)] = v

    # parse factors
    raw_factors = input_config['factors']
    factors = []
    for raw_factor in raw_factors:
        factor = Factor(var=np.array(raw_factor['var']), card=np.array(raw_factor['card']),
                        val=np.array(raw_factor['val']))
        factors.append(factor)
    return nodes, edges, evidence, factors


# MAIN ENTRY POINT
def main():
    """ Entry function to handle loading inputs and saving outputs. DO NOT EDIT THIS FUNCTION. """
    argparser = ArgumentParser()
    argparser.add_argument('--case', type=int, required=True,
                           help='case number to create observations e.g. 1 if 1.json')
    args = argparser.parse_args()

    case = args.case
    input_file = os.path.join(INPUT_DIR, '{}.json'.format(case))
    nodes, edges, evidence, factors = parse_input_file(input_file=input_file)

    # solution part:
    query_nodes, query_conditional_probabilities = get_conditional_probabilities(all_nodes=nodes, edges=edges,
                                                                                 factors=factors, evidence=evidence)

    predictions = {}
    for i, node in enumerate(query_nodes):
        probability = query_conditional_probabilities[i].val
        predictions[int(node)] = list(np.array(probability, dtype=float))

    if not os.path.exists(PREDICTION_DIR):
        os.makedirs(PREDICTION_DIR)
    prediction_file = os.path.join(PREDICTION_DIR, '{}.json'.format(case))
    with open(prediction_file, 'w') as f:
        json.dump(predictions, f, indent=1)
    print('INFO: Results for test case {} are stored in {}'.format(case, prediction_file))


if __name__ == '__main__':
    main()

