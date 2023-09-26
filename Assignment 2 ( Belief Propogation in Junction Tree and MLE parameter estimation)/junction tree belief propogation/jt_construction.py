import copy

import numpy as np
import networkx as nx
from networkx.algorithms import tree
from factor import Factor
from factor_utils import *
import matplotlib.pyplot as plt

""" ADD HELPER FUNCTIONS HERE (IF NEEDED) """
def create_nx_graph(nodes, edges):
    edges = add_other_direction_edges_to_existing_edges(edges)
    g = nx.Graph()
    # Add nodes and edges to the graph
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return g

def visualize_graph(graph, weighted=True):
    pos = nx.spring_layout(graph)
    nx.draw_networkx(graph, pos=pos, with_labels=True, font_weight='bold',
                     node_size=1000, arrowsize=20)
    # only for weighted
    if weighted:
        edge_labels = {(u, v): d['weight'] for u, v, d in graph.edges(data=True)}
        nx.draw_networkx_edge_labels(graph, pos=pos, edge_labels=edge_labels)
    plt.axis('off')
    plt.show()


def add_other_direction_edges_to_existing_edges(edges):
    '''
    Assumes that the edges passed is a np.array
    @param edges:
    @return:
    '''
    # ensure that bidirectional edges are present
    other_edges = []
    edges = edges.tolist()
    for edge in edges:
        # reverse it
        other_edges.append(edge[::-1])

    # extend it with the bidirectional edges as well
    edges.extend(other_edges)
    return np.array(edges)


def create_reconstituted_graph(nodes, edges, elimination_order):
    '''

    @param nodes: all nodes
    @param edges: all edges in bidirectional order
    @param elimination_order: some order of elimination of nodes
    @return: networkX graph
    '''
    # Create a graph
    reconstituted_graph = nx.Graph()

    # Add nodes and edges to the graph
    reconstituted_graph.add_nodes_from(nodes)
    reconstituted_graph.add_edges_from(edges)

    # Initialize a copy on which we will remove variables
    graph_to_operate_on = reconstituted_graph.copy()

    # Perform variable elimination
    for node_to_eliminate in elimination_order:
        # Find neighbors of the node to eliminate
        neighbors = list(graph_to_operate_on.neighbors(node_to_eliminate))

        # Add elimination edges (fill edges) between neighbors
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                # add elimination edge to graph_to_operate as that will required later
                graph_to_operate_on.add_edge(neighbors[i], neighbors[j])
                graph_to_operate_on.add_edge(neighbors[j], neighbors[i])

                # add elimination edge to reconstituted graph as well right now only
                reconstituted_graph.add_edge(neighbors[i], neighbors[j])
                reconstituted_graph.add_edge(neighbors[j], neighbors[i])

        # Remove the node being eliminated
        graph_to_operate_on.remove_node(node_to_eliminate)

    return reconstituted_graph


def get_clique_graph_edges(cliques, reconstituted_graph):
    '''

    @param cliques:
    @param reconstituted_graph:
    @return: cliques list, clique edges, clique graph
    '''

    clique_edges = []

    # For each clique find the intersections between them and store that
    for i in range(len(cliques)):
        for j in range(i + 1, len(cliques)):
            clique_i = set(cliques[i])
            clique_j = set(cliques[j])
            intersection_between_cliques = clique_i.intersection(clique_j)
            # store it only if there is a sepset
            if len(intersection_between_cliques) > 0:
                clique_edges.extend([
                    [i, j],
                    [j, i]
                ])

    # return list of cliques, edges from clique to clique dictionary with sepsets, and the clique graph itself
    return np.array(clique_edges)


def create_junction_tree_nx_graph(jt_cliques, jt_edges):
    '''

    @param jt_cliques:
    @param jt_edges:
    @return: a network x graph
    '''
    clique_graph = nx.Graph()

    # find out the weights
    clique_edges_with_weights = []
    for jt_edge in jt_edges:
        i, j = jt_edge
        clique_i = jt_cliques[i]
        clique_j = jt_cliques[j]
        # cardinality of the sepset
        weight = len(
            set(clique_i).intersection(set(clique_j))
        )
        clique_edges_with_weights.append([i, j, weight])

    # add weighted edges
    clique_graph.add_weighted_edges_from(clique_edges_with_weights)

    # add all nodes where each node number i represents jt_cliques[i]
    clique_graph.add_nodes_from(list(range(len(jt_cliques))))

    # find the maximal spanning tree in this to form junction tree (can be more than 1 which one to pick?) (networkx.tree.maximum_spanning_edges)
    junction_tree = nx.maximum_spanning_tree(clique_graph)

    return junction_tree


""" END ADD HELPER FUNCTIONS HERE """


def _get_jt_clique_and_edges(nodes, edges):
    """
    Construct the structure of the junction tree and return the list of cliques (nodes) in the junction tree and
    the list of edges between cliques in the junction tree. [i, j] in jt_edges means that cliques[j] is a neighbor
    of cliques[i] and vice versa. [j, i] should also be included in the numpy array of edges if [i, j] is present.
    You can use nx.Graph() and nx.find_cliques().

    Args:
        nodes: numpy array of nodes [x1, ..., xN] e.g. [0,1,3,4,5]
        edges: numpy array of edges e.g. [x1, x2] implies that x1 and x2 are neighbors. e.g [[3,4],[3,5],[4,5]]

    Returns:
        list of junction tree cliques. each clique should be a maximal clique. e.g. [[X1, X2],[X2,X4], ...]
        numpy array of junction tree edges e.g. [[0,1], ...], [i,j] means that cliques[i]
            and cliques[j] are neighbors.
    """

    # ensure that you have both direction edges
    edges = add_other_direction_edges_to_existing_edges(edges)  #is already a np array

    """ YOUR CODE HERE """

    # Fix some elimination order (say descending order)
    elimination_order = list(sorted(nodes, reverse=True))

    # Create reconstituted graph based on elimination order (make it all interconnected.)
    reconstituted_graph = create_reconstituted_graph(nodes, edges, elimination_order)

    # find the cliques in the reconstitured graphs using nx.find_cliques()
    jt_cliques = list(nx.find_cliques(reconstituted_graph))

    # Construct a new clique graph considering all the sepsets in between them (keep that cardinality of intersection as the weight)
    jt_edges = get_clique_graph_edges(jt_cliques, reconstituted_graph)

    """ END YOUR CODE HERE """
    return jt_cliques, jt_edges


def _get_clique_factors(jt_cliques, factors):
    """
    Assign node factors to cliques in the junction tree and derive the clique factors.

    Args:
        jt_cliques: list of junction tree maximal cliques e.g. [[x1, x2, x3], [x2, x3], ... ]
        factors: list of factors from the original graph

    Returns:
        list of clique factors where the factor(jt_cliques[i]) = clique_factors[i]
    """
    clique_factors = [Factor() for _ in jt_cliques]
    unseen_factor_indices = set(list(range(len(factors))))

    """ YOUR CODE HERE """
    # find out which factors belong to this clique and do a factor product but use everything only once?, keep track
    for i, jt_clique in enumerate(jt_cliques):
        jt_clique = set(jt_clique)

        # get set of vars in clique
        jt_clique_used = set()

        # for iterating through factors list, start with the lowest unseen index
        j = min(unseen_factor_indices)

        # go through factor by factor and try to use it only once!
        while jt_clique_used.issubset(jt_clique) and jt_clique_used != jt_clique and \
                len(unseen_factor_indices) > 0 and j < len(factors):
            # take the current factor
            current_factor = factors[j]
            current_factor_var_set = set(current_factor.var)

            # check if the current factor's variables are a subset of the clique , if so use it
            are_unseen_factor_vars_subset_of_clique = current_factor_var_set.issubset(jt_clique)

            # check if the current factor is already used, if it is not used you can still use it
            is_current_factor_not_used = j in unseen_factor_indices

            if is_current_factor_not_used and are_unseen_factor_vars_subset_of_clique:
                # compute the clique factor by doing a pairwise factor_product with current_factor
                clique_factors[i] = factor_product(clique_factors[i], current_factor)

                # keep adding it to the set of seen vars
                jt_clique_used = jt_clique_used.union(current_factor_var_set)

                # remove the factor[j] from unseen as it is now seen and used!
                unseen_factor_indices.remove(j)

            # move to next factor
            j += 1

    # we may still be left with factors as the above loop picks everything greedily
    for unseen_factor_index in unseen_factor_indices:
        current_factor = factors[unseen_factor_index]
        current_factor_var_set = set(current_factor.var)

        # for iterating through cliques and clique_factors
        for i, jt_clique in enumerate(jt_cliques):
            jt_clique = set(jt_clique)

            # check if the current factor's variables are a subset of the clique , if so use it
            are_unseen_factor_vars_subset_of_clique = current_factor_var_set.issubset(jt_clique)

            if are_unseen_factor_vars_subset_of_clique:
                # compute the clique factor by doing a pairwise factor_product with current_factor
                clique_factors[i] = factor_product(clique_factors[i], current_factor)

                # break as we are done with this
                break

    # NOTE: here the clique_factor in some cases does not have a factor with all of the same random variables (e.g. (2,3,5) clique but factor has only (3,5). so we just proceed as is for now assuming this is okay!

    """ END YOUR CODE HERE """

    assert len(clique_factors) == len(jt_cliques), 'there should be equal number of cliques and clique factors'
    return clique_factors


def construct_junction_tree(nodes, edges, factors):
    """
    Constructs the junction tree and returns it's the cliques, edges and clique factors in the junction tree.
    DO NOT EDIT THIS FUNCTION.

    Args:
        nodes: numpy array of random variables e.g. [X1, X2, ..., Xv]
        edges: numpy array of edges e.g. [[X1,X2], [X2,X1], ...]
        factors: list of factors in the graph

    Returns:
        list of cliques e.g. [[X1, X2], ...]
        numpy array of edges e.g. [[0,1], ...], [i,j] means that cliques[i] and cliques[j] are neighbors.
        list of clique factors where jt_cliques[i] has factor jt_factors[i] where i is an index
    """
    jt_cliques, jt_edges = _get_jt_clique_and_edges(nodes=nodes, edges=edges)
    jt_factors = _get_clique_factors(jt_cliques=jt_cliques, factors=factors)
    return jt_cliques, jt_edges, jt_factors
