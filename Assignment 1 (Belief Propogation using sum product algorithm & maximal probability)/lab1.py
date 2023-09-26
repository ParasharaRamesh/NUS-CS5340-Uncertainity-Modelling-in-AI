""" CS5340 Lab 1: Belief Propagation and Maximal Probability
See accompanying PDF for instructions.

Name: Parashara Ramesh
Email: e1216292@u.nus.edu
Student ID: A0285647M_lab1
"""

import copy
from collections import OrderedDict, defaultdict
from typing import List
import numpy as np
from factor import Factor, index_to_assignment, assignment_to_index, generate_graph_from_factors, visualize_graph
from functools import reduce

"""For sum product message passing"""


def factor_product(A, B):
    """Compute product of two factors.

    Suppose A = phi(X_1, X_2), B = phi(X_2, X_3), the function should return
    phi(X_1, X_2, X_3)
    """
    if A.is_empty():
        return B
    if B.is_empty():
        return A

    # Create output factor. Variables should be the union between of the
    # variables contained in the two input factors
    out = Factor()
    out.var = np.union1d(A.var, B.var)

    # Compute mapping between the variable ordering between the two factors
    # and the output to set the cardinality
    out.card = np.zeros(len(out.var), np.int64)
    mapA = np.argmax(out.var[None, :] == A.var[:, None], axis=-1)
    mapB = np.argmax(out.var[None, :] == B.var[:, None], axis=-1)

    # NOTE: eventually mapA and mapB corresponds to the list of random variables present A & B respectively
    out.card[mapA] = A.card
    out.card[mapB] = B.card

    # For each assignment in the output, compute which row of the input factors it comes from
    out.val = np.zeros(np.prod(out.card))
    assignments = out.get_all_assignments()  # corresponds to all possible row values of its constituent RVs
    idxA = assignment_to_index(assignments[:, mapA],
                               A.card)  # across the A factor's rows; which row indices have : in what order?
    idxB = assignment_to_index(assignments[:, mapB],
                               B.card)  # across the B factor's rows; which row indices have : in what order?
    """ YOUR CODE HERE
    You should populate the .val field with the factor product
    Hint: The code for this function should be very short (~1 line). Try to
      understand what the above lines are doing, in order to implement
      subsequent parts.
    """
    # Multiply the corresponding rows/assignments from factor A & factor B
    out.val = np.array([A.val[rowAB[0]] * B.val[rowAB[1]] for i, rowAB in enumerate(zip(idxA, idxB))])
    return out


def factor_marginalize(factor, var):
    """Sums over a list of variables.

    Args:
        factor (Factor): Input factor
        var (List): Variables to marginalize out

    Returns:
        out: Factor with variables in 'var' marginalized out.
    """
    out = Factor()

    """ YOUR CODE HERE
    Marginalize out the variables given in var
    """

    # 0. keep a dict of all vars and its cardinalities from the original factor
    allCardinalities = OrderedDict()
    for v, c in zip(factor.var, factor.card):
        allCardinalities[v] = c

    # 1. set the output factor's vars to be the set of remaining vars
    # a. get the set difference between original factor and the vars to be marginalized away!
    remainingVars = set(factor.var).difference(set(var))

    # b. now get it in the same order as it was in the original factor!
    outputVars = []
    for var in allCardinalities.keys():  # this ensures it's in the same order as the original factor!
        if var in remainingVars:
            outputVars.append(var)
    out.var = np.array(outputVars)

    # 2. next set the cardinalities of the out.card
    out.card = np.array([allCardinalities[var] for var in out.var], np.int64)

    # 3. now find out the output factors values
    # a. find the output size of the output vars
    numPossibilitiesWithCards = reduce(lambda x, y: x * y, out.card)  # product of all the cardinalities

    # b. get all rows in the original factor
    allRows = factor.get_all_assignments()

    # c. now get a list of indices where the output vars are present in the input vars (e.g. if input vars is X1,X2, X3 and output vars is X1,X3 it should return [0,2]
    indicesWhereOutVarsArePresentInOrigFactorVars = [i for i, item in enumerate(factor.var) if item in out.var]

    # d. find the sum by grouping across each combo of output factor in the input factor's table
    outProbabilities = OrderedDict()  # contains the marginalized probabilities for each combo of the remaining vars
    for i, inRowCombo in enumerate(allRows):
        inRowComboOnlyForOutVars = tuple(
            [inRowCombo[index] for index in indicesWhereOutVarsArePresentInOrigFactorVars])
        if inRowComboOnlyForOutVars not in outProbabilities.keys():
            outProbabilities[inRowComboOnlyForOutVars] = factor.val[i]
        else:
            outProbabilities[inRowComboOnlyForOutVars] += factor.val[i]

    # e. assign outVals to the out factor
    out.val = np.array(list(outProbabilities.values()))

    assert len(out.val) == numPossibilitiesWithCards
    return out


def observe_evidence(factors, evidence=None):
    """Modify a set of factors given some evidence

    Args:
        factors (List[Factor]): List of input factors
        evidence (Dict): Dictionary, where the keys are the observed variables
          and the values are the observed values.

    Returns:
        List of factors after observing evidence
    """
    if evidence is None:
        return factors
    out = copy.deepcopy(factors)

    """ YOUR CODE HERE
    Set the probabilities of assignments which are inconsistent with the 
    evidence to zero.
    """
    for factor in out:
        # check if there is some intersection between the evidence set and this particular factor's variables
        varsPresentInFactorWhichAreObserved = set(factor.var).intersection(set(evidence))
        shouldOperateOnFactor = len(varsPresentInFactorWhichAreObserved) > 0

        if shouldOperateOnFactor:
            # This means that this factor needs to be operated on as it has an evidence variable
            indicesInFactorWhichHaveObservedEvidenceVariables = [(observedVar, list(factor.var).index(observedVar)) for
                                                                 observedVar in varsPresentInFactorWhichAreObserved]
            allFactorRows = factor.get_all_assignments()

            # go through each row/assignment in the factor table
            for i, row in enumerate(allFactorRows):
                variableValuesMatchingGivenEvidenceVariables = []
                for observedVar, possibleObservedIndex in indicesInFactorWhichHaveObservedEvidenceVariables:
                    variableValuesMatchingGivenEvidenceVariables.append(
                        row[possibleObservedIndex] == evidence[observedVar]
                    )
                # checking if this row has values which correspond to the given evidence variable values
                isRowMatchingObservedEvidenceValues = all(variableValuesMatchingGivenEvidenceVariables)

                # set anything not matching to 0
                if not isRowMatchingObservedEvidenceValues:
                    factor.val[i] = 0

    return out


"""For max sum meessage passing (for MAP)"""


def factor_sum(A, B):
    """Same as factor_product, but sums instead of multiplies
    """
    if A.is_empty():
        return B
    if B.is_empty():
        return A

    # Create output factor. Variables should be the union between of the
    # variables contained in the two input factors
    out = Factor()
    out.var = np.union1d(A.var, B.var)

    # Compute mapping between the variable ordering between the two factors
    # and the output to set the cardinality
    out.card = np.zeros(len(out.var), np.int64)
    mapA = np.argmax(out.var[None, :] == A.var[:, None], axis=-1)
    mapB = np.argmax(out.var[None, :] == B.var[:, None], axis=-1)
    out.card[mapA] = A.card
    out.card[mapB] = B.card

    # For each assignment in the output, compute which row of the input factors
    # it comes from
    out.val = np.zeros(np.prod(out.card))
    assignments = out.get_all_assignments()
    idxA = assignment_to_index(assignments[:, mapA], A.card)
    idxB = assignment_to_index(assignments[:, mapB], B.card)

    """ YOUR CODE HERE
    You should populate the .val field with the factor sum. The code for this
    should be very similar to the factor_product().
    """
    out.val = np.array([A.val[rowAB[0]] + B.val[rowAB[1]] for i, rowAB in enumerate(zip(idxA, idxB))])
    return out


def factor_max_marginalize(factor, var):
    """Marginalize over a list of variables by taking the max.

    Args:
        factor (Factor): Input factor
        var (List): Variable to marginalize out.

    Returns:
        out: Factor with variables in 'var' marginalized out. The factor's
          .val_argmax field should be a list of dictionary that keep track
          of the maximizing values of the marginalized variables.
          e.g. when out.val_argmax[i][j] = k, this means that
            when assignments of out is index_to_assignment[i],
            variable j has a maximizing value of k.
          See test_lab1.py::test_factor_max_marginalize() for an example.
    """
    out = Factor()

    """ YOUR CODE HERE
    Marginalize out the variables given in var. 
    You should make use of val_argmax to keep track of the location with the
    maximum probability.
    """

    # 0. keep a dict of all vars and its cardinalities from the original factor
    allCardinalities = OrderedDict()
    for v, c in zip(factor.var, factor.card):
        allCardinalities[v] = c

    # 1. set the output factor's vars to be the set of remaining vars
    # a. get the set difference between original factor and the vars to be marginalized away!
    remainingVars = set(factor.var).difference(set(var))

    # b. now get it in the same order as it was in the original factor!
    outputVars = []
    for var in allCardinalities.keys():  # this ensures it's in the same order as the original factor!
        if var in remainingVars:
            outputVars.append(var)
    out.var = np.array(outputVars)

    # 2. next set the cardinalities of the out.card
    out.card = np.array([allCardinalities[var] for var in out.var], np.int64)

    # 3. now find out the output factors values
    # b. get all rows in the original factor
    allRows = factor.get_all_assignments()

    # c. now get a list of indices where the output vars are present in the input vars (e.g. if input vars is X1,X2, X3 and output vars is X1,X3 it should return [0,2]
    indicesWhereOutVarsArePresentInOrigFactorVars = [i for i, item in enumerate(factor.var) if item in out.var]
    indicesOfRemainingVarsInInp = [i for i, item in enumerate(factor.var) if item not in out.var]

    # d. find the sum by grouping across each combo of output factor in the input factor's table
    outProbabilities = OrderedDict()  # contains the marginalized probabilities for each combo of the remaining vars
    for i, inRowCombo in enumerate(allRows):
        inRowComboOnlyForOutVars = tuple(
            [inRowCombo[index] for index in indicesWhereOutVarsArePresentInOrigFactorVars])
        if inRowComboOnlyForOutVars not in outProbabilities.keys():
            # store it in this manner
            outProbabilities[inRowComboOnlyForOutVars] = [(
                factor.val[i],
                {
                    factor.var[index]: inRowCombo[index] for index in indicesOfRemainingVarsInInp
                }
            )]

        else:
            # append to list
            outProbabilities[inRowComboOnlyForOutVars].append((
                factor.val[i],
                {
                    factor.var[index]: inRowCombo[index] for index in indicesOfRemainingVarsInInp
                }
            ))

    out.val = []

    out.val_argmax = []
    for vals in outProbabilities.values():
        max_val, individual_arg_max = max(vals, key=lambda x: x[0])
        out.val.append(max_val)
        out.val_argmax.append(individual_arg_max)
    out.val = np.array(out.val, np.float64)
    return out


def compute_joint_distribution(factors):
    """Computes the joint distribution defined by a list of given factors

    Args:
        factors (List[Factor]): List of factors

    Returns:
        Factor containing the joint distribution of the input factor list
    """
    joint = Factor()

    """ YOUR CODE HERE
    Compute the joint distribution from the list of factors. You may assume
    that the input factors are valid so no input checking is required.
    """
    if len(factors) == 1:
        # edge case
        return factors[0]

    joint = reduce(lambda f1, f2: factor_product(f1, f2), factors)
    return joint


def compute_marginals_naive(V, factors, evidence):
    """Computes the marginal over a set of given variables

    Args:
        V (int): Single Variable to perform inference on
        factors (List[Factor]): List of factors representing the graphical model
        evidence (Dict): Observed evidence. evidence[k] = v indicates that
          variable k has the value v.

    Returns:
        Factor representing the marginals
    """

    output = Factor()

    """ YOUR CODE HERE
    Compute the marginal. Output should be a factor.
    Remember to normalize the probabilities!
    """
    # 1. compute the joint probability distribution
    joint = compute_joint_distribution(factors)

    # 2. reduce the joint distribution by the evidence
    jointDistributionAfterObservingEvidence = observe_evidence([joint], evidence)[0]

    # 3. now marginalize out all the other irrelevant variables
    output = factor_marginalize(jointDistributionAfterObservingEvidence,
                                list(set(jointDistributionAfterObservingEvidence.var).difference(set([V]))))

    # 4. normalize the output
    output.val = output.val / sum(output.val)

    return output


'''
Sum Product related functions
'''


def find_root(graph):
    '''
    On observation in the graph.nodes only one of the keys has a factor inside it which indicates that it is the root!

    @param graph: networkX graph
    @return:
    '''
    nodes = graph.nodes
    for node in nodes:
        nodeInfo = nodes[node]
        if "factor" in nodeInfo:
            return node, nodeInfo["factor"]


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


def compute_mij_upward(i, j, messages, graph, parentToChildren):
    '''
    Compute the message between ith node to jth node when going upwards.

    mij = marginalize_i (binary potential(i,j) * product of all k children(mki))
    @param i: child node
    @param j: parent node
    @param messages: stored messages
    @param graph: networkX graph
    @param parentToChildren: dict of parent -> children
    @return:
    '''
    is_ij_edge_present = (i, j) in graph.edges
    binary_potential_factor = graph.edges[(i, j)] if is_ij_edge_present else graph.edges[(j, i)]
    mij = binary_potential_factor["factor"]
    for child in parentToChildren[i]:
        mChildToI = messages[child][i]
        mij = factor_product(mij, mChildToI)

    # marginalizing away i
    mij_marginalized_i = factor_marginalize(mij, [i])
    return mij_marginalized_i


def compute_mij_downward(i, j, messages, graph, root, rootFactor):
    '''

    @param i:
    @param j:
    @param messages:
    @param graph:
    @return:
    '''
    # Get the binary potential
    is_ij_edge_present = (i, j) in graph.edges
    binary_potential_factor = graph.edges[(i, j)] if is_ij_edge_present else graph.edges[(j, i)]

    # if i is the root consider the unary potential also and init mij
    mij = None
    if i == root:
        mij = factor_product(binary_potential_factor["factor"], rootFactor)
    else:
        mij = binary_potential_factor["factor"]

    # get set of all the neighbours for i (apart from j)
    allNeighbours = graph.neighbors(i)
    neighboursApartFromJ = list(set(allNeighbours).difference(set([j])))

    # use already computed messages here and consider the binary potential also
    for neighbour in neighboursApartFromJ:
        messageFromNeighbourToI = messages[neighbour][i]
        mij = factor_product(mij, messageFromNeighbourToI)

    # marginalizing away i
    mij_marginalized_i = factor_marginalize(mij, [i])
    return mij_marginalized_i


def distribute_downward(i, j, messages, graph, parentToChildren, root, rootFactor):
    '''
    Do the downward pass

    @param i:
    @param j:
    @param messages:
    @param graph:
    @param parentToChildren:
    @return:
    '''
    # compute normalized message
    mij = compute_mij_downward(i, j, messages, graph, root, rootFactor)

    # store the computed message
    messages[i][j] = mij

    # further propagate that downward message
    for child in parentToChildren[j]:
        # returning messages here
        messages = distribute_downward(j, child, messages, graph, parentToChildren, root, rootFactor)

    return messages


def compute_marginals_bp(V, factors, evidence):
    """Compute single node marginals for multiple variables
    using sum-product belief propagation algorithm

    Args:
        V (List): Variables to infer single node marginals for
        factors (List[Factor]): List of factors representing the grpahical model
        evidence (Dict): Observed evidence. evidence[k]=v denotes that the
          variable k is assigned to value v.

    Returns:
        marginals: List of factors. The ordering of the factors should follow
          that of V, i.e. marginals[i] should be the factor for variable V[i].
    """
    # Dummy outputs, you should overwrite this with the correct factors
    marginals = []

    # Setting up messages which will be passed
    factors = observe_evidence(factors, evidence)
    graph = generate_graph_from_factors(factors)

    # Uncomment the following line to visualize the graph. Note that we create
    # an undirected graph regardless of the input graph since 1) this
    # facilitates graph traversal, and 2) the algorithm for undirected and
    # directed graphs is essentially the same for tree-like graphs.

    # visualize_graph(graph)

    # You can use any node as the root since the graph is a tree. For simplicity
    # we always use node 0 for this assignment.
    root = 0

    # Create structure to hold messages
    num_nodes = graph.number_of_nodes()

    # This is the place where store each mij
    messages = [[None] * num_nodes for _ in range(num_nodes)]

    """ YOUR CODE HERE
    Use the algorithm from lecture 4 and perform message passing over the entire
    graph. Recall the message passing protocol, that a node can only send a
    message to a neighboring node only when it has received messages from all
    its other neighbors.
    Since the provided graphical model is a tree, we can use a two-phase 
    approach. First we send messages inward from leaves towards the root.
    After this is done, we can send messages from the root node outward.
    
    Hint: You might find it useful to add auxilliary functions. You may add 
      them as either inner (nested) or external functions.
    """
    # Phase 0. Preprocessing:
    # 0.1 find out which factor has unary potential, that will be our root for the purpose of calculating the messages
    root, rootFactor = find_root(graph)

    # 0.2. make a custom linkage of parents and children dictionary given a particular root ( assuming it is directed )
    childToParent, parentToChildren = find_parent_and_children(graph, root)

    # -------------------------------------------------------------------------------------------------------------------------------------------
    # Phase 1. Upward pass
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
                mij = compute_mij_upward(i, j, messages, graph, parentToChildren)

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

    # -------------------------------------------------------------------------------------------------------------------------------------------
    # Phase 2. Downward pass
    for child in parentToChildren[root]:
        # returning messages here
        messages = distribute_downward(root, child, messages, graph, parentToChildren, root, rootFactor)

    # -------------------------------------------------------------------------------------------------------------------------------------------
    # Phase 3. use the computed mij's to do the magic! ( make this a separate function)
    for v in V:
        # if v has unary potential then multiply that also
        marginal_v = Factor() if v != root else rootFactor

        neighbours = graph.neighbors(v)
        # 3.1 here just call the mij's of the neighbours and multiply it all
        for neighbour in neighbours:
            messageFromNeighbour = messages[neighbour][v]
            marginal_v = factor_product(marginal_v, messageFromNeighbour)

        # normalize it (or) marginalize away the last variable!
        marginal_v.val = marginal_v.val / sum(marginal_v.val)

        marginals.append(marginal_v)

    # -------------------------------------------------------------------------------------------------------------------------------------------
    return marginals


def log_transform(factors):
    logFactors = []
    for factor in factors:
        factor.val = np.log(factor.val)
        logFactors.append(factor)
    return logFactors


def compute_mij_max_upward(i, j, messages, graph, parentToChildren):
    '''
    Compute the message between ith node to jth node when going upwards.

    mij = max_marginalize_i (binary potential(i,j) * product of all k children(mki))
    @param i: child node
    @param j: parent node
    @param messages: stored messages
    @param graph: networkX graph
    @param parentToChildren: dict of parent -> children
    @return:
    '''
    is_ij_edge_present = (i, j) in graph.edges
    binary_potential_factor = graph.edges[(i, j)] if is_ij_edge_present else graph.edges[(j, i)]
    mij = binary_potential_factor["factor"]
    for child in parentToChildren[i]:
        mChildToI = messages[child][i]
        # as we are in the log scale now!
        mij = factor_sum(mij, mChildToI)

    # max marginalizing away i
    mij_max_marginalized_i = factor_max_marginalize(mij, [i])
    return mij_max_marginalized_i


def map_eliminate(factors, evidence):
    """Obtains the maximum a posteriori configuration for a tree graph
    given optional evidence

    Args:
        factors (List[Factor]): List of factors representing the graphical model
        evidence (Dict): Observed evidence. evidence[k]=v denotes that the
          variable k is assigned to value v.

    Returns:
        max_decoding (Dict): MAP configuration
        log_prob_max: Log probability of MAP configuration. Note that this is
          log p(MAP, e) instead of p(MAP|e), i.e. it is the unnormalized
          representation of the conditional probability.
    """

    max_decoding = {}
    log_prob_max = 0.0

    """ YOUR CODE HERE
    Use the algorithm from lecture 5 and perform message passing over the entire
    graph to obtain the MAP configuration. Again, recall the message passing 
    protocol.
    Your code should be similar to compute_marginals_bp().
    To avoid underflow, first transform the factors in the probabilities
    to **log scale** and perform all operations on log scale instead.
    You may ignore the warning for taking log of zero, that is the desired
    behavior.
    """

    factors = observe_evidence(factors, evidence)
    factors = log_transform(factors)
    graph = generate_graph_from_factors(factors)

    # Create structure to hold messages
    num_nodes = graph.number_of_nodes()

    # This is the place where store each mij
    messages = [[None] * num_nodes for _ in range(num_nodes)]

    # Phase 0. Preprocessing:
    # 0.1 find out which factor has unary potential, that will be our root for the purpose of calculating the messages
    root, rootFactor = find_root(graph)

    # 0.2. make a custom linkage of parents and children dictionary given a particular root ( assuming it is directed )
    childToParent, parentToChildren = find_parent_and_children(graph, root)

    # -------------------------------------------------------------------------------------------------------------------------------------------
    # Phase 1. Upward pass
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
                mij = compute_mij_max_upward(i, j, messages, graph, parentToChildren)

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
    # -------------------------------------------------------------------------------------------------------------------------------------------
    # Phase 2. compute the MAP.
    map_factor = rootFactor
    neighbours = graph.neighbors(root)
    for neighbour in neighbours:
        messageFromNeighbour = messages[neighbour][root]
        map_factor = factor_sum(map_factor, messageFromNeighbour)
    # 2.1 finally max marginalize away the root as well
    final_map_factor = factor_max_marginalize(map_factor, [root])

    # 2.2 there will only be one value here
    log_prob_max = final_map_factor.val[0]

    # -------------------------------------------------------------------------------------------------------------------------------------------
    # Phase3. find the max decoding

    # initialize max_decoding with the maximizing value from root
    _, argMaxAtRoot = max(zip(final_map_factor.val, final_map_factor.val_argmax), key=lambda x: x[0])
    max_decoding.update(argMaxAtRoot)

    # Do a dfs from the root and check the messages array
    frontier = [root]

    while len(frontier) > 0:
        node = frontier.pop()
        children = parentToChildren[node]
        for child in children:
            messageFromChildToNode = messages[child][node]
            argMaxAtThisChild = messageFromChildToNode.val_argmax[max_decoding[node]]
            max_decoding.update(argMaxAtThisChild)
        frontier.extend(children)

    # remove already observed evidences
    for e in evidence:
        assert evidence[e] == max_decoding[e]
        del max_decoding[e]

    # -------------------------------------------------------------------------------------------------------------------------------------------
    return max_decoding, log_prob_max
