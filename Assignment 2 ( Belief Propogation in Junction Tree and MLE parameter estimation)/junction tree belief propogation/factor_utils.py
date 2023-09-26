# taken from part 1
import copy
from collections import OrderedDict
from functools import reduce

import numpy as np
from factor import Factor, index_to_assignment, assignment_to_index


def factor_product(A, B):
    """
    Computes the factor product of A and B e.g. A = f(x1, x2); B = f(x1, x3); out=f(x1, x2, x3) = f(x1, x2)f(x1, x3)

    Args:
        A: first Factor
        B: second Factor

    Returns:
        Returns the factor product of A and B
    """
    out = Factor()

    """ YOUR CODE HERE """
    # NOTE: re-used the same code as lab1
    if A.is_empty():
        return B
    if B.is_empty():
        return A

    # Create output factor. Variables should be the union between of the
    # variables contained in the two input factors
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

    # Multiply the corresponding rows/assignments from factor A & factor B
    out.val = np.array([A.val[rowAB[0]] * B.val[rowAB[1]] for i, rowAB in enumerate(zip(idxA, idxB))])

    """ END YOUR CODE HERE """
    return out


def factor_marginalize(factor, var):
    """
    Returns factor after variables in var have been marginalized out.

    Args:
        factor: factor to be marginalized
        var: numpy array of variables to be marginalized over

    Returns:
        marginalized factor
    """
    # NOTE. commented this out as my implementation builds out from scratch
    # out = copy.deepcopy(factor)
    """ YOUR CODE HERE
     HINT: Use the code from lab1 """
    out = Factor()

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

    """ END YOUR CODE HERE """
    return out

def factor_evidence(factor, evidence):
    """
    Observes evidence and retains entries containing the observed evidence. Also removes the evidence random variables
    because they are already observed e.g. factor=f(1, 2) and evidence={1: 0} returns f(2) with entries from node1=0
    Args:
        factor: factor to reduce using evidence
        evidence:  dictionary of node:evidence pair where evidence[1] = evidence of node 1.
    Returns:
        Reduced factor that does not contain any variables in the evidence. Return an empty factor if all the
        factor's variables are observed.
    """
    out = copy.deepcopy(factor)

    """ YOUR CODE HERE, HINT: copy from lab2 part 1! """
    # check if there is some intersection between the evidence set and this particular factor's variables
    varsPresentInFactorWhichAreObserved = set(out.var).intersection(set(evidence))
    shouldOperateOnFactor = len(varsPresentInFactorWhichAreObserved) > 0

    if shouldOperateOnFactor:
        # This means that this factor needs to be operated on as it has an evidence variable
        indicesInFactorWhichHaveObservedEvidenceVariables = [(observedVar, list(out.var).index(observedVar)) for
                                                             observedVar in varsPresentInFactorWhichAreObserved]
        allFactorRows = out.get_all_assignments()

        # go through each row/assignment in the factor table
        for i, row in enumerate(allFactorRows):
            variableValuesMatchingGivenEvidenceVariables = []
            for observedVar, possibleObservedIndex in indicesInFactorWhichHaveObservedEvidenceVariables:
                variableValuesMatchingGivenEvidenceVariables.append(
                    row[possibleObservedIndex] == evidence[observedVar]
                )
            # checking if this row has values which correspond to the given evidence variable values
            isRowMatchingObservedEvidenceValues = all(variableValuesMatchingGivenEvidenceVariables)

            # set anything not matching to None
            if not isRowMatchingObservedEvidenceValues:
                #setting to dummy value
                out.val[i] = -1

    #post processing to remove unobserved variables
    out = remove_unobserved_vars(out, evidence)

    """ END YOUR CODE HERE """
    return out


# Helper fuctions
def remove_unobserved_vars(factor, evidence):
    '''
    Remove unobserved variables, and remove those values in the input factor which are 0

    @param factor:
    @param evidence:
    @return:
    '''
    out = Factor()

    #initialize with vars which are unobserved
    out_vars = []
    out_cards = []
    for i in range(len(factor.var)):
        var = factor.var[i]
        card = factor.card[i]
        if var not in evidence:
            out_vars.append(var)
            out_cards.append(card)

    out.var = np.array(out_vars)
    out.card = np.array(out_cards)

    #ignore values which are zero
    out_vals = []
    for val in factor.val:
        #checking if it was None
        if val != -1:
            out_vals.append(val)
    out.val = np.array(out_vals)

    return out