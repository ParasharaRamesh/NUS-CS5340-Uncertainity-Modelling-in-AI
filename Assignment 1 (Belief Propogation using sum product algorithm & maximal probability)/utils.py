def cardinality_product(cardinalities):
    '''
    Given a list of cardinalities ( in reversed order ) -> return all possible combinations
    '''
    args = [list(range(cardinality)) for cardinality in cardinalities]
    if not args:
        return [[]]

    result = []
    for x in cardinality_product(cardinalities[:-1]):
        for y in args[-1]:
            result.append([y,] + x)

    return result

if __name__ == '__main__':
    cards = [3,2,4]
    cards = list(reversed(cards)) # this step is really important!
    print(cardinality_product(cards))
