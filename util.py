from math import sqrt


def print_scientific_notation(n):
    # Only works with numbers > 1
    # Used to compare large mae values lol

    exp = 0
    while n > 10:
        exp += 1
        n /= 10

    print('%fe(10 ** %i)' % (n, exp))


def in_to_cm(n):
    return 2.54 * n


def kg_to_lb(n):
    return 2.205 * n


def MAE(l1, l2):
    assert len(l1) == len(l2), 'The lists must be the same length to compare'

    def temp(temp_l1, temp_l2):

        if len(temp_l1) == 1:
            return abs(temp_l1[0] - temp_l2[0])
        else:
            return abs(temp_l1[0] - temp_l2[0]) + temp(temp_l1[1:], temp_l2[1:])

    return temp(l1, l2) / len(l1)


def RMSE(l1, l2):
    # Probably would have been easier to do these two iteratively, but I just learned recursion
    # in CS61A and I wanted to practice
    assert len(l1) == len(l2), 'The lists must be the same length to compare'

    def temp(temp_l1, temp_l2):
        if len(temp_l1) == 1:
            return (temp_l1[0] - temp_l2[0]) ** 2
        else:
            return (temp_l1[0] - temp_l2[0]) ** 2 + temp(temp_l1[1:], temp_l2[1:])

    return sqrt(temp(l1, l2) / len(l1))


first = [10, 10, 10]
second = [0, 8, 9]

# print(MAE(first, second))
# print(RMSE(first, second))
