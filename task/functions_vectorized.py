import math

import numpy as np


def prod_non_zero_diag(x):
    return np.prod(np.diag(x)[np.diag(x) != 0])


def are_multisets_equal(x, y):
    return np.array_equal(np.sort(x), np.sort(y))


def max_after_zero(x):
    res = np.argwhere(x == 0) + 1
    res = res[res < len(x)]
    a = np.take(x, res)
    return a.max()


def convert_image(img, coefs):
    return np.dot(img, coefs)


def run_length_encoding(x):
    a = np.diff(x)
    res = [0]
    res = np.append(res, np.reshape(np.argwhere(a != 0), -1) + 1)
    res = np.append(res, len(x))

    occurrence = np.diff(res)

    res = np.delete(res, -1)

    elements = np.array(x[res])

    return elements, occurrence


def pairwise_distance(x, y):
    return np.linalg.norm(x[:, np.newaxis] - y, axis=2)
