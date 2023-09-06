import numpy as np
import math


def prod_non_zero_diag(x):
    n = len(x)
    ans = 1
    if n == 0:
        return ans
    m = len(x[0])
    for i in range(min(n, m)):
        if x[i][i] == 0:
            continue
        ans *= x[i][i]
    return ans


def are_multisets_equal(x, y):
    a = sorted(x)
    b = sorted(y)
    return a == b


def max_after_zero(x):
    Found = False
    ans = 0

    for i in range(len(x) - 1):
        if x[i] == 0:
            if not Found:
                ans = x[i + 1]
                Found = True
            else:
                ans = max(ans, x[i + 1])

    return ans


def convert_image(img, coefs):

    ans = []
    for i in range(len(img)):
        x = []
        for j in range(len(img[i])):
            color = 0
            for k in range(len(img[i][j])):
                color += k * (img[i][j][k] * coefs[k])
            x.append(color)
        ans.append(x)

    ans = np.array(ans)

    return ans

    """Sum up image channels with weights from coefs array

    input:
    img -- 3-d numpy array (H x W x 3)
    coefs -- 1-d numpy array (length 3)
    output:
    img -- 2-d numpy array

    Not vectorized implementation.
    """

    pass


def run_length_encoding(x):
    elements = []
    occurrence = []

    last = 0
    last_element = 0

    for i in range(len(x)):
        if last_element == x[i]:
            last += 1
        else:
            if last > 0:
                elements.append(last_element)
                occurrence.append(last)

            last = 1
            last_element = x[i]

    if last > 0:
        elements.append(last_element)
        occurrence.append(last)

    elements = np.array(elements)
    occurrence = np.array(occurrence)

    return elements, occurrence


def pairwise_distance(x, y):
    ans = []

    for i in range(len(x)):
        z = []
        for j in range(len(y)):
            z.append(math.dist(x[i], y[j]))
        ans.append(z)

    return ans


# print(convert_image())
