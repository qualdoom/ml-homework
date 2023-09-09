import numpy as np


def generate_array_np(n, low=0, high=100):
    x = np.array([np.random.randint(low, high) for i in range(n)])
    return x


def generate_array(n, low=0, high=100):
    x = np.array([np.random.randint(low, high) for i in range(n)])
    return x


def generate_array_of_points(n, low=0, high=100):
    x = np.array([(np.random.randint(low, high), np.random.randint(low, high)) for i in range(n)])
    return x


def generate_matrix(n, m, low=0, high=100):
    x = np.array([[np.random.randint(low, high) for i in range(m)] for j in range(n)])
    return x


def generate_number(low=1, high=100):
    return np.random.randint(low, high)
