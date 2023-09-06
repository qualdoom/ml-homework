import numpy as np

values = np.array([0, 1, 2, 1, 2, 4, 5, 6, 1, 2, 1])

searchval = [1, 2]

x = ((values[:-1]==searchval[0]) & (values[1:]==searchval[1]))

print(x)