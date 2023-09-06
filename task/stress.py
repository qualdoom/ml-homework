import functions as f
import functions_vectorized as fv

import scipy

import generator

Tests = 10000

for _ in range(Tests):
    n = generator.generate_number(1, 5)
    A = generator.generate_array(int(n), 0, 4)

    print('Hi')
    print(A)
    print(f.run_length_encoding(A))
    print(fv.run_length_encoding(A))
    if f.run_length_encoding(A) == fv.run_length_encoding(A):
        print(_ + 1)
    else:
        print(A)
        print(f.run_length_encoding(A))
        print(fv.run_length_encoding(A))
        break
