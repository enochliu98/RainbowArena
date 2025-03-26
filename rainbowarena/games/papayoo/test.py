import numpy as np

np_random = np.random.RandomState()

print([np_random.randint(0,3) for _ in range(10)])
