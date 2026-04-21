import numpy as np

class Sigmoid:

    def __init__(self):
        pass

    def __str__(self):
        return 'Sigmoid'

    def function(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return x * (1 - x)
