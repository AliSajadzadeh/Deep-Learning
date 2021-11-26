import numpy as np


class Constant:
    def __init__(self, weight_initialization_constant=0.1):
        self.weight_initialization_constant = weight_initialization_constant

    def initialize(self, weights_shape, fan_in, fan_out):
        weights = np.full(weights_shape, self.weight_initialization_constant)
        return weights


class UniformRandom:
    def initialize(self, weights_shape, fan_in, fan_out):
        weights = np.random.rand(weights_shape[0], weights_shape[1])
        return weights


class Xavier:
    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2) / np.sqrt(fan_in + fan_out)
        normal_distribution = np.random.normal(0, sigma, weights_shape[0]*weights_shape[1])
        weights = normal_distribution.reshape(weights_shape)
        return weights


class He:
    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2) / np.sqrt(fan_in)
        normal_distribution = np.random.normal(0, sigma, weights_shape[0]*weights_shape[1])
        weights = normal_distribution.reshape(weights_shape)
        return weights
