from copy import *
from CNN.src_to_implement.Layers import Base
from scipy import ndimage
from scipy import signal
import numpy as np


class Conv(Base.BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        if len(self.stride_shape) > 1:
            self.kernel_H_T = int(convolution_shape[1]/2)
            self.kernel_H_B = int(convolution_shape[1]/2)
            self.kernel_W_L = int(convolution_shape[2]/2)
            self.kernel_W_R = int(convolution_shape[2]/2)
            if convolution_shape[1] % 2 == 0:
                self.kernel_H_B -= 1
            if convolution_shape[2] % 2 == 0:
                self.kernel_W_R -= 1
        else:
            self.kernel_W_L = int(convolution_shape[1]/2)
            self.kernel_W_R = int(convolution_shape[1]/2)
            if convolution_shape[1] % 2 == 0:
                self.kernel_W_R -= 1
        self.num_kernels = num_kernels
        self.weights = np.random.rand(self.num_kernels, *convolution_shape)
        self.gradient_weights = np.zeros((self.num_kernels, *convolution_shape))
        self.bias = np.random.rand(self.num_kernels)
        self.gradient_bias = np.zeros(self.num_kernels)
        self.input_tensor = 0
        self.optimizer = 0

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        if len(self.stride_shape) > 1:
            stride_x = self.stride_shape[0]
            stride_y = self.stride_shape[1]
            #stride_input_tensor = input_tensor[:, :, 0::stride_x, 0::stride_y]
            x_samples = int(np.shape(input_tensor)[2]/stride_x)
            if np.shape(input_tensor)[2] % stride_x != 0:
                x_samples = x_samples+1
            y_samples = int(np.shape(input_tensor)[3]/stride_y)
            if np.shape(input_tensor)[3] % stride_y != 0:
                y_samples = y_samples + 1
            output_tensor = np.zeros((input_tensor.shape[0], self.num_kernels, x_samples, y_samples))
        else:
            stride_x = self.stride_shape[0]
            x_samples = int(np.shape(input_tensor)[2]/stride_x)
            if np.shape(input_tensor)[2] % stride_x != 0:
                x_samples = x_samples+1
            #stride_input_tensor = input_tensor[:, :, 0::stride_x]
            output_tensor = np.zeros((input_tensor.shape[0], self.num_kernels, x_samples))
        # for in batch
        for i in range(input_tensor.shape[0]):
            # for in kernels
            for j in range(self.num_kernels):
                output_tensor_conv = 0
                # for in channels
                for k in range(input_tensor.shape[1]):
                    output_tensor_conv = ndimage.correlate(input_tensor[i, k], self.weights[j, k], mode='constant') + output_tensor_conv
                if len(self.stride_shape) > 1:
                    output_tensor[i, j] = output_tensor_conv[0::stride_x, 0::stride_y]
                else:
                    output_tensor[i, j] = output_tensor_conv[0::stride_x]
                output_tensor[i, j] = output_tensor[i, j] + self.bias[j]
        return output_tensor

    def backward(self, error_tensor):
        bp_error_tensor = np.zeros(self.input_tensor.shape)
        if self.optimizer is not None:
            weight_optimizer = deepcopy(self.optimizer)
            bias_optimizer = deepcopy(self.optimizer)
        # for in batch
        for i in range(error_tensor.shape[0]):
            # for in channels
            for j in range(self.input_tensor.shape[1]):
                # for in kernels
                bp_error_tensor[i, j] = 0
                for k in range(self.num_kernels):
                    if len(self.stride_shape) > 1:
                        edited_error_tensor = np.zeros((self.input_tensor.shape[2], self.input_tensor.shape[3]))
                        edited_error_tensor[0::self.stride_shape[0], 0::self.stride_shape[1]] = error_tensor[i, k]
                    else:
                        edited_error_tensor = np.zeros((self.input_tensor.shape[2]))
                        edited_error_tensor[0::self.stride_shape[0]] = error_tensor[i, k]
                    bp_error_tensor[i, j] = ndimage.convolve(edited_error_tensor, self.weights[k, j], mode='constant') + bp_error_tensor[i, j]
            # kernel gradian
            for j in range(self.num_kernels):
                self.gradient_bias[j] = np.sum(error_tensor[i, j, :])
                if len(self.stride_shape) > 1:
                    edited_error_tensor = np.zeros((self.input_tensor.shape[2], self.input_tensor.shape[3]))
                    edited_error_tensor[0::self.stride_shape[0], 0::self.stride_shape[1]] = error_tensor[i, j]
                else:
                    edited_error_tensor = np.zeros((self.input_tensor.shape[2]))
                    edited_error_tensor[0::self.stride_shape[0]] = error_tensor[i, j]
                # for channels
                for k in range(self.input_tensor.shape[1]):
                    if len(self.stride_shape) > 1:
                        edited_input_tensor = np.pad(self.input_tensor[i, k], [(self.kernel_H_T, self.kernel_H_B), (self.kernel_W_L, self.kernel_W_R)], mode="constant")
                    else:
                        edited_input_tensor = np.pad(self.input_tensor[i, k], [(self.kernel_W_L, self.kernel_W_R)], mode="constant")
                    weight_gradian_tensor = signal.correlate(edited_input_tensor, edited_error_tensor, mode='valid')
                    self.gradient_weights[j, k] = weight_gradian_tensor
            # update weights and bias
            #if self.optimizer is not None:
                #self.bias = bias_optimizer.calculate_update(self.bias, self.gradient_bias)
                #self.weights = weight_optimizer.calculate_update(self.weights, self.gradient_weights)

        return bp_error_tensor

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize((self.num_kernels, self.convolution_shape), np.prod(self.convolution_shape), np.prod(self.convolution_shape[1:]) * self.num_kernels)
        self.bias = bias_initializer.initialize(self.num_kernels, bias_initializer.fan_in, bias_initializer.fan_out)

