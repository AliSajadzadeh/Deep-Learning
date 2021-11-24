import numpy as np
import sys
class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.step_update = 0


    def calculate_update(self,weight_tensor, gradient_tensor):

        self.weight_tensor = weight_tensor
        self.gradient_tensor = gradient_tensor

        self.momentum_term = np.multiply(self.momentum_rate,self.step_update)

        self.gradient_term = np.multiply(self.learning_rate, self.gradient_tensor)
        self.step_update = np.subtract(self.momentum_term,self.gradient_term)
        self.new_weight = np.add(self.step_update,self.weight_tensor)

        return self.new_weight
class Adam:
    def __init__(self,learning_rate,mu,rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v_k = 0
        self.r_k = 0
        self.k = 0

    def calculate_update(self,weight_tensor, gradient_tensor):
        self.weight_tensor = weight_tensor
        self.gradient_tensor = gradient_tensor
        self.k = self.k + 1


        #update v_k
        self.v_pass_term = np.multiply(self.mu, self.v_k)
        self.v_gradient_term  = np.multiply((1-self.mu), self.gradient_tensor)
        self.v_k = np.add(self.v_pass_term,self.v_gradient_term)

        #update r_k
        self.r_pass_term = np.multiply(self.rho, self.r_k)
        #self.r_gradient_term = np.multiply((1-self.rho),(np.dot(self.gradient_tensor,self.gradient_tensor)))
        self.r_gradient_term = np.multiply((1 - self.rho),self.gradient_tensor**2)
        self.r_k = np.add(self.r_pass_term,self.r_gradient_term)

        #bias correction

        self.updated_v_k = np.divide(self.v_k,(1-self.mu**self.k))
        self.updated_r_k = np.divide(self.r_k,(1-self.rho**self.k))
        #epsilon = sys.float_info.epsilon
        self.epsilon = np.finfo(float).eps
        #self.epsilon = 5
        print('eps',self.epsilon)

        self.new_weight_update = self.learning_rate * (np.divide(self.updated_v_k,(np.add(np.sqrt(self.updated_r_k),self.epsilon))))

        self.new_weights = np.subtract(self.weight_tensor,self.new_weight_update)

        return self.new_weights






