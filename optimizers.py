"""
Optimizers
    Helpful Resource: https://deepnotes.io/sgd-momentum-adaptive
"""

from numpy import *

EPS = 1e-8


class Stochastic_GD:

    def __init__(self, learning_rate=.01, friction=0.9, nesterov=False): 
        self.lr = learning_rate
        self.mu = friction
        self.nesterov = nesterov

    def update_params(self, param):
        if not hasattr(param,'v'):
            param.v = zeros(param().shape)
        if self.nesterov:
            param.val += self.mu*param.v
        param.v = self.mu*param.v+self.lr*param.grad
        param.val -= param.v

    def __repr__(self):
        s =  'Stochastic Gradience Descent\n'
        s += '\tLearning Rate: %.3f\n'%self.lr
        s += '\tFriction: %.3f\n'%self.mu
        s += '\tNesterov: %s'%str(self.nesterov)
        return s


class AdaGrad:
    
    def __init__(self, learning_rate=.01):
        self.lr = learning_rate
    
    def update_params(self, param):
        if not hasattr(param,'c'):
            param.c = zeros(param().shape)
        param.c += param.grad**2
        param.val -= self.lr*param.grad/(sqrt(param.c)+EPS)

    def __repr__(self):
        s =  'Adaptive Gradient Algorithm\n'
        s += '\tLearning Rate: %.3f'%self.lr
        return s 


class RMSProp:

    def __init__(self, learning_rate=.01, decay_rate=0.9):
        self.lr = learning_rate
        self.dr = decay_rate

    def update_params(self, param):
        if not hasattr(param,'c'):
            param.c = zeros(param().shape)
        param.c = self.dr*param.c+(1-self.dr)*param.grad**2
        param.val -= self.lr*param.grad/(sqrt(param.c)+EPS)

    def __repr__(self):
        s = 'Root Mean Square Propogation\n'
        s += '\tLearning Rate: %.3f\n'%self.lr
        s += '\tDecay Rate: %.3f'%self.dr    
        return s 


class Adam:

    def __init__(self, learning_rate=.01, beta1=.9, beta2=.999):
        self.lr = learning_rate
        self.b1 = beta1
        self.b2 = beta2

    def update_params(self, param):
        if not hasattr(param,'c'): 
            param.c = zeros(param().shape)
        if not hasattr(param,'v'):
            param.v = zeros(param().shape)
        if not hasattr(param,'i'):
            param.t = 1
        param.c = self.b1*param.c+(1-self.b1)*param.grad
        mt = param.c/(1-self.b1**param.t)
        param.v = self.b2*param.v+(1-self.b2)*(param.grad**2)
        vt = param.v/(1-self.b2**param.t)
        param.val -= self.lr*mt/(sqrt(vt)+EPS)
        param.t += 1

    def __repr__(self):
        s =  'Adam (Adaptive Momentum Estimation)\n'
        s += '\tLearning Rate: %.3f\n'%self.lr
        s += '\tBeta1: %.3f\n'%self.b1
        s += '\tBeta2: %.3f'%self.b2
        return s 
        