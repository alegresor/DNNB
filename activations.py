''' Activation Functions '''

from .neural_network import Layer
from numpy import *


class Sigmoid(Layer):

    def f(self, x):
        x = clip(x,-100,100)
        s = 1/(1+exp(-x))
        return s
    
    def fp(self, x, g):
        x = clip(x,-100,100)
        sp = exp(-x)/((1+exp(-x))**2)
        return g*sp

    def __repr__(self):
        return 'Sigmoid Activation Function'


class Tanh(Layer):
    
    def f(self, x):
        t =  2/(1+exp(-2*x))-1
        return t
    
    def fp(self, x, g):
        tp = 4*exp(-2*x)/((1+exp(-2*x))**2)
        return g*tp

    def __repr__(self):
        return 'Tanh Activation Function'


class ReLu(Layer):

    def f(self, x):
        r =  maximum(0,x)
        return r
    
    def fp(self, x, g):
        rp = maximum(0,x)
        return g*rp
    
    def __repr__(self):
        return 'ReLu (Rectified Linear) Activation Function'
