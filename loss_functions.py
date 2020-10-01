''' Loss Functions '''

from numpy import *


class MeanSquareError:

    def f(self, y_hat, y):
        n = y.shape[0]
        return ((y_hat-y)**2).sum()/n
    
    def fp(self, y_hat, y):
        n = y.shape[0]
        return 2*(y_hat-y)/n
    
    def __repr__(self):
        return 'MSE (Mean Square Error)'


class BinaryCrossEntropy:

    def __init__(self, weights=(1,1)):
        self.w = array(weights)
    
    def f(self, y_hat, y):
        n = y.shape[0]
        return -(self.w[0]*(1-y)*log(1-y_hat) + self.w[1]*y*log(y_hat)).sum()/n
    
    def fp(self, y_hat, y):
        n = y.shape[0]
        return (self.w[0]*(1-y)/(1-y_hat) - self.w[1]*y/y_hat)/n

    def __repr__(self):
        s =  'Binary Cross Entropy\n'
        s += '\tWeights: %s'%str(self.w)
        return s


class CategoricalCrossEntropyWithSoftmax:

    def softmax(self, x):
        x = x-x.max(1,keepdims=True)
        s = exp(x)/exp(x).sum(1,keepdims=True)
        return s

    def f(self, x, y):
        n = y.shape[0]
        self.y_hat = self.softmax(x)
        log_likelihood = -log(self.y_hat[range(n),y.flatten()])
        loss = log_likelihood.sum()/n
        return loss

    def fp(self, x, y):
        n = y.shape[0]
        self.y_hat[range(n),y.flatten()] -= 1
        grad = self.y_hat/n
        return grad
    
    def __repr__(self):
        return 'Categorical Cross Entropy (with softmax prior)'
