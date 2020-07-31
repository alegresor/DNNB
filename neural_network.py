''' Neural Network wrapper class '''

from multiprocessing.pool import ThreadPool
from numpy import array, arange, zeros


class NeuralNetwork:
    ''' High level wrapper for a neural network model '''

    def __init__(self, name, layers, loss_obj, optimizer_obj, parallel=False, processes=2, chunksize=10):
        '''       
        Args:
            name (string): name for this neural network model
            layers (list): list of instances of classes from 
                           layers.py, activations.py, or transforms.py
            loss_obj (Object): instance of class from loss_functions.py
            optimizer_obj (Object): instance of class from optimizers.py
        '''
        self.name = name
        self.layers = layers
        self.loss_obj = loss_obj
        self.optimizer_obj = optimizer_obj
        self.parallel = parallel
        self.processes = processes
        self.chunksize = chunksize
        
    def forward(self, x):
        n = x.shape[0]
        for layer in self.layers:
            layer.init_x(n)
        if self.parallel:
            pool = ThreadPool(processes=self.processes)
            y_hat = array(pool.starmap(self.f, zip(arange(n),x), self.chunksize))
        else:
            y_hat = [None]*n
            for i,x_i in enumerate(x):
                y_hat[i] = self.f(i,x_i)
            y_hat = array(y_hat)
        return y_hat

    def f(self, i, x):
        for layer in self.layers:
            x = layer.forward(i,x)
        return x

    def backward(self, y_hat, y):
        n = y_hat.shape[0]
        loss = self.loss_obj.f(y_hat,y)
        g = self.loss_obj.fp(y_hat,y)
        if self.parallel:
            pool = ThreadPool(processes=self.processes)
            pool.starmap(self.fp, zip(arange(n),g), self.chunksize)
        else:
            for i,g_i in enumerate(g):
                self.fp(i,g_i)
        for layer in reversed(self.layers):
            if not hasattr(layer,'params'):
                continue
            for param in layer.params:
                self.optimizer_obj.update_params(param)
                param.zero_grad()
        return loss
    
    def fp(self, i, g):
        for layer in reversed(self.layers):
            g = layer.backward(i,g)  
        return
    
    def __repr__(self):
        s =  '%s Neural Network\n'%self.name
        s_tmp = '\tLayers:\n'
        params = 0
        for layer_obj in self.layers:
            s_tmp += '\t\t'
            s_tmp += str(layer_obj).replace('\n','\n\t\t')
            s_tmp += '\n'
            if hasattr(layer_obj, 'params'):
                params += layer_obj.total_params
        s += '\tTotal Network Parameters: %d\n'%params
        s += s_tmp
        s += '\tLoss Function:\n\t\t'
        s += str(self.loss_obj).replace('\n','\n\t\t')+'\n'
        s += '\tOptimizer:\n\t\t'
        s += str(self.optimizer_obj).replace('\n','\n\t\t')
        return s


class Parameter:
    
    def __init__(self,val):
        self.val = val
        self.shape = self.val.shape
        self.grad = zeros(self.shape)
    
    def __call__(self):
        return self.val

    def add_grad(self, g):
        self.grad += g

    def zero_grad(self):
        self.grad = zeros(self.shape)


class Layer:

    def __init__(self):
        self._n = None
    
    def init_x(self, n):
        self._x = [None]*n

    def forward(self, i, x):
        self._x[i] = x
        return self.f(x[:])
    
    def backward(self, i, g):
        x = self._x[i]
        return self.fp(x,g)
