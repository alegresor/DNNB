''' Transform the Shape and Density of Data '''

from .neural_network import Layer
from .util import KernelShapeError
from numpy import zeros, argmax


class Reshape(Layer):

    def __init__(self, *new_shape):
        self.new_shape = new_shape
    
    def f(self, x):
        return x.reshape(self.new_shape)
    
    def fp(self, x, g):
        return g.reshape(x.shape)

    def __repr__(self):
        return 'Reshape Transform'
        

class MaxPooling_2D(Layer):

    def __init__(self, input_features, kernel_features, stride):
        self.x_f = input_features
        self.k_f = kernel_features
        self.s = stride
        self.z_f = (self.x_f-self.k_f)/self.s+1
        if self.z_f.is_integer():
            self.z_f = int(self.z_f)
        else:
            raise KernelShapeError(int(self.x_f),int(self.k_f),int(self.s))
    
    def f(self, x):
        ll,f = x.shape
        z = zeros((ll,self.z_f))
        for l in range(ll):
            for j in range(self.z_f):
                jj = j*self.s
                z[l,j] = x[l,jj:jj+self.k_f].max()
        return z

    def fp(self, x, g):
        ll,f = g.shape
        dg_dx = zeros((ll,self.x_f))
        for l in range(ll):
            for j in range(self.z_f):
                jj = j*self.s
                max_j = x[l,jj:jj+self.k_f].argmax()+jj
                dg_dx[l,max_j] += g[l,j]
        return dg_dx
    
    def __repr__(self):
        s =  '2D Max Pooling Layer\n'
        s += '\tPool Shape: %s\n'%str((self.k_f,))
        s += '\tStride Shape: %s'%str((self.s,))
        return s

class MaxPooling_3D(Layer):

    def __init__(self, input_features, kernel_features, stride):
        self.x_f1, self.x_f2 = input_features
        self.k_f1,self.k_f2 = kernel_features
        self.s1,self.s2 = stride
        self.z_f1 = (self.x_f1-self.k_f1)/self.s1+1
        self.z_f2 = (self.x_f2-self.k_f2)/self.s2+1
        if self.z_f1.is_integer() and self.z_f2.is_integer():
            self.z_f1 = int(self.z_f1)
            self.z_f2 = int(self.z_f2)
        else:
            x_f = (int(self.x_f1),int(self.x_f2))
            k_f = (int(self.k_f1),int(self.k_f2))
            s = (int(self.s1),int(self.s2))
            raise KernelShapeError(x_f,k_f,s)

    def f(self, x):
        ll,*f = x.shape
        z = zeros((ll,self.z_f1,self.z_f2))
        self.idxs = zeros((ll,self.z_f1,self.z_f2)).astype('int')
        for l in range(ll):
            for j1 in range(self.z_f1):
                for j2 in range(self.z_f2):
                    jj1 = j1*self.s1
                    jj2 = j2*self.s2
                    z[l,j1,j2] = x[l,jj1:jj1+self.k_f1,jj2:jj2+self.k_f2].max()
        return z

    def fp(self, x, g):
        ll,*f = g.shape
        dg_dx = zeros((ll,self.x_f1,self.x_f2))
        for l in range(ll):
            for j1 in range(self.z_f1):
                for j2 in range(self.z_f2):
                    jj1 = j1*self.s1
                    jj2 = j2*self.s2
                    max_j = x[l,jj1:jj1+self.k_f1,jj2:jj2+self.k_f2].argmax()
                    max_j1 = (max_j//self.k_f2)+jj1
                    max_j2 = (max_j%self.k_f2)+jj2
                    dg_dx[l,max_j1,max_j2] += g[l,j1,j2]
        return dg_dx
    
    def __repr__(self):
        s =  '3D Max Pooling Layer\n'
        s += '\tPool Shape: %s\n'%str((self.k_f1,self.k_f2))
        s += '\tStride Shape: %s'%str((self.s1,self.s2))
        return s


class AveragePooling_2D(Layer):
    
    def __init__(self, input_features, kernel_features, stride):
        self.x_f = input_features
        self.k_f = kernel_features
        self.s = stride
        self.z_f = (self.x_f-self.k_f)/self.s+1
        if self.z_f.is_integer():
            self.z_f = int(self.z_f)
        else: 
            raise KernelShapeError(int(self.x_f),int(self.k_f),int(self.s))

    def f(self, x):
        ll,f = x.shape
        z = zeros((ll,self.z_f))
        for l in range(ll):
            for j in range(self.z_f):
                jj = j*self.s
                z[l,j] = x[l,jj:jj+self.k_f].mean()
        return z

    def fp(self, x, g):
        ll,f = g.shape
        dg_dx = zeros((ll,self.x_f))
        for l in range(ll):
            for j in range(self.z_f):
                jj = j*self.s
                dg_dx[l,jj:jj+self.k_f] += g[l,j]/self.k_f
        return dg_dx
    
    def __repr__(self):
        s =  '2D Average Pooling Layer\n'
        s += '\tPool Shape: %s\n'%str((self.k_f,))
        s += '\tStride Shape: %s'%str((self.s,))
        return s


class AveragePooling_3D(Layer):

    def __init__(self, input_features, kernel_features, stride):
        self.x_f1,self.x_f2 = input_features
        self.k_f1,self.k_f2 = kernel_features
        self.s1,self.s2 = stride
        self.z_f1 = (self.x_f1-self.k_f1)/self.s1+1
        self.z_f2 = (self.x_f2-self.k_f2)/self.s2+1
        if self.z_f1.is_integer() and self.z_f2.is_integer():
            self.z_f1 = int(self.z_f1)
            self.z_f2 = int(self.z_f2)
        else:
            x_f = (int(self.x_f1),int(self.x_f2))
            k_f = (int(self.k_f1),int(self.k_f2))
            s = (int(self.s1),int(self.s2))
            raise KernelShapeError(x_f,k_f,s)
    
    def f(self, x):
        ll,*f = x.shape
        z = zeros((ll,self.z_f1,self.z_f2))
        for l in range(ll):
            for j1 in range(self.z_f1):
                for j2 in range(self.z_f2):
                    jj1 = j1*self.s1
                    jj2 = j2*self.s2
                    z[l,j1,j2] = x[l,jj1:jj1+self.k_f1,jj2:jj2+self.k_f2].mean()
        return z
    
    def fp(self, x, g):
        ll,*f = g.shape
        dg_dx = zeros((ll,self.x_f1,self.x_f2))
        for l in range(ll):
            for j1 in range(self.z_f1):
                for j2 in range(self.z_f2):
                    jj1 = j1*self.s1
                    jj2 = j2*self.s2
                    dg_dx[l,jj1:jj1+self.k_f1,jj2:jj2+self.k_f2] += g[l,j1,j2]/(self.k_f1*self.k_f2)
        return dg_dx

    def __repr__(self):
        s =  '3D Average Pooling Layer\n'
        s += '\tPool Shape: %s\n'%str((self.k_f1,self.k_f2))
        s += '\tStride Shape: %s'%str((self.s1,self.s2))
        return s
