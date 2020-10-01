''' Layers With Parameters'''

from .neural_network import Parameter, Layer
from .util import KernelShapeError
from .activations import *
from numpy import *
from scipy import stats


class Linear(Layer):

    def __init__(self, input_features, output_features, itype='Kaiming'):
        self.f_in = input_features
        self.f_out = output_features
        self.itype = itype.lower()
        self.w = Parameter(zeros((self.f_in,self.f_out)))
        self.b = Parameter(zeros(self.f_out))
        self.params = [self.w,self.b]
        self.total_params = self.w().size + self.b().size

    def f(self, x):
        z = dot(x,self.w())+self.b()
        return z
    
    def fp(self, x, g):
        dg_dx = dot(g,self.w().T)
        dg_dw = outer(x,g)
        dg_db = g
        self.w.add_grad(dg_dw)
        self.b.add_grad(dg_db)
        return dg_dx

    def __repr__(self):
        s =  'Linear (Fully Connected) Layer\n'
        s += '\tWeight Shape: %s\n'%str(self.w().shape)
        s += '\tBias Shape: %s\n'%str(self.b().shape)
        s += '\tTotal Parameters: %d'%self.total_params
        return s
    
    def _init_weights(self, x0, next_layer, alpha):
        if self.itype == 'kaiming':
            if next_layer != None and type(next_layer) != ReLu:
                raise Exception("Kaiming initialization requires ReLu next_layer.")
            std = sqrt(2/self.f_in)
        elif  self.itype == 'xavier':
            if next_layer != None and type(next_layer) != Tanh:
                raise Exception("Xavier initialization requires Tanh next_layer")
            std = sqrt(1/self.f_in)
        elif 'yamchow' in self.itype:
            if next_layer == None or type(next_layer) == Tanh:
                s_bar = 2.29
            elif type(next_layer) == Sigmoid:
                s_bar = 4.59
            else:
                raise Exception("YamChow initialization requires either Tanh or Sigmoid next_layer")
            if 'paper' in self.itype and 'uniform' in self.itype:
                bnd = lambda a: s_bar*sqrt(3/(self.f_in*(a**2).sum()))
            elif 'paper' in self.itype and 'gaussian' in self.itype:
                bnd = lambda a: s_bar*sqrt(1/(self.f_in*(a**2).sum()))
            elif 'novel' in self.itype and 'gaussian' in self.itype:
                bnd = lambda a: s_bar*sqrt(1/(stats.chi2.ppf(1-alpha,df=self.f_in)*(a**2).sum()))
            else:
                raise Exception("YamChow itype must include ('paper','uniform'), ('paper','gaussian'), or ('novel,'gaussian'). ")
            std = bnd(x0)
        return std
    
    def _set_weights(self, std):
        if 'uniform' not in self.itype:
            self.w.val = random.randn(self.f_in,self.f_out)*std
        else:
            self.w.val = random.rand(self.f_in,self.f_out)*2*std-std


class Conv_2D(Layer):

    def __init__(self, input_filters, input_features, kernel_filters, kernel_features, stride, padding, itype='Kaiming'):
        self.x_l = input_filters
        self.x_f = input_features
        self.k_l = kernel_filters
        self.k_f = kernel_features
        self.s = stride
        self.p = padding
        self.itype = itype.lower()
        self.x_f += 2*self.p
        self.z_f = (self.x_f-self.k_f)/self.s+1
        if self.z_f.is_integer():
            self.z_f = int(self.z_f)
        else:
            raise KernelShapeError(int(self.x_f),int(self.k_f),int(self.s))
        self.k = Parameter(zeros((self.k_l,self.x_l,self.k_f)))
        self.b = Parameter(zeros(self.k_l))
        self.params = [self.k,self.b]
        self.total_params = self.k().size + self.b().size
    
    def f(self, x_r):
        x = zeros((self.x_l,self.x_f))
        x[:,self.p:self.x_f-self.p] = x_r
        z = zeros((self.k_l,self.z_f))
        for l in range(self.k_l):
            for j in range(self.z_f):
                jj = j*self.s
                z[l,j] = (x[:,jj:jj+self.k_f]*self.k()[l,:,:]).sum()+self.b()[l]
        return z
    
    def fp(self, x_r, g):
        x = zeros((self.x_l,self.x_f))
        x[:,self.p:self.x_f-self.p] = x_r
        dg_dx = zeros((self.x_l,self.x_f))
        dg_dk = zeros((self.k_l,self.x_l,self.k_f))
        dg_db = zeros(self.k_l)
        for l in range(self.k_l):
            for j in range(self.z_f):
                jj = j*self.s
                g_lj = g[l,j]
                dg_dx[:,jj:jj+self.k_f] += self.k()[l,:,:]*g_lj
                dg_dk[l,:,:] += x[:,jj:jj+self.k_f]*g_lj
                dg_db[l] += g_lj
        dg_dx = dg_dx[:,self.p:self.x_f-self.p]
        self.k.add_grad(dg_dk)
        self.b.add_grad(dg_db)
        return dg_dx
    
    def __repr__(self):
        s =  '2D Convolution Layer\n'
        s += '\tKernel Shape: %s\n'%str(self.k().shape)
        s += '\tBias Shape: %s\n'%str(self.b().shape)
        s += '\tTotal Parameters: %d'%self.total_params
        return s
    
    def _init_weights(self, x0, next_layer, alpha):
        if self.itype == 'kaiming':
            if next_layer != None and type(next_layer) != ReLu:
                raise Exception("Kaiming initialization requires ReLu next_layer.")
            std = sqrt(2/(self.x_l*self.k_f))
        elif self.itype == 'xavier':
            if next_layer != None and type(next_layer) != Tanh:
                raise Exception("Xavier initialization requires Tanh next_layer")
            std = sqrt(1/(self.x_l*self.k_f))
        elif 'yamchow' in self.itype:
            if next_layer == None or type(next_layer) == Tanh:
                s_bar = 2.29
            elif type(next_layer) == Sigmoid:
                s_bar = 4.59
            else:
                raise Exception("YamChow initialization requires either Tanh or Sigmoid next_layer")
            if 'paper' in self.itype and 'uniform' in self.itype:
                bnd = lambda a: s_bar*sqrt(3/(self.x_l*self.k_f*(a**2).sum()))
            elif 'paper' in self.itype and 'gaussian' in self.itype:
                bnd = lambda a: s_bar*sqrt(1/(self.x_l*self.k_f*(a**2).sum()))
            elif 'novel' in self.itype and 'gaussian' in self.itype:
                bnd = lambda a: s_bar*sqrt(1/(stats.chi2.ppf(1-alpha,df=self.x_l*self.k_f)*(a**2).sum()))
            else:
                raise Exception("YamChow itype must include ('paper','uniform'), ('paper','gaussian'), or ('novel,'gaussian'). ")
            x = zeros((self.x_l,self.x_f))
            x[:,self.p:self.x_f-self.p] = x0
            std = inf
            for l in range(self.k_l):
                for j in range(self.z_f):
                    jj = j*self.s
                    bnd_i = bnd(x[:,jj:jj+self.k_f])
                    if bnd_i < std:
                        std = bnd_i
        return std
    
    def _set_weights(self, std):
        if 'uniform' not in self.itype:
            self.k.val = random.randn(self.k_l,self.x_l,self.k_f)*std
        else:
            self.k.val = random.rand(self.k_l,self.x_l,self.k_f)*2*std-std


class Conv_3D(Layer):

    def __init__(self, input_filters, input_features, kernel_filters, kernel_features, stride, padding, itype='Kaiming'):
        self.x_l = input_filters
        self.x_f1,self.x_f2 = input_features
        self.k_l = kernel_filters
        self.k_f1,self.k_f2 = kernel_features
        self.s1,self.s2 = stride
        self.p = padding
        self.itype = itype.lower()
        self.x_f1 += 2*self.p
        self.x_f2 += 2*self.p
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
        self.k = Parameter(zeros((self.k_l,self.x_l,self.k_f1,self.k_f2)))
        self.b = Parameter(zeros(self.k_l))
        self.params = [self.k,self.b]
        self.total_params = self.k().size + self.b().size
    
    def f(self, x_r):
        x = zeros((self.x_l,self.x_f1,self.x_f2))
        x[:,self.p:self.x_f1-self.p,self.p:self.x_f2-self.p] = x_r
        z = zeros((self.k_l,self.z_f1,self.z_f2))
        for l in range(self.k_l):
            for j1 in range(self.z_f1):
                for j2 in range(self.z_f2):
                    jj1 = j1*self.s1
                    jj2 = j2*self.s2
                    z[l,j1,j2] = (x[:,jj1:jj1+self.k_f1,jj2:jj2+self.k_f2]*self.k()[l,:,:,:]).sum()+self.b()[l]
        return z

    def fp(self, x_r, g):
        x = zeros((self.x_l,self.x_f1,self.x_f2))
        x[:,self.p:self.x_f1-self.p,self.p:self.x_f2-self.p] = x_r
        dg_dx = zeros((self.x_l,self.x_f1,self.x_f2))
        dg_dk = zeros((self.k_l,self.x_l,self.k_f1,self.k_f2))
        dg_db = zeros(self.k_l)
        for l in range(self.k_l):
            for j1 in range(self.z_f1):
                for j2 in range(self.z_f2):
                    jj1 = j1*self.s1
                    jj2 = j2*self.s2
                    g_lj = g[l,j1,j2]
                    dg_dx[:,jj1:jj1+self.k_f1,jj2:jj2+self.k_f2] += self.k()[l,:,:,:]*g_lj
                    dg_dk[l,:,:,:] += x[:,jj1:jj1+self.k_f1,jj2:jj2+self.k_f2]*g_lj
                    dg_db[l] += g_lj
        dg_dx = dg_dx[:,self.p:self.x_f1-self.p,self.p:self.x_f2-self.p]
        self.k.add_grad(dg_dk) 
        self.b.add_grad(dg_db)
        return dg_dx
    
    def __repr__(self):
        s =  '3D Convolution Layer\n'
        s += '\tKernel Shape: %s\n'%str(self.k().shape)
        s += '\tBias Shape: %s\n'%str(self.b().shape)
        s += '\tTotal Parameters: %d'%self.total_params
        return s
    
    def _init_weights(self, x0, next_layer, alpha):
        if self.itype == 'kaiming':
            if next_layer != None and type(next_layer) != ReLu:
                raise Exception("Kaiming initialization requires ReLu next_layer.")
            std = sqrt(2/(self.x_l*self.k_f1*self.k_f2))
        elif self.itype == 'xavier':
            if next_layer != None and type(next_layer) != Tanh:
                raise Exception("Xavier initialization requires Tanh next_layer")
            std = sqrt(1/(self.x_l*self.k_f1*self.k_f2))
        elif 'yamchow' in self.itype:
            if next_layer == None or type(next_layer) == Tanh:
                s_bar = 2.29
            elif type(next_layer) == Sigmoid:
                s_bar = 4.59
            else:
                raise Exception("YamChow initialization requires either Tanh or Sigmoid next_layer")
            if 'paper' in self.itype and 'uniform' in self.itype:
                bnd = lambda a: s_bar*sqrt(3/(self.x_l*self.k_f1*self.k_f2*(a**2).sum()))
            elif 'paper' in self.itype and 'gaussian' in self.itype:
                bnd = lambda a: s_bar*sqrt(1/(self.x_l*self.k_f1*self.k_f2*(a**2).sum()))
            elif 'novel' in self.itype and 'gaussian' in self.itype:
                bnd = lambda a: s_bar*sqrt(1/(stats.chi2.ppf(1-alpha,df=self.x_l*self.k_f1*self.k_f2)*(a**2).sum()))
            else:
                raise Exception("YamChow itype must include ('paper','uniform'), ('paper','gaussian'), or ('novel,'gaussian'). ")
            x = zeros((self.x_l,self.x_f1,self.x_f2))
            x[:,self.p:self.x_f1-self.p,self.p:self.x_f2-self.p] = x0
            std = inf
            for l in range(self.k_l):
                for j1 in range(self.z_f1):
                    for j2 in range(self.z_f2):
                        jj1 = j1*self.s1
                        jj2 = j2*self.s2
                        bnd_i = bnd(x[:,jj1:jj1+self.k_f1,jj2:jj2+self.k_f2])
                        if bnd_i < std:
                            std = bnd_i
        return std
    
    def _set_weights(self, std):
        if 'uniform' not in self.itype:
            self.k.val = random.randn(self.k_l,self.x_l,self.k_f1,self.k_f2)*std
        else:
            self.k.val = random.rand(self.k_l,self.x_l,self.k_f1,self.k_f2)*2*std-std


class GraphConv(Layer):
    
    def __init__(self, adjacency_matrix, input_features, output_features, itype='Kaiming'):
        self.a = adjacency_matrix
        self.f_in = input_features
        self.f_out = output_features
        self.itype = itype.lower()
        self.nodes = self.a.shape[0]
        self.w = Parameter(zeros((self.f_in,self.f_out)))
        self.b = Parameter(zeros(self.f_out))
        self.params = [self.w,self.b]
        self.total_params = self.w().size + self.b().size

    def f(self, xf):
        x = dot(self.a,xf)
        z = dot(x,self.w()) + self.b()
        return z
    
    def fp(self, xf, g):
        x = dot(self.a,xf)
        dg_dx = zeros((self.nodes,self.f_in))
        dg_dxf = zeros((self.nodes,self.f_in))
        dg_dw = zeros((self.f_in,self.f_out))
        dg_db = zeros(self.f_out)
        for v in range(self.nodes):
            dg_dx[v,:] = (self.w()*g[v,:]).sum(1)
            dg_dw += outer(x[v,:],g[v,:])
        dg_db = g.sum(0)
        for v in range(self.nodes):
            dg_dxf += outer(self.a[v,:],dg_dx[v,:])
        self.w.add_grad(dg_dw)
        self.b.add_grad(dg_db)
        return dg_dxf

    def __repr__(self):
        s =  'Graph Convolution Layer\n'
        s += '\tNodes: %s\n'%str(self.nodes)
        s += '\tWeight Shape: %s\n'%str(self.w().shape)
        s += '\tBias Shape: %s\n'%str(self.b().shape)
        s += '\tTotal Parameters: %d'%self.total_params
        return s
    
    def _init_weights(self, x0, next_layer, alpha):
        if self.itype == 'kaiming':
            if next_layer != None and type(next_layer) != ReLu:
                raise Exception("Kaiming initialization requires ReLu next_layer.")
            std = sqrt(2/self.f_in)
        elif self.itype == 'xavier':
            if next_layer != None and type(next_layer) != Tanh:
                raise Exception("Xavier initialization requires Tanh next_layer")
            std = sqrt(1/self.f_in)
        elif 'yamchow' in self.itype:
            if next_layer == None or type(next_layer) == Tanh:
                s_bar = 2.29
            elif type(next_layer) == Sigmoid:
                s_bar = 4.59
            else:
                raise Exception("YamChow initialization requires either Tanh or Sigmoid next_layer")
            if 'paper' in self.itype and 'uniform' in self.itype:
                bnd = lambda a: s_bar*sqrt(3/(self.f_in*(a**2).sum()))
            elif 'paper' in self.itype and 'gaussian' in self.itype:
                bnd = lambda a: s_bar*sqrt(1/(self.f_in*(a**2).sum()))
            elif 'novel' in self.itype and 'gaussian' in self.itype:
                bnd = lambda a: s_bar*sqrt(1/(stats.chi2.ppf(1-alpha,df=self.f_in)*(a**2).sum()))
            else:
                raise Exception("YamChow itype must include ('paper','uniform'), ('paper','gaussian'), or ('novel,'gaussian'). ")
            x = dot(self.a,x0)
            std = bnd(x)
        return std
    
    def _set_weights(self, std):
        if 'uniform' not in self.itype:
            self.w.val = random.randn(self.f_in,self.f_out)*std
        else:
            self.w.val = random.rand(self.f_in,self.f_out)*2*std-std
