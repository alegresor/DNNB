''' Functions Related to Constructing Adjacency Matricies '''

from numpy import *

def construct_a1(x, y, n_lines):
    x -= 1
    y -= 1
    a1 = zeros((n_lines,n_lines))
    for i in range(len(x)):
        i_0 = min(x[i],y[i])
        i_1 = max(x[i],y[i])
        a1[i_0,i_1] = 1
    return (a1+a1.T+diag(ones(n_lines))).astype(int)
