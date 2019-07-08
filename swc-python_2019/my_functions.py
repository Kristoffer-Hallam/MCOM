import numpy as np

def dot(x,y):
    '''Calculates the dot product between two arrays
    
    input >
    x:    1D array - first vector
    y:    1D array - second vector
    
    output >
    dot_product: scalar function
    '''
    assert x.size == y.size, 'Arrays must have the same size'
    c = 0
    for i in range(x.size):
        c = c + x[i]*y[i]
    return c

def hadamard(x,y):
    '''Calculates the Hadamard product between two arrays
    
    input >
    x:    1D array - first vector
    y:    1D array - second vector
    
    output >
    hadamard_product: scalar function
    '''
    assert x.size == y.size, 'Arrays must have the same size'
    assert x.shape == y.shape, 'Arrays must have the same shape'
    z = np.empty_like(x)
    for i in range(x.size):
        z[i] = x[i]*y[i]
    return z

def outer1(x,y):
    '''Calculates the outer product between two arrays
    
    input >
    x:    1D array - first vector
    y:    1D array - second vector
    
    output >
    outer_product: scalar function
    '''
    assert x.size != y.size, 'Arrays must have different sizes'
    assert x.shape != y.shape, 'Arrays must have different shapes'
    M = np.empty((x.size,y.size))
    for i in range(x.size):
        for j in range(y.size):
            M[i,j] = x[i]*y[j]
    return M

def outer2(x,y):
    '''Calculates the outer product between two arrays
    
    input >
    x:    1D array - first vector
    y:    1D array - second vector
    
    output >
    outer_product: scalar function
    '''
    assert x.size != y.size, 'Arrays must have different sizes'
    assert x.shape != y.shape, 'Arrays must have different shapes'
    M = np.empty((x.size,y.size))
    for i in range(x.size):
        M[i,:] = x[i]*y
    return M

def outer3(x,y):
    '''Calculates the outer product between two arrays
    
    input >
    x:    1D array - first vector
    y:    1D array - second vector
    
    output >
    outer_product: scalar function
    '''
    assert x.size != y.size, 'Arrays must have different sizes'
    assert x.shape != y.shape, 'Arrays must have different shapes'
    M = np.empty((x.size,y.size))
    for j in range(y.size):
        M[:,j] = x*y[j]
    return M

def vec_norm(x, p):
    '''Calculates the p-norm of a vector
    
    input >
    x:    1D array - vector
    p:    int      - p value
    
    output >
    norm: scalar function
    '''
    assert type(p) == int, 'p value is not an integer'
    assert 0 <= p <= 2, 'p value must be in the inteval [0,2]'
    if p == 0:
        norm = np.max(x)
    elif p == 1:
        norm = np.sum(np.abs(x))
    else:
        norm = np.sqrt(dot(x,x))
    return norm