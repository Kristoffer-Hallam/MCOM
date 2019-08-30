import numpy as np

def sma1d(data, window_size):
    '''Apply a simple moving average filter with
    size window_size to data.
    
    input
    data: numpy array 1D - data set to be filtered.
    window_size: int - number of points forming the window.
                 It must be odd. If not, it will be increased
                 by one.
                 
    output
    filtered_data: numpy array 1D - filtered data. This array has the
                   same number of elementos of the original data.
    '''
    
    assert data.size >= window_size, \
        'data must have more elements than window_size'
    
    assert window_size%2 != 0, 'window_size must be odd'

    assert window_size >= 3, 'window_size must be greater than or equal to 3'

    # lost points at the extremities
    i0 = window_size//2

    # non-null data
    N = data.size - 2*i0

    filtered_data = np.empty_like(data)

    filtered_data[:i0] = 0.
    filtered_data[-1:-i0-1:-1] = 0.

    for i in range(N):
        filtered_data[i0+i] = np.mean(data[i:i+window_size])
        
    return filtered_data

def dot(x,y):
    '''Calculates the dot product between two arrays
    
    input >
    x:    1D or 2D array - first vector or matrix
    y:    1D or 2D array - second vector or matrix
    
    output >
    dot_product: scalar function
    '''
    c = 0
    if x.ndim == 1 and y.ndim == 1:
        assert x.size == y.size, '1D arrays must have the same size'
        for i in range(x.size):
            c += x[i]*y[i]
    elif x.ndim == 1 and y.ndim == 2:
        assert x.size == y.shape[0], 'Columns of array x must have the same \
                                            size as the rows of array y.'
        for i in range(y.shape[0]):
            c += x[i]*y[i]
    elif x.ndim == 2 and y.ndim == 1:
        assert x.shape[1] == y.size, 'Columns of array x must have the same \
                                            size as the rows of array y.'
        for i in range(x.shape[1]):
            c += x[i]*y[i]
    else:
        assert x.shape[1] == y.shape[0], 'Columns of array x must have the same \
                                            size as the rows of array y.'
        for i in range(y.shape[0]):
            c += x[i]*y[i]
    return c

def hadamard(x,y):
    '''Calculates the Hadamard product between two arrays
    
    input >
    x:    1D or 2D array - first vector or matrix
    y:    1D or 2D array - second vector or matrix
    
    output >
    hadamard_product: scalar function
    '''
    z = np.empty_like(x)
    if x.ndim == 1 and y.ndim == 1:
        assert x.size == y.size, '1D arrays must have the same size'
        for i in range(x.size):
            z[i] = x[i]*y[i]
    else:
        assert x.shape == y.shape, '2D arrays must have the same shape.'
        for i in range(x.size):
            z[i] = x[i]*y[i]
    return z

def outer1(x,y):
    '''Calculates the outer product between two vectors
    
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
    norm = 0.
    if p == 0:
        norm = np.max(np.abs(x))
    elif p == 1:
        for i in range(len(x)):
            norm += np.abs(x[i])
    else:
        for i in range(len(x)):
            norm += x[i]*x[i]
        # or norm = dot(x,x)**0.5
        norm = norm**0.5
    return norm

def matvec_prod1(A, x):
    '''Calculates the product between a matrix and a vector
    also called 'The triply nested for'
    input >
    A:    2D array - matrix
    x:    1D array - vector
    
    output >
    y:    1D array - product between matrix and vector
    '''
    assert A.shape[1] == x.size, 'Matrix columns do not match vector lines'
    y = np.zeros_like(x)
    for i in range(A.shape[0]):
        for j in range(x.size):
            y[i] += A[i,j]*x[j]
    return y

def matvec_prod2(A, x):
    '''Calculates the product between a matrix and a vector
    also called 'dot product formulation'
    input >
    A:    2D array - matrix
    x:    1D array - vector
    
    output >
    y:    1D array - product between matrix and vector
    '''
    assert A.shape[1] == x.size, 'Matrix columns do not match vector lines'
    y = np.zeros_like(x)
    for i in range(A.shape[0]):
        y[i] = dot(A[i,:], x)
    return y
    
def matvec_prod3(A, x):
    '''Calculates the product between a matrix and a vector
    also called 'linear combination formulation'
    input >
    A:    2D array - matrix
    x:    1D array - vector
    
    output >
    y:    1D array - product between matrix and vector
    '''
    assert A.shape[1] == x.size, 'Matrix columns do not match vector lines'
    y = np.zeros_like(x)
    for j in range(x.size):
        y += A[:,j]*x[j]
    return y

def mat_sma(data, ws):
    '''Calculates the moving average filtered data by matrix vector
    product
    
    input >
    vector:    1D array - vector
    window:    int      - window size
    
    output >
    filtered:  1D array - filtered data
    '''
    assert data.size >= ws, \
        'data must have more elements than window_size'
    
    assert ws%2 != 0, 'window_size must be odd'

    assert ws >= 3, 'window_size must be greater than or equal to 3'
    ws = 3 # window size
    i0 = ws//2
    A = np.array(np.hstack(((1./ws)*np.ones(ws), np.zeros(data.size - ws + 1))))
    A = np.resize(A, (data.size-2*i0, data.size))
    A = np.vstack((np.zeros(data.size), A, np.zeros(data.size)))
    filtered = matvec_prod1(A, data)
    return filtered

def deriv1d(data, spacing):
    '''Calculates the 1D first derivative based on central finite
    difference approach
    
    input >
    data:      1D array - vector
    spacing:   float    - spacing between points
    
    output >
    der:       1D array - vector of first derivatives
    '''
    h = 3 # window size
    i0 = h//2
    D = np.array(np.hstack(((1./(2.*spacing))*np.ones(h), np.zeros(data.size - h + 1))))
    D[0] = -1./(2.*spacing)
    D[1] = 0.
    D = np.resize(D, (data.size-2*i0, data.size))
    D = np.vstack((np.zeros(data.size), D, np.zeros(data.size)))
    der = matvec_prod2(D, data)
    return der

def matmat_prod1(A, B):
    '''Calculates a new matrix from the product of A and B
    also known as 'The triply nested for'
    input >
    A:          2D array - matrix
    B:          2D array - matrix
    
    output >
    C:          2D array - matrix
    '''
    assert A.shape[1] == B.shape[0], 'Number of columns of A is not the same as the lines of B'
    C = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                C[i,j] += A[i,k]*B[k,j]
    return C

def matmat_prod2(A, B):
    '''Calculates a new matrix from the product of A and B
    also known as 'Dot product formulation'
    input >
    A:          2D array - matrix
    B:          2D array - matrix
    
    output >
    C:          2D array - matrix
    '''
    assert A.shape[1] == B.shape[0], 'Number of columns of A is not the same as the lines of B'
    C = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            C[i,j] = dot(A[i,:], B[:,j])
    return C

def matmat_prod3(A, B):
    '''Calculates a new matrix from the product of A and B
    also known as 'Vector-matrix product formulation (row update)'
    input >
    A:          2D array - matrix
    B:          2D array - matrix
    
    output >
    C:          2D array - matrix
    '''
    assert A.shape[1] == B.shape[0], 'Number of columns of A is not the same as the lines of B'
    C = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        # C[i,:] = dot(A[i,:],B)
        C[i,:] = matvec_prod2(B.T, A[i,:])
    return C

def matmat_prod4(A, B):
    '''Calculates a new matrix from the product of A and B
    also known as 'Matrix-vector product formulation (colunm update)'
    input >
    A:          2D array - matrix
    B:          2D array - matrix
    
    output >
    C:          2D array - matrix
    '''
    assert A.shape[1] == B.shape[0], 'Number of columns of A is not the same as the lines of B'
    C = np.zeros((A.shape[0], B.shape[1]))
    for j in range(B.shape[1]):
        C[:,j] = dot(B[:,j], A.T)
    return C

def matmat_prod5(A, B):
    '''Calculates a new matrix from the product of A and B
    also known as 'Outer product formulation'
    input >
    A:          2D array - matrix
    B:          2D array - matrix
    
    output >
    C:          2D array - matrix
    '''
    assert A.shape[1] == B.shape[0], 'Number of columns of A is not the same as the lines of B'
    C = np.zeros((A.shape[0], B.shape[1]))
    for k in range(A.shape[1]):
        C += outer1(A[:,k],B[k,:])
    return C

def R1(theta):
    '''Creates R1 rotation matrix
    
    input >
    theta:       float - coordinate in degrees
    
    output >
    R1:          2D array - rotation matrix
    '''
    theta = np.deg2rad(theta)
    assert 0 <= theta <= 2.*np.pi, 'Theta is not in radians'
    R1 = np.array([1., 0., 0., 0., np.cos(theta), np.sin(theta), \
                   0., -np.sin(theta), np.cos(theta)])
    R1 = np.resize(R1, (3,3))
    return R1

def R2(theta):
    '''Creates R2 rotation matrix
    
    input >
    theta:       float - coordinate in degrees
    
    output >
    R2:          2D array - rotation matrix
    '''
    theta = np.deg2rad(theta)
    assert 0 <= theta <= 2.*np.pi, 'Theta is not in radians'
    R2 = np.array([np.cos(theta), 0., -np.sin(theta), 0., 1., 0., np.sin(theta), \
                   0.,  np.cos(theta)])
    R2 = np.resize(R2, (3,3))
    return R2

def R3(theta):
    '''Creates R2 rotation matrix
    
    input >
    theta:       float - coordinate in degrees
    
    output >
    R3:          2D array - rotation matrix
    '''
    theta = np.deg2rad(theta)
    assert 0 <= theta <= 2.*np.pi, 'Theta is not in radians'
    R3 = np.array([np.cos(theta), np.sin(theta), 0., -np.sin(theta), np.cos(theta), 0., 0., 1.])
    R3 = np.resize(R3, (3,3))
    return R3

def matvec_diag_prod(d, x):
    '''Makes the diagonal matrix-vector product through Hadamard product
    
    input >
    d:         1D array - vector extracted from the diagonal of a matrix
    x:         1D array - vector
    
    output >
    prod:      1D array - Hadamard product
    '''
    assert d.size == x.size, 'Sizes of vectors mismatch'
    prod = hadamard(d,x)
    return prod

def matmat_diagfull_prod(d, B):
    '''Makes the diagonal matrix-vector product through Hadamard product
    
    input >
    d:         1D array - vector extracted from the diagonal of a matrix
    B:         2D array - matrix
    
    output >
    prod:      1D array - Hadamard product
    '''
    C = np.zeros_like(B)
    assert d.size == B.shape[0], 'Incompatibility between lines of matrix\
                                    and size of diagonal vector'
    for i in range(d.size):
        C[i,:] = d[i]*B[i,:]
    return C

def matmat_fulldiag_prod(B, d):
    '''Makes the diagonal matrix-vector product through Hadamard product
    
    input >
    B:         2D array - matrix
    d:         1D array - vector extracted from the diagonal of a matrix
    
    output >
    prod:      1D array - Hadamard product
    '''
    E = np.zeros_like(B)
    assert d.size == B.shape[0], 'Incompatibility between lines of matrix\
                                    and size of diagonal vector'
    for j in range(d.size):
        E[:,j] = d[j]*B[:,j]
    return E

def matvec_diagk_prod(d, k, x):
    '''Makes the dislocated diagonal matrix-vector product through Hadamard product
    
    input >
    d:         1D array - vector extracted from the diagonal of a matrix
    k:         int      - defines direction of diagonal change
    x:         1D array - vector
    
    output >
    prod:      1D array - Hadamard product
    '''
    assert type(k) == int, 'Factor k is not integer'
    assert -x.size < np.abs(k) < x.size, 'Factor k must reside in interval [{limin},{limax}]'.format(limin=-x.size, limax=x.size)
    assert d.size == x.size - np.abs(k), 'Factor k decides the size of d according\
                                            to the size of x'
    z = np.zeros(x.size)
    if k < 0:
        z[np.abs(k):] = hadamard(d, x[:k])
    elif k == 0:
        z = hadamard(d, x)
    else:
        z[:d.size] = hadamard(d, x[k:])
    return z

def matvec_triu_prod3(U, x):
    '''Product between upper triangular matrix and a vector
    (Algorithm 3)
    
    input >
    U:         2D array - upper triangular matrix
    d:         1D array - vector
    
    output >
    prod:      1D array - vector
    '''
    assert U.shape[1] == x.size, 'Columns of matrix incompatible with size of vector'
    y = np.zeros_like(x)
    for i in range(x.size):
        y[i] = dot(U[i,i:], x[i:])
    return y

def matvec_triu_prod5(U, x):
    '''Product between upper triangular matrix and a vector
    (Algorithm 5)

    input >
    U:         2D array - upper triangular matrix
    d:         1D array - vector
    
    output >
    prod:      1D array - vector
    '''
    assert U.shape[1] == x.size, 'Columns of matrix incompatible with size of vector'
    y = np.zeros_like(x)
    for j in range(x.size):
        y[:j+1] += U[:j+1,j]*x[j]
    return y

def matvec_tril_prod8(L, x):
    '''Product between upper triangular matrix and a vector
    (Algorithm 8)

    input >
    L:         2D array - lower triangular matrix
    d:         1D array - vector
    
    output >
    prod:      1D array - vector
    '''
    assert L.shape[1] == x.size, 'Columns of matrix incompatible with size of vector'
    z = np.zeros_like(x)
    for i in range(x.size):
        z[i] = dot(L[i,:i+1],x[:i+1])
    return z

def matvec_tril_prod10(L, x):
    '''Product between upper triangular matrix and a vector
    (Algorithm 10)

    input >
    L:         2D array - lower triangular matrix
    d:         1D array - vector
    
    output >
    prod:      1D array - vector
    '''
    assert L.shape[1] == x.size, 'Columns of matrix incompatible with size of vector'
    z = np.zeros_like(x)
    for j in range(x.size):
        z[j:] += L[j:,j]*x[j]
    return z

def matvec_triu_opt_prod(u, x):
    '''Product between non-null element vector extracted from
    upper triangle matrix and a vector

    input >
    u:         1D array - non-null element vector extracted from
                            upper triangle matrix
    d:         1D array - vector
    
    output >
    prod:      1D array - vector
    '''
    assert u.size == (x.size*(x.size+1))/2, 'Number of elements is not equal to {size}'.format(size=(x.size*(x.size+1))/2)
    y = np.zeros(x.size)
    k = 0
    for i in range(x.size):
        y[:x.size-i] += u[i+k:x.size+k]*x[i:]
        k = x.size-(i+1)
    return y

def matvec_tril_opt_prod(l, x):
    '''Product between non-null element vector extracted from
    lower triangle matrix and a vector

    input >
    l:         1D array - non-null element vector extracted from
                            lower triangle matrix
    d:         1D array - vector
    
    output >
    y:      1D array - vector
    '''
    assert l.size == (x.size*(x.size+1))/2, 'Number of elements is not equal to {size}'.format(size=(x.size*(x.size+1))/2)
    y = np.zeros(x.size)
    k = 0
    for i in range(x.size):
        y[x.size-i-1:] += l[k:i+k+1]*x[:i+1]
        k += i+1
    return y

def matvec_symm_opt_prod(s, x):
    '''Product between vector whose elements are same as the elements
    of the diagonals of half a matrix and a generic vector

    input >
    s:         1D array - vector whose elements are same as the elements 
                            of the diagonals of half a matrix
    d:         1D array - vector
    
    output >
    y:      1D array - vector
    '''
    y = hadamard(s[:x.size], x)
    j1 = x.size
    j2 = x.size+4
    for i in range(1,x.size):
        y += matvec_diagk_prod(s[j1:j2],i,x)
        y += matvec_diagk_prod(s[j1:j2],-i,x)
        j1 = j2
        j2 = j2+x.size-(i+1)
    return y

def matvec_band_opt_prod(b, x):
    '''Product between vector whose elements are same as the elements
    of the diagonals of half a matrix and a generic vector

    input >
    b:         1D array - vector whose elements are parts of the
                            different diagonals of a block matrix
    x:         1D array - vector
    
    output >
    y:      1D array - vector
    '''

    j2 = 0
    y = np.zeros_like(x)
    for i in range(-1,3):
        j1 = j2
        j2 += x.size-np.abs(i)
        y += matvec_diagk_prod(b[j1:j2],i,x)
    return y