import numpy as np
import numpy.testing as npt
from pytest import raises

import my_functions as mf

def test_dot_var_size():
    '''Checks variable size compatibility'''
    x = np.empty(5)
    y = np.empty(4)
    raises(AssertionError, mf.dot, x, y)
    
    x = np.empty(5)
    y = np.empty((3,5))
    raises(AssertionError, mf.dot, x, y)

    x = np.empty((3,2))
    y = np.empty(5)
    raises(AssertionError, mf.dot, x, y)

    x = np.empty((3,2))
    y = np.empty((3,2))
    raises(AssertionError, mf.dot, x, y)
    
def test_dot_comparison():
    '''Compares my dot function with the numpy.dot'''
    x = np.arange(5)
    y = np.arange(5,10)
    my = mf.dot(x,y)
    py = np.dot(x,y)
    npt.assert_almost_equal(my, py, decimal=15)
    
def test_dot_true_calc():
    '''Comparing a simple result with the result of my function'''
    result = 80
    x = np.arange(5)
    y = np.arange(5,10)
    my = mf.dot(x,y)
    npt.assert_almost_equal(my, result, decimal=15)
    
def test_hadamard_var_size():
    '''Checks variable size compatibility'''
    x = np.empty(5)
    y = np.empty(4)
    raises(AssertionError, mf.dot, x, y)
    
    x = np.empty(4)
    y = np.empty(5)
    raises(AssertionError, mf.dot, x, y)
    
def test_hadamard_comparison():
    '''Compares my dot function with the numpy.dot'''
    x = np.arange(5)
    y = np.arange(5,10)
    my = mf.hadamard(x,y)
    py = x*y
    
    npt.assert_almost_equal(my, py, decimal=15)
    
def test_hadamard_result():
    '''Comparing a simple result with the result of my function'''
    result = np.array([ 0,  6, 14, 24, 36])
    x = np.arange(5)
    y = np.arange(5,10)
    my = mf.hadamard(x,y)
    
    npt.assert_almost_equal(my, result, decimal=15)
    
def test_outer1_var_size():
    '''Checks variable size incompatibility'''
    x = np.empty(4)
    y = np.empty(4)
    raises(AssertionError, mf.outer1, x, y)
    
def test_outer2_var_size():
    '''Checks variable size incompatibility'''
    x = np.empty(4)
    y = np.empty(4)
    raises(AssertionError, mf.outer2, x, y)
    
def test_outer3_var_size():
    '''Checks variable size incompatibility'''
    x = np.empty(4)
    y = np.empty(4)
    raises(AssertionError, mf.outer3, x, y)
    
def test_outer1_result():
    '''Checks result compatibility'''
    result = np.array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.], \
                       [ 5.,  6.,  7.,  8.,  9., 10., 11.], \
                       [10., 12., 14., 16., 18., 20., 22.], \
                       [15., 18., 21., 24., 27., 30., 33.], \
                       [20., 24., 28., 32., 36., 40., 44.]])
    x = np.arange(5)
    y = np.arange(5,12)
    my = mf.outer1(x,y)
    npt.assert_almost_equal(my, result, decimal=15)
    
def test_outer2_result():
    '''Checks result compatibility'''
    result = np.array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.], \
                       [ 5.,  6.,  7.,  8.,  9., 10., 11.], \
                       [10., 12., 14., 16., 18., 20., 22.], \
                       [15., 18., 21., 24., 27., 30., 33.], \
                       [20., 24., 28., 32., 36., 40., 44.]])
    x = np.arange(5)
    y = np.arange(5,12)
    my = mf.outer2(x,y)
    npt.assert_almost_equal(my, result, decimal=15)
    
def test_outer3_result():
    '''Checks result compatibility'''
    result = np.array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.], \
                       [ 5.,  6.,  7.,  8.,  9., 10., 11.], \
                       [10., 12., 14., 16., 18., 20., 22.], \
                       [15., 18., 21., 24., 27., 30., 33.], \
                       [20., 24., 28., 32., 36., 40., 44.]])
    x = np.arange(5)
    y = np.arange(5,12)
    my = mf.outer3(x,y)
    npt.assert_almost_equal(my, result, decimal=15)
    
def test_my_outer_Ipy_outer_comparison():
    '''Compares np.outer and our function'''
    x = np.arange(5)
    y = np.arange(5,12)
    py = np.outer(x,y)
    my1 = mf.outer1(x,y)
    npt.assert_almost_equal(my1, py, decimal=15)
    
    my2 = mf.outer2(x,y)
    npt.assert_almost_equal(my2, py, decimal=15)
    
    my3 = mf.outer3(x,y)
    npt.assert_almost_equal(my3, py, decimal=15)
    
    npt.assert_almost_equal(my2, my1, decimal=15)
    
    npt.assert_almost_equal(my3, my1, decimal=15)
    
    npt.assert_almost_equal(my3, my2, decimal=15)
    
def test_vec_norm_p_int():
    '''Checks if p value is integer'''
    p = 1.
    x = np.arange(-2,3)
    raises(AssertionError, mf.vec_norm, x, p)
    
def test_vec_norm_p_interval():
    '''Checks if p value is within the interval [0,2]'''
    p = -1
    x = np.arange(-2,3)
    raises(AssertionError, mf.vec_norm, x, p)
    
    p = 3
    x = np.arange(-2,3)
    raises(AssertionError, mf.vec_norm, x, p)
    
def test_vec_norm_first_inequality():
    '''Checks first inequality'''
    p = 2
    x = np.arange(-2,3)
    y = np.linspace(2,8,5)
    left = mf.vec_norm(x+y, p) 
    right = mf.vec_norm(x, p) + mf.vec_norm(y, p)
    npt.assert_(left <= right, 'Left-side term of inequality is not less than right-side term')
    
def test_vec_norm_second_inequality():
    '''Checks second inequality'''
    p = 2
    x = np.arange(-2,3)
    alpha = -2
    left = mf.vec_norm(alpha*x, p) 
    right = np.abs(alpha)*mf.vec_norm(x, p)
    npt.assert_(left <= right, 'Left-side term of inequality is not less than right-side term')
    
def test_vec_norm_third_inequality():
    '''Checks third inequality'''
    p = 2
    x = np.arange(-2,3)
    y = np.linspace(2,8,5)
    left = np.abs(mf.dot(x,y))
    right = mf.vec_norm(x, p)*mf.vec_norm(y, p)
    npt.assert_(left <= right, 'Left-side term of inequality is not less than right-side term')
    
def test_vec_norm_fourth_inequality():
    '''Checks fourth inequality'''
    p = [1,2]
    x = np.arange(-2,3)
    left = mf.vec_norm(x,p[1])
    middle = mf.vec_norm(x,p[0])
    right = np.sqrt(x.size)*mf.vec_norm(x,p[1])
    npt.assert_(left <= middle <= right, 'Left-side term of inequality is not less than right-side term')
    
def test_vec_norm_fifth_inequality():
    '''Checks fifth inequality'''
    p = [0,2]
    x = np.arange(-2,3)
    left = mf.vec_norm(x,p[0])
    middle = mf.vec_norm(x,p[1])
    right = np.sqrt(x.size)*mf.vec_norm(x,p[0])
    npt.assert_(left <= middle <= right, 'Left-side term of inequality is not less than right-side term')
    
def test_vec_norm_sixth_inequality():
    '''Checks sixth inequality'''
    p = [0,1]
    x = np.arange(-2,3)
    left = mf.vec_norm(x,p[0])
    middle = mf.vec_norm(x,p[1])
    right = x.size*mf.vec_norm(x,p[0])
    npt.assert_(left <= middle <= right, 'Left-side term of inequality is not less than right-side term')

def test_matvec_prod1():
    '''Checks matvec_prod1 function'''
    result = np.array([6., 15., 24.])
    A = np.array([[1,2,3],[4,5,6],[7,8,9]])
    x = np.ones(3)
    y = mf.matvec_prod1(A, x)
    npt.assert_almost_equal(y, result, decimal=15)
    
def test_matvec_prod2():
    '''Checks matvec_prod1 function'''
    result = np.array([6., 15., 24.])
    A = np.array([[1,2,3],[4,5,6],[7,8,9]])
    x = np.ones(3)
    y = mf.matvec_prod2(A, x)
    npt.assert_almost_equal(y, result, decimal=15)
    
def test_matvec_prod3():
    '''Checks matvec_prod1 function'''
    result = np.array([6., 15., 24.])
    A = np.array([[1,2,3],[4,5,6],[7,8,9]])
    x = np.ones(3)
    y = mf.matvec_prod3(A, x)
    npt.assert_almost_equal(y, result, decimal=15)
    
def test_matvec_prod_against_numpy_dot():
    '''Compares np.dot and our functions'''
    result = np.array([6., 15., 24.])
    A = np.array([[1,2,3],[4,5,6],[7,8,9]])
    x = np.ones(3)
    y1 = mf.matvec_prod1(A, x)
    y2 = mf.matvec_prod2(A, x)
    y3 = mf.matvec_prod3(A, x)
    py = np.dot(A, x)
    npt.assert_almost_equal(y1, py, decimal=15)
    
    npt.assert_almost_equal(y2, py, decimal=15)
    
    npt.assert_almost_equal(y3, py, decimal=15)
    
def test_matvec_prod_dimension_compatibility():
    '''Checks dimension compatibility'''
    A = np.array([[1,2],[4,5],[7,8]])
    x = np.ones(3)
    raises(AssertionError, mf.matvec_prod1, A, x)
    
    raises(AssertionError, mf.matvec_prod2, A, x)
    
    raises(AssertionError, mf.matvec_prod3, A, x)
    
def test_mat_sma_smaller_size_data():
    '''Checks if size of data is larger than size of window'''
    ws = 7
    data = np.empty(5)
    raises(AssertionError, mf.mat_sma, data, ws)
    
def test_mat_sma_ws_is_not_odd():
    '''Checks if window is odd'''
    ws = 8
    data = np.empty(20)
    raises(AssertionError, mf.mat_sma, data, ws)
    
def test_mat_sma_ws_is_not_larger_than_three():
    '''Checks if window is larger than three'''
    ws = 1
    data = np.empty(20)
    raises(AssertionError, mf.mat_sma, data, ws)
    
def test_mat_sma_comparison():
    '''Compares both sma functions'''
    ws = 3
    x = np.random.random(30)
    filt = mf.mat_sma(x, 5)
    y = mf.sma1d(x, ws)
    npt.assert_almost_equal(y, filt, decimal=15)

def test_deriv1d():
    '''Compares the true cossine function with the
    one calculated from our function'''
    spacing = 2.*np.pi/1000.
    theta = np.arange(0., 2*np.pi, spacing)
    y = np.sin(theta)
    z_true = np.cos(theta)
    z_calc = mf.deriv1d(y, spacing)
    npt.assert_almost_equal(z_calc[1:-1], z_true[1:-1], decimal=5)

def test_matmat_prod1_true_calc():
    '''Checks true and calcuted from matmat_prod1 function matrix'''
    C_true = np.array([[ 7., 11., 12.], [10., 16., 16.], [17., 28., 24.]])
    A = np.array([[1,3],[2,4],[5,6]])
    B = np.array([[1,2,0],[2,3,4]])
    npt.assert_almost_equal(C_true, mf.matmat_prod1(A, B), decimal=15)

def test_matmat_prod1_dot_versus_matmat_prod1():
    '''Checks true and calcuted from matmat_prod1 function matrix'''
    A = np.array([[1,3],[2,4],[5,6]])
    B = np.array([[1,2,0],[2,3,4]])
    npt.assert_almost_equal(np.dot(A, B), mf.matmat_prod1(A, B), decimal=15)

def test_matmat_prod2_true_calc():
    '''Checks true and calcuted from matmat_prod1 function matrix'''
    C_true = np.array([[ 7., 11., 12.], [10., 16., 16.], [17., 28., 24.]])
    A = np.array([[1,3],[2,4],[5,6]])
    B = np.array([[1,2,0],[2,3,4]])
    npt.assert_almost_equal(C_true, mf.matmat_prod2(A, B), decimal=15)

def test_matmat_prod2_dot_versus_matmat_prod1():
    '''Checks true and calcuted from matmat_prod1 function matrix'''
    A = np.array([[1,3],[2,4],[5,6]])
    B = np.array([[1,2,0],[2,3,4]])
    npt.assert_almost_equal(np.dot(A, B), mf.matmat_prod2(A, B), decimal=15)

def test_matmat_prod3_true_calc():
    '''Checks true and calcuted from matmat_prod1 function matrix'''
    C_true = np.array([[ 7., 11., 12.], [10., 16., 16.], [17., 28., 24.]])
    A = np.array([[1.,3.],[2.,4.],[5.,6.]])
    B = np.array([[1.,2.,0.],[2.,3.,4.]])
    npt.assert_almost_equal(C_true, mf.matmat_prod3(A, B), decimal=15)

def test_matmat_prod3_dot_versus_matmat_prod3():
    '''Checks true and calcuted from matmat_prod1 function matrix'''
    A = np.array([[1,3],[2,4],[5,6]])
    B = np.array([[1,2,0],[2,3,4]])
    npt.assert_almost_equal(np.dot(A, B), mf.matmat_prod3(A, B), decimal=15)

def test_matmat_prod4_true_calc():
    '''Checks true and calcuted from matmat_prod1 function matrix'''
    C_true = np.array([[ 7., 11., 12.], [10., 16., 16.], [17., 28., 24.]])
    A = np.array([[1.,3.],[2.,4.],[5.,6.]])
    B = np.array([[1.,2.,0.],[2.,3.,4.]])
    npt.assert_almost_equal(C_true, mf.matmat_prod4(A, B), decimal=15)

def test_matmat_prod4_dot_versus_matmat_prod3():
    '''Checks true and calcuted from matmat_prod1 function matrix'''
    A = np.array([[1,3],[2,4],[5,6]])
    B = np.array([[1,2,0],[2,3,4]])
    npt.assert_almost_equal(np.dot(A, B), mf.matmat_prod4(A, B), decimal=15)

def test_matmat_prod5_true_calc():
    '''Checks true and calcuted from matmat_prod1 function matrix'''
    C_true = np.array([[ 7., 11., 12.], [10., 16., 16.], [17., 28., 24.]])
    A = np.array([[1.,3.],[2.,4.],[5.,6.]])
    B = np.array([[1.,2.,0.],[2.,3.,4.]])
    npt.assert_almost_equal(C_true, mf.matmat_prod5(A, B), decimal=15)

def test_matmat_prod5_dot_versus_matmat_prod3():
    '''Checks true and calcuted from matmat_prod1 function matrix'''
    A = np.array([[1,3],[2,4],[5,6]])
    B = np.array([[1,2,0],[2,3,4]])
    npt.assert_almost_equal(np.dot(A, B), mf.matmat_prod5(A, B), decimal=15)
    
def test_R1():
    '''Checks if R1 is orthogonal'''
    theta = 90.
    R1 = mf.R1(theta)
    result1 = np.dot(R1, R1.T)
    I = np.identity(3)
    npt.assert_almost_equal(result1, I, decimal=15)
    
    result2 = np.dot(R1.T, R1)
    npt.assert_almost_equal(result2, I, decimal=15)
    
def test_R2():
    '''Checks if R1 is orthogonal'''
    theta = 90.
    R2 = mf.R2(theta)
    result1 = np.dot(R2, R2.T)
    I = np.identity(3)
    npt.assert_almost_equal(result1, I, decimal=15)
    
    result2 = np.dot(R2.T, R2)
    npt.assert_almost_equal(result2, I, decimal=15)
    
def test_R3():
    '''Checks if R1 is orthogonal'''
    theta = 90.
    R3 = mf.R2(theta)
    result1 = np.dot(R3, R3.T)
    I = np.identity(3)
    npt.assert_almost_equal(result1, I, decimal=15)
    
    result2 = np.dot(R3.T, R3)
    npt.assert_almost_equal(result2, I, decimal=15)
    
def test_matvec_diag_prod_true_calc():
    '''Comparison between true and result from function'''
    true = np.array([-8., -4.,  0.,  4.,  8.])
    D = np.identity(5)*4.
    x = np.arange(-2,3)
    calc = mf.matvec_diag_prod(np.diag(D), x)
    npt.assert_almost_equal(calc, true, decimal=15)
    
def test_matvec_diag_prod_comparing_functions():
    '''Comparison between true and result from function'''
    d = np.random.rand(5)
    x = np.arange(-2.,3.)
    diag_prod = mf.matvec_diag_prod(d, x)

    D = np.diag(d)
    matvec = mf.matvec_prod1(D, x)
    npt.assert_almost_equal(matvec, diag_prod, decimal=15)

def test_matvec_diag_var_size():
    '''Checks sizes of variables'''
    d = np.empty(5)
    x = np.empty(6)
    raises(AssertionError, mf.matvec_diag_prod, d, x)

def test_matmat_diagfull_prod_var_size():
    '''Checks sizes of variables'''
    d = np.empty(5)
    B = np.empty((6,6))
    raises(AssertionError, mf.matmat_diagfull_prod, d, B)

def test_matmat_diagfull_prod_true_calc():
    '''Compares true and calculated values'''
    true = np.array([[1, 2],[2, 3]])
    B = np.array([[1,2],[2,3]])
    d = np.diag(np.identity(2))
    calc = mf.matmat_diagfull_prod(d, B)
    npt.assert_almost_equal(calc, true, decimal=15)

def test_matmat_diagfull_prod_comparing_functions():
    '''Comparison between true and result from function'''
    d = np.random.rand(5)
    B = np.array([[1.,2.,3.,4.,5.],[2.,3.,4.,5.,6.],\
        [3.,4.,5.,6.,7.],[4.,5.,6.,7.,8.],[5.,6.,7.,8.,9.]])
    full_prod = mf.matmat_diagfull_prod(d, B)

    D = np.diag(d)
    matvec = mf.matmat_prod1(D, B)
    npt.assert_almost_equal(matvec, full_prod, decimal=15)

def test_matmat_fulldiag_prod_var_size():
    '''Checks sizes of variables'''
    d = np.empty(5)
    B = np.empty((6,6))
    raises(AssertionError, mf.matmat_fulldiag_prod, B, d)

def test_matmat_fulldiag_prod_true_calc():
    '''Compares true and calculated values'''
    true = np.array([[1, 2],[2, 3]])
    B = np.array([[1,2],[2,3]])
    d = np.diag(np.identity(2))
    calc = mf.matmat_fulldiag_prod(B, d)
    npt.assert_almost_equal(calc, true, decimal=15)

def test_matmat_diagfull_prod_comparing_functions():
    '''Comparison between true and result from function'''
    d = np.random.rand(5)
    B = np.array([[1.,2.,3.,4.,5.],[2.,3.,4.,5.,6.],\
        [3.,4.,5.,6.,7.],[4.,5.,6.,7.,8.],[5.,6.,7.,8.,9.]])
    full_prod = mf.matmat_fulldiag_prod(B, d)

    D = np.diag(d)
    matvec = mf.matmat_prod1(B, D)
    npt.assert_almost_equal(matvec, full_prod, decimal=15)

def test_matvec_diagk_prod_k_factor_type():
    '''Checks type of factor k'''
    d = np.empty(5)
    x = np.empty(5)
    k = 'Hi'
    raises(AssertionError, mf.matvec_diagk_prod, d, k, x)

    k = 3.
    raises(AssertionError, mf.matvec_diagk_prod, d, k, x)

    k = (1,2)
    raises(AssertionError, mf.matvec_diagk_prod, d, k, x)

    k = True
    raises(AssertionError, mf.matvec_diagk_prod, d, k, x)

    k = False
    raises(AssertionError, mf.matvec_diagk_prod, d, k, x)

    k = np.zeros_like(x)
    raises(AssertionError, mf.matvec_diagk_prod, d, k, x)

def test_matvec_diagk_prod_k_factor_interval():
    '''Checks if factor k resides within the correct interval'''
    d = np.empty(5)
    x = np.empty(5)
    k = -6
    raises(AssertionError, mf.matvec_diagk_prod, d, k, x)

    k = -5
    raises(AssertionError, mf.matvec_diagk_prod, d, k, x)

    k = 5
    raises(AssertionError, mf.matvec_diagk_prod, d, k, x)

    k = 6
    raises(AssertionError, mf.matvec_diagk_prod, d, k, x)

def test_matvec_diagk_prod_k_factor_xsize_dsize():
    '''Checks size compatibility'''
    k = 2
    x = np.empty(5)
    d = np.empty(5)
    raises(AssertionError, mf.matvec_diagk_prod, d, k, x)

def test_matvec_diagk_prod_true_calc():
    '''Compares true and calculated values'''
    true = np.array([4., 6., 0., 0., 0.])
    k = 3
    x = np.array([1.,2.,1.,4.,6.])
    d = np.ones(x.size-np.abs(k))
    mf.matvec_diagk_prod(d, k, x)
    npt.assert_almost_equal(true, mf.matvec_diagk_prod(d, k, x), decimal=15)

def test_matvec_diagk_prod_comparing_functions():
    '''Comparison between true and result from function'''
    k = -3
    x = np.array([1.,2.,1.,4.,6.])
    d = np.random.rand(x.size-np.abs(k))
    D = np.zeros((x.size, x.size))
    for i in range(x.size-np.abs(k)):
        D[np.abs(k)+i, i] = d[i]
    npt.assert_almost_equal(mf.matvec_prod1(D, x), \
        mf.matvec_diagk_prod(d, k, x), decimal=15)
    
    k = 0
    d = np.random.rand(x.size-np.abs(k))
    D = np.zeros((x.size, x.size))
    for i in range(x.size-np.abs(k)):
        D[i, i] = d[i]
    npt.assert_almost_equal(mf.matvec_prod1(D, x), \
        mf.matvec_diagk_prod(d, k, x), decimal=15)

    k = 3
    d = np.random.rand(x.size-np.abs(k))
    D = np.zeros((x.size, x.size))
    for i in range(x.size-np.abs(k)):
        D[i, k+i] = d[i]
    npt.assert_almost_equal(mf.matvec_prod1(D, x), \
        mf.matvec_diagk_prod(d, k, x), decimal=15)

def test_matvec_triu_prod3_prod5_dot_comparison():
    '''Compares results of created functions and np.dot'''
    x = np.array([1.,2.,1.,4.,6.])
    A = np.ones((5,5))
    U = np.triu(A)
    npt.assert_almost_equal(np.dot(U, x), mf.matvec_triu_prod3(U, x), decimal=15)

    npt.assert_almost_equal(np.dot(U, x), mf.matvec_triu_prod5(U, x), decimal=15)

def test_matvec_tril_prod8_prod10_dot_comparison():
    '''Compares results of created functions and np.dot'''
    x = np.array([1.,2.,1.,4.,6.])
    A = np.ones((5,5))
    L = np.tril(A)
    npt.assert_almost_equal(np.dot(L, x), mf.matvec_tril_prod8(L, x), decimal=15)

    npt.assert_almost_equal(np.dot(L, x), mf.matvec_tril_prod10(L, x), decimal=15)

def test_matvec_triu_opt_prod_u_size():
    '''Checks size of vector u'''
    x = np.empty(5)
    u = np.empty(5)
    raises(AssertionError, mf.matvec_triu_opt_prod, u, x)

def test_matvec_triu_opt_prod_comparing_functions():
    '''Compares matvec_triu_opt_prod function and
    matvec_triu_prod3'''
    x = np.array([1.,2.,1.,4.,6.])
    A = np.ones((5,5))
    U = np.triu(A)
    u = np.reshape(U, (U.shape[0]*U.shape[1]))
    ind = np.where(u == 0.)[0]
    u = np.delete(u, ind)
    npt.assert_almost_equal(mf.matvec_triu_opt_prod(u, x), \
                            mf.matvec_triu_prod3(U, x), decimal=15)

def test_matvec_tril_opt_prod_l_size():
    '''Checks size of vector u'''
    x = np.empty(5)
    l = np.empty(5)
    raises(AssertionError, mf.matvec_tril_opt_prod, l, x)

def test_matvec_tril_opt_prod_comparing_functions():
    '''Compares matvec_triu_opt_prod function and
    matvec_triu_prod3'''
    x = np.array([1.,2.,1.,4.,6.])
    A = np.ones((5,5))
    L = np.tril(A)
    l = np.reshape(L, (L.shape[0]*L.shape[1]))
    ind = np.where(l == 0.)[0]
    l = np.delete(l, ind)
    npt.assert_almost_equal(mf.matvec_tril_opt_prod(l, x), \
                            mf.matvec_tril_prod8(L, x), decimal=15)

def test_matvec_symm_opt_prod_true_calc():
    '''Compares the true value with the calculated from the function'''
    x = np.array([1.,2.,1.,4.,6.])
    S = np.triu(np.array([[5.,4.,3.,2.,1.],[4.,5.,4.,3.,2.],\
        [3.,4.,5.,4.,3.],[2.,3.,4.,5.,4.],[1.,2.,3.,4.,5.]]))
    s = np.reshape(S,(S.shape[0]*S.shape[1]))
    ind = np.where(s==0)[0]
    s = np.sort(np.delete(s, ind))[s.size::-1]

    true = np.array([30., 42., 50., 56., 54.])
    npt.assert_almost_equal(mf.matvec_symm_opt_prod(s,x),true,decimal=15)

def test_matvec_symm_opt_prod_comparing_functions():
    '''Compares results from different implemented functions'''
    x = np.array([1.,2.,1.,4.,6.])
    S = np.array([[5.,4.,3.,2.,1.],[4.,5.,4.,3.,2.],\
        [3.,4.,5.,4.,3.],[2.,3.,4.,5.,4.],[1.,2.,3.,4.,5.]])
    Sup = np.triu(S)
    s = np.reshape(Sup,(Sup.shape[0]*Sup.shape[1]))
    ind = np.where(s==0)[0]
    s = np.sort(np.delete(s, ind))[s.size::-1]

    npt.assert_almost_equal(mf.matvec_symm_opt_prod(s,x), mf.matvec_prod1(S, x),decimal=15)