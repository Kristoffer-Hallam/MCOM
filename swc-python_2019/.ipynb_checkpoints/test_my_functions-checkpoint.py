import numpy as np
import numpy.testing as npt
from pytest import raises

import my_functions as mf

def test_dot_var_size():
    '''Checks variable size compatibility'''
    x = np.empty(5)
    y = np.empty(4)
    raises(AssertionError, mf.dot, x, y)
    
    x = np.empty(4)
    y = np.empty(5)
    raises(AssertionError, mf.dot, x, y)
    
def test_dot_comparison():
    '''Compares my dot function with the numpy.dot'''
    x = np.arange(5)
    y = np.arange(5,10)
    my = mf.dot(x,y)
    py = np.dot(x,y)
    npt.assert_almost_equal(my, py, decimal=15)
    
def test_dot_result():
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
    A = np.array([[1,2,3], [4,5,6],[7,8,9]])
    x = np.ones(3)
    y = mf.matvec_prod1(A, x)
    npt.assert_almost_equal(y, result, decimal=15)
    
def test_matvec_prod2():
    '''Checks matvec_prod1 function'''
    result = np.array([6., 15., 24.])
    A = np.array([[1,2,3], [4,5,6],[7,8,9]])
    x = np.ones(3)
    y = mf.matvec_prod2(A, x)
    npt.assert_almost_equal(y, result, decimal=15)
    
def test_matvec_prod3():
    '''Checks matvec_prod1 function'''
    result = np.array([6., 15., 24.])
    A = np.array([[1,2,3], [4,5,6],[7,8,9]])
    x = np.ones(3)
    y = mf.matvec_prod3(A, x)
    npt.assert_almost_equal(y, result, decimal=15)
    
def test_matvec_prod_against_numpy_dot():
    '''Compares np.dot and our functions'''
    result = np.array([6., 15., 24.])
    A = np.array([[1,2,3], [4,5,6],[7,8,9]])
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
    A = np.array([[1,2], [4,5],[7,8]])
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
#    x = np.random.randint(0., 0.1, size=18)
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
    
def test_R1():
    '''Checks if R1 is orthogonal'''
    theta = 90.
    R1 = mf.R1(theta)
    I = np.identity(3)
    npt.assert_almost_equal(np.dot(R1, R1.T), I, decimal=15)
    
    npt.assert_almost_equal(np.dot(R1.T, R1), I, decimal=15)
    
def test_R2():
    '''Checks if R1 is orthogonal'''
    theta = 90.
    R2 = mf.R2(theta)
    I = np.identity(3)
    npt.assert_almost_equal(np.dot(R2, R2.T), I, decimal=15)
    
    npt.assert_almost_equal(np.dot(R2.T, R2), I, decimal=15)
    
def test_R3():
    '''Checks if R1 is orthogonal'''
    theta = 90.
    R3 = mf.R2(theta)
    I = np.identity(3)
    npt.assert_almost_equal(np.dot(R3, R3.T), I, decimal=15)
    
    npt.assert_almost_equal(np.dot(R3.T, R3), I, decimal=15)
    
def test_matvec_diag_prod_true_calc():
    '''Comparison between true and result from function'''
    true = np.array([-8., -4.,  0.,  4.,  8.])
    D = np.identity(5)*4.
    x = np.arange(-2,3)
    calc = mf.matvec_diag_prod(np.diag(D), x)
    npt.assert_almost_equal(calc, true, decimal=15)
    
def test_matvec_diag_prod_comparison_functions():
    '''Comparison between true and result from function'''
    d = np.random.rand(5)
    D = np.diag(d)
    x = np.arange(-2,3)
    diag_prod = mf.matvec_diag_prod(np.diag(D), x)
    matvec = mf.matvec_prod1(D, x)
    print 'diag_prod =', diag_prod
    print 'matvec =', matvec
    npt.assert_almost_equal(matvec, diag_prod, decimal=15)