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