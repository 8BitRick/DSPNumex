# -*- coding: utf-8 -*-
"""
Created on Wed Dec 02 17:45:56 2015

@author: Richard
"""

def ks_loop(x, alpha, D) :
    import numpy as np
    ''' 
    Length of the output signal must be larger than the length of the input signal,
    that is, D must be larger than 1 
    '''
    if D < 1:
        raise ValueError('Duration D must be greater than 1')
        
    # Make sure the input is a row-vector
    if x.ndim != 1:
        raise ValueError('The array entered must be a row-vector')
        return None
    
    # Number of input samples
    M = len(x)
    
    # N umber of output samples
    size_y = D*M
    
    # Initialize with random input x
    y = np.zeros((size_y,1))
    for i in range(M):
        y[i] = x[i]
    
    for index in range(M,size_y):
        y[index] = float(alpha * y[index - M])
    
    return y
    
import numpy as np
x = np.random.randn(100)
y = ks_loop(x, 0.9, 10)

#%pylab inline
stem(np.arange(x.size),x,)
xlabel('Samples')
ylabel('x')
show()

stem(np.arange(y.size),y)
xlabel('Samples')
ylabel('y')
show()


#np.zeros(x.size)
#np.transpose
