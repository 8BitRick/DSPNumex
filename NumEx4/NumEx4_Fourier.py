# -*- coding: utf-8 -*-
"""
Created on Tue Dec 08 09:00:15 2015

@author: Rick
"""

# function to make DFT
def dftmatrix(N):
    '''construct DFT matrix'''
    import numpy as np
    
    # create a 1xN matrix containing indices 0 to N-1
    a = np.expand_dims(np.arange(N), 0)
    
    # take advantage of numpy broadcasting to create the matrix
    WN = np.exp(-2j*np.pi*a.T*a/N)
    
    return WN

#---------------
# Messing with a line of 1's to show error    
%pylab inline

N = 6
WN = dftmatrix(N)

x = np.ones(N)
X = np.dot(WN, x)

x2 = 1./N * np.dot(WN.T.conjugate(), X)
print 'Error: ',(np.abs(x - x2).sum())
print '(This small error is due to machine precision.)'

#=======================================================
# Now let's do an example with unitary step function

x = np.append(np.ones(64),np.zeros(64))

y = np.linspace(0,127,128) # makes 0..127 inclusive
# The plotting is done by the pylab.stem function
stem(y,x)
xlabel('Samples')
ylabel('x')
ylim([0,2])
show()

# Transform!
x = x.flatten();
N = 128
W128 = dftmatrix(N)
X  = np.dot(W128,x)

# Magnitude
magnitude = abs(X)

y = np.linspace(0,127,128)
y[:] = y[:]/len(y) #we are normalizing the frequencies
# Plotting is done by the pylab.stem function
stem(y,magnitude)
ylabel('Magnitude')
xlabel('Normalized Frequencies')
show()

# Phase
phase = np.angle(X)
stem(y,phase)
ylabel('Angle(X)')
xlabel('Normalized Frequencies')
show()

# Numerical Intermezzo
# Shows what we should get
# All even values [2,4,6,...,] should be zero
# The math calls these "odd" values though because X[2] in code is X(3) in math
Theoretical_phase = np.linspace(-1.5,1.5,128)
Theoretical_phase[0:127:2] = 0
stem(y,Theoretical_phase)
ylabel('Angle(X)')
xlabel('Normalized Frequencies')
xlim([0,1])
ylim([-2,2])
show()

# Now synthesize back to signal
xx = np.real(np.dot(np.conjugate(W128),X) *1/N)
stem(np.arange(xx.size),xx)
ylim([0,2])
xlim([0,128])
ylabel('x')
xlabel('Samples')
show()

#======================================================
# load sample data

import scipy.io
mat = scipy.io.loadmat('NumEx4/frequencyRepresentation.mat')
print mat

#-------------------------------------------------------
# My attempt to analyze this signal
x = mat['x'][0]
N = len(x) # 4000

# Let's look at it
num_samples = 50 # Can set to any number, 4000 is whole set
axis = np.arange(num_samples)
stem(axis, x[0:num_samples])
ylabel('Amplitude')
xlabel('Sample #')
#xlim([0,4000])
#ylim([-2,2])
show()

# Transform!
dft4000 = dftmatrix(4000)
X = np.dot(dft4000, x)

# Get magnitude and angle
Xmag = abs(X)
Xangle = np.angle(X)

# Let's look at the magnitude
num_samples = N
axis = np.linspace(0.,1.,num_samples)
stem(axis, Xmag[0:num_samples])
ylabel('Amplitude')
xlabel('Frequency(normalized)')
show()

# Let's look at the phase
num_samples = 100
axis = np.linspace(0.,1.,num_samples)
stem(axis, Xangle[0:num_samples])
ylabel('Angle')
xlabel('Frequency(normalized)')
show()

#-------------------------------------------------------
# Now doing the NumEx4 investigation of the data

signal = mat['x']
print signal 
type(signal)

signal = signal.reshape(signal.size,)
plot(np.arange(signal.size),signal)
xlim([0,500])
xlabel('Samples')
ylabel('x')
show()

signal = signal.reshape(signal.size,1)
W4000 = dftmatrix(4000)
X = np.dot(W4000,signal)

normFrequ = np.arange(1,X.size+1,dtype=float)/float(X.size)
plot(normFrequ,abs(X))
ylabel('|X|')
xlabel('Normalized Frequencies')
show()

#-------------------------------------------------------
# DFS and DFT as projections on a base of complex sinusoids

# Looking at one complex sinusoid
x = np.append(np.ones(64),np.zeros(64))
W128 = dftmatrix(128)
X1 = np.dot(W128[0],x)
xx1 = np.real(np.conjugate(W128[:,0])*X1*1/128)

X2 = np.dot(W128[0:2],x)
xx2 = np.real(np.dot(np.conjugate(W128[:,0:2]),X2)*1/128)

X3 = np.dot(W128[0:3],x)
xx3 = np.real(np.dot(np.conjugate(W128[:,0:3]),X3)*1/128)

X4 = np.dot(W128[0:4],x)
xx4 = np.real(np.dot(np.conjugate(W128[:,0:4]),X4)*1/128)

X5 = np.dot(W128[0:5],x)
xx5 = np.real(np.dot(np.conjugate(W128[:,0:5]),X5)*1/128)

X6 = np.dot(W128[0:6],x)
xx6 = np.real(np.dot(np.conjugate(W128[:,0:6]),X6)*1/128)

X7 = np.dot(W128[0:7],x)
xx7 = np.real(np.dot(np.conjugate(W128[:,0:7]),X7)*1/128)

X8 = np.dot(W128[0:8],x)
xx8 = np.real(np.dot(np.conjugate(W128[:,0:8]),X8)*1/128)

X9 = np.dot(W128[0:9],x)
xx9 = np.real(np.dot(np.conjugate(W128[:,0:9]),X9)*1/128)

X10 = np.dot(W128[0:10],x)
xx10 = np.real(np.dot(np.conjugate(W128[:,0:10]),X10)*1/128)

plot(np.arange(xx1.size),xx1)
plot(np.arange(xx2.size),xx2)
plot(np.arange(xx3.size),xx3)
plot(np.arange(xx4.size),xx4)
plot(np.arange(xx5.size),xx5)
plot(np.arange(xx6.size),xx6)
plot(np.arange(xx7.size),xx7)
plot(np.arange(xx8.size),xx8)
plot(np.arange(xx9.size),xx9)
plot(np.arange(xx10.size),xx10)
xlabel('Samples')
ylabel('x')
xlim([0,128])
show()

#==============================
# Three tones
from scipy import constants as c
y = np.linspace(0,999,1000)
x1 = np.sin((y*2*c.pi*40)/1000)
x2 = np.sin((y*2*c.pi*80)/1000)
x3 = np.sin((y*2*c.pi*160)/1000)

x = np.append(x1,[np.zeros(1000),x2,np.zeros(1000),x3])

W5000 = dftmatrix(5000)
X = np.dot(W5000,x)

normFrequ = np.arange(1,X.size+1,dtype=float)/float(X.size)
plot(normFrequ,abs(X))
ylabel('|X|')
xlabel('Normalized Frequencies')
show()

# Use less N=400
W5000reduced = np.append(W5000[0:600],W5000[4399:5000],axis=0)
iW5000reduced = np.conj(np.append(W5000[:,0:600],W5000[:,4399:5000],axis=-1))

Xapprox = np.dot(W5000reduced,x)  # DFT/DFS on a reduced base
xapprox = np.dot(iW5000reduced,Xapprox)*1/5000 # inverse DFT/DFS of the DFT/DFS on a reduced base
xapprox = np.real(xapprox) # real value of the result to get rid of imaginary numerical errors

X = np.dot(W5000,xapprox)

plot(np.arange(1,5001,dtype=float)/float(5000),abs(X))
show()

#============================================
# Fast Fourier Transform

from scipy import fftpack as f
X = f.fft(x)

y = np.linspace(0,3999,4000)
x1 = np.sin((y*2*c.pi*40)/1000)
x2 = np.sin((y*2*c.pi*50)/1000)
x = x1 + x2

M = 4000
X = f.fft(x,M)
#Plot versus the normalized frequencies
normFrequ = np.arange(1,M+1,dtype=float)/float(M)
plot(normFrequ,abs(X))
ylabel('|X|')
xlabel('Normalized Frequencies')
show()

M = 500
X = f.fft(x,500)

#Plot versus the normalized frequencies
normFrequ = np.arange(1,M+1,dtype=float)/float(M)
plot(normFrequ,abs(X))
ylabel('|X|')
xlabel('Normalized Frequencies')
ylim([0,300])
show()

M = 50
X = f.fft(x,50)

#Plot versus the normalized frequencies
normFrequ = np.arange(1,M+1,dtype=float)/float(M)
plot(normFrequ,abs(X))
ylabel('|X|')
xlabel('Normalized Frequencies')
ylim([0,35])
show()

M = 50
X = f.fft(x,M)   # X is of length 50
x50 = f.ifft(X)

M = 4000
X = f.fft(x50,M)

#Plot versus the normalized Frequencies
normFrequ = np.arange(1,M+1,dtype=float)/float(M)
plot(normFrequ,abs(X))
ylabel('|X|')
xlabel('Normalized Frequencies')
show()


