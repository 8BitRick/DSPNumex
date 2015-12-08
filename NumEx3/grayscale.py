# -*- coding: utf-8 -*-

# see the image
from IPython.display import Image
Image(filename='Num_Ex_3/camera_blurred_big_1.jpg')

# Put into a 2D array
%pylab inline
import matplotlib.pylab as plt
I = np.array(plt.imread('Num_Ex_3/camera_blurred.jpg'), dtype=float64)

I[:,0]

# Show using pylab.imshow
import matplotlib.pylab as plt
plt.imshow(I, cmap=plt.cm.gray, interpolation='none')  #The cmap=plt.cm.gray renders the image in gray
plt.show()


# Here we calculate error of the image during transfer
# ----------------------------------------------------
# Initialization of image
import matplotlib.pylab as plt
I_approx = np.array(plt.imread('Num_Ex_3/camera_blurred.jpg'), dtype=float64)

# Grey out half the image 
# pretending like half did not make it through transfer
I_approx[:,I_approx.shape[1]/2:] = 0

plt.imshow(I_approx, cmap=plt.cm.gray, interpolation='none')
plt.title('Approximation')
plt.show()

import math as m
# Error calculation
error = I - I_approx
distance = m.sqrt(sum(sum(error*error)))
print 'The distance between the original and approximate image is: ',distance

# This just shows the image is cut off and not good
# The next phase will cut into fidelity layers to give us detail in layers
# ----------------------------------------------------


# ----------------------------------------------------
# Begin Haar Wavelets
# Pretty cool - will let us layer our image in details

def haar(N):
    
    #Assuming N is a power of 2
    import numpy as np
    import math as m
    import scipy as sc
    h = np.zeros((N,N), dtype = float)
    h[0] = np.ones(N)/m.sqrt(N)
    for k in range(1,N) :
        
        p = sc.fix(m.log(k)/m.log(2))
        q = float(k - pow(2,p))
        k1 = float(pow(2,p))
        t1 = float(N / k1)
        k2 = float(pow(2,p+1))
        t2 = float(N / k2)
        
        for i in range(1,int(sc.fix(t2))+1):
            h[k,i+q*t1-1] = pow(2,(p/2))/m.sqrt(N)
            h[k,i+q*t1+t2-1] = -pow(2,(p/2))/m.sqrt(N)
        
    return h

import numpy as np 
    
#Load image
import matplotlib.pylab as plt
I = np.array(plt.imread('Num_Ex_3/camera_blurred.jpg'), dtype='float64')
size = I.shape

#Arrange image in column vector
I = I.flatten()
#Generate Haar basis vector (rows of H)
H = haar(4096)
    
#Project image on the new basis
I_Haar = np.dot(H,I)
    
#Remove the second half of the coefficient
I_Haar[2048 : 4095] = 0

#Recover the image by inverting change of basis
I_Haar = np.dot(H.T,I_Haar)

#Rearrange pixels of the image
I_Haar = I_Haar.reshape(size)    

imshow(I_Haar, cmap=plt.cm.gray)
show()

I = np.array(plt.imread('Num_Ex_3/camera_blurred.jpg'), dtype='float64')
error_h = I - I_Haar
import math as m
distance = m.sqrt(sum(sum(error_h*error_h)))
print 'The distance between the original image and the Haar approximation is ',distance

# ----------------------------------------------------
