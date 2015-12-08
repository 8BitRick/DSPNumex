# coding: utf-8
runfile('D:/dev/DSP/DSPNumex/NumEx3/grayscale.py', wdir='D:/dev/DSP/DSPNumex/NumEx3')
Image(filename='Num_Ex_3/camera_blurred_big.jpg')
from IPython.display import Image
Image(filename='Num_Ex_3/camera_blurred_big.jpg')
Image(filename='cameraman.jpg')
Image(filename='cameraman_down.jpg')
Image(filename='camera_down.jpg')
Image(filename='Num_Ex_3/TeX_7.jpg')
get_ipython().magic(u'pwd ')
Image(filename='Num_Ex_3\\TeX_7.jpg')
Image(filename='Num_Ex_3\\camera_blurred_1.jpg')
Image(filename='Num_Ex_3\\camera_blurred_1.png')
Image(filename='Num_Ex_3/camera_blurred_1.png')
Image(filename='Num_Ex_3/camera_blurred_big.jpg')
Image(filename='Num_Ex_3/camera_blurred.jpg')
Image(filename='Num_Ex_3/camera_blurred_big_1.jpg')
runfile('D:/dev/DSP/DSPNumex/NumEx3/grayscale.py', wdir='D:/dev/DSP/DSPNumex/NumEx3')
from IPython.display import Image
Image(filename='Num_Ex_3/camera_blurred_big_1.jpg')
get_ipython().magic(u'pylab inline')
import matplotlib.pylab as plt
I = np.array(plt.imread('Num_Ex_3/camera_blurred.jpg'), dtype=float64)
I[:,0]
len(I)
len(I[0])
I[0]
import matplotlib.pylab as plt
plt.imshow(I, cmap=plt.cm.gray, interpolation='none')  #The cmap=plt.cm.gray renders the image in gray
plt.show()
import matplotlib.pylab as plt
I_approx = np.array(plt.imread('Num_Ex_3/camera_blurred.jpg'), dtype=float64)
I_approx[:,0]
I_approx[0]
Image(filename='Num_Ex_3/camera_blurred.jpg')
I_approx[:,I_approx.shape[1]/2:] = 0
plt.imshow(I_approx, cmap=plt.cm.gray, interpolation='none')
plt.title('Approximation')
plt.show()
import math as m
error = I - I_approx
distance = m.sqrt(sum(sum(error*error)))
print 'The distance between the original and approximate image is: ',distance
I_approx.shape
I_approx.shape[1]
I_approx[:,0]
I_approx[:,1]
I_approx[:,63]
I_approx[:,62]
I_approx[:,62:]
I_approx[:,63:]
I_approx[:,61:]
error
sum(error)
sum(sum(error))
sum(error * error)
sum(sum(error * error))
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
import matplotlib.pylab as plt
I = np.array(plt.imread('Num_Ex_3/camera_blurred.jpg'), dtype='float64')
size = I.shape
I = I.flatten()
#Generate Haar basis vector (rows of H)
H = haar(4096)
I_Haar = np.dot(H,I)
I_Haar[2048 : 4095] = 0
I_Haar = np.dot(H.T,I_Haar)
I_Haar = I_Haar.reshape(size)    
size
imshow(I_Haar, cmap=plt.cm.gray)
show()
I = np.array(plt.imread('Num_Ex_3/camera_blurred.jpg'), dtype='float64')
error_h = I - I_Haar
import math as m
distance = m.sqrt(sum(sum(error_h*error_h)))
print 'The distance between the original image and the Haar approximation is ',distance
get_ipython().magic(u'reset')
get_ipython().magic(u'clear ()')
