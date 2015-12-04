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

#np.eye
#np.tile
#np.zeros(x.size)
#alphaMatrix = np.eye(D,M)
#xMatrix = np.tile(x, (D,1))
#np.transpose

def ks_matrix(x, alpha, D) :
    import numpy as np
    #   Length of the output signal must be larger than the length of the input signal,
    #   that is, D must be larger than 1 
    if D < 1:
        raise ValueError('Duration D must be greater than 1')
        
    #   Make sure the input is a row-vector
    if x.ndim != 1:
        raise ValueError('The array entered must be a row-vector')
        return None
    #   Number of input samples
    M = len(x)
    
    #   Create a vector of the powers of alpha, [alpha^0 alpha^1 ....]
    a = np.ones((1,D)) * alpha
    b = np.arange(D)
    alphaVector = pow(a,b)
    
    #Create a matrix with M columns, each being the vector of the powers of alpha
    alphaMatrix = np.eye(D,M) 
    for index in range(M):
        alphaMatrix[:,index] = alphaVector
        
    #Create a matrix with D rows filled by the input signal x  
    xMatrix = np.tile(x,(D,1))
    
    #Multipliy the two, so we can read it out
    #column-by-column
    yMatrix = alphaMatrix * xMatrix
    
    #Read out the output column by columnn
    y = yMatrix.flatten()
        
    return y

z = ks_matrix(x, 0.9, 10)

stem(np.arange(z.size),z)
xlabel('Samples')
ylabel('z')
show()


###############################################################################
## Begin Music Example - Hard day's night

# Parameters:
#
# - Fs       : sampling frequency
# - F0       : frequency of the notes forming chord
# - gain     : gains of individual notes in the chord
# - duration : duration of the chord in second
# - alpha    : attenuation in KS algorithm

Fs = 48000

import numpy as np
# D2, D3, F3, G3, F4, A4, C5, G5
F0 = 440*np.array([pow(2,(-31./12.)), pow(2,(-19./12.)), pow(2,(-16./12.)), pow(2,(-14./12.)), pow(2,(-4./12.)), 1, pow(2,(3./12.)), pow(2,(10./12.))])
gain = np.array([1.2, 3.0, 1.0, 2.2, 1.0, 1.0, 1.0, 3.5])
duration = 4
alpha = 0.9785

# Number of samples in the chord
nbsample_chord = Fs * duration

# This is used to correct alpha later, so that all the notes decay together
# (with the same decay rate)
first_duration = np.ceil(float(nbsample_chord)/round(float(Fs)/float(F0[0])))

# Initialization
chord = np.zeros(nbsample_chord)

for i in range(len(F0)):
    
    # Get M and duration parameter
    current_M = round(float(Fs)/float(F0[i]));
    current_duration = np.ceil(float(nbsample_chord)/float(current_M))
    
    # Correct current alpha so that all the notes decay together (with the
    # same decay rate)
    current_alpha = pow(alpha,(float(first_duration)/float(current_duration)))
    
    # Let Paul's high D on the bass ring a bit longer
    if i == 1:
        current_alpha = pow(current_alpha,.8)
    
    # Generate input and output of KS algorithm
    x = np.random.rand(current_M)
    y = ks(x, current_alpha, current_duration)
    y = y[0:nbsample_chord]
    
    # Construct the chord by adding the generated note (with the
    # appropriate gain)
    chord = chord + gain[i] * y

import numpy as np
from scipy.io.wavfile import write

data = chord
scaled = np.int16(data/np.max(np.abs(data)) * 32767)

write('hard_days.wav', 44100, scaled)

###############################################################################
## Try to make Joy to the World (beeps)

import winsound
winsound.PlaySound('hard_days.wav')

import winsound
notes = [15,14,12,10,8,7,5,3]
def beep(note, dur=500):
    winsound.Beep(int(440. * pow(2.,((note + 12.)/12.))), int(dur))

# Play with no pauses
for n in notes: beep(n)
    
# Add durations
durations = np.array([1.0,0.75,0.5,0.75,0.5,0.75,0.75,0.75]) * 1000
note_pairs = zip(notes, durations)
for n in note_pairs: beep(n[0], n[1])
# That's the best we can do with "beep". Can't control the pauses.


###############################################################################
## Try to make Joy to the World (wav)

def w(file_name):
    write(file_name,44100,scaled)

Fs = 48000
np.average
np.pi
def sin_sample(samples):
    return np.sin(np.arange(samples)*2*np.pi / samples)

def square_sample(samples):
    return np.concatenate((np.zeros(samples/2), np.ones(samples/2)))

sin_samples = sin_sample

# Sine wave
x = sin_sample(current_M)
write('C4_clean.wav', 44100, scaled)

x = np.concatenate((np.zeros(current_M/2), np.ones(current_M/2)))

sample_method=sin_sample
def make_note(n):
    freq = 440.0 * pow(2.,(float(n)/12.))
    Fs = 48000.0
    Play_time = 4.0 # seconds
    Play_samples = Fs * Play_time
    M = round(Fs/freq)
    dur = np.ceil(Play_samples/M)
    print "M: {}, Dur: {}".format(int(M), int(dur))
    #alpha = 0.9785
    alpha = 0.99694289968288352 # this makes a huge difference
    x = sample_method(M)
    y = ks(x, alpha, dur)
    y = y[0:Play_samples]
    return y

# Here are our notes again, we are adjusting durations from last time
notes = [15,14,12,10,8,7,5,3]
durations = np.array([2.,1.5,0.5,3.0,1.,2.,2.,4.]) * 0.4 * 1000
note_pairs = zip(notes, durations)

# Map our notes into samples
note_samples = map(make_note, np.array(note_pairs)[:,0])

# Calculate total time of our song
play_times = np.array(note_pairs)[:,1] / 1000.0
total_time = np.sum(play_times) + (4.0)
total_samples = int(np.ceil((total_time) * Fs))

# build our final output buffer
final_buffer = np.zeros(total_samples)

# find the start times for each note
note_start_times = np.roll(play_times,1)
note_start_times[0] = 0.
note_start_times = np.cumsum(note_start_times)

start_time_and_samples = zip(note_start_times, note_samples)

for ts in start_time_and_samples:
    start = int(np.floor(ts[0] * Fs))
    end = int(np.floor(start + len(ts[1])))
#    print "ts len: {}".format(len(ts[1]))
#    print "start:end len: {}".format(end - start)
    final_buffer[start:end] += ts[1]

data = final_buffer
scaled = np.int16(data/np.max(np.abs(data)) * 32767)
import os
os.remove('Joy.wav')
write('Joy.wav', 48000, scaled)
# AWESOME! Sounds like a xylophone playing Joy to the World intro!
