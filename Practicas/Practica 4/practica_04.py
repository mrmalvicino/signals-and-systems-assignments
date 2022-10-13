import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import scipy
import sys, os

from IPython.display import Audio
from scipy.signal import fftconvolve
from scipy import signal

sys.path.insert(0, os.path.normpath('C:/Users/Admin/Desktop/signals-and-systems-assignments'))
import file_management as fm


def sq_pulse(t, tau, f_s):
    
    """
    Generates a square pulse centered at the center a given time vector.
    
    Parameters
    ----------
    
    t : NUMPY ARRAY
        Time vector.
    
    tau : FLOAT
        Width of the pulse in seconds.
    
    f_s : INTEGER
        Sampling frequency in samples per second.
    
    Raises
    ------
    
    ValueError
        Parameter tau can not be larger than the time vector lenght.
    
    Returns
    -------
    
    x : NUMPY ARRAY
        Amplitude vector.
    
    """
    
    if tau > len(t):
        raise ValueError('Parameter tau can not be larger than the time vector lenght.')
    
    x = np.zeros(t.size)
    on  = int(0.5 * t.size) - int(0.5 * tau * f_s)
    off = int(0.5 * t.size) + int(0.5 * tau * f_s)
    x[on:off] = 1
    
    return x


def tr_pulse(t, m, f_s):
    
    if m <= 0 or len(t)/2 < m:
        raise ValueError('The parameter m must be greater than 0 and less than (n_end-n_start)/2.')
    
    if m < 1/f_s:
        raise ValueError('Can not define such a narrow triangle for the requested sampling frequency.')
    
    x = np.zeros(t.size)
    t_center = int((t.max()-t.min())/2)*f_s
    
    for i in range(-int(m*f_s), int(m*f_s), 1):
        t_i = t_center + i
        x[t_i] = 1 - abs(i/(m*f_s))
    
    return x


def conv_interval(t_1, t_2, f_s):
    
    a, b = t_1.min(), t_1.max()
    c, d = t_2.min(), t_2.max()
    
    if d >= a:
        conv_start = -abs(d-a)
    else:
        conv_start = abs(d-a)
        
    conv_end = conv_start + abs(a-b) + abs(c-d) - (1/f_s)
    
    return conv_start, conv_end


f_s = 100
t_start, t_end = 0, 10
t = np.linspace(t_start, t_end, int((t_end-t_start)*f_s))

x_1 = sq_pulse(t, 5, f_s)
x_2 = tr_pulse(t, 3, f_s)
x_3 = np.sin(2*np.pi*(1/1)*t)

conv_12 = np.convolve(x_1, x_2)
conv_13 = np.convolve(x_1, x_3)
conv_23 = np.convolve(x_2, x_3)

conv_start, conv_end = conv_interval(t, t, f_s)
t_conv = np.linspace(conv_start, conv_end, int(x_1.size+x_2.size-1))

fig, (axis11, axis21, axis31) = plt.subplots(3,1, figsize=(10,15))
axis11.plot(t, x_1, color='blue')
axis21.plot(t, x_2, color='blue')
axis31.plot(t_conv, conv_12, color='red')

fig, (axis11, axis21, axis31) = plt.subplots(3,1, figsize=(10,15))
axis11.plot(t, x_1, color='blue')
axis21.plot(t, x_3, color='blue')
axis31.plot(t_conv, conv_13, color='red')

fig, (axis11, axis21, axis31) = plt.subplots(3,1, figsize=(10,15))
axis11.plot(t, x_2, color='blue')
axis21.plot(t, x_3, color='blue')
axis31.plot(t_conv, conv_23, color='red')






