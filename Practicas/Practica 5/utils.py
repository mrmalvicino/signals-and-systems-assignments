import numpy as np 
import matplotlib.pyplot as plt

def plot_sin(t, signal, xlim, ylim, title = False):
    plt.plot(t, signal)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.xlim(0, xlim)
    plt.ylim(-ylim, ylim)
    if title:
        plt.title(title)
    plt.show()