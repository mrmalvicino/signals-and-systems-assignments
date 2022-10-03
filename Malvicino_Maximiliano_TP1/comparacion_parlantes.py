import numpy as np
from matplotlib import pyplot as plt
import functions

# Load signals

signalA = np.load('files\\loudspeaker_Genelec.npy')
signalB = np.load('files\\loudspeaker_JBL.npy')

freq_A = signalA[0,:]
x_A = signalA[1,:]
phi_A = signalA[2,:]

freq_B = signalB[0,:]
x_B = signalB[1,:]
phi_B = signalB[2,:]


'''
------------------------------------------------------------
GRAPH 1: Amplitude A vs. Phase A // Amplitude B vs. Phase B
------------------------------------------------------------
'''


# Plot axis definitions

fig, (axisTL, axisBL) = plt.subplots(2,1, figsize=(10,10), sharex=False)

axisTR = axisTL.twinx()
axisBR = axisBL.twinx()

axisTL.plot(freq_A, x_A, color='blue')
axisTR.plot(freq_A, phi_A, color='red')
axisBL.plot(freq_B, x_B, color='blue')
axisBR.plot(freq_B, phi_B, color='red')


# Plot configurations

axisBL.set_xlabel('Frequency [Hz]')
axisBL.set_ylabel('Amplitude [dB]')    
axisBR.set_ylabel('Phase [Deg]')
axisTL.set_xlabel('Frequency [Hz]')
axisTL.set_ylabel('Amplitude [dB]')    
axisTR.set_ylabel('Phase [Deg]')

axisTL.set_xscale("log")
axisBL.set_xscale("log")

axisTL.grid()
axisBL.grid()

axisTL.set_title('Genelec')
axisBL.set_title('JBL')

octaves_float = functions.gen_ticks(preset='octaves')[0]
octave_str = functions.gen_ticks(preset='octaves')[1]
SPL = functions.make_list(np.arange(68, 92, 2))
axes = (axisTL, axisBL)
plt.setp(axes, xticks=octaves_float , xticklabels=octave_str , yticks=SPL, yticklabels=SPL)

axisTL.legend(['Amplitude'], loc='lower left')
axisTR.legend(['Phase'], loc='lower right')
axisBL.legend(['Amplitude'], loc='upper left')
axisBR.legend(['Phase'], loc='upper right')

plt.tight_layout()


# Plot saving

save_plot = 'y'

while save_plot != 'y' and save_plot != 'n':
    save_plot = input('Do you want to save the plot? [y/n] ')

if save_plot == 'y':
    savefig_kwargs = {'bbox_inches': 'tight', 'dpi': 300, 'transparent': False}
    graph = plt.gcf()
    graph.savefig('images\\comparacion_parlantes.png', **savefig_kwargs)


'''
------------------------------------------------------------
GRAPH 2: Amplitude A vs. Amplitude B // Phase A vs. Phase B
------------------------------------------------------------
'''

'''
# Plot axis definitions

fig, (axisTL, axisBL) = plt.subplots(2,1, figsize=(10,10), sharex=False)

axisTR = axisTL.twinx()
axisBR = axisBL.twinx()

axisTL.plot(freq_A, x_A, color='blue')
axisTR.plot(freq_A, x_B, color='red', linestyle=':')
axisBL.plot(freq_B, phi_A, color='blue')
axisBR.plot(freq_B, phi_B, color='red', linestyle=':')


# Plot configurations

axisBL.set_xlabel('Frequency [Hz]')
axisBL.set_ylabel('Amplitude [dB]')    
axisBR.set_ylabel('Phase [Deg]')
axisTL.set_xlabel('Frequency [Hz]')
axisTL.set_ylabel('Amplitude [dB]')    
axisTR.set_ylabel('Phase [Deg]')

axisTL.set_xscale("log")
axisBL.set_xscale("log")

axisTL.grid()
axisBL.grid()

axisTL.set_title('Amplitude')
axisBL.set_title('Phase')

octaves_float = functions.gen_ticks(preset='octaves')[0]
octave_str = functions.gen_ticks(preset='octaves')[1]
SPL = functions.make_list(np.arange(68, 92, 2))
axes = (axisTL, axisBL)
plt.setp(axes, xticks=octaves_float , xticklabels=octave_str , yticks=SPL, yticklabels=SPL)

axisTL.legend(['Amplitude'], loc='lower left')
axisTR.legend(['Phase'], loc='lower right')
axisBL.legend(['Amplitude'], loc='upper left')
axisBR.legend(['Phase'], loc='upper right')

plt.tight_layout()


# Plot saving

save_plot = 'y'

while save_plot != 'y' and save_plot != 'n':
    save_plot = input('Do you want to save the plot? [y/n] ')

if save_plot == 'y':
    savefig_kwargs = {'bbox_inches': 'tight', 'dpi': 300, 'transparent': False}
    graph = plt.gcf()
    graph.savefig('images\\comparacion_parlantes.png', **savefig_kwargs)
'''