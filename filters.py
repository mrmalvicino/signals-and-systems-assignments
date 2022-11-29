import numpy as np
import os
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal


def autoplot(x_11, y_11L, y_11R='empty', x_21='empty', y_21L='empty', y_21R='empty', **kwargs):
    
    """
    Generates 1x1 or 2x1 plots with one or two vertical axes depending on which inputs are given.

    Parameters
    ----------
    
    x_11 : NUMPY ARRAY
        Top graph x axis data.
    
    y_11L : NUMPY ARRAY
        Top graph left y axis data.
    
    y_11R : NUMPY ARRAY, optional
        Top graph right y axis data. The default is 'empty'.
    
    x_21 : NUMPY ARRAY, optional
        Bottom graph x axis data. The default is 'empty'.
    
    y_21L : NUMPY ARRAY, optional
        Bottom graph left y axis data. The default is 'empty'.
    
    y_21R : NUMPY ARRAY, optional
        Bottom graph right y axis data. The default is 'empty'.
    
    **kwargs : UNPACKED DICTIONARY
        Object orientated kwargs values for matplotlib.pyplot.plot() method.
        
        kwargs = {
            'subplot_size': (10,5),
            'title_11': '',
            'title_21': '',
            'x_11_label': '',
            'y_11L_label': '',
            'y_11R_label': '',
            'x_21_label': '',
            'y_21L_label': '',
            'y_21R_label': '',
            'x_11_scale': 'linear',
            'y_11L_scale': 'linear',
            'y_11R_scale': 'linear',
            'x_21_scale': 'linear',
            'y_21L_scale': 'linear',
            'y_21R_scale': 'linear',
            'y_11L_legend': '',
            'y_11R_legend': '',
            'y_21L_legend': '',
            'y_21R_legend': '',
            'x_ticks': 'default',
            'yL_ticks': 'default',
            'yR_ticks': 'default',
            'x_ticklabels': 'default',
            'yL_ticklabels': 'default',
            'yR_ticklabels': 'default',
            'x_lim': 'default',
            'yL_lim': 'default',
            'yR_lim': 'default'
            }
    
    Returns
    -------
    
    None.

    """
    
    parameters = kwargs
    
    kwargs = {
        'subplot_size': (10,5),
        'title_11': '',
        'title_21': '',
        'x_11_label': '',
        'y_11L_label': '',
        'y_11R_label': '',
        'x_21_label': '',
        'y_21L_label': '',
        'y_21R_label': '',
        'x_11_scale': 'linear',
        'y_11L_scale': 'linear',
        'y_11R_scale': 'linear',
        'x_21_scale': 'linear',
        'y_21L_scale': 'linear',
        'y_21R_scale': 'linear',
        'y_11L_legend': '',
        'y_11R_legend': '',
        'y_21L_legend': '',
        'y_21R_legend': '',
        'x_ticks': 'default',
        'yL_ticks': 'default',
        'yR_ticks': 'default',
        'x_ticklabels': 'default',
        'yL_ticklabels': 'default',
        'yR_ticklabels': 'default',
        'x_lim': 'default',
        'yL_lim': 'default',
        'yR_lim': 'default'
        }
    
    for key, value in parameters.items():
        if key in kwargs:
            kwargs[key] = value
    
    setup_valid_keys = {'x_ticks', 'yL_ticks', 'yR_ticks', 'x_ticklabels', 'yL_ticklabels', 'yR_ticklabels', 'x_lim', 'yL_lim', 'yR_lim'}
    setup = dict()
    setup_L = dict()
    setup_R = dict()
    
    for key, value in kwargs.items():
        if value != 'default' and key in setup_valid_keys:
            setup.update({key: value})


    if x_21 == 'empty':
        
# ------
# CASE 1
# ------
        
        if y_11R == 'empty':
            
            fig, (axisTL) = plt.subplots(1, 1, figsize=kwargs['subplot_size'])
            
            axisTL.plot(x_11, y_11L, color='blue')
            axisTL.set_xlabel(kwargs['x_11_label'])
            axisTL.set_ylabel(kwargs['y_11L_label'])
            axisTL.set_xscale(kwargs['x_11_scale'])
            axisTL.set_yscale(kwargs['y_11L_scale'])
            axisTL.grid()
            axisTL.set_title(kwargs['title_11'])
            
            setup_L_valid_keys = {'x_ticks': 'xticks', 'yL_ticks': 'yticks', 'x_ticklabels': 'xticklabels', 'yL_ticklabels': 'yticklabels', 'x_lim': 'xlim', 'yL_lim': 'ylim'}
            
            for key, value in setup.items():
                if key in setup_L_valid_keys:
                    setup_L.update({setup_L_valid_keys[key]: value})
            
            plt.setp(axisTL, **setup_L)
            
            plt.tight_layout()
            # graph = plt.gcf()
            # plt.show()
            
# ------    
# CASE 2
# ------
        
        else:
            
            fig, (axisTL) = plt.subplots(1,1, figsize=kwargs['subplot_size'])
            axisTR = axisTL.twinx()
            
            axisTL.plot(x_11, y_11L, color='blue')
            axisTR.plot(x_11, y_11R, color='red', linestyle='--')
            
            axisTL.set_xlabel(kwargs['x_11_label'])
            axisTL.set_ylabel(kwargs['y_11L_label'])
            axisTR.set_ylabel(kwargs['y_11R_label'])
            
            axisTL.set_xscale(kwargs['x_11_scale'])
            axisTL.set_yscale(kwargs['y_11L_scale'])
            axisTR.set_yscale(kwargs['y_11R_scale'])
            
            axisTL.grid()
            
            axisTL.set_title(kwargs['title_11'])
            
            setup_L_valid_keys = {'x_ticks': 'xticks', 'yL_ticks': 'yticks', 'x_ticklabels': 'xticklabels', 'yL_ticklabels': 'yticklabels', 'x_lim': 'xlim', 'yL_lim': 'ylim'}
            setup_R_valid_keys = {'yR_ticks': 'yticks', 'yR_ticklabels': 'yticklabels', 'yR_lim': 'ylim'}
            
            for key, value in setup.items():
                if key in setup_L_valid_keys:
                    setup_L.update({setup_L_valid_keys[key]: value})
            
            for key, value in setup.items():
                if key in setup_R_valid_keys:
                    setup_R.update({setup_R_valid_keys[key]: value})
            
            plt.setp(axisTL, **setup_L)
            plt.setp(axisTR, **setup_R)
            
            axisTL.legend([kwargs['y_11L_legend']], loc='lower left')
            axisTR.legend([kwargs['y_11R_legend']], loc='lower right')
            
            plt.tight_layout()
            # graph = plt.gcf()
            # plt.show()
    
    else:
        
# ------
# CASE 3
# ------
        
        if y_11R == 'empty' and y_21R == 'empty':
            
            fig, (axisTL, axisBL) = plt.subplots(2,1, figsize=(kwargs['subplot_size'][0], 2*kwargs['subplot_size'][1]), sharex=False)
            
            axisTL.plot(x_11, y_11L, color='blue')
            axisBL.plot(x_21, y_21L, color='blue')
            
            axisTL.set_xlabel(kwargs['x_11_label'])
            axisTL.set_ylabel(kwargs['y_11L_label'])
            axisBL.set_xlabel(kwargs['x_21_label'])
            axisBL.set_ylabel(kwargs['y_21L_label'])
            
            axisTL.set_xscale(kwargs['x_11_scale'])
            axisTL.set_yscale(kwargs['y_11L_scale'])
            axisBL.set_xscale(kwargs['x_21_scale'])
            axisBL.set_yscale(kwargs['y_21L_scale'])
            
            axisTL.grid()
            axisBL.grid()
            
            axisTL.set_title(kwargs['title_11'])
            axisBL.set_title(kwargs['title_21'])
            
            axesL = (axisTL, axisBL)
            
            setup_L_valid_keys = {'x_ticks': 'xticks', 'yL_ticks': 'yticks', 'x_ticklabels': 'xticklabels', 'yL_ticklabels': 'yticklabels', 'x_lim': 'xlim', 'yL_lim': 'ylim'}
            
            for key, value in setup.items():
                if key in setup_L_valid_keys:
                    setup_L.update({setup_L_valid_keys[key]: value})
            
            plt.setp(axesL, **setup_L)
            
            axisTL.legend([kwargs['y_11L_legend']], loc='lower left')
            axisBL.legend([kwargs['y_21L_legend']], loc='upper left')
            
            plt.tight_layout()
            # graph = plt.gcf()
            # plt.show()
            
# ------
# CASE 4
# ------
        
        elif y_11R != 'empty' and y_21R != 'empty':
            
            fig, (axisTL, axisBL) = plt.subplots(2,1, figsize=(kwargs['subplot_size'][0], 2*kwargs['subplot_size'][1]), sharex=False)
            axisTR = axisTL.twinx()
            axisBR = axisBL.twinx()
            
            axisTL.plot(x_11, y_11L, color='blue')
            axisTR.plot(x_11, y_11R, color='red', linestyle='--')
            axisBL.plot(x_21, y_21L, color='blue')
            axisBR.plot(x_21, y_21R, color='red', linestyle='--')
            
            axisTL.set_xlabel(kwargs['x_11_label'])
            axisTL.set_ylabel(kwargs['y_11L_label'])
            axisTR.set_ylabel(kwargs['y_11R_label'])
            axisBL.set_xlabel(kwargs['x_21_label'])
            axisBL.set_ylabel(kwargs['y_21L_label'])
            axisBR.set_ylabel(kwargs['y_21R_label'])
            
            axisTL.set_xscale(kwargs['x_11_scale'])
            axisTL.set_yscale(kwargs['y_11L_scale'])
            axisTR.set_yscale(kwargs['y_11R_scale'])
            axisBL.set_xscale(kwargs['x_21_scale'])
            axisBL.set_yscale(kwargs['y_21L_scale'])
            axisBR.set_yscale(kwargs['y_21R_scale'])
            
            axisTL.grid()
            axisBL.grid()
            
            axisTL.set_title(kwargs['title_11'])
            axisBL.set_title(kwargs['title_21'])
            
            axesL = (axisTL, axisBL)
            axesR = (axisTR, axisBR)
            
            setup_L_valid_keys = {'x_ticks': 'xticks', 'yL_ticks': 'yticks', 'x_ticklabels': 'xticklabels', 'yL_ticklabels': 'yticklabels', 'x_lim': 'xlim', 'yL_lim': 'ylim'}
            setup_R_valid_keys = {'yR_ticks': 'yticks', 'yR_ticklabels': 'yticklabels', 'yR_lim': 'ylim'}
            
            for key, value in setup.items():
                if key in setup_L_valid_keys:
                    setup_L.update({setup_L_valid_keys[key]: value})
            
            for key, value in setup.items():
                if key in setup_R_valid_keys:
                    setup_R.update({setup_R_valid_keys[key]: value})
            
            plt.setp(axesL, **setup_L)
            plt.setp(axesR, **setup_R)
            
            axisTL.legend([kwargs['y_11L_legend']], loc='lower left')
            axisTR.legend([kwargs['y_11R_legend']], loc='lower right')
            axisBL.legend([kwargs['y_21L_legend']], loc='upper left')
            axisBR.legend([kwargs['y_21R_legend']], loc='upper right')
            
            plt.tight_layout()
            # graph = plt.gcf()
            # plt.show()
    
    return


def make_list(v): # Branched on 6/11/22

    """
    Attempts to convert a given variable into a list.

    Parameters
    ----------
    
    v : ANY TYPE

    Returns
    -------
    
    lst : LIST

    """
    
    if type(v) == list:
        lst = v
    elif type(v) == np.ndarray:
        lst = v.tolist()
    else:
        lst = list(v)
    return lst


def root_dir(param, open_root_dir=False): # Forked on 6/11/22
    
    """
    Traces a folder relative to where the script is being executed.
    Defines this folder as "root directory" and returns it absolute path.
    
    Parameters
    ----------
    
    param : STRING OR INTEGER
        Name of the folder (str) to trace or level of hierarchy (int) to define as root directory.
    
    open_root_dir : BOOLEAN, optional
        Determines whether the root directory will be opened after being defined. The default is False.
    
    Raises
    ------
    
    ValueError
        Invalid input.

    Returns
    -------
    
    root_dir : STRING
        Path of the folder defined as root directory.
    """
    
    if type(param) == int:
        root_dir = os.path.dirname(__file__)
        
        for i in range(1, param + 1, 1):
            root_dir = os.path.realpath(os.path.join(root_dir, '..'))
    
    elif type(param) == str:
        root_dir = ''
        
        for i in __file__:
            if param not in root_dir:
                root_dir = root_dir + i
    
    else:
        raise ValueError(f'{type(param)} is not a valid input.')
    
    if open_root_dir == True:
        os.startfile(root_dir)
    
    return root_dir


def round_array(v, sig_digits=3):
    
    """
    Attempts to round a given float or the floats of a given array to a certain number of significant figures. Contemplates that the decimal (. ,) and negative (-) symbols are not digits.

    Parameters
    ----------
    
    v : FLOAT, NUMPY ARRAY OF FLOATS
        Input that is going to be rounded.
    
    sig_digits : TYPE, optional
        Significant figures or digits. The default is 3.

    Returns
    -------
    
    w : FLOAT, NUMPY ARRAY OF FLOATS
        Rounded output.

    """
    
    if type(v) == np.ndarray:
        w = np.array([])
        for v_i in v:
            if '-' not in str(v_i):
                int_digits = len(str(int(v_i)))
            else:
                int_digits = len(str(int(v_i))) - 1
            v_i = v_i/10**int_digits
            v_i = round(v_i, sig_digits)
            v_i = v_i * (10**int_digits)
            if type(v_i) == float and str(v_i)[-2:] == '.0':
                v_i = int(v_i)
            w = np.append(w, v_i)
    elif type(v) == float:
        if '-' not in str(v):
            int_digits = len(str(int(v)))
        else:
            int_digits = len(str(int(v))) - 1
        dec_digits = len(str(v)) - int_digits - 1
        v = v/10**int_digits
        v = round(v, sig_digits)
        v = v * (10**int_digits)
        if type(v) == float and str(v)[-2:] == '.0':
            w = int(v)
        if type(v) == float and str(v)[-dec_digits+1:] == '9'*(dec_digits-1):
            w = round(v, 1)
        w = v
    
    return w


def f_m(x, b=1, G=2, f_r=1000):
    
    """
    Calculates the central frequency for a given integer index or array of integer indexes. Definitions, notation and formulas are set according to UNE-EN 61260.

    Parameters
    ----------
    
    x : INTEGER, NUMPY ARRAY OF INTEGERS
        Determines which band's central frequencies are going to be calculated. The index x=0 generates the reference frequency, 1k Hz.
    
    b : INTEGER, optional
        Positive integer that determines the bandwith. The default is 1.
    
    G : FLOAT, optional
        Octave ratio. Usual values are 2 or 10^(3/10). The default is 2.
    
    f_r : FLOAT, optional
        Reference frequency. The default is 1000.

    Returns
    -------
    
    f_m : INTEGER, NUMPY ARRAY OF INTEGERS
        Central frequency (or frequencies) of the band (or bands).

    """
    
    if b%2 == 0:
        f_m = round_array(f_r * G**(x/b + 1/(2*b)), sig_digits=5)
    else:
        f_m = round_array(f_r * G**(x/b), sig_digits=5)
    
    return f_m


def f_c(c, f_m, b=1, G=2):
    
    """
    Calculates the cutoff frequencies for a given central frequency. Definitions, notation and formulas are set according to UNE-EN 61260.

    Parameters
    ----------
    
    c : INTEGER
        Determines whether to calculate the lowcut or highcut frequency. The only 2 input values are c=1 that stands for lowcut and c=2 that stands for highcut.
    
    f_m : INTEGER, NUMPY ARRAY OF INTEGERS
        Central frequency (or frequencies) of the band (or bands).
    
    b : INTEGER, optional
        Positive integer that determines the bandwith. The default is 1.
    
    G : FLOAT, optional
        Octave ratio. Usual values are 2 or 10^(3/10). The default is 2.

    Returns
    -------
    
    f_c : INTEGER, NUMPY ARRAY OF INTEGERS
        Cutoff frequency of the band (or bands).

    """
    
    f_c = f_m * (G**(((-1)**c)/(2*b)))
    
    return f_c


def bandpass_filter(x, f_s, b=1, G=2, N_order=3, worN=5120):
    
    """
    Generates a bandpass filter using signal.butter() according to UNE-EN 61260 norm.

    Parameters
    ----------
    
    x : INTEGER
        Determines which band's central frequencies are going to be calculated. The index x=0 generates the reference frequency, 1k Hz.
        
    f_s : INTEGER
        Sampling rate, which determines the Nyquist frequency.
    
    b : INTEGER, optional
        Positive integer that determines the bandwith. The default is 1.
    
    G : FLOAT, optional
        Octave ratio. Usual values are 2 or 10^(3/10). The default is 2.
    
    N_order : INTEGER, optional
        The order of the filter. Parameter of signal.butter(). The default is 3.
    
    worN : INTEGER, optional
        Number of frequencies computed by signal.sosfreqz(). The default is 5120.
    
    Returns
    -------
    
    sos : ARRAY
        Array of second-order filter coefficients.
    
    f_k : ARRAY
        Array of frequency bins.
    
    H_mag : ARRAY
        Array of magnitudes or absolute values of the filter.
    
    H_phase : ARRAY
        Array of phases of the filter.
    
    """
    
    f_nyq = f_s / 2
    f_1 = f_c(1, f_m(x, b, G), b, G)
    f_2 = f_c(2, f_m(x, b, G), b, G)
    f_1_norm = f_1 / f_nyq
    f_2_norm = f_2 / f_nyq
    
    sos = signal.butter(N_order, [f_1_norm, f_2_norm], btype='band', output='sos')
    omega_k, H = signal.sosfreqz(sos, worN)
    f_k = (omega_k/(2*np.pi))*f_s
    H_mag = 20*np.log10(abs(H))
    H_phase = np.arctan2(np.imag(H),np.real(H))
    
    return sos, f_k, H_mag, H_phase


def fft(x, f_s):
    
    """
    Fast Fourier transform of a given time signal.

    Parameters
    ----------
    
    x : NUMPY ARRAY
        Function of time.
    
    f_s : INTEGER
        Sampling rate, which determines the Nyquist frequency..

    Returns
    -------
    
    X_frequencies : NUMPY ARRAY
        Array of frequencies which the function of time is composed of.
    
    X_magnitude : NUMPY ARRAY
        Magnitudes of the frequencies.
    
    X_phase : NUMPY ARRAY
        Phase of the frequencies.

    """
    
    fft_raw = np.fft.fft(x)
    fft = fft_raw[:len(fft_raw)//2]
    X_magnitude = abs(fft)/len(fft)
    X_phase = np.arctan2(np.imag(fft),np.real(fft))
    X_frequencies = np.linspace(0,f_s/2, len(fft))
    
    return X_frequencies, X_magnitude, X_phase


def mean(v, k=2):
    
    """
    Calculates the generalized mean of a given vector.

    Parameters
    ----------
    
    v : NUMPY ARRAY
        Vector which values are going to be evaluated.
    
    k : INTEGER, optional
        Positive integer that defines the root, being k=2 the RMS value. The default is 2.

    Returns
    -------
    
    v_mean : FLOAT
        Mean value.

    """
    
    N = len(v)
    v_sum = 0
    
    for v_i in v:
        v_sum = v_sum + v_i**k
    
    v_mean = (v_sum/N)**(1/k)
    
    return v_mean


def SPL(v, p_ref=0.00002):
    
    """
    Calculates the sound pressure level for a given reference value by definition.
    
    Parameters
    ----------
    
    v : FLOAT, NUMPY ARRAY
        Pressure [Pa].
    
    p_ref : FLOAT, optional
        Reference value. The default is 20u Pa.
    
    Returns
    -------
    
    SPL : FLOAT, NUMPY ARRAY
        Sound pressure level.
    
    """
    
    SPL = 10*np.log10((v/p_ref)**2)
    
    return SPL


def SPL_ave(SPL):
    
    """
    Calculates the sound pressure level average by definition.
    
    Parameters
    ----------
    
    SPL : NUMPY ARRAY
        Array of sound pressure levels to evaluate.
    
    Returns
    -------
    
    SPL_ave : FLOAT
        Average sound pressure level.
    
    """
    
    N = len(SPL)
    SPL_sum = 0
    
    for SPL_i in SPL:
        SPL_sum = SPL_sum + 10**(SPL_i/20)
    
    SPL_ave = 20*np.log10(SPL_sum/N)
    
    return SPL_ave


def filter_bank(audio_input, f_s, p_ref=0.00002, b=1, G=2, N_order=3):
    
    """
    Generates a filter bank and applies it to a given input to obtain the frequency spectrum averaged by fractions of octaves.
    
    Parameters
    ----------
    
    audio_input : ARRAY
        Data to be filtered.
    
    f_s : INTEGER
        Sampling rate, which determines the Nyquist frequency.
    
    b : INTEGER, optional
        Positive integer that determines the bandwith. The default is 1.
    
    G : FLOAT, optional
        Octave ratio. Usual values are 2 or 10^(3/10). The default is 2.
    
    N_order : INTEGER, optional
        The order of the filter. Parameter of signal.butter(). The default is 3.
    
    Returns
    -------
    
    sos_bank : LIST OF ARRAYS
        List of arrays of second-order filter coefficients for each band.
    
    bands : ARRAY OF FLOATS
        Array of bands' central frequencies.
    
    SPL_averages : ARRAY OF FLOATS
        List of bands' sound pressure level average.
    
    """
    
    x_vec = range(-5*b-int(b/3), 5*b-int(b/3), 1) # Frequencies index array
    sos_bank = []
    SPL_averages = np.array([])
    bands = f_m(np.array(x_vec), b)

    for x in x_vec:
        sos_x = bandpass_filter(x, f_s, b, G, N_order)[0]
        sos_bank.append(sos_x)
    
    for sos_x in sos_bank:
        filtered_signal_x = signal.sosfilt(sos_x, audio_input)
        SPL_x = SPL(filtered_signal_x, p_ref)
        SPL_averages = np.concatenate(( SPL_averages, np.array([SPL_ave(SPL_x)]) ))
    
    return sos_bank, bands, SPL_averages


# PLOT KWARGS

octaves_ticks = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
octaves_ticklabels = ['31.5', '63', '125', '250', '500', '1k', '2k', '4k', '8k', '16k']
dB_ticks = [-6, -5, -4, -3, -2, -1, 0, 1]
phase_ticks = make_list(round_array(np.arange(-180*(2*np.pi)/360, 180*(2*np.pi)/360 + 45*(2*np.pi)/360, 45*(2*np.pi)/360)))

freq_kwargs = {
    'subplot_size': (10,5),
    'title_11': 'Frequency spectrum',
    'title_21': '',
    'x_11_label': 'Frequency [Hz]',
    'y_11L_label': 'Magnitude [dB]',
    'y_11R_label': 'Phase [rad]',
    'x_21_label': 'Frequency [Hz]',
    'y_21L_label': '',
    'y_21R_label': '',
    'x_11_scale': 'log',
    'y_11L_scale': 'linear',
    'y_11R_scale': 'linear',
    'x_21_scale': 'log',
    'y_21L_scale': 'linear',
    'y_21R_scale': 'linear',
    'y_11L_legend': 'Magnitude',
    'y_11R_legend': 'Phase',
    'y_21L_legend': '',
    'y_21R_legend': '',
    'x_ticks': octaves_ticks,
    'yL_ticks': dB_ticks,
    'yR_ticks': phase_ticks,
    'x_ticklabels': octaves_ticklabels,
    'yL_ticklabels': 'default',
    'yR_ticklabels': 'default',
    'x_lim': (20,20000),
    'yL_lim': (-6, 1),
    'yR_lim': 'default'
    }

time_kwargs = {
    'subplot_size': (10,5),
    'title_11': 'Sound Pressure Level',
    'title_21': '',
    'x_11_label': 'Time [s]',
    'y_11L_label': 'Magnitude [dB SPL]',
    'y_11R_label': '',
    'x_21_label': 'Time [s]',
    'y_21L_label': '',
    'y_21R_label': '',
    'x_11_scale': 'linear',
    'y_11L_scale': 'linear',
    'y_11R_scale': 'linear',
    'x_21_scale': 'linear',
    'y_21L_scale': 'linear',
    'y_21R_scale': 'linear',
    'y_11L_legend': '',
    'y_11R_legend': '',
    'y_21L_legend': '',
    'y_21R_legend': '',
    'x_ticks': 'default',
    'yL_ticks': np.arange(0, 160+10, 10),
    'yR_ticks': 'default',
    'x_ticklabels': 'default',
    'yL_ticklabels': 'default',
    'yR_ticklabels': 'default',
    'x_lim': (5,5.05),
    'yL_lim': (20,120),
    'yR_lim': 'default'
    }


if __name__ == "__main__":
    # CALIPER RECORDING
    audio_cal, sr_cal = sf.read(os.path.join(root_dir(0), 'audios', 'processed', 'calibrador_L.wav'))
    p_ref = (20*10**(-6))*mean(audio_cal)
    
    # PINK NOISE RECORDING PLOT
    audio_pinknoise, sr_pinknoise = sf.read(os.path.join(root_dir(0), 'audios', 'processed', '1_noise_L.wav'))
    autoplot(np.linspace(0, len(audio_pinknoise)/sr_pinknoise, len(audio_pinknoise)), SPL(audio_pinknoise, p_ref), **time_kwargs)
    
    # BACKGROUND NOISE RECORDING PLOT
    audio_background, sr_background = sf.read(os.path.join(root_dir(0), 'audios', 'processed', '2_background_L_sape.wav'))
    autoplot(np.linspace(0, len(audio_background)/sr_background, len(audio_background)), SPL(audio_background, p_ref), **time_kwargs)
    
    # FILTER SPECTRUM GRAPH
    f_s = 44100
    x = 0
    b = 1
    f_k = bandpass_filter(x, f_s, b)[1]
    H_mag = bandpass_filter(x, f_s, b)[2]
    H_phase = bandpass_filter(x, f_s, b)[3]
    autoplot(x_11=f_k, y_11L=H_mag, y_11R=H_phase, **freq_kwargs)
    
    # FILTER BANK PINK NOISE OCTAVES
    audio = audio_pinknoise
    sr = sr_pinknoise
    b = 1
    (array_x,array_y,array_z) = filter_bank(audio, sr, p_ref, b)
    autoplot(array_y, array_z, x_11_scale='log', x_ticks=octaves_ticks, x_ticklabels=octaves_ticklabels)
    
    # FILTER BANK PINK NOISE THIRDS
    audio = audio_pinknoise
    sr = sr_pinknoise
    b = 3
    (array_x,array_y,array_z) = filter_bank(audio, sr, p_ref, b)
    autoplot(array_y, array_z, x_11_scale='log', x_ticks=octaves_ticks, x_ticklabels=octaves_ticklabels)
    
    # FILTER BANK BACKGROUND NOISE OCTAVES
    audio = audio_background
    sr = sr_background
    b = 1
    (array_x,array_y,array_z) = filter_bank(audio, sr, p_ref, b)
    autoplot(array_y, array_z, x_11_scale='log', x_ticks=octaves_ticks, x_ticklabels=octaves_ticklabels)
    
    # FILTER BANK BACKGROUND NOISE THIRDS
    audio = audio_background
    sr = sr_background
    b = 3
    (array_x,array_y,array_z) = filter_bank(audio, sr, p_ref, b)
    autoplot(array_y, array_z, x_11_scale='log', x_ticks=octaves_ticks, x_ticklabels=octaves_ticklabels)




