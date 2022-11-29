import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.ticker
import soundfile as sf
import os
from itertools import product

BASEDIR = os.getcwd()

def sweep(T:int, f1:int, f2:int, sample_rate:int, folder:str = "./", save_audio=False):
    
    """
    Generates a logarithmic sweep signal, with the possibility of saving it in wav format.

    Parameters
    ----------
    
    T : int
        Sweep signal duration.
    
    f1 : int
        Sweep start frequency.
    
    f2 : int
        Sweep end frequency.
    
    sample_rate : int
        Sweep sample rate frequency.
    
    folder : str, optional
        Directory where the audio file can be saved, by default "./"
    
    save_audio : bool, optional
        If is True, you can save an audio file at your chosen sample rate and 24-bit, by default False

    Returns
    -------
    
    tuple
        Tuple with data time from 0 to T, and magnitude of sweep.
    
    """
    
    t = np.linspace(0, T, sample_rate*T)
    R = np.log(f2/f1) # rate sweep
    K = (2*np.pi*f1*T)/R
    L = T/R
    
    sweep_signal = np.sin(K*(np.exp(t/L) - 1))
    
    if save_audio == True:
        sf.write(file=os.path.join(BASEDIR, 'audios', f'sweep_{T}_{f1}_{f2}_{sample_rate}_{24}.wav'), data=sweep_signal, samplerate=sample_rate, subtype='PCM_24')
    
    return (t, sweep_signal)


def make_list(v):
    
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
    
    return sos_bank, (bands, SPL_averages)


def inverse_filter(x:tuple, T:int, f1:int, f2:int):
    
    """
    Generates an inverse filter from a sweep signal over time.

    Parameters
    ----------
    
    x : tuple
        Tuple of a sweep signal with information of the time and magnitude.
    
    T : int
        Sweep signal duration.
    
    f1 : int
        Sweep start frequency.
    
    f2 : int
        Sweep end frequency.

    Returns
    -------
    
    tuple
        Tuple with data time from 0 to T, and magnitude of inverse filter.
    
    """
    
    t = x[0]
    R = np.log(f2/f1) # rate sweep
    L = T/R 
    x = x[1][::-1] #inverse the numpy array
    f = x*np.exp(-t/L)
    return (t, f)

    
def list_udim(list1:list):
    
    """
    Transform a matrix in horizontal list.

    Parameters
    ----------
    
    list1 : list
        List to convert.

    Returns
    -------
    
    list
        List converted.
    
    """
    
    list2 = []
    dim = np.array(list1).shape

    if len(dim) > 1:    
        for x in range(len(list1)):
            for y in range(len(list1)):
                list2.append(list1[x][y])
    else:
        list2 = list1
        
    return list2


def frequency_labels(freqs:list, octave:int):
    
    """
    Converts a list of frequency values ​​of type int to another list of frequencies of type str with suffix notation k, to use like a labels.

    Parameters
    ----------
    
    freqs : list
        Frequency values in type int.

    Returns
    -------
    
    list
        Frequency labels in type str.
    
    """
    
    f_labels = []
    
    
    if octave == 3:
        norm_freqs = [25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000]
        
        for f in norm_freqs:
            if f>=1000:
                nomen = f/1000
                f_labels.append(f'{nomen}k')
            else:
                f_labels.append(str(f))
    else:
        for f in freqs:
            if f>=1000:
                nomen = f/1000
                f_labels.append(f'{nomen}k')
            else:
                f_labels.append(str(f))
            
    return f_labels


def grapher(signals:list, dimension:tuple, bars:bool = False, oct_axis:int = 1, semilogx:bool = False, **options):
    
    """
    Allow plot a list of functions in linear axis or semilog x axis.

    Parameters
    ----------
    
    signals : list
        A list of signals coming in as a tuple containing the X values ​​and Y values.
    
    dimension : tuple
        Dimension of the subplots that will make up the plot area in format (rows, columns).
    
    bars : bool, optional
        Show data how bars graphics, by default False.
    
    oct_axis : int
        When bars or semilogx are True, then can set the x axis type. Set 1 for octaves or 3 por thirds octave. By default is 1.
    
    semilogx : bool, optional
        Allow plot in logarithmic scale in X axis, by default False.
        
    **optionals
    -----------
    
    figsize : tuple
        Determinate the size of the plot area, by default (15,8).
    
    fontsize : int
        Fontsize base of label axis, this is scaled to 120%\ to titles, 80%\ to axis ticks labels. By default 12.
    
    title : str or list
        Title of each figure. When its unidimensional plot (1,1) then title must be a str.
    
    legends : list
        List that describes the each plots, this will be localized upper right in the subplot area.
    
    xlabel : str
        Name of x axis, this will be the same in each subplots when plots multiple signals in differents subplots.
    
    ylabel : str
        Name of y axis, this will be the same in each subplots when plots multiple signals in differents subplots.
    
    xlim : list
        Limit of x axis, from a to b value. This will be the same in each subplots when plots multiple signals in differents subplots.
    
    ylim : list
        Limit of y axis, from a to b value. This will be the same in each subplots when plots multiple signals in differents subplots.
    
    """
    
    fig, axs = plt.subplots(*dimension, figsize=options['figsize'] if 'figsize' in options else (15,8))
    fontsize = options['fontsize'] if 'fontsize' in options else 12
    i = 0
    
    
    
    if dimension == (1,1):
        for sg in signals:
            x_values = sg[0]
            y_values = sg[1]
            
            if semilogx:
                # eje de f_m maxi funcion
                y_values_db = 20*np.log10(y_values/np.max(abs(y_values)))   
                axs.semilogx(x_values, y_values_db, label= options['legends'][i] if 'legends' in options else None)
                
            elif bars:

                index_freqs = np.array(list(range(-5*oct_axis-int(oct_axis/3), 5*oct_axis-int(oct_axis/3), 1)))
                axis_freqs = f_m(x=index_freqs, b=oct_axis)

                axs.bar(range(len(x_values)), y_values, color='g')
                axs.set_xticks(range(len(x_values)))
                axs.set_xticklabels(frequency_labels(axis_freqs, oct_axis), rotation=45 if oct_axis==3 else 0)
                # axs.xlabel("Frecuencia [Hz]", fontsize=14)
                # axs.ylabel("ruiduski [dB SPL]", fontsize=14)
                # plt.ylim(20, 95)
                # plt.title('sapatuki', fontsize=18)
                # plt.grid()  

                                
            else:
                axs.plot(x_values, y_values, label= options['legends'][i] if 'legends' in options else None)
                
            i = i + 1
            
        axs.set_title(options['title'] if 'title' in options else None, fontsize=fontsize*1.2)
        axs.set_xlabel(options['xlabel'] if 'xlabel' in options else None, fontsize = fontsize )
        axs.set_ylabel(options['ylabel'] if 'ylabel' in options else None, fontsize = fontsize)
        axs.set_xlim(options['xlim'] if 'xlim' in options else None)
        axs.set_ylim(options['ylim'] if 'ylim' in options else None)
        plt.setp(axs.get_xticklabels(), fontsize=fontsize*0.8)
        plt.setp(axs.get_yticklabels(), fontsize=fontsize*0.8)
        plt.grid()
        if 'legends' in options:
            plt.legend(loc="upper right", fontsize=fontsize)
            
            
        
    else:
        axs = list_udim(axs) #matrix of subplots changes to a list unidimensional
        for ax in axs:
            x_values = signals[i][0]
            y_values = signals[i][1]

            if semilogx:
                index_freqs = np.array(list(range(-5*oct_axis-int(oct_axis/3), 5*oct_axis-int(oct_axis/3), 1)))
                axis_freqs = f_m(x=index_freqs, b=oct_axis)
                y_values_db = 20*np.log10(y_values/np.max(abs(y_values)))  
                 
                ax.semilogx(x_values, y_values_db, label= options['legends'][i] if 'legends' in options else None)
                ax.set_xticks(axis_freqs)
                ax.set_xticklabels(frequency_labels(axis_freqs, oct_axis), rotation=45 if oct_axis==3 else 0)
                ax.get_xaxis().set_tick_params(which='minor', size=0)
                ax.get_xaxis().set_tick_params(which='minor', width=0)
            elif bars:
                index_freqs = np.array(list(range(-5*oct_axis-int(oct_axis/3), 5*oct_axis-int(oct_axis/3), 1)))
                axis_freqs = f_m(x=index_freqs, b=oct_axis)
                ax.bar(range(len(x_values)), y_values, color='g')
                ax.set_xticks(range(len(x_values)))
                ax.set_xticklabels(frequency_labels(axis_freqs, oct_axis), rotation=45 if oct_axis==3 else 0)
                
            else:  
                ax.plot(x_values, y_values, label= options['legends'][i] if 'legends' in options else None)
                    
            ax.set_title(options['title'][i] if 'title' in options else None, fontsize=fontsize*1.2)
            ax.set_xlabel(options['xlabel'] if 'xlabel' in options else None, fontsize = fontsize )
            ax.set_ylabel(options['ylabel'] if 'ylabel' in options else None, fontsize = fontsize)
            ax.set_xlim(options['xlim'] if 'xlim' in options else None)
            ax.set_ylim(options['ylim'] if 'ylim' in options else None)
            plt.setp(ax.get_xticklabels(), fontsize=fontsize*0.8)
            plt.setp(ax.get_yticklabels(), fontsize=fontsize*0.8)
            
            if 'legends' in options:
                ax.legend(loc="upper right", fontsize=fontsize)  
            ax.grid() 
            
            i = i + 1
            
        plt.tight_layout()
        
    plt.show()
    
    
def fft(x:np.ndarray):
    """
    Transform a temporal domain signal on a frequency domain signal by FFT

    Parameters
    ----------
    
    x : np.ndarray
        Magnitude of the signal to transform

    Returns
    -------
    
    np.ndarray
        Magnitude transformed to frequency domain
    
    """
    
    fft_signal = np.fft.fft(x)
    fft_signal = fft_signal[:len(fft_signal)//2]
    magnitude = abs(fft_signal)/len(fft_signal)
    
    return magnitude


def impulse_response(x:np.ndarray, y:np.ndarray, sr:int):
    
    """
    Get the impulse response of a system using fft and ifft instead of convolve in time, h(t) = ifft(fft(x)*fft(y)) . The x and y inputs must have the same dimension.
    
    Parameters
    ----------
    
    x : np.ndarray
        magnitudes array of the signal x 
    
    y : np.ndarray
        magnitudes array of the signal y 
    
    sr : int
        Sample rate

    Returns
    -------
    
    tuple
        Tuple containing time and magnitude information of the impulse response
    
    """
    
    if x.shape == y.shape:
        X = fft(x)
        Y = fft(y)
        h = np.fft.ifft(X*Y) 
        h = h/max(abs(h)) # normalize
        t = np.linspace(0, h.size/sr, h.size)
        return (t, h)
    else:
        ValueError('Dimension of x and y are not same')


def frec_sum(frequencies,sampling,duration):
    
    """
    Performs the sum of the sinusoidal signals with the frequencies entered, 
    with a number of samples entered and with a duration entered. Then graph the result. 

    Parameters
    ----------
    
    frequencies : tuple
        Frequencies of the sinusoidal signals to sum
    
    sampling : int
        Number of samples of the result signal (must be between 10 and 100k)
    
    duration : int
        Seconds of the signal duration, begins at 0 and stops at this parameter value
        
    Returns
    Graph with the resulting sum
    -------
    out_signal : tuple
        Tuple with the time values and generated sum
    
    """
    
    out_signal=0
    time = np.linspace(0,duration,sampling*duration)
    for count,frequency in enumerate(frequencies):
        out_signal=out_signal+np.sin(2*np.pi*time*frequency)
    
    return (time, out_signal)


def load_audio(folder, files):
    
    """
    _summary_

    Parameters
    ----------
    
    folder : _type_
        _description_
    
    files : _type_
        _description_

    Returns
    -------
    
    _type_
        List with audio data, audio and sample rate in each position
    
    """
    
    audios = []
    
    if type(files) is list:
        for file in files:
            audios.append(sf.read(os.path.join(folder, file)))
    elif type(files) is str:
        audios = sf.read(os.path.join(folder, files))
    else:
        print('The file must be a list or str')
        
    return audios