import numpy as np
import os
from matplotlib import pyplot as plt


def root_dir(param, open_root_dir=False):
    
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


def save(param, **kwargs):
    
    """
    Saves a given numpy array or matplotlib plot.
    
    Parameters
    ----------
    
    param : FIGURE OR ARRAY
        Object that is going to be saved.
    
    **kwargs : UNPACKED DICTIONARY
    
        **save_kwargs : UNPACKED DICTIONARY
            Kwargs for internal use.
    
            file_dir : STRING
                Path of the directory where the file is going to be saved.
    
            file_name : STRING
                Name of the file which is going to be saved.
    
            ask_for_confirmation : BOOLEAN
                Determines whether the script should ask for user input confirmation.
    
        **savefig_kwargs : UNPACKED DICTIONARY
            Kwargs for the savefig() method.
    
            bbox_inches : STRING
    
            dpi : INTEGER
    
            transparent : BOOLEAN
    
    Raises
    ------
    
    ValueError
        Invalid input.
    
    Returns
    -------
    
    None.
    """
    
    save_kwargs = {'file_dir': root_dir(0), 'file_name': 'saved_by_' + os.getlogin(), 'ask_for_confirmation': False}
    
    for key, value in kwargs.items():
        if key in save_kwargs and value != save_kwargs[key]:
            save_kwargs[key] = value
    
    if save_kwargs['ask_for_confirmation'] == True:
        save = 'ask'
    else:
        save = 'y'

    while save != 'y' and save != 'n':
        save = input('Do you really want to save? [y/n] ')
    
    if save == 'y':
        if type(param) == plt.Figure:
            savefig_kwargs = {'bbox_inches': 'tight', 'dpi': 300, 'transparent': False}
            
            for key, value in kwargs.items():
                if key in savefig_kwargs and value != savefig_kwargs[key]:
                    savefig_kwargs[key] = value
            
            param.savefig(os.path.join(save_kwargs['file_dir'], save_kwargs['file_name'] + '.png'), **savefig_kwargs)
        
        elif type(param) == np.ndarray:
            np.save(os.path.join(save_kwargs['file_dir'], save_kwargs['file_name']), param)
        
        else:
            raise ValueError(f'{type(param)} input not supported.')
    
    return


def info(v):
    
    """
    Gives descriptive information about a given variable.

    Parameters
    ----------
    
    v : (Any type)
        Variable of which information is being asked for.

    Returns
    -------
    
    None.

    """
    
    if type(v) == int:
        print(f'{v} is an integer.')
    
    elif type(v) == float:
        print(f'{v} is a float.')
    
    elif type(v) == str:
        print(f'"{v}" is a string.')
    
    elif type(v) == list:
        print(f'The input is a list which contains {len(v)} elements.')
    
    elif type(v) == tuple:
        print(f'The input is a tuple of {len(v)} components.')
    
    elif type(v) == dict:
        print(f'The input is a dictionary of {len(v)} elements:')
        print(v.items())
    
    elif type(v) == np.ndarray:
        N = str(v.shape[0])
        
        if len(v.shape) == 1:
            N = 'one dimention'
        else:
            for i in range(1,len(v.shape),1):
                N = N + 'x' + str(v.shape[i])
        
        print(f'The input is a {N} array of {v.size} elements.')
    
    else:
        print(f'There is no information available for {type(v)}.')
    
    return


def gen_discrete_signals(signal_name, n_start=-10, n_end=10, n_0=10, on=5, off=15, m=5, mu=0, sigma=1, isClosedInterval = True, **kwargs):
    
    """
    Generates a custom discrete signal and optionally saves the plot and arrays involved.
    
    Parameters
    ----------
    
    signal_name : STR
        Name of the signal which is going to be generated. E.g., unitImpulse, unitStep, sqPulse, triangPulse, rnd.
    
    n_start : INT, optional
        Starting sample. The default is -10.
    
    n_end : INT, optional
        Ending sample. The default is 10.
    
    n_0 : INT, optional
        Sample at which the unitImpulse is and at which the unitStep begins. The default is 10.
    
    on : INT, optional
        Sample at which the sqPulse goes from 0 to 1. The default is 5.
    
    off : INT, optional
        Sample at which the sqPulse goes from 1 to 0. The default is 15.
    
    m : INT, optional
        Half of the triangPulse base lenght. The default is 5.
    
    mu : INT, optional
        Mean or expected value. The default is 0.
    
    sigma : FLOAT, optional
        Standard deviation. The default is 1.
    
    isClosedInterval : BOOL, optional
        Determines whether the bound of the samples interval belongs to it or not. The default is True.
    
    **kwargs : UNPACKED DICT
        Optional saving parameters.
    
        save_plot : BOOL
            Determines whether the plot is saved to a .png file.
            
        save_array : BOOL
            Determines whether the sample and amplitude arrays are saved to .npy files.

    Raises
    ------
    
    ValueError
        The parameter n_start must be less than n_end by definition.
    
    ValueError
        The parameter n_0 does not belong to the interval [n_start,n_end].
    
    ValueError
        Duty Cycle can not be greater than 100%.
    
    ValueError
        The parameter m must be greater than 0 and less than (n_end-n_start)/2.
    
    ValueError
        Invalid input.

    Returns
    -------
    
    None.

    """
    
    
    # Defines samples interval
    
    if not n_start < n_end:
        raise ValueError('The parameter n_start must be less than n_end by definition.')
    
    n = np.arange(n_start, n_end + int(isClosedInterval), 1)
    
    
    # Defines signal waveform
    
    if signal_name == 'unitImpulse':
        
        if n_0 > len(n):
            raise ValueError('The parameter n_0 does not belong to the interval [n_start,n_end].')
        
        x = np.zeros(len(n))
        x[n_0] = 1    
    
    elif signal_name == 'unitStep':
        
        if n_0 > len(n):
            raise ValueError('The parameter n_0 does not belong to the interval [n_start,n_end].')
        
        x = np.concatenate((np.zeros(n_0), np.ones(len(n)-n_0)), axis=0)
    
    elif signal_name == 'sqPulse':
        
        dutyCycle = (off-on)*100/len(n)
        
        if dutyCycle > 100:
            raise ValueError('Duty Cycle can not be greater than 100%.')
        
        x = np.concatenate((np.zeros(on), np.ones(off-on), np.zeros(len(n)-off)), axis=0)
    
    elif signal_name == 'triangPulse':
        if m <= 0 or m > len(n)/2:
            raise ValueError('The parameter m must be greater than 0 and less than (n_end-n_start)/2.')
        
        x = np.zeros(len(n))
        
        for i in range(-m, m, 1):
            n_i = int((n_end-n_start)/2 + i)
            x[n_i] = 1-abs(i*(1/m))
    
    elif signal_name == 'rnd':
        x = np.random.normal(mu, sigma, len(n))
    
    else:
        raise ValueError('Invalid input.')
    
    
    # Plot
    
    plot_kwargs = {'alpha': 1, 'color': 'black', 'linestyle': '', 'linewidth': 1, 'marker': 'o'}
    
    plt.figure(figsize=(8,4))
    plt.plot(n,x, **plot_kwargs)
    plt.grid()
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    
    if len(n) < 21:
        plt.xticks(n)
    else:
        plt.xticks(gen_ticks(n, N=21)[0])
    
    
    # Kwargs
    
    for key, value in kwargs.items():
        
        if key == 'save_plot' and value == True:
            savefig_kwargs = {'bbox_inches': 'tight', 'dpi': 300, 'transparent': False}
            plt.savefig('images\\' + signal_name + 'Plot.png', **savefig_kwargs)
        
        if key == 'save_array' and value == True:
            np.save('files\\' + signal_name + 'Array_x', x)
            np.save('files\\' + signal_name + 'Array_n', n)
    
    return


def gen_n_sin(*frequencies, A=1, f_s = 44100, isClosedInterval = True):
    
    """
    Generates a sine wave for each input frequency.

    Parameters
    ----------
    
    *frequencies : UNPACKED TUPLE OF FLOATS
        Unpacked tuple containing the frequency values of the sine waves which are going to be generated.
    
    A : FLOAT, optional
        Amplitude of all the sine waves which are going to be generated. The default is 1.
    
    f_s : FLOAT, optional
        Sampling frequency rate. The default is 44100.
        
    isClosedInterval : BOOL, optional
        Determines whether the bound of the samples interval belongs to it or not. The default is True.

    Returns
    -------
    
    output : LIST OF TUPLES
        For each sine wave generated, the function returns a list of one tuple for every signal.
        Each tuple has three components that contains the time vector, the amplitude vector and a label respectively.
        Thus, the output for n signals generated would be:
        [ (time_1, amplitude_1, label_1) , (time_2, amplitude_2, label_2) , ... , (time_n, amplitude_n, label_n) ]
        Where time_i and amplitude_i are numpy arrays which holds the signal data and label_i is a string with descriptive porposes, being i a natural number between 1 and n.
        The average frequency is specified between brackets in its' label.
    """
    
    t = np.arange(0, 1/closest_to_average(frequencies) + int(isClosedInterval)/f_s , 1/f_s)
    
    output = []
    
    for i in range(0, len(frequencies), 1):
        omega_i = 2*np.pi*frequencies[i]
        y_i = A*np.sin(omega_i*t)
        if frequencies[i] == closest_to_average(frequencies):
            label = f'sin_{i+1}_freq_{frequencies[i]}Hz(ave)'
        else:
            label= f'sin_{i+1}_freq_{frequencies[i]}Hz'
        signal_i = (t , y_i , label)
        output.append(signal_i)
    
    return output


def plot_sin_list(tuples_list, **plot_kwargs):
    
    """
    Plots a list of sine waveforms in an interval determined by the average period of all the signals.

    Parameters
    ----------
    tuples_list : LIST OF TUPLES
        List of tuples containing each tuple the x-y axes data in the first two components.
        The third component of each tuple have to be a string carring the label of the respective signal, with the following format:
            'sin_N_freq_FHz' being N any natural number and F the frequency of the sinewave.
        The label of the sinewave with the average frequency must have the word 'ave' between brackets, have no spaces between characters and have the following format:
            'sin_N_freq_FHz(ave)' being N any natural number and F the frequency of the sinewave.
    
    **plot_kwargs : UNPACKED DICT
        Arguments for the matplotlib.plot() function.

    Returns
    -------
    
    None.

    """
    
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    
    labels = []
    
    for i in range(0, len(tuples_list), 1):
        x = tuples_list[i][0]
        y = tuples_list[i][1]
        plt.plot(x,y, **plot_kwargs)
        labels.append(tuples_list[i][2])
        if 'ave' in tuples_list[i][2]:
            cut = len(tuples_list[i][2]) - 7
            freq_ave = float(tuples_list[i][2][11:cut])
    
    plt.xticks(gen_ticks(preset='periodic', freq=freq_ave, N=5)[0])
    plt.legend(labels, loc="upper right")
    
    return


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


def closest_to_average(v):
    
    """
    Returns the value from a given list which is closest to the average of all the values from it.

    Parameters
    ----------
    v : LIST OF FLOATS
        Input variable of floats in which the value that aproximates to the average is going to be the output. The variable may also be a set, a tuple or a numpy array.

    Returns
    -------
    closest : FLOAT
        Float from the input which difference with the exact average is the smallest.

    """
    
    floats_list = make_list(v)
    average = sum(floats_list)/len(floats_list)
    closest = 0
    
    if average in floats_list:
        closest = average
    else:
        for i in floats_list:
            difference = abs(average - i)
            if difference < abs(average - closest):
                closest = i
    return closest 


def gen_ticks(n=[], N=5, freq=1, preset='octaves'):
    
    '''gen_ticks() is under development'''
    
    n = make_list(n)
    ticks = []
    ticklabels = []
    
    
    # Octaves preset
    
    '''
    ticks = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    ticklabels = ['31.5', '63', '125', '250', '500', '1k', '2k', '4k', '8k', '16k']
    '''
    
    if n == [] and preset == 'octaves':
        for i in range(0, 10, 1):
            ticks.append(31.25*(2**i))
            if 31.25*(2**i) < 1000:
                ticklabels.append(str(int(31.25*(2**i))))
            else:
                ticklabels.append(str(int((31.25/1000)*(2**i)))+'k')
    
    
    # Periodic preset
    
    if n == [] and preset == 'periodic':
        ticks = np.linspace(0, 1/freq, N)
    
    
    # Generate N ticks along a list
    
    if n != []:    
        if len(n)%2 != 0:
            del n[-1] # Set n to even lenght
        
        while len(n)%N != 0:
            N = N - 1 # Set N to greatest common divisor
        
        ticks = [None]*N
        K = int(len(n)/N)
        
        for i in range(1, N+1, 1):
            ticks[i-1] = n[K*i-1]
    
    
    # Default ticklabels
    
    if ticklabels == []:
        ticklabels  = ticks
    
    return ticks, ticklabels
