import numpy as np
from matplotlib import pyplot as plt


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
        print(f'{v} is a real number.')
    
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


def gen_discrete_signals(signal_name, a=-10, b=10, n_0=10, on=5, off=15, m=5, mu=0, sigma=1, isClosedInterval = True):
    """
    Generates a custom discrete signal and optionally saves the plot and arrays involved.
    
    Parameters
    ----------
    
    signal_name : STR
        Name of the signal which is going to be generated. E.g., unitImpulse, unitStep, sqPulse, triangPulse, rnd.
    
    a : INT, optional
        Starting sample. The default is -10.
    
    b : INT, optional
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

    Raises
    ------
    
    ValueError
        The parameter a must be less than b by definition.
    
    ValueError
        The parameter n_0 does not belong to the interval [a,b].
    
    ValueError
        Duty Cycle can not be greater than 100%.
    
    ValueError
        The parameter m must be greater than 0 and less than (b-a)/2.
    
    ValueError
        Invalid input.

    Returns
    -------
    
    None.

    """
    # Defines samples interval
    
    if not a < b:
        raise ValueError('The parameter a must be less than b by definition.')
    
    n = np.arange(a, b+int(isClosedInterval), 1)
    
    
    # Defines signal waveform
    
    if signal_name == 'unitImpulse':
        
        if n_0 > len(n):
            raise ValueError('The parameter n_0 does not belong to the interval [a,b].')
        
        x = np.zeros(len(n))
        x[n_0] = 1    
    
    elif signal_name == 'unitStep':
        
        if n_0 > len(n):
            raise ValueError('The parameter n_0 does not belong to the interval [a,b].')
        
        x = np.concatenate((np.zeros(n_0), np.ones(len(n)-n_0)), axis=0)
    
    elif signal_name == 'sqPulse':
        
        dutyCycle = (off-on)*100/len(n)
        
        if dutyCycle > 100:
            raise ValueError('Duty Cycle can not be greater than 100%.')
        
        x = np.concatenate((np.zeros(on), np.ones(off-on), np.zeros(len(n)-off)), axis=0)
    
    elif signal_name == 'triangPulse':
        if m <= 0 or m > len(n)/2:
            raise ValueError('The parameter m must be greater than 0 and less than (b-a)/2.')
        
        x = np.zeros(len(n))
        
        for i in range(-m, m, 1):
            n_i = int((b-a)/2 + i)
            x[n_i] = 1-abs(i*(1/m))
    
    elif signal_name == 'rnd':
        x = np.random.normal(mu, sigma, len(n))
    
    else:
        raise ValueError('Invalid input.')
    
    
    # Plot
    
    plot_kwargs = {'alpha': 1, 'color': 'black', 'linestyle': '', 'linewidth': 1, 'marker': 'o'}
    
    samples_ticks = []
    
    if len(n.tolist()) > 21:
        samples_ticks = gen_ticks(n, N=21)
    else:
        samples_ticks = n.tolist()
    
    plt.figure(figsize=(8,4))
    plt.plot(n,x, **plot_kwargs)
    plt.xticks(samples_ticks)
    plt.grid()
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    
    # Saving
    
    save_plot = 'ask'
    save_array = 'ask'
    
    while save_plot != 'y' and save_plot != 'n':
        save_plot = input('Do you want to save the plot? [y/n] ')
    
    while save_array != 'y' and save_array != 'n':
        save_array = input('Do you want to save the arrays? [y/n] ')
    
    if save_plot == 'y':
        savefig_kwargs = {'bbox_inches': 'tight', 'dpi': 300, 'transparent': False}
        plt.savefig('images\\' + signal_name + 'Plot.png', **savefig_kwargs)
    
    if save_array == 'y':
        np.save('files\\' + signal_name + 'Array_x', x)
        np.save('files\\' + signal_name + 'Array_n', n)
    
    return


def gen_n_sin(*frequencies, a = 0, b=1, f_s = 44100, A=1, isClosedInterval = True):
    """
    Generates an arbitrary quantity of sine waves from the defined input frequencies.

    Parameters
    ----------
    
    *frequencies : UNPACKED TUPLE OF FLOATS
        Unpacked tuple containing the frequency values of the sine waves which are going to be generated.
    
    a : FLOAT, optional
        Starting point in seconds. The default is 0.
    
    b : FLOAT, optional
        Ending point in seconds. The default is 1.
    
    f_s : FLOAT, optional
        Sampling frequency rate. The default is 44100.
    
    A : FLOAT, optional
        Amplitude of all the sine waves which are going to be generated. The default is 1.
    
    isClosedInterval : BOOL, optional
        Determines whether the bound of the samples interval belongs to it or not. The default is True.

    Returns
    -------
    
    output : LIST OF TUPLES
        For each sine wave generated, the function returns a list of one tuple for every signal. Each tuple has three components that contains the time vector, the amplitude vector and a label respectively.

    """
    t = np.arange(a/max(frequencies), b/max(frequencies) + int(isClosedInterval)/f_s , 1/f_s)
    
    output = []
    
    for i in range(0, len(frequencies), 1):
        omega_i = 2*np.pi*frequencies[i]
        y_i = A*np.sin(omega_i*t)
        signal_i = (t , y_i , f'sin_{i+1}_freq_{frequencies[i]}')
        output.append(signal_i)
    
    return output


def plot_sin_list(tuples_list, **plot_kwargs):
    """
    Plots a list of signals.

    Parameters
    ----------
    tuples_list : LIST OF TUPLES
        List of tuples containing each tuple the x-y axes data in the first two components.
    
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
    
    plt.legend(labels, loc="upper right")
    
    return


def gen_ticks(n=[], N=21, scale='octave'):
    
    # Default outputs if input is nule
    
    ticks = []
    ticklabels = []
    
    if n == [] and scale == 'octave':
        for i in range(0, 10, 1):
            ticks.append(31.25*(2**i))
            if 31.25*(2**i) < 1000:
                ticklabels.append(str(int(31.25*(2**i))))
            else:
                ticklabels.append(str(int((31.25/1000)*(2**i)))+'k')
    
    return ticks, ticklabels
    
    
    # Check if the input is a list
    
    if type(n) == np.ndarray:
        n = n.tolist()
    elif type(n) != list:
        raise ValueError('The input can not be converted to a list.')
    
    
    # Set n to even lenght and N to greatest common divisor
    
    if len(n)%2 != 0:
        del n[-1]
    
    while len(n)%N != 0:
        N = N - 1
    
    
    # Create ticks list
    
    ticks = [None]*N
    
    K = int(len(n)/N)
    
    for i in range(1, N+1, 1):
        ticks[i-1] = n[K*i-1]
    
    return ticks
