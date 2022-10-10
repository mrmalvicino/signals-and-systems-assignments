import os
import numpy as np
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
