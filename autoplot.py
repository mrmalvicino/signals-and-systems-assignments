import matplotlib.pyplot as plt


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
