import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import scipy

from IPython.display import Audio
from scipy.signal import fftconvolve
from scipy import signal

import copy_functions
from functions_copy import gen_discrete_signals

gen_discrete_signals('sqPulse')
