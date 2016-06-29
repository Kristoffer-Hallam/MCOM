import numpy as np

def noisy_signal(func, mean, std):
    '''
    Inputs random normal noise to a chosen signal.
    
    input:
    func: array - vector containing the original data signal.
    mean: float - float designating the mean value for the noise.
    std: float - float designating the standard deviation value for the noise.    
    
    output:
    noisy_data: array - vector containing the noisy data signal.
    '''
    noise = np.random.normal(mean,std,len(func))
    noisy_data = func + ruido
    return noisy_data
    
def moving_average(func, window):
    '''
    Smoothes a noisy signal according to the averages of the values of points embodied
    by a window.
    
    input:
    func: array - vector containing the noisy data signal.
    window: int - integer designating the window size.
    
    output:
    filt: array - vector containing the smoothed data signal.
    '''
    position = window//2 # position where the average will be allocated
    Nwin = len(func)-(window-position) # Number of windows fitted on the data size
    filt = np.zeros(Nwin) # Initiating filtered data variable having the size of the
    						# number of windows
    
    for i in range(Nwin):
        filt[i] = np.sum(func[i:i+window])/window
#        print i, func[i], filt[i]
    return filt