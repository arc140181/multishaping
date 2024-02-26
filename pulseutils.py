import numpy as np
import math as mat

## Pulse generation

def pulse_crrc(T,Ts,tau1,tau2):
    """ Create CR-RC pulse with time constants tau1 and tau2, shaping
        time Ts and duration T"""
    t = np.linspace(0,T*Ts,T)
    h = (tau1/((tau1 - tau2)))*(np.exp(-t/tau1) - np.exp(-t/tau2))    
    return h/np.max(h)


def pulse_crrcn(T,Ts,tau1,n):
    """ Create CR-(RC)^n pulse of order n with constant tau1, shaping
        time Ts and duration T"""
    t = np.linspace(0,T*Ts,T)
    h = (1/mat.factorial(n)) * (t / tau1)**n * np.exp(-t / tau1)
    return h/np.max(h)

def pulse_sh(ka, ma, kb, mb, A, B, da, db):
    """ Pulse creation according to:
        Jordanov, V. T. (2003). Real time digital pulse shaper with
        variable weighting function. Nuclear Instruments and Methods
        in Physics Research Section A 505 (1-2), 347-351. """
    ha = np.zeros((2*ka)+ma)
    hb = np.zeros((2*kb)+mb)

    for j in range(1,(2*ka)+ma):
        if (j<ka):
            ha[j-1] = ((j**2) + j) /2
        elif (j >= ka and j<= (ka + ma)):
            ha[j-1] = ka*(ka+1)/2
        elif (j > (ka + ma) and j < (2*ka + ma)):
            ha[j-1] = ((((2*ka)+ma-j)**2)+((2*ka)+ma-j))/2
    
    for j in range(1,(2*kb)+mb):
        if j < kb:
            hb[j-1] = ((((kb + 1)*2)*j)-(j**2)-j)/2
        elif j >= kb and j<= (kb + mb):
            hb[j-1] = kb*(kb+1)/2
        elif j > (kb + mb) and j < (2*kb + mb):
            hb[j-1] = (-((2*kb+mb-j)**2)+(2*kb+mb-j)*(2*kb+1))/2

    h1 = np.concatenate((np.zeros(da),A*ha,np.zeros(da))) 
    h2 = np.concatenate((np.zeros(db),B*hb,np.zeros(db)))
    h = h1 + h2
    
    return h/np.max(h)

def gdiff(y, x, gam):
    """ Differintegral operator to create fractional noise indexes """ 
    h = x[1] - x[0]
    
    lx = x.size
    dy = np.zeros(lx)
    w = np.ones(lx)
    
    for j in range(1,lx):
        w[j] = w[j-1]*(1-(gam+1)/j)
        
    for i in range(1,lx):
        dy[i] = np.sum(w[0:i+1] * y[i::-1]) / (h ** gam)
        
    return dy

# Create dataset

# Noise color dependent from pulse shape. 0 1 2 3 4 0 1 2 3 4 0 1 2 3 4...
def create_real_dataset_seg(pulse, length, smm, ssat, nmax, noisetype, freq, maxpileup=4):
    """ Create pulse dataset.
        pulse: Shape of the pulse
        length: Length of the dataset
        smm : Interval of pulse heigths
        ssat : Level of saturation
        nmax : Noise amplitude
        noisetype: Type of noise [-1 -> brownian noise, 0 -> white noise]
        freq: Frequency of appearance of pulses. The higher it is, the more risk of pile-up
        maxpileup: Number of maximum pile-ups
    """     
    
    smin = smm[0]
    smax = smm[1]
    x = np.random.random_sample(length)
    x = (x > (1 - freq)) * ((smax - smin) * np.random.random_sample(length) + smin)
    x[x < (smax/1000)] = 0

    noise = nmax * gdiff(np.random.random_sample(length + 1) - 0.5, np.arange(0, length + 1, 1), noisetype)
    noise = np.diff(noise)
    
    x_noise = x + noise
        
    x_h = x

    x_sep_tmp = np.zeros((length,maxpileup))
    x_mask = np.zeros(length)
    c = 0
    for n in range(0, length):
        x_mask[n] = c
        if x[n] > 0:
            x_sep_tmp[n,c] = x[n]
            c = (c + 1) % maxpileup
            
    x = np.convolve(x, pulse)[0:length]
    x_noise = np.convolve(x_noise, pulse)[0:length]
    
    x[x > ssat] = ssat
    x_noise[x_noise > ssat] = ssat
    
    #x_sep = np.zeros((len(x),maxpileup))
    #for n in range(0, maxpileup):
    #    x_sep[:, n] = np.convolve(x_sep_tmp[:, n], pulse)
    #x_sep = x_sep[0:length, :]
    
    x_sep = x_sep_tmp
    
    return x_noise, x, x_mask, x_h, x_sep
