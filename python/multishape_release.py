
import numpy as np
from pulseutils import pulse_sh, gdiff

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter


def CreatePulses(samples, pulse, length, smm, ssat, noiseampl, noisetype, freq):
    """
    Creates a time sequence of length 'length' along which pulses with shape 'pulse' turn up.
    The pulses are generated ramdomly in time and amplitude.
        
    Parameters
    ----------
    samples : int
        Number of samples.
    pulse : list of np.array
        Pulse shape (normalized to height=1).
    length : integer
        Length of samples.
    smm : (float, float)
        Mininum and maximum pulse amplitude.
    ssat : float
        Saturation value, if it is below the pulse amplitude, pulses with higher height will appear cut off.
    noiseampl : float | np.array
        Noise amplitude
    noisetype : float | np.array
        Noise type at the output of the detector (e.g. 0 -> white, -0.5 -> 1/f noise, -1 -> brownian).
        Note that the pulse at the output of the detector is modelled as a step pulse and the noise
        is be filtered with the shape 'pulse'.
    freq : float
        Frequency of the appearance of pulses.    

    Returns
    -------
    x_noise : np.array
        The time-sequence with noise.

    x_h : np.array
        The time-sequence with unfolded pulses.
    """    
    
    if len(noisetype)!=len(noiseampl):
        print("Length of noise types and noise amplitudes do not match.")
        
    smin = smm[0]
    smax = smm[1]
    
    x_noise = np.zeros((samples, length, len(pulse)))
    x_h = np.zeros((samples, length, 1))

    for k in range(samples):
        x = np.random.random_sample(length)
        x = (x > (1 - freq)) * ((smax - smin) * np.random.random_sample(length) + smin)
        x[x < (smax/1000)] = 0
    
        x_h[k,:,0] = x    
        x_h_noise = x
        for i in range(len(noisetype)):
            #noise = noiseampl[i] * gdiff(np.random.random_sample(length + 1) - 0.5, np.arange(0, length + 1, 1), noisetype[i])
            
            white_noise = np.random.normal(0, noiseampl[i], length + 1)
            noise = gdiff(white_noise, np.arange(0, length + 1, 1), noisetype[i])
            
            
            noise = np.diff(noise)
            x_h_noise += noise
        
        for i in range(len(pulse)):
            x_noise[k, :, i] = np.convolve(x_h_noise, pulse[i])[0:length]
            
    x_noise[x_noise > ssat] = ssat
    
    return x_noise, x_h


# Length of pulses
PULSELEN = [10, 22, 62]

pulse = []

for i in range(len(PULSELEN)):
    # Pulse type: Triangular
    pulse.append(pulse_sh(ka=PULSELEN[i]//2, ma=0, kb=PULSELEN[i]//2, mb=0, A=1, B=1, da=0, db=0))

SAMPLES = 10
SMAX = 1
NAMPL = [0.01] #[0.05]
NOISETYPE = [0]
FREQ = 0.01 #0.01
NPULSES = 1
LEN = 1024

THR0 = 0.15

colorpal = ['slateblue', 'red', 'limegreen', 'orange']




#%% PHA bare

x_bare, x_bare_ast = CreatePulses(SAMPLES, [pulse[0]], LEN, (0.5, 0.5), SMAX, NAMPL, NOISETYPE, FREQ)

def pha(x, thr):
    heights = []
    flag0, flag1 = 0, 0
    ph = 0
    y_i = np.zeros(LEN)
    for pos in range(LEN):  
        val = x[pos]
        if flag0==0 and flag1==1:
            if val < thr:
                flag0, flag1 = 0, 0
        elif flag0==0:
            ph = 0
            if val > thr:
                flag0 = 1
        elif flag1==0:
            if val > ph:
                ph = val
            elif val < (ph / 2):
                flag0, flag1 = 0, 1
                y_i[pos] = ph
                heights.append(ph)
    return y_i, heights

# Check concept

n_sample = 0

y_bare, h = pha(x_bare[n_sample,:,0], THR0)


# Plot example of pulses

#NAMPL = [0.01, 0.05, 0.1, 0.15]
NAMPL = [0.005, 0.01, 0.02, 0.05]

fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8,6))
for i in range(2):
    for j in range(2):
        x, x_ast = CreatePulses(SAMPLES, pulse, 500, (0.0, 1.0), SMAX, [NAMPL[j+2*i]], NOISETYPE, FREQ)
        title = r'$P_N$=%.3f $\cdot 10^{-3}$' % ((NAMPL[j+2*i]**2)*1e3) 
        ax[i, j].set_title(title)
        if i==1:
            ax[i, j].set_xlabel('time[s]')
        if j==0:
            ax[i, j].set_ylabel('amplitude [a.u.]')
        for k in range(3):
            ax[i, j].plot(x[0,:,k] * 1e-3, color=colorpal[k])
    
#%% PHA multi

#x, x_ast = CreatePulses(SAMPLES, pulse, LEN, (0.0, 1.0), SMAX, NAMPL, NOISETYPE, FREQ)

def pha_multi(x, thr, debug=0, perc=0.1):
    flag0_r = np.zeros(x.shape, dtype=int)
    flag1_r = np.zeros(x.shape, dtype=int)
    ph_r = np.zeros(x.shape, dtype=float)
    last_r = np.zeros(x.shape, dtype=float)
    active_r = np.zeros(x.shape[0], dtype=int)
    
    flag0 = np.zeros(x.shape[1], dtype=int)
    flag1 = np.zeros(x.shape[1], dtype=int)
    ph = np.zeros(x.shape[1], dtype=float)
    last = np.zeros(x.shape[1], dtype=float)
    active = 0
    y_i = np.zeros(x.shape)
    y = np.zeros(x.shape[0])
    heights = []
    count = np.zeros(x.shape[1])
    
    
    
    # PHA
    for pos in range(x.shape[0]):  
        for i in range(x.shape[1]):  
            val = x[pos, i]
            if flag0[i]==0 and flag1[i]==1:
                last[i] = ph[i]
                if val < thr:
                    flag0[i], flag1[i] = 0, 0
            elif flag0[i]==0:
                ph[i] = 0
                if val > thr:
                    flag0[i] = 1
            else:
                if val > ph[i]:
                    ph[i] = val
                elif val < ph[i] - thr: #(ph[i] / 2):
                    flag0[i], flag1[i] = 0, 1
                    y_i[pos, i] = ph[i]
    
            flag0_r[pos,i] = flag0[i]
            flag1_r[pos,i] = flag1[i]
            ph_r[pos,i] = ph[i]
            last_r[pos, i] = last[i]
        
        # FSM
        if active==0 and y_i[pos,0]>0:
            active = 1
        elif active==1 and y_i[pos,0]>0:
            y[pos] = last[0]
            count[0] += 1
            heights.append(last[0])
            active = 1
        elif active==1 and y_i[pos,1]>0:
            active = 2
        elif active==1 and y_i[pos,2]>0:
            y[pos] = last[0]
            count[0] += 1
            heights.append(last[0])
            active = 0
        elif active==2 and (y_i[pos,1]>0 or y_i[pos,0]>0):
            if abs(last[1] - last[0])/last[0] > perc:
                y[pos] = last[0]
                count[0] += 1
                heights.append(last[0])           
            else:
                y[pos] = last[1]
                count[1] += 1
                heights.append(last[1])
            active = 1
        elif active==2 and y_i[pos,2]>0:
            if False: #abs(y_i[pos, 2] - last[1])/last[1] > perc:
                y[pos] = last[1]  # The current value
                count[1] += 1
                heights.append(last[1]) # The current value
            else:
                y[pos] = y_i[pos, 2]  # The current value
                count[2] += 1
                heights.append(y_i[pos, 2]) # The current value
            active = 0
            
        active_r[pos] = active
    
    # End condition
    if active==2:
        y[-1] = last[1]
        count[1] += 1
        heights.append(last[1])
    elif active==1:
        y[-1] = last[0]
        count[0] += 1
        heights.append(last[0])
    
    if debug==1:
        fig, ax = plt.subplots(6, 1, sharex=True, sharey=False, figsize=(8,6))
        
        ax[0].plot(x[:,0], color=colorpal[0])
        ax[0].plot(x[:,1], color=colorpal[1])
        ax[0].plot(x[:,2], color=colorpal[2])
        ax[0].set_ylabel('height [a.u.]')
        ax[0].set_xlim(0, x.shape[0])
        ax[0].set_ylim(0, SMAX)
        ax[0].grid(color='lightgray', linestyle='-', linewidth=1)
        ax[0].text(20, 0.80, r'input $x[n]$')
        
        ax[1].stem(np.arange(0, x.shape[0])[y>THR0], y[y>THR0], linefmt=colorpal[-1], markerfmt='o')
        ax[1].set_xlim(0, x.shape[0])
        ax[1].set_ylim(0, SMAX)
        ax[1].grid(color='lightgray', linestyle='-', linewidth=1)
        ax[1].set_ylabel('height [a.u.]')
        ax[1].text(20, 0.80, r'output $y[n]$')
        
        for k in range(x.shape[1]):
            y_i_tmp = y_i[:,k]
            ax[2].stem(np.arange(0, x.shape[0])[y_i_tmp>THR0], y_i_tmp[y_i_tmp>THR0], linefmt=colorpal[k])
        ax[2].set_xlim(0, x.shape[0])
        ax[2].set_ylim(0, SMAX)
        ax[2].set_ylabel('height [a.u.]')
        ax[2].grid(color='lightgray', linestyle='-', linewidth=1)
        ax[2].text(20, 0.80, r'$y[n]$ of each shaper')
        ax[2].legend(['shaper 0', 'shaper 1', 'shaper 2'], frameon=False, loc='upper right')
        
        for k in range(x.shape[1]):
            ax[3].plot(np.arange(0, x.shape[0]), ph_r[:,k], color=colorpal[k])
        ax[3].set_xlim(0, x.shape[0])
        ax[3].set_ylim(0, SMAX)
        ax[3].set_ylabel('height [a.u.]')
        ax[3].grid(color='lightgray', linestyle='-', linewidth=1)
        ax[3].text(20, 0.80, r'pulse height $h[n]$')
        ax[3].legend(['shaper 0', 'shaper 1', 'shaper 2'], frameon=False, loc='upper right')
        
        for k in range(x.shape[1]):
            ax[4].plot(np.arange(0, x.shape[0]), last_r[:,k], color=colorpal[k])
        ax[4].set_xlim(0, x.shape[0])
        ax[4].set_ylim(0, SMAX)
        ax[4].set_ylabel('height [a.u.]')
        ax[4].text(20, 0.80, r'height of the last pulse $l[n]$')
        ax[4].grid(color='lightgray', linestyle='-', linewidth=1)
        ax[4].legend(['shaper 0', 'shaper 1', 'shaper 2'], frameon=False, loc='upper right')
        
        ax[5].set_xlim(0, x.shape[0])
        ax[5].set_xlabel("time step")
        ax[5].set_ylabel("state")
        ax[5].plot(np.arange(0, x.shape[0]), active_r)
        #ax[5].fill(np.arange(0, x.shape[0]), active_r, -0.1)
        ax[5].grid(color='lightgray', linestyle='-', linewidth=1)
        
    return y, heights, count

#%% Example of pulses

y = np.zeros((SAMPLES, LEN, 1))
n_sample = 1
_, heights, count = pha_multi(x[n_sample], THR0, debug=1)

#%% Multiple histograms 1

NSAMPLHIST = 2000
RANGE = (0.42, 0.58)
BINS = 100

ALPHA = 0.1

#NAMPL = [0.01, 0.05, 0.1, 0.15]
NAMPL = [0.005, 0.01, 0.02, 0.05]

left, bottom, width, height = [0.15, 0.6, 0.07, 0.2]


#%% Histograms

fig, ax = plt.subplots(len(NAMPL) // 2, 2, sharex=False, sharey=False, figsize=(8,6))

bx = []
for j in range(len(NAMPL)):    
    bx.append(fig.add_axes([left + 0.42*(j%2), bottom - 0.42*(j//2), width, height]))
    bx[j].set_axis_off()

for j in range(len(NAMPL)):
    x, x_ast = CreatePulses(NSAMPLHIST, pulse, LEN, (0.5, 0.5), SMAX, [NAMPL[j]], NOISETYPE, FREQ)
    
    # Create histogram
    height_a = []
    height_b = []
    height_c = []
    height_d = []
    count = np.zeros(x.shape[2])
    lhast = 0
    
    for i in range(x.shape[0]):
        _, ha = pha(x[i,:,0], THR0)
        _, hb, c = pha_multi(x[i,:,:], THR0)
        count += c
        _, hc = pha(x[i,:,1], THR0)
        _, hd = pha(x[i,:,2], THR0)
        height_a.extend(ha)
        height_b.extend(hb)
        height_c.extend(hc)
        height_d.extend(hd)
        lhast += (x_ast[i,:,0] > 0.0).sum()
    
    height_a = np.array(height_a)
    height_b = np.array(height_b)
    height_c = np.array(height_c)
    height_d = np.array(height_d)
        
    #%% Plot results

    sns.histplot(height_a, element="poly", binrange=RANGE, bins=BINS, alpha=ALPHA, ax=ax[j//2,j%2], kde=False, color=colorpal[0])
    sns.histplot(height_c, element="poly", binrange=RANGE, bins=BINS, alpha=ALPHA, ax=ax[j//2,j%2], kde=False, color=colorpal[1])
    sns.histplot(height_d, element="poly", binrange=RANGE, bins=BINS, alpha=ALPHA, ax=ax[j//2,j%2], kde=False, color=colorpal[2])
    sns.histplot(height_b, element="poly", binrange=RANGE, bins=BINS, alpha=ALPHA, ax=ax[j//2,j%2], kde=False, color=colorpal[3])

    pa = len(height_a) * 100 / lhast
    pb = len(height_b) * 100 / lhast
    pc = len(height_c) * 100 / lhast
    pd = len(height_d) * 100 / lhast

    height_a = height_a[(height_a > RANGE[0]) & (height_a < RANGE[1])]
    height_b = height_b[(height_b > RANGE[0]) & (height_b < RANGE[1])]
    height_c = height_c[(height_c > RANGE[0]) & (height_c < RANGE[1])]
    height_d = height_d[(height_d > RANGE[0]) & (height_d < RANGE[1])]
        
    σa = np.std(height_a)
    σb = np.std(height_b)
    σc = np.std(height_c)
    σd = np.std(height_d)
    
    legend = []
    legend.append("S0. %.2f$\%%$\n $\sigma$=%.2f$\cdot 10^{-2}$" % (pa, σa*100))
    legend.append("S1. %.2f$\%%$\n $\sigma$=%.2f$\cdot 10^{-2}$" % (pc, σc*100))
    legend.append("S2. %.2f$\%%$\n $\sigma$=%.2f$\cdot 10^{-2}$" % (pd, σd*100))
    legend.append("Multi. %.2f$\%%$\n $\sigma$=%.2f$\cdot 10^{-2}$" % (pb, σb*100))
    
    ax[j//2,j%2].legend(legend, frameon=False)

    if j>1:
        ax[j//2,j%2].set_xlabel("pulse height")
    else:
        ax[j//2,j%2].set_xlabel(None)
    
    if j%2==0:
        ax[j//2,j%2].set_ylabel("counts")
    else:
        ax[j//2,j%2].set_ylabel(None)

    ax[j//2,j%2].text(0.05, 0.95, ' $P_N$=%.3f $\cdot 10^{-3}$ \n $f$=%.2f ev/s'  % ((NAMPL[j]**2)*1e3, FREQ*1000), transform=ax[j//2,j%2].transAxes, fontsize=10, verticalalignment='top')
    # Percentages

    bottom = 0
    width = 1.1

    count /= sum(count)
    count *= 100

    for i in range(len(count)):
        bx[j].bar(' ', count[i], 1.1, bottom, color=colorpal[i], alpha=ALPHA*3)
        bottom += count[i]
        bx[j].text(-0.48, -3+sum(count[0:i])+count[i]/2, "%.1f%%" % (count[i]))
        
#%% Multiple histograms 2

NSAMPLHIST = 2000
RANGE = (0.48, 0.52)
BINS = 100

ALPHA = 0.1

FREQU = [0.005, 0.01, 0.05, 0.1]
left, bottom, width, height = [0.15, 0.6, 0.07, 0.2]

fig, ax = plt.subplots(len(FREQU) // 2, 2, sharex=False, sharey=False, figsize=(8,6))

bx = []
for j in range(len(FREQU)):    
    bx.append(fig.add_axes([left + 0.42*(j%2), bottom - 0.42*(j//2), width, height]))
    bx[j].set_axis_off()

for j in range(len(FREQU)):
    x, x_ast = CreatePulses(NSAMPLHIST, pulse, LEN, (0.5, 0.5), SMAX, [NAMPL[0]], NOISETYPE, FREQU[j])
    
    # Create histogram
    height_a = []
    height_b = []
    height_c = []
    height_d = []
    count = np.zeros(x.shape[2])
    lhast = 0
    
    for i in range(x.shape[0]):
        _, ha = pha(x[i,:,0], THR0)
        _, hb, c = pha_multi(x[i,:,:], THR0)
        count += c
        _, hc = pha(x[i,:,1], THR0)
        _, hd = pha(x[i,:,2], THR0)
        height_a.extend(ha)
        height_b.extend(hb)
        height_c.extend(hc)
        height_d.extend(hd)
        lhast += (x_ast[i,:,0] > 0.0).sum()
    
    height_a = np.array(height_a)
    height_b = np.array(height_b)
    height_c = np.array(height_c)
    height_d = np.array(height_d)
        
    #%% Plot results
    
    sns.histplot(height_a, element="poly", binrange=RANGE, bins=BINS, alpha=ALPHA, ax=ax[j//2,j%2], kde=False, color=colorpal[0])
    sns.histplot(height_c, element="poly", binrange=RANGE, bins=BINS, alpha=ALPHA, ax=ax[j//2,j%2], kde=False, color=colorpal[1])
    sns.histplot(height_d, element="poly", binrange=RANGE, bins=BINS, alpha=ALPHA, ax=ax[j//2,j%2], kde=False, color=colorpal[2])
    sns.histplot(height_b, element="poly", binrange=RANGE, bins=BINS, alpha=ALPHA, ax=ax[j//2,j%2], kde=False, color=colorpal[3])

    pa = len(height_a) * 100 / lhast
    pb = len(height_b) * 100 / lhast
    pc = len(height_c) * 100 / lhast
    pd = len(height_d) * 100 / lhast

    height_a = height_a[(height_a > RANGE[0]) & (height_a < RANGE[1])]
    height_b = height_b[(height_b > RANGE[0]) & (height_b < RANGE[1])]
    height_c = height_c[(height_c > RANGE[0]) & (height_c < RANGE[1])]
    height_d = height_d[(height_d > RANGE[0]) & (height_d < RANGE[1])]
        
    σa = np.std(height_a)
    σb = np.std(height_b)
    σc = np.std(height_c)
    σd = np.std(height_d)
    
    
    legend = []
    legend.append("S0. %.2f$\%%$\n $\sigma$=%.2f$\cdot 10^{-2}$" % (pa, σa*100))
    legend.append("S1. %.2f$\%%$\n $\sigma$=%.2f$\cdot 10^{-2}$" % (pc, σc*100))
    legend.append("S2. %.2f$\%%$\n $\sigma$=%.2f$\cdot 10^{-2}$" % (pd, σd*100))
    legend.append("Multi. %.2f$\%%$\n $\sigma$=%.2f$\cdot 10^{-2}$" % (pb, σb*100))
    
    ax[j//2,j%2].legend(legend, frameon=False)
    
    if j>1:
        ax[j//2,j%2].set_xlabel("pulse height")
    else:
        ax[j//2,j%2].set_xlabel(None)
    
    if j%2==0:
        ax[j//2,j%2].set_ylabel("counts")
    else:
        ax[j//2,j%2].set_ylabel(None)

    ax[j//2,j%2].text(0.05, 0.95, ' $P_N$=%.3f $\cdot 10^{-3}$ \n $f$=%.3f ev/s'  % ((NAMPL[0]**2)*1e3, FREQU[j]*1000), transform=ax[j//2,j%2].transAxes, fontsize=10, verticalalignment='top')

    # Percentages

    bottom = 0
    width = 1.1

    count /= sum(count)
    count *= 100

    for i in range(len(count)):
        bx[j].bar(' ', count[i], 1.1, bottom, color=colorpal[i], alpha=ALPHA*3)
        bottom += count[i]
        bx[j].text(-0.48, -3+sum(count[0:i])+count[i]/3, "%.1f%%" % (count[i]))

