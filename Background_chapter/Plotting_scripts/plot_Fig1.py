import matplotlib.pyplot as plt
from numpy.random import default_rng

import sys
sys.path.insert(1, '../')

import HP_scripts as HP # import module containing functions for the Masters project
import pandas as pd
import HP_scripts_debug as HPD
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


from matplotlib import pyplot as plt
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rcParams["figure.figsize"] = (16, 9) #set default figure size (w, h) 
plt.style.use("ggplot")

mu = 0.5; alpha = 0.3; beta = 0.8 # true parametric values


seed = 1 # uncomment (or comment) this cell to set (or unset) seed value
tstamp1, num_of_tstamp1 = HPD.simulate_univ_ts(mu, alpha, beta, num_of_tstamp = 10, seed=seed)

# HPD.plot_cif(tstamp1, np.array([mu]), np.array([alpha]), np.array([beta]), num_of_nodes=-1, Time_horizon=-999)
# plt.show()
# exit()
T_horizon = tstamp1[-1] # simulate events of HP till time T

t = np.linspace(0,T_horizon+0.5,1000) # moments of time between [0,T_horizon] linearly spaced with linear space of 0.01 betwen every t_i and t_i+1
#t = np.linspace(0,T,int(T/0.01)) # moments of time between [0,T] linearly spaced with linear space of 0.01 betwen every t_i and t_i+1
lambdas = [HPD.intensity_func(s, tstamp1, mu, alpha, beta) for s in t]
plt.clf()

plt.plot(t,lambdas, '-', color='blue',linewidth = 2)
#plt.plot(tstamp1, [HPD.intensity_func(s, tstamp1, mu, alpha, beta) for s in tstamp1], 'o', 0.3) # plot occurence of events

plt.vlines(x = tstamp1, ymin = 10*[0], ymax = [HPD.intensity_func(s, tstamp1, mu, alpha, beta) for s in tstamp1], 
           colors = 'blue', 
           ls='--', linewidth = 1.5) 

plt.plot(tstamp1, [0 for i in range(num_of_tstamp1)], 'o', 0.3, color = 'blue') # plot occurence of events
plt.xlabel(r'$t$', fontsize=20)
plt.ylabel(r'$\lambda(t)$',fontsize=20)

ax = plt.gca() # grab the current axis
ax.set_xticks(tstamp1) # choose which x locations to have ticks
xtick_lebels = [r'$t_1$',r'$t_2$',r'$t_3$',r'$t_4$',r'$t_5$',r'$t_6$',r'$t_7$',r'$t_8$',r'$t_9$',r'$t_{10}$']
ax.set_xticklabels(xtick_lebels) # set the labels to display at those ticks
ax.set_yticks([0.5])
ax.set_yticklabels([r'$\mu$'])
#plt.plot(event_times,event_intensity, '-o')

plt.savefig('Fig1_cif_events.png')
plt.show()

