import matplotlib.pyplot as plt
from numpy.random import default_rng
import pandas as pd
import numpy as np
import sys
sys.path.insert(1, '../')

# import module containing functions for the MSc project
import HP_scripts as HP 
import HP_scripts_debug as HPD

from matplotlib import pyplot as plt
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rcParams["figure.figsize"] = (16, 9) #set default figure size (w, h) 
plt.style.use("ggplot")

mu = 0.5; alpha = 0.3; beta = 0.8 # true parametric values

seed = 1 # uncomment (or comment) this cell to set (or unset) seed value
tstamp, num_of_tstamp = HPD.simulate_univ_ts(mu, alpha, beta,
                        num_of_tstamp = 10, seed=seed)

T_horizon = tstamp[-1] # simulate events of HP till time T

''' Plot an example of point process realisation {t1, t2,} and the
associated counting process N(t). '''


xmin_val = [0]
xmin_val += [i for i in tstamp]
xmax_val = []
xmax_val += [i for i in tstamp]
xmax_val += [tstamp[-1]+0.5]


plt.hlines(y = np.arange(0, num_of_tstamp+1, 1), xmin= xmin_val,
            xmax=xmax_val, colors = 'blue', ls='-', linewidth = 2.5)
plt.vlines(x = tstamp, ymin = 10*[0],
            ymax = [i for i in range(1,num_of_tstamp+1)], 
           colors = 'blue', ls='--', linewidth = 1.5) 

plt.xlabel(r'$t$', fontsize=20)
plt.ylabel(r'$N(t)$',fontsize=20)

plt.plot(tstamp, np.arange(1, num_of_tstamp+1, 1) , 'o', color='blue')
ax = plt.gca() # grab the current axis
ax.set_xticks(tstamp) # choose which x locations to have ticks
xtick_labels = [r'$t_1$',r'$t_2$',r'$t_3$',r'$t_4$',r'$t_5$',
                r'$t_6$',r'$t_7$',r'$t_8$',r'$t_9$',r'$t_{10}$']
ax.set_xticklabels(xtick_labels) # set the labels to display at those ticks
ax.set_yticks([i for i in range(0,num_of_tstamp+1)])
plt.show()


#########################################################################
# Plot intensity of a univariate Hawkes process over time with an
# exponential kernel 
#########################################################################

# t contains 1000 linearly spaced moments of time between [0, T_horizon + 0.5] 
t = np.linspace(0,T_horizon+0.5,1000)
lambdas = [HPD.intensity_func(s, tstamp, mu, alpha, beta) for s in t]
plt.clf()

plt.plot(t,lambdas, '-', color='blue',linewidth = 2)

plt.vlines(x = tstamp, ymin = 10*[0], ymax = 
        [HPD.intensity_func(s, tstamp, mu, alpha, beta) for s in tstamp],
        colors = 'blue', ls='--', linewidth = 1.5) 
# plot occurence of events
plt.plot(tstamp, [0 for i in range(num_of_tstamp)], 'o', 0.3, color = 'blue') 
plt.xlabel(r'$t$', fontsize=20)
plt.ylabel(r'$\lambda(t)$',fontsize=20)

ax = plt.gca() # grab the current axis
ax.set_xticks(tstamp) # choose which x locations to have ticks
xtick_labels = [r'$t_1$',r'$t_2$',r'$t_3$',r'$t_4$',r'$t_5$',
                r'$t_6$',r'$t_7$',r'$t_8$',r'$t_9$',r'$t_{10}$']
ax.set_xticklabels(xtick_labels) # set the labels to display at those ticks
ax.set_yticks([0.5])
ax.set_yticklabels([r'$\mu$'])

#plt.savefig('Fig1_cif_events.png')
plt.show()


