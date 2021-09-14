import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import copy
import scipy.sparse as sparse
from numpy.random import default_rng
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
plt.style.use("ggplot")
import pandas as pd



# define exponential triggering kernel
def exp_kernel(alpha, beta, x):
    return alpha*np.exp(-beta*x)

''' Define non-vectorised function that computes conditional intensity 
function of univariate HP with exponential kernel at time t
'''
def intensity_func(t, event_times, baseline, alpha, beta):
    """ Conditional intensity function of a univariate Hawkes process with
    exponential triggering kernel.
    Parameters:
    - t: moment of time t
    - event_times: list/array containing time stamps of arrival of events
    - baseline: baseline intensity of the Hawkes process
    - alpha: excitation parameter of HP
    - beta: decay parameter of HP
    """
    rate = baseline # baseline intensity
     # total number of events observed between [0,T]
    num_of_events = np.shape(event_times)[0]
    i = 0 # event counter
    
    if (num_of_events == 0):
        return rate

    while (t > event_times[i]):
        rate = rate + exp_kernel(alpha, beta, t-event_times[i])
        i += 1 # next event
        if (i >= num_of_events):
            break
    return rate


''' Define vectorised function that computes conditional intensity 
function of univariate HP with exponential kernel at time t
'''
def univ_cif(t, timestamp, mu, alpha, beta):
    """ Conditional intensity function of a univariate Hawkes process with
    exponential triggering kernel.
    Parameters:                                                                                  
    - mu corresponds to the background intensity of the HP.                                        
    - alpha corresponds to the jump intensity that indicates the jump in
        intensity upon arrival of an event.  
    - beta is the decay parameter that governs the exponential decay
    of intensity.
    """

    cif = mu # baseline intensity fucntion
    mask_index = np.where(timestamp < t)
    cif += np.sum(alpha*np.exp(-beta*np.subtract(t, timestamp[mask_index])))
    return cif




# Define vectorised multivariate conditional intensity function at time t
def multiv_cif(t, timestamp, mu, alpha, beta):

    """ Conditional intensity function of a (M-variate) Hawkes process
    with exponential triggering kernel.
    Parameters:                                                                                  
    - mu corresponds to the background intensity of the HP.                                        
    - alpha corresponds to the jump intensity that indicates the jump in
        intensity upon arrival of an event.  
    - beta is the decay parameter that governs the exponential decay
    of intensity.
    """

    num_of_nodes = np.shape(mu)[0]
    lambdas = copy.deepcopy(mu) # baseline intensity
    
    # compute cif of node M at time t
    for M in range(num_of_nodes):
        # compute contribution of node n to the cif of node M
        for n in range(num_of_nodes):
            ''' masks/select out all index of timestamps across type-n
                event such that t precedes moments of time in timstamp
                array i.e., mask_index contains all i such that t_i < t'''

            mask_index = np.where(timestamp[n] < t)
            # contribution of node n
            lambdas[M] += np.sum(alpha[M][n]*np.exp(-beta[M][n]* \
                np.subtract(t, timestamp[n][mask_index])))
    return lambdas



## Generate N number of timestamps of univariate HP
def simulate_univ_ts(mu, alpha, beta, num_of_tstamp = 100, seed=None, \
    output_rejected_data=False):
    """
    Inputs:
    mu, alpha, beta are parameters of intensity function of HP
    """
    #################
    # Initialisation
    #################
    rng = default_rng(seed)  # get instance of random generator
    t = 0 # initialise current time to be 0
    i = 0 # set event counter to be 0 
    epsilon = 10**(-10) # This was used in many HP code
    lambda_star = mu # upper bound at current time t = 0
    ts = np.array([]); accepted_event_intensity = [lambda_star]
    # containter for rejected time points and their correspodning intensities
    rejected_points = []; rpy = []
    # M_y stores upper bound of current times while M_x stores their x-values
    M_x = []; M_y = []
    #################
    # Begin loop
    #################
    while(i < num_of_tstamp):
        previous_lambda_star = lambda_star; previous_t = t
        '''compute upper bound of intensity using conditional
        intensity function '''
        lambda_star = intensity_func(t+epsilon, ts, mu, alpha, beta)
        u = rng.uniform(0,1) # draw a uniform random number between (0,1)
        tau = -np.log(u)/lambda_star # sample inter-arrival time
        t = t + tau # update current time by adding tau to current time
        M_x += [previous_t,t]
        M_y += [previous_lambda_star]
        s = rng.uniform(0,1) # draw another standard uniform random number
        # compute intensity function at current time t
        lambda_t = intensity_func(t, ts, mu, alpha, beta) 
        ##########################
        ## Rejection Sampling test
        if s <= lambda_t/lambda_star:
            ts = np.append(ts, float(t))
            accepted_event_intensity.append(lambda_t)
            i += 1
        else:
            rejected_points += [t]
            rpy += [lambda_t]
    if output_rejected_data:
        return ts, num_of_tstamp, accepted_event_intensity, rejected_points, rpy
    return ts, num_of_tstamp




## Generate univariate HP
""" Simulate timestamps of HP till time T_horizon """
def simulate_timestamps_till_horizon(mu, alpha, beta, Thorizon = 60, \
    seed=None, node=None, output_rejected_data=False):
    """
    Inputs:
    mu, alpha, beta are parameters of intensity function of HP
    """
    #################
    # Initialisation
    #################
    rng = default_rng(seed)  # get instance of random generator
    t = 0 # initialise current time to be 0
    i = 0 # set event counter to be 0 
    epsilon = 10**(-10) # This was used in many HP code
    lambda_star = mu # upper bound at current time t = 0
    ts = np.array([]); accepted_event_intensity = [lambda_star]
    # containter for rejected time points and their correspodning intensities
    rejected_points = []; rpy = []
    # M_y stores upper bound of current times while M_x stores their x-values
    M_x = []; M_y = []
    #################
    # Begin loop
    #################
    while(t < Thorizon):
        previous_lambda_star = lambda_star; previous_t = t
        # compute upper bound of intensity using intensity function
        lambda_star = intensity_func(t+epsilon, ts, mu, alpha, beta) 
        
        u = rng.uniform(0,1) # draw a uniform random number between (0,1)
        tau = -np.log(u)/lambda_star # sample inter-arrival time
        t = t + tau # update current time by adding tau to current time
        M_x += [previous_t,t]
        M_y += [previous_lambda_star]
        s = rng.uniform(0,1)# draw another standard uniform random number
        # compute intensity function at current time t
        lambda_t = intensity_func(t, ts, mu, alpha, beta) 
        if (t >= Thorizon):
            break
        ##########################
        ## Rejection Sampling test
        if s <= lambda_t/lambda_star:
            ts = np.append(ts, float(t))
            if (node != None):
                ts = np.append(ts, [float(t), np.array([node])])
            accepted_event_intensity.append(lambda_t)
            i += 1
        else:
            rejected_points += [t]
            rpy += [lambda_t]
    if output_rejected_data:
        return ts, i, accepted_event_intensity, rejected_points, rpy
    return ts




""" Simulate timestamps of HP till time T_horizon """
## Generate timestamps of multivariate HP till time horizon
def simulate_multivariate_ts(mu, alpha, beta, num_of_nodes=-1,\
    Thorizon = 60, seed=None, output_rejected_data=False):
    """
    Inputs: 
    mu: baseline intesnities M X 1 array
    alpha: excitiation rates of multivariate kernel pf HP M X M array
    beta: decay rates of kernel of multivariate HP
    node: k-th node of multivariate HP
    """
    #################
    # Initialisation
    #################
    if num_of_nodes < 0:
        num_of_nodes = np.shape(mu)[0] 
    
    rng = default_rng(seed)  # get instance of random generator

    ts = [num_of_nodes * np.array([])] # create M number of empty lise to store ordered set of timestamps of each nodes
    t = 0 # initialise current time to be 0
    num_of_events = np.zeros(num_of_nodes) # set event counter to be 0 for all nodes
    epsilon = 10**(-10) # This was used in many HP code
    M_star = copy.copy(mu) # upper bound at current time t = 0

    accepted_event_intensity = [num_of_nodes * np.array([])]
    rejected_points = [num_of_nodes * np.array([])]; rpy = [num_of_nodes * np.array([])] # containter for rejected time points and their correspodning intensities
    M_x = [num_of_nodes * []]; M_y = [num_of_nodes * np.array([])] # M_y stores Maximum or upper bound of current times while M_x stores their x-values

    #################
    # Begin loop
    #################
    while(t < Thorizon):
        previous_M_star = M_star; previous_t = t
        M_star = np.sum(multiv_cif(t+epsilon, ts, mu, alpha, beta)) # compute upper bound of intensity using conditional intensity function
        
        u = rng.uniform(0,1) # draw a uniform random number between interval (0,1)
        tau = -np.log(u)/M_star # sample inter-arrival time
        t = t + tau # update current time by adding tau to current time (hence t is the candidate point)
        M_x += [previous_t,t]
        M_y += [previous_M_star]
        s = rng.uniform(0,1) # draw another standard uniform random number
        M_t = np.sum(multiv_cif(t, ts, mu, alpha, beta)) # compute intensity function at current time t

        if t <= Thorizon:  
            ##########################
            ## Rejection Sampling test where probability of acceptance: M_t/M_star 
            if s <= M_t/M_star:
                k = 0 # initialise k to be the first node '0'
                # Search for node k such that the 'while condition' below is satisfied
                while s*M_star <= np.sum(multiv_cif(t, ts, mu, alpha, beta)[0:k+1]):
                    k += 1
                num_of_events[k] += 1 # update number of points in node k
                ts[k] = np.append(ts[k], float(t)) # accept candidate point t in node k
                accepted_event_intensity.append(M_t)
            else:
                rejected_points += [t]
                rpy += [M_t]
        else:
            break
    
    if output_rejected_data:
        return ts, num_of_events, accepted_event_intensity, rejected_points, rpy
    else:
        return ts, num_of_events



########################################################
# Following 3 functions From Steve Morse 
########################################################
def plot_cif(ts, mu, alpha, beta, num_of_nodes=-1, Time_horizon=-999):
    #ts = np.array(ts)
    if num_of_nodes < 0:
        num_of_nodes = np.shape(mu)[0]
    if Time_horizon < 0:
        Time_horizon = np.max([np.max(n) for n in ts])
        
    time_vals = np.linspace(0, Time_horizon, 1000) 
    if num_of_nodes == 1:
        ci = np.array([[univ_cif(i, ts, mu[0], alpha[0], beta[0]) for i in time_vals]])
        ci = ci.reshape((1000,1))
        ts = [ts]
    else:
        # col 1 of ci represents intensities of type 1 event and similar for col 2, col3,...
        ci = np.array([multiv_cif(s, ts, mu, alpha, beta) for s in time_vals]) 
    
    fig, axs = plt.subplots(num_of_nodes*2,1, sharex='col', 
                            gridspec_kw = {'height_ratios':sum([[3,1] for i in range(num_of_nodes)],[])}, 
                            figsize=(8,num_of_nodes*2))
    #time_vals = np.linspace(0, Time_horizon, int((Time_horizon/100.)*1000)+1)
    for node_i in range(int(num_of_nodes)):
        row = node_i * 2

        # plot intensity
        ci_of_node_i = ci[:,node_i]
        axs[row].plot(time_vals,ci_of_node_i , 'k-')
        axs[row].set_ylim([-0.01, np.amax(ci_of_node_i)+(np.amax(ci_of_node_i)/2.)])
        axs[row].set_ylabel(r'$\lambda^{%d}(t)$' % node_i, fontsize=14)

        # plot event times
        ts_across_node_i = ts[node_i]
        axs[row+1].plot(ts_across_node_i, np.zeros(np.shape(ts_across_node_i)[0]) - 0.5, 'bo', alpha=0.3)
        axs[row+1].yaxis.set_visible(False)

        axs[row+1].set_xlim([0, Time_horizon])
        axs[row+1].set_xlabel(r'Time period, $t$')
    plt.tight_layout()
    #plt.savefig('Cif.png')


def plot_intensity_function(data, mu, alpha, beta, num_of_nodes=-1, Time_horizon=-999):
    
    if num_of_nodes < 0:
        num_of_nodes = np.shape(mu)[0]

    if Time_horizon < 0:
        Time_horizon = np.amax(data[:,0])

    f, axarr = plt.subplots(num_of_nodes*2,1, sharex='col', 
                            gridspec_kw = {'height_ratios':sum([[3,1] for i in range(num_of_nodes)],[])}, 
                            figsize=(8,num_of_nodes*2))

    
    time_vals = np.linspace(0, Time_horizon, int((Time_horizon/100.)*1000)+1)
    for node in range(int(num_of_nodes)):
        row = node * 2

        # plot rate
        #r = np.array([intensity_func([t], event_times, mu, alpha, beta) for t in time_vals])
        #event_times_of_node_i_ = data[data[:,1]==i][:,0]

        r = np.array(intensity_func(time_vals, data,node, mu, alpha, beta))
        r = r.flatten()
        axarr[row].plot(time_vals, r, 'k-')
        axarr[row].set_ylim([-0.01, np.amax(r)+(np.amax(r)/2.)])
        axarr[row].set_ylabel('$\lambda^{%d}(t)$' % node, fontsize=14)
        r = []

        # plot events
        subseq = data[data[:,1]==node][:,0]
        axarr[row+1].plot(subseq, np.zeros(len(subseq)) - 0.5, 'bo', alpha=0.3)
        axarr[row+1].yaxis.set_visible(False)

        axarr[row+1].set_xlim([0, Time_horizon])
        axarr[row+1].set_xlabel('Time period, t')
    plt.tight_layout()



def plot_event_times(data, num_of_nodes, Time_horizon=-999, company_ticker=[], xaxislabel=None, show_time_periods=True, labeled=True):
    """ 
    nodes: dimension of the multivariate Hawkes process
    
    """
    plt.rcdefaults()
    plt.rcParams["figure.figsize"] = (16, 10) #set default figure size (w, h) 
    plt.style.use("ggplot")
    # change the fontsize of the xtick and ytick labels
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)

    if Time_horizon < 0:
        Time_horizon = np.amax(data[:,0]) # find maximum of the first column where the column is sequence of timestamps

    fig = plt.figure(figsize=(10,5))
    ax = plt.gca()
    for node_i in range(num_of_nodes):
        sequence = data[data[:,1]==node_i][:,0]
        plt.plot(sequence, np.zeros(len(sequence)) - node_i, 'bo', alpha=0.2, markersize = 25)
    if show_time_periods:
        for j in range(1,int(Time_horizon)):
            plt.plot([j,j], [-num_of_nodes, 1], 'k:', alpha=0.2)
    if labeled:
        if len(company_ticker) != 0:

            # Ensure that the axis ticks only show up on the bottom and left of the plot.
            # Ticks on the right and top of the plot are generally unnecessary.
            ax.yaxis.tick_left()
            ax.set_yticks(-np.arange(0,num_of_nodes), minor=False)
            ax.set_yticklabels(company_ticker, minor=False)

            # Limit the range of the plot to only where the data is.
            # Avoid unnecessary whitespace.
            ax.set_xlim([0,Time_horizon+100])
            ax.set_ylim([-num_of_nodes, 1])
        else:
            ax.set_yticks(-range(0, num_of_nodes), minor=True)
            ytick_labels = [r'$e_{%d}$' % i for i in range(10)]

            ax.set_yticks(ytick_labels)
            # Limit the range of the plot to only where the data is.
            # Avoid unnecessary whitespace.
            ax.set_xlim([0,Time_horizon])
            ax.set_ylim([-num_of_nodes, 1])
    else:
        ax.yaxis.set_visible(False)
    if xaxislabel == None:
        ax.set_xlabel('Time period, t')
    else:
        ax.set_xlabel(xaxislabel, fontsize = 20)
    #plt.tight_layout()
    plt.show()



########################################################
# Negative log-likelihood function of univaritae HP modified using MMATLAB code from Dr. Naratip 
########################################################

# define vectorised negative log-likelihood of univariate HP

def Hawkes_Nloglike(time_stamp, mu, alpha, beta, node = -1, num_of_tstamp=-1):
    """ Need to figure what to do with prameter 'node' which is a univariate HP of networks"""
    if node < 0:
        node = 0
        num_of_pts = 1
    else:
        num_of_pts = np.shape(alpha)[0]
    #time_stamp = data[data[:,1] == node, 0] # filter out timestamps that do not belong to the node type specified
    if num_of_tstamp < 0:
        num_of_tstamp = np.shape(time_stamp)[0]
    t_n = np.amax(time_stamp) # final event time of specified node

    A_i = np.zeros((num_of_tstamp,num_of_pts))
    ###########################
    # Compute First summation term
    ###########################

    for i in range(1,num_of_tstamp):
        A_i[i,:] = np.exp(-beta*(time_stamp[i]-time_stamp[i-1]))*(1+A_i[i-1,:]) # recursive computation
     # repmat returns repeated copies of array mu and alpha N by M times where N = num_of_tstamp and M = 1
    log_of_sums = np.log(np.matlib.repmat(mu,num_of_tstamp,1)+np.matlib.repmat(alpha,num_of_tstamp,1)*A_i)

    sum_term1 = np.sum(log_of_sums, axis=0) # f_event
    #############################
    # Second summation term: part of f_int term
    sum_term2 = 1*mu*t_n

    ###########################
    # Compute Final summation term: part of f_int term
    ###########################
    sum_term3 = alpha/beta*((num_of_tstamp - 1)- A_i[-1,-1]) # third summation term

    ##### Unoptimised
    # temp_sum = np.sum([np.exp(-beta[node]*(t_n - time_stamp[t_i])) for t_i in range(num_of_time_stamp)]) # third summation term
    # loglikelihood_val += alpha[node]*(temp_sum-num_of_time_stamp)
    
    return -sum_term1 + sum_term2 + sum_term3





########################################################
# Expectation-Maximisation function modified using MATLAB code from Dr. Naratip
# Note: There is a bug in one of the line of this function and needs to be fixed
########################################################

def EM(mu_guess, alpha_guess, beta_guess, timestamp, Maxiter=100, num_of_tstamp = -1):

    mu = mu_guess; alpha = alpha_guess; beta = beta_guess # initialise the guess values of parameters 
    NLL = np.zeros(Maxiter) # store negative loglikelihoods

    if num_of_tstamp < 0:
        num_of_tstamp = np.shape(timestamp)[0]

    Tdiff = sparse.lil_matrix((num_of_tstamp, num_of_tstamp), dtype=np.float) # little sparse matrix to store lower triangular matrix contatning time diff between i and j pair where i >= j
    for j in range(num_of_tstamp):
        temp = timestamp - timestamp[j] # time difference between varying i-th and fixed j-th pair


        Tdiff[temp > 0,j] = temp[temp > 0] # store time diff in every row i=0,...,num_of_timestamp-1 and column j if the time difference t_i - t_j > 0
    
    dt = timestamp[1:num_of_tstamp] - timestamp[0:num_of_tstamp-1] # shifted time differences
    dt = np.concatenate([[0], dt])
  
    alpha_tilde = alpha*beta # EM algorithm Mohler et. al notation (exponential excitation rate = alpha*beta)
    Pij = Tdiff.toarray() # initialise Pij to be array representation of Tdiff 

    """
    # variable indices is a tuple contaning all pairs of indices (i,j) with non-zero
    # entries where all 'i' is in row array and correspondig 'j' indices are in column array that contains
    """
    row, col, Non_zero_entries = sparse.find(Tdiff)
    indices = (row, col) # row represents row indices and col represetns column indices of non-zero entreis of sparse matrix Tdiff

    for k in range(Maxiter):
        ########################################################
        # E-step
        ########################################################

        # Compute probability matrix P_ij  where p_ij is prob that event j creates event i
        Pij = sparse.csr_matrix((Non_zero_entries, indices)) # sparse matrix representation of probability matrix Pij

        #Pij = sparse.lil_matrix((Non_zero_entries, indices)) # sparse matrix representation of probability matrix Pij
        Pij[indices] = alpha_tilde*np.exp(-beta*Pij[indices])
        
        row_Pij, col_Pij, Non_zero_entries_Pij = sparse.find(Pij)
        indices_Pij = (row_Pij, col_Pij)
       
        # steps to compute probability matrix Pii  where p_ii is prob that event i creates itself
        arr = mu*np.ones(num_of_tstamp)
        diags = np.array([0])
         # retrun array representaion of sparse diagonal matrix where leading diagoanl enries are non-zero
        PP = sparse.spdiags(arr, diags, num_of_tstamp, num_of_tstamp).toarray()

        #tril_index = np.tril_indices(num_of_tstamp,-1) # triangular index
        PP[indices_Pij] = sparse.find(Pij)[-1] # full probability matrix, i.e. PP contains info of Pii and Pij's
        arr_sum = np.sum(PP, axis = 1)
        arr_sum = num_of_tstamp*[arr_sum]
        PP = PP/np.array(arr_sum).T
        
        ########################################################
        # M-step
        ########################################################

        mu_k = np.sum(np.diag(PP))
        mu_k = mu_k/timestamp[-1]

        low_tr = np.tril(PP, -1)
        
        alpha_tilde_k = np.sum(np.sum(low_tr, axis=0))/num_of_tstamp        
        low_tr_times_dt = low_tr*np.array(num_of_tstamp*[dt]).T
        #omega_k = Nlength*alpha_new/sum(nonzeros(PP.*tdiff))
        #omega_k = num_of_tstamp*alpha_tilde_k/np.sum(np.sum(low_tr_times_dt, axis=0)) # tril(arr) return lower triangular matrix of an array arr

        omega_k =  num_of_tstamp*alpha_tilde_k/np.sum(sparse.find(PP*Tdiff)[-1]) # This line needds to be fixed
        mu = mu_k; alpha_tilde = alpha_tilde_k*omega_k; beta = omega_k
 
        NLL[k] = Hawkes_Nloglike(timestamp, mu, [alpha_tilde], beta, node=1 ) # negative likelihood
        
    return mu, alpha_tilde, beta, NLL




# Use the if condition below to run functions in this script
if __name__ == "main":
    print("Used as a standalone scipt")

 