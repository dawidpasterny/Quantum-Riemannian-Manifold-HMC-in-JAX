import jax.numpy as jnp
from jax.numpy.fft import fft, ifft
import numpy as np
from functools import wraps, partial, update_wrapper
from jax.random import normal, uniform, split, randint
import jax
import matplotlib.pyplot as plt
from scipy.stats import t

########################################## STATS ######################################

def autocorrelation(sequence, lag=100, barlett=True, p_val=0.05):
    """ Autocorrelation estimator for every lag k of the input sequence.
        Biased for stability reasons, see Geyer (1992).
        Uses spectral decomposition using FFT.
        Inspired by Stan implementation https://github.com/stan-dev/stan/blob/develop/src/stan/analyze/mcmc/autocovariance.hpp.
        If barlett==True compute also Barlett confidence intervals at p value = p,
        by default computes 95% confidence intervals
        
    """
    def abs2(x):
        " Computes |x|^2 ~80x faster than jnp.absolute(x)**2 "
        return x.real**2 + x.imag**2
    
    N = len(sequence)
    # Pad signal with zeros to a nice length for FFT and double that length to avoid periodicity issues 
    # M = int(2**(1 + np.ceil(np.log2(N)))) 
    M = 2*fft_next_good_size(N)
    centered_signal = sequence - jnp.mean(sequence)

    # FFT -> 
    freqvec = fft(centered_signal, n=M) # returns array of complex numbers
    freqvec = abs2(freqvec)
    ac_tmp = ifft(freqvec)

    ac = ac_tmp[:lag].real/(N*N*2) # biased
    ac /= ac[0] # divide by variance

    if barlett:
        varac = jnp.ones_like(ac)/N
        varac = varac.at[0].set(0)
        # varac.at[1].set(1.0/N)
        varac = varac.at[2:].multiply(2*jnp.cumsum(ac[1:-1]**2)) # biased
        interval = jax.scipy.stats.norm.ppf(1 - p_val/2) * jnp.sqrt(varac)
        return ac, interval
    
    return ac


def ESS(draws):
    """ Returns (min,med,max) of effective sample size (ESS) 
        Several chains can be used
        Note that the effective sample size can be estimated with no less than four draws.
    
    """
    N = len(draws)
    means = np.mean(draws, axis=0)
    vars
    ac = autocorrelation(draws, N, False)
    return N/(1+2*sum(ac))



def ESS(draws, sizes):
    """ Returns (min,med,max) of effective sample size (ESS) 
        Several chains can be used
        Note that the effective sample size can be estimated with no less than four draws.
    
    """
    def abs2(x):
        " Computes |x|^2 ~80x faster than jnp.absolute(x)**2 "
        return x.real**2 + x.imag**2
    
    N = len(num_draws)
    M = 2*N
    centered_signal = sequence - jnp.mean(sequence)

    # FFT -> 
    freqvec = fft(centered_signal, n=M) # returns array of complex numbers
    freqvec = abs2(freqvec)
    ac_tmp = ifft(freqvec)

    ac = ac_tmp.real/(N*N*2) # autocovariance


    num_chains = len(sizes)
    num_draws = sizes[0]
    for chain in range(1, num_chains):
        num_draws = min(num_draws, sizes[chain])

    acov = np.zeros((num_chains,), dtype=object)
    chain_mean = np.zeros((num_chains,))
    chain_var = np.zeros((num_chains,))
    for chain in range(num_chains):
        draw = np.array(draws[chain][:sizes[chain]])
        acov[chain] = autocorrelation(draw)
        chain_mean[chain] = draw.mean()
        chain_var[chain] = acov[chain][0] * num_draws / (num_draws - 1)

    mean_var = chain_var.mean()
    var_plus = mean_var * (num_draws - 1) / num_draws
    if num_chains > 1:
        var_plus += np.var(chain_mean)
    rho_hat_s = np.zeros((num_draws,))
    rho_hat_s[0] = 1.0
    acov_s = np.zeros((num_chains,))
    for chain in range(num_chains):
        acov_s[chain] = acov[chain][1]
    rho_hat_even = 1.0
    rho_hat_s[1] = rho_hat_odd = 1 - (mean_var - acov_s.mean()) / var_plus
    for s in range(1, num_draws - 4):
        for chain in range(num_chains):
            acov_s[chain] = acov[chain][s + 1]
        rho_hat_even = 1 - (mean_var - acov_s.mean()) / var_plus
        for chain in range(num_chains):
            acov_s[chain] = acov[chain][s + 2]
        rho_hat_odd = 1 - (mean_var - acov_s.mean()) / var_plus
        if rho_hat_even + rho_hat_odd >= 0:
            rho_hat_s[s + 1] = rho_hat_even
            rho_hat_s[s + 2] = rho_hat_odd

    max_s = s
    # this is used in the improved estimate, which reduces variance
    # in antithetic case -- see tau_hat below
    if rho_hat_even > 0:
        rho_hat_s[max_s + 1] = rho_hat_even

    # Convert Geyer's initial positive sequence into an initial
    # monotone sequence
    for s in range(1, max_s - 3, 2):
        if rho_hat_s[s + 1] + rho_hat_s[s] > 0 and rho_hat_s[s + 2] + rho_hat_s[s + 1] > 0:
            rho_hat_s[s + 2] = max(rho_hat_s[s + 1], rho_hat_s[s + 2])

            



# def compute_potential_scale_reduction(draws, sizes):
#     num_chains = len(sizes)
#     num_draws = sizes[0]
#     for chain in range(1, num_chains):
#         num_draws = min(num_draws, sizes[chain])

#     # check if chains are constant; all equal to first draw's value
#     are_all_const = False
#     init_draw = np.zeros(num_chains)

#     for chain in range(num_chains):
#         draw = draws[chain][:sizes[chain]]

#         for n in range(num_draws):
#             if not np.isfinite(draw[n]):
#                 return float('nan')

#         init_draw[chain] = draw[0]

#         if np.allclose(draw, draw[0]):
#             are_all_const = True

#     if are_all_const:
#         # If all chains are constant then return NaN
#         # if they all equal the same constant value
#         if np.allclose(init_draw, init_draw[0]):
#             return float('nan')

#     from scipy.stats import t, f
#     chain_mean = np.zeros(num_chains)
#     acc_chain_mean = []
#     chain_var = np.zeros(num_chains)
#     unbiased_var_scale = num_draws / (num_draws - 1.0)

#     for chain in range(num_chains):
#         acc_draw = []
#         for n in range(num_draws):
#             acc_draw.append(draws[chain][n])
#         acc_draw = np.array(acc_draw)

#         chain_mean[chain] = np.mean(acc_draw)
#         acc_chain_mean.append(chain_mean[chain])
#         chain_var[chain] = np.var(acc_draw) * unbiased_var_scale

#     var_between = num_draws * np.var(acc_chain_mean) * num_chains / (num_chains - 1)
#     var_within = np.mean(chain_var)

#     # rewrote [(n-1)*W/n + B/n]/W as (n-1+ B/W)/n
#     return np.sqrt((var_between / var_within + num_draws - 1) / num_draws)



# def effective_sample_size(draws, sizes):
#     """ Returns (min,med,max) of effective sample size (ESS) 
#         Several chains can be used
#         Note that the effective sample size can be estimated with no less than four draws.
    
#     """
#     num_chains = len(sizes)
#     num_draws = sizes[0]
#     for chain in range(1, num_chains):
#         num_draws = min(num_draws, sizes[chain])

#     if num_draws < 4:
#         return float('nan')

#     # check if chains are constant; all equal to first draw's value
#     are_all_const = False
#     init_draw = np.zeros(num_chains)

#     for chain_idx in range(num_chains):
#         draw = np.array(draws[chain_idx][:sizes[chain_idx]])

#         if not np.all(np.isfinite(draw)):
#             return float('nan')

#         init_draw[chain_idx] = draw[0]

#         if np.allclose(draw, draw[0]):
#             are_all_const |= True

#     if are_all_const:
#         # If all chains are constant then return NaN
#         # if they all equal the same constant value
#         if np.allclose(init_draw, init_draw[0]):
#             return float('nan')

#     acov = np.zeros((num_chains,), dtype=object)
#     chain_mean = np.zeros((num_chains,))
#     chain_var = np.zeros((num_chains,))
#     for chain in range(num_chains):
#         draw = np.array(draws[chain][:sizes[chain]])
#         acov[chain] = autocorrelation(draw)
#         chain_mean[chain] = draw.mean()
#         chain_var[chain] = acov[chain][0] * num_draws / (num_draws - 1)

#     mean_var = chain_var.mean()
#     var_plus = mean_var * (num_draws - 1) / num_draws
#     if num_chains > 1:
#         var_plus += np.var(chain_mean)
#     rho_hat_s = np.zeros((num_draws,))
#     rho_hat_s[0] = 1.0
#     acov_s = np.zeros((num_chains,))
#     for chain in range(num_chains):
#         acov_s[chain] = acov[chain][1]
#     rho_hat_even = 1.0
#     rho_hat_s[1] = rho_hat_odd = 1 - (mean_var - acov_s.mean()) / var_plus
#     for s in range(1, num_draws - 4):
#         for chain in range(num_chains):
#             acov_s[chain] = acov[chain][s + 1]
#         rho_hat_even = 1 - (mean_var - acov_s.mean()) / var_plus
#         for chain in range(num_chains):
#             acov_s[chain] = acov[chain][s + 2]
#         rho_hat_odd = 1 - (mean_var - acov_s.mean()) / var_plus
#         if rho_hat_even + rho_hat_odd >= 0:
#             rho_hat_s[s + 1] = rho_hat_even
#             rho_hat_s[s + 2] = rho_hat_odd

#     max_s = s
#     # this is used in the improved estimate, which reduces variance
#     # in antithetic case -- see tau_hat below
#     if rho_hat_even > 0:
#         rho_hat_s[max_s + 1] = rho_hat_even

#     # Convert Geyer's initial positive sequence into an initial
#     # monotone sequence
#     for s in range(1, max_s - 3, 2):
#         if rho_hat_s[s + 1] + rho_hat_s[s] > 0 and rho_hat_s[s + 2] + rho_hat_s[s + 1] > 0:
#             rho_hat_s[s + 2] = max(rho_hat_s[s + 1], rho_hat_s[s + 2])



def compute_CI(draws, p):
    """ Copmutes mean and its confidence intervals up to desired p value"""
    # Descriptive statistics
    n = len(draws)  # Sample size
    x_bar = np.mean(draws)  # Mean
    s = np.std(draws, ddof=1)  # Sample standard deviation
    t_star = t.ppf(p/2, n-1)

    return x_bar, t_star*s/np.sqrt(n)

    

##################################### STEP SIZE GENERATION #######################################

def dual_averagerator(init_step_size, tune_steps, target_accept, gamma=0.05, t_0=10, kappa=0.75):
    """ Appends the step size dual averaging functionality the sampling kernel
        Based on implementation of (Hoffman and Gelman 2013) 
        Carries running statistics and tunes the step size to the
        desired target accceptance rate for the first #tune_steps steps.
        During tuning yields the most recent step_size, then its average.
    """
    h = 0.0
    t = 0
    log_averaged_step = jnp.log(init_step_size)
    # log_averaged_step = 0.0
    mu = jnp.log(10 * init_step_size)

    def decorator(kernel):
     
        @wraps(kernel) # not necesary but makes __name__() work correctly
        def dual_avg_kernel(carry, key): 
            q, step_size, M, h, t, log_averaged_step = carry
            (q,_,M), out = kernel((q, step_size, M), key) # updates q

            cond =  (t<=tune_steps) & (~jnp.isnan(out[1]))
            h, t, new_step_size, log_averaged_step = jax.lax.cond(cond, 
                                                    update, dont_update,
                                                    h, t, out[1], log_averaged_step)
            
            return (q, new_step_size, M, h, t, log_averaged_step), (*out, step_size) # no need to .copy()

        def update(h, t, accept_prob, log_averaged_step):
            t+=1 # iteration count
            h += ((target_accept - accept_prob) - h)/(t+t_0) # running mean
            log_step = mu - jnp.sqrt(t)*h/gamma
            eta = t**(-kappa)
            log_averaged_step = eta*log_step + (1 - eta)*log_averaged_step

            return h, t, jnp.exp(log_step), log_averaged_step
        
        
        def dont_update(h, t, accept_prob, log_averaged_step):
            "For the first t_0 iterations outputs the initial step size"
            return h, t, jnp.exp(log_averaged_step), log_averaged_step


        return dual_avg_kernel
    return [h, t, log_averaged_step], decorator

########################################### MISCALL #######################################

def check_nans(integrator):
    """ Caches last output of the integrator, if the next one is NaN
        it outputs the previous outupt and rises a nan_flag
    """

    @wraps(integrator) # not necesary but makes __name__() work correctly
    def wrapper(states): 
        " A bit ugly, lax.cond as a workaround for not working lax.select on a tuple"
        def return_old(): return *states[:-1], True # True to stop further iteration
        def return_new(): return i,q,p,False 
        
        i,q,p,_ = integrator(states)
        nan_flag = jnp.isnan(q).any() | jnp.isnan(p).any()
        return jax.lax.cond(nan_flag, return_old, return_new) 

    return wrapper


def fft_next_good_size(N):
  if (N <= 2):
    return 2
  while True:
    m = N;
    while ((m % 2) == 0):
      m /= 2;
    while ((m % 3) == 0):
      m /= 3;
    while ((m % 5) == 0):
      m /= 5;
    if (m <= 1):
      return N;
    N+=1;


def plot_ac(sequence, lag=100, plot_every=1,ax=None):
    ac, conf = autocorrelation(sequence, lag)

    if ax==None:
        fig, ax = plt.subplots(figsize=(15,5))
    else:
        x = np.arange(1, len(ac)+1, plot_every)
        ax.stem(x, ac[::plot_every], markerfmt='.k', linefmt='grey', basefmt='grey')
        ax.fill_between(x, -conf[::plot_every], conf[::plot_every], color='grey', alpha=.2)
        ax.set_ylabel("Autocorrelation")
        ax.set_xlabel(f"Lag (ploted every {plot_every}-th)")