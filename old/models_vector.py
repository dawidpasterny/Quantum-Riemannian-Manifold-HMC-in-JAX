import numpy as np
from numpy.random import default_rng
import jax
import jax.numpy as jnp
from jax.random import PRNGKey, normal, uniform, split
from functools import partial

def random_walk(q, rng, step_size, num_steps):
    """Simulate random walk with isotropic normal"""
    for _ in range(num_steps):
        q += step_size*rng.normal(size=q.shape) 
    return q
    
def mcmc_kernel(prob, q, rng, step_size):
    step_size = rng.uniform(*step_size) if isinstance(step_size, list) else step_size
    # q_new = random_walk(q.copy(), rng, step_size, num_steps) # obtain proposal
    q_new = q + rng.normal(scale=step_size, size=q.shape) 
    # Metropolis acceptance rate (q is symmetric in its parameters)
    accept_prob = np.nan_to_num(prob(q_new)/prob(q), nan=1.0)
    mask = rng.uniform(size=len(accept_prob))<accept_prob.flatten()
    return np.where(mask[:, np.newaxis],q_new,q), sum(mask)

def mcmc_sampler(prob, dim_prob, num_walkers, N, root_seed, step_size, burn=100):
    rng = default_rng(root_seed)
    samples = [] 
    accepted = 0
    try: q = rng.normal(scale=step_size[0], size=(num_walkers,dim_prob)) # arbitrary start
    except: q = rng.normal(scale=step_size, size=(num_walkers,dim_prob)) # arbitrary start
    for _ in range(burn):
        q,_ = mcmc_kernel(prob, q, rng, step_size)
    for i in range(N//num_walkers):
        print('\r iter:', str(i),end='')
        q, acc = mcmc_kernel(prob, q, rng, step_size)
        samples.append(q)
        accepted += acc
    traj = np.stack(samples, axis=2) # disregard the leftovers for ease of collecting trajectories
    q, acc = mcmc_kernel(prob, q[:N%num_walkers,:], rng, step_size)
    samples.append(q)
    accepted += acc
    # Assemble the samples and compute stats
    samples = np.vstack(samples)
    mean = np.cumsum(samples, axis=0)/np.arange(1,N+1)[:, np.newaxis]
    est_mean = np.cumsum(mean, axis=0)/np.arange(1,N+1)[:, np.newaxis]
    est_std = np.sqrt(np.cumsum((mean - est_mean)*(mean - est_mean), axis=0)/np.arange(1,N+1)[:, np.newaxis])

    return samples, traj, mean, est_std, accepted/N


def leapfrog(q, p, grad_U, M, step_size, num_steps):
    """ Helper function to perform sympletic integration of Hamiltonian
        dynamics. It's explicit for separable Hamiltonians 
    """
    p -= step_size*grad_U(q) / 2 # initial half step
    for i in range(num_steps-1):
        q += step_size * p / M
        p -= step_size * grad_U(q)
    q += step_size * p / M
    return q, p

def hmc_kernel(grad_U, H, q, key, M, step_size, num_steps):
    step_size = uniform(key, minval=step_size[0], maxval=step_size[1]) if isinstance(step_size, list) else step_size
    key,split_key = split(key)
    p = M*normal(split_key, q.shape) # sample momentum
    E = H(q,p) # current energy
    q_new, p_new = leapfrog(q, p, grad_U, M, step_size, num_steps)
    E_new = H(q_new, p_new) # new energy
    # Metropolis acceptance rate (q is symmetric in its parameters)
    accept_prob = np.nan_to_num(np.exp(E-E_new), nan=1.0)
    mask = uniform(key, (len(accept_prob), ))<accept_prob.flatten()
    return np.where(mask[:, np.newaxis],q_new,q), np.sum(mask)

# def langevin_kernel(grad_log_prob, H, dim_prob, q, key, num_walkers, M):
#     p = M*normal(key, (num_walkers, dim_prob)) # sample momentum
#     E = H(q,p) # current energy
#     # Single step
#     p_new = p + .5*grad_log_prob(q)/2
#     q_new = q + .5*p_new/M
#     E_new = H(q_new, p_new) # new energy
#     # Metropolis acceptance rate (q is symmetric in its parameters)
#     accept_prob = np.exp(E-E_new) # min(1, prob(q_new)/prob(q)) is unneccessary
#     mask = uniform(key, (num_walkers, ))<accept_prob.flatten()
#     return np.where(mask[:, np.newaxis],q_new,q), np.sum(mask)

def hmc_sampler(U, dim_prob, num_walkers, N, root_seed, kernel=hmc_kernel, M=1, burn=100, **kwargs):
    """ U is the negative log probability function"""
    key = PRNGKey(root_seed)
    samples = [] 
    accepted = 0    
    
    grad_U = jax.vmap(jax.grad(U))
    def H(q,p): return U(q) + np.sum(p**2, axis=1)/(2*M)

    # try: q = kwargs.get("step_size")[0]*normal(key, (num_walkers, dim_prob)) # intial state
    # except: q = kwargs.get("step_size")*normal(key, (num_walkers, dim_prob)) # intial state
    q = normal(key, (num_walkers, dim_prob))
    for _ in range(burn):
        key,split_key = split(key)
        q,_ = kernel(grad_U, H, q, split_key, M, **kwargs)
    for i in range(N//num_walkers):
        print('\r iter:', str(i),end='')
        key,split_key = split(key)
        q,acc = kernel(grad_U, H, q, split_key, M, **kwargs)
        samples.append(q)
        accepted += acc
    traj = np.stack(samples, axis=2) # disregard the leftovers for ease of collecting trajectories
    key,split_key = split(key)
    q,acc = kernel(grad_U, H, q[:N%num_walkers,:], split_key, M, **kwargs)
    samples.append(q)
    accepted += acc
    # Assemble the samples and compute stats
    samples = np.vstack(samples)
    mean = np.cumsum(samples, axis=0)/np.arange(1,N+1)[:, np.newaxis]
    est_mean = np.cumsum(mean, axis=0)/np.arange(1,N+1)[:, np.newaxis]
    est_std = np.sqrt(np.cumsum((mean - est_mean)*(mean - est_mean), axis=0)/np.arange(1,N+1)[:, np.newaxis])

    # return samples, mean, est_std, accepted/N
    return samples, traj, mean, est_std, accepted/N

def leapfrog_sampler(U, q_init, p_init, num_steps=20, M=1, step_size=.1, temper=1.0):
    """ Just walking around according to Hamiltonian dynamics, deterministically"""
    samples = []     
    grad_log_prob = jax.vmap(jax.grad(U))
    q = q_init
    p = p_init
    for i in range(num_steps//2):
        print('\r iter:', str(i),end='')
        q,p = leapfrog(q.copy(), temper*p.copy(), grad_log_prob, M, step_size, 1)
        samples.append(q)
    for j in range(num_steps//2):
        print('\r iter:', str(i+j),end='')
        q,p = leapfrog(q.copy(), p.copy()/temper, grad_log_prob, M, step_size, 1)
        samples.append(q)
    
    traj = jnp.stack(samples, axis=2)

    return traj
##################################### Riemanian HMC ##################################################


##################################### Quantum HMC ##################################################

    

if __name__=="__main__":
    import time
    import matplotlib.pyplot as plt

    root_seed = 0
    dim = 20
    mu = np.ones(dim)
    sigma = np.diag(np.linspace(1/dim,1,dim))
    gaussian = lambda x: jax.scipy.stats.multivariate_normal.pdf(x, mu, sigma)
    U = lambda x: -jax.scipy.stats.multivariate_normal.logpdf(x, mu, sigma)
    num_steps = 150
    hmc_step_size = [0.0104, 0.0156] # range to sample from
    mcmc_step_size = [0.0176, 0.0264]
    num_walkers = 10
    num_samples = 100

    tic = time.time()
    # samples, trajectories, mean, mean_std, a_rate = mcmc_sampler(gaussian, dim, num_walkers, num_steps*num_samples, root_seed, step_size=mcmc_step_size, num_steps=1)
    samples, trajectories, mean, mean_std, a_rate = hmc_sampler(U, dim, num_walkers, num_samples, root_seed, step_size=hmc_step_size, num_steps=num_steps, burn=10)
    toc = time.time()

    print(f"\nAcceptance rate {a_rate}")
    print(f'Execution time:{toc - tic} seconds')

    # plt.scatter(np.linspace(1/dim,1,dim),mean[-1])
    # plt.show()

    for i in np.arange(1,dim,5):
        plt.plot(mean_std[:,i])
        plt.fill_between(np.arange(len(mean_std)), mean[:,i]-mean_std[:,i], mean[:,i]+mean_std[:,i], alpha=0.3)

    plt.show()