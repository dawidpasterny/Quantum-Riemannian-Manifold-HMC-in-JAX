import numpy as np
import multiprocessing as mp
from numpy.random import default_rng
import jax
import jax.numpy as jnp
from jax.random import PRNGKey, normal, uniform, split
from numpy.random import SeedSequence, default_rng



def leapfrog(q, p, grad_log_prob, M, step_size=0.1, num_steps=10):
    """ Helper function to perform sympletic integration of Hamiltonian
        dynamics. It's explicit for separable Hamiltonians 
    """
    for _ in range(num_steps):
        p = p - step_size * grad_log_prob(q) / 2
        q = q + step_size * p / M
        p = p - step_size * grad_log_prob(q) / 2
    return q, p


def hmc_kernel(prob, dim_p, work_queue, rng, lock, N, i_tot, n_tot, burn_in = 0, M = 1):
    log_prob = lambda x: jnp.log(prob(x))
    grad_log_prob = jax.grad(log_prob)
    H = lambda q,p: log_prob(q) - jnp.sum(p**2)/(2*M)

    q, p = rng.normal(size=(2, dim_p)) # intial state and momentum
    i, n = 0, 0
    while n<N:
        E = H(q,p) # current energy
        q_new, p_new = leapfrog(q, p, grad_log_prob, M, .2,5)
        E_new = H(q_new, p_new) # new energy
        accept_prob = min(1, jnp.exp(E-E_new)) # Metropolis acceptance rate (q is symmetric)
        if rng.random() < accept_prob:
            q = q_new
            i += 1
        if n<burn_in:
            continue
        n += 1
        work_queue.put(q)
        p = M*rng.normal(size = dim_p) # resample momentum
    with lock:
        i_tot.value += i
        n_tot.value += n 


def mcmc_kernel(prob, dim_prob, work_queue, rng, lock, N, accepted_tot, vec_len, burn_in = 100): # MCMC simulation
    samples = [] 
    accepted = 0
    q = rng.normal(size=(vec_len,dim_prob)) # arbitrary start
    for _ in range(burn_in):
        q,_ = mcmc_iter(prob, dim_prob, q.copy(), rng, vec_len)
    for _ in range(N//vec_len):
        q, acc = mcmc_iter(prob, dim_prob, q.copy(), rng, vec_len)
        samples.append(q)
        accepted += acc
    # Pick up the slack
    q, acc = mcmc_iter(prob, dim_prob, q[:N%vec_len,:].copy(), rng, N%vec_len)
    samples.append(q)
    accepted += acc
    work_queue.put(np.vstack(samples))
    with lock:
        accepted_tot.value += accepted

def mcmc_iter(prob, dim_prob, q, rng, vec_len):
    # Proposal is an isotropic normal
    q_new = q + .5*rng.normal(size=(vec_len,dim_prob)) # Tweak variance!
    # Metropolis acceptance rate (q is symmetric in its parameters)
    accept_prob = prob(q_new)/prob(q) # min(1, prob(q_new)/prob(q)) is unneccessary
    mask = rng.uniform(size=vec_len)<accept_prob.flatten()
    q[mask] = q_new[mask]
    return q, sum(mask)

def mcmc_sampler (prob, dim_prob, kernel=mcmc_kernel, N=100000, num_walkers = mp.cpu_count()-1, vec_len = 10, root_seed=123875938745):
    """
        p: function handle for the target distribution
        dim_p: dimenions of the domain of p
        num_walkers: numbers of parallel processes
        N: number of samples to generate
    """

    work_queue = mp.Queue(maxsize=N)
    lock = mp.Lock()
    accepted_tot = mp.Value('f', 0.0)
    seeds = SeedSequence(root_seed).spawn(num_walkers)
    thread_N = ((N//num_walkers)*np.ones(num_walkers) + np.pad(np.ones(N%num_walkers),(0,num_walkers-N%num_walkers),constant_values=0)).astype(int)
    
    data_proc_list = []
    for worker_id in range(num_walkers):
        # rng = default_rng([worker_id, root_seed]) # an independent seed
        worker = mp.Process(target=kernel, args=(prob, dim_prob, work_queue, default_rng(seeds[worker_id]), lock, thread_N[worker_id], accepted_tot, vec_len))
        worker.start()
        data_proc_list.append(worker)

    try:   
        samples = np.vstack([work_queue.get() for _ in range(num_walkers)])
        mean = np.cumsum(samples, axis=0)/np.arange(1,N+1)[:, np.newaxis]
        est_mean = np.cumsum(mean, axis=0)/np.arange(1,N+1)[:, np.newaxis]
        est_std = np.sqrt(np.cumsum((mean[1:,:] - est_mean[:-1,:])*(mean - est_mean)[1:,:], axis=0)/np.arange(2,N+1)[:, np.newaxis])

    finally:
        for worker in data_proc_list: # wait till all workers terminate
            # worker.join() # because it will block until que is emptied with get() which we don't do
            worker.terminate()
    return samples, mean, est_std, accepted_tot.value/N

