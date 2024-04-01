import numpy as np
import jax
import jax.scipy as jsp
import jax.numpy as jnp
from jax.random import normal, uniform, split

from functools import partial
from utils import *
from copy import copy


def mass_randomizer(mu, sigma):
    """ Appends the mass matrix sampling functionality to the kernel
        Follows the diagonal Quantum-Inspired HMC by (Liu & Zhang, 2020)
        - mu: float or 1D array
        - sigma: float, 1D or 2D array
    """
    def decorator(kernel):
        @wraps(kernel) # not necesary but makes __name__() work correctly
        def rand_M_kernel(carry, key): 
            _, subkey = split(key) # spawn a subkey from the key passed to kernel
            log10_M = mu + jnp.dot(sigma, normal(subkey, carry[0].shape))
            M = jnp.power(10, log10_M)
            carry, out = kernel((*carry[:-1], M), key)
            return carry, out
        return rand_M_kernel
    return decorator


def stats_stripper(func):
    @wraps(func) # not necesary but makes __name__() work correctly
    def wrapper(*args, **kwargs): 
        carry, stats = func(*args, **kwargs)
        return carry, stats[0]
    return wrapper


def kernel_factory(kernel:str, q_init, init_step_size, U:callable, integrator:callable, **kwargs):
    """ Initializes a kernel:
        - kernel: str - "mcmc", "hmc", "diag_rmhmc", "chol_rmhmc"
        - U: Callable - negative log probability = potential energy function
            - in the case of mcmc sampler just the pdf
        - integrator: Callable - sympletic integrator
        - init_step_size - required! all heuristics suck. Always deterministic, we very path length instead!
        - **kwargs: (Optional)
            - gather_stats:bool - whether to output only samples or also internal stats from the kernel
            - dual_avg_kwargs: if none provided, dual averaging won't be used!
                - tune_steps
                - target_accept
            - SoftAbs_alpha:float - metric eigenvalue cutoff
            - random_mass_kwargs - mean and variance (scalar, diagonal or full covariance) 
        
        Returns: tuple
            - initial_state - first carry value for the (decorated) kernel 
            - kernel handle
    """
    
    # Kernel setups   
    if kernel=="hmc":
        if integrator.__name__ == 'implicit_leapfrog':
            raise TypeError("Vanilla HMC cannot run with implicit integrators")

        def H(q, p, M): return U(q) + jnp.dot(p, p/(2*M))
        integrator = partial(integrator, dUdq=jax.grad(U))
        def sample_p(key, q): return normal(key, q.shape)
        

    elif kernel=="p_rmhmc":
        if integrator.__name__ != 'implicit_leapfrog':
            raise TypeError("Spatially dependent M implies non-separable Hamiltonian and requires implicit integrator")
        
        alpha = kwargs.get("SoftAbs_alpha", 10)
        def diagHU(q): return jax.jvp(jax.grad(U), (q,), (jnp.ones_like(q),))[1] # diagonal Hessian (faster than computed with diag(jax.hessian))
        def SoftAbsdiagHU(q): return diagHU(q)/jnp.tanh(alpha*diagHU(q)) # THE metric
        def eff_U(q): return U(q) + jnp.sum(jnp.log(SoftAbsdiagHU(q)))/2
        def T(q,p): return jnp.dot(p, p/(2*SoftAbsdiagHU(q)))
        def H(q, p, M): return eff_U(q) + T(q,p)
        def sample_p(key, q): return normal(key, q.shape)*jnp.nan_to_num(SoftAbsdiagHU(q), 1e3)
        integrator = partial(integrator, 
                                dUdq=jax.grad(eff_U), 
                                dTdq=jax.grad(T, argnums=0), 
                                dTdp=jax.grad(T, argnums=1))
        
    elif kernel=="q_rmhmc":
        alpha = kwargs.get("SoftAbs_alpha", 1e6)
        # def eff_U(q):  # full Hessian  
        #     hess_U = jax.hessian(U)
        #     eig, Q = jnp.linalg.eigh(hess_U(q)) # eigendecomp of a Hermitian matrix
        #     sigma = jnp.sqrt(eig/jnp.tanh(alpha*eig)) # negative eigenvalue treatment
        #     return U(Q@q/sigma) + jnp.sum(jnp.log(sigma))
            # return U(Q@q/sigma) - jnp.sum(jnp.log(sigma))
        
        def diagHU(q): return jax.jvp(jax.grad(U), (q,), (jnp.ones_like(q),))[1]
        def eff_U(q): 
            """ Diagonal Hessian"""   
            eig = diagHU(q) # eigendecomp of a Hermitian matrix, Q=I
            sigma = jnp.sqrt(eig/jnp.tanh(alpha*eig)) # negative eigenvalue treatment
            return U(q/sigma) + jnp.sum(jnp.log(sigma))
        
        # from test_functions import neg_log_mvnormal
        # eff_U = neg_log_mvnormal(np.zeros_like(q_init), jnp.eye(q_init.shape[0]))
        
        integrator = partial(integrator, dUdq=jax.grad(eff_U))
        def sample_p(key, q): return normal(key, q.shape)
        def H(q, p, M=1): return eff_U(q) + jnp.dot(p, p/(2*M))

    elif kernel=="qp_rmhmc":
        if integrator.__name__ != 'implicit_leapfrog':
            raise TypeError("Spatially dependent M implies non-separable Hamiltonian and requires implicit integrator")
        
        alpha = kwargs.get("SoftAbs_alpha", 10)
        def diagHU(q): return jax.jvp(jax.grad(U), (q,), (jnp.ones_like(q),))[1] # diagonal Hessian (faster than computed with diag(jax.hessian))
        def SoftAbsdiagHU(q): return diagHU(q)/jnp.tanh(alpha*diagHU(q)) # THE metric
        def eff_U(q): return U(q/SoftAbsdiagHU(q)) + jnp.sum(jnp.log(SoftAbsdiagHU(q)))/2
        def T(q,p, M=1): return jnp.dot(p/M, p*SoftAbsdiagHU(q))/(2)
        def H(q, p, M): return eff_U(q) + T(q,p,M)
        def sample_p(key, q): return normal(key, q.shape)*jnp.nan_to_num(SoftAbsdiagHU(q), 1e-3)
        integrator = partial(integrator, 
                                dUdq=jax.grad(eff_U), 
                                dTdq=jax.grad(T, argnums=0), 
                                dTdp=jax.grad(T, argnums=1))

    elif kernel=="full_rmhmc":
        alpha = kwargs.get("SoftAbs_alpha", 10)
        def SoftAbsHU(q):  # Softabs of the Hessian 
            hess_U = jax.hessian(U)
            eig, Q = jnp.linalg.eigh(hess_U(q)) # eigendecomp of a Hermitian matrix
            eig = jnp.diag(eig/jnp.tanh(alpha*eig)) # negative eigenvalue treatment
            return eig, Q
        
    elif kernel!="mcmc":
        raise TypeError("Unsuported kernel")
    

    # Some defaults
    init_M = kwargs.get("M", 1.0)
    init_num_steps = kwargs.get("num_steps", 20)
    initial_state = [q_init, init_step_size, init_M]


    # Kernel definition
    if kernel=="mcmc": # mcmc kernel is fundamentally different
        def kernel(carry, key):
            q, step_size, _ = carry
            q_new = q + step_size*normal(key, q.shape) 
            key, _ = split(key)
            accept_prob = jnp.nan_to_num(jnp.minimum(1.0, U(q_new)/U(q)), 0.0)
            q = jax.lax.select(uniform(key)<accept_prob, q_new, q) 
            return (q, step_size, 1.0), (q, accept_prob) # jax operates out-of-place, no need to .copy()
        

    else: # all hmc kernels
        def kernel(carry, key): 
            """ Implemented to work with jax.lax.scan.
                - carry: a "carry" value, previous sample
                - key: jax.random.key
                Returns: carry, sample
            """
            q, step_size, M = carry
            keys = split(key, num=3) # generate keys for this run
            p = sample_p(keys[0], q)*jnp.sqrt(M)
            E = H(q,p,M) # current energy
            random_step_size = uniform(keys[1], minval=.8*step_size, maxval=1.2*step_size) # step size jittering
            q_new, p_new = integrator(q, p, random_step_size, init_num_steps, M)
            E_new = H(q_new, p_new, M) # new energy
            accept_prob = jnp.minimum(1.0, jnp.exp(E-E_new)) # Metropolis acceptance
            q = jax.lax.select(uniform(keys[2])<accept_prob, q_new, q) 
            return (q, step_size, M), (q, accept_prob, E_new) # jax operates out-of-place, no need to .copy()
        
        

    # Decorate:
    # Random mass matrix
    random_mass_kwargs = kwargs.get("random_mass_kwargs")
    if random_mass_kwargs!=None:
        kernel = mass_randomizer(**random_mass_kwargs)(kernel)
        initial_state[-1] = jnp.ones(q_init.shape) # in order for the randomizer not to change shape of carry

    # Dual step averaging
    dual_avg_kwargs = kwargs.get("dual_avg_kwargs")
    if dual_avg_kwargs!=None: # return dual averaging generator
        da_initial_state, decorator = dual_averagerator(init_step_size, **dual_avg_kwargs)
        kernel = decorator(kernel)
        initial_state += da_initial_state

    # Stats gathering
    if not kwargs.get("gather_stats", False):
        kernel = stats_stripper(kernel)

    return tuple(initial_state), kernel


