import numpy as np
import jax.numpy as jnp
import jax
from utils import check_nans

# def explicit_leapfrog(q, p, step_size, num_steps, M, dUdq:callable):
#     """ Sympletic integrator of Hamiltonian dynamics. 
#         Only for separable Hamiltonians.
#     """
#     p -= step_size * dUdq(q)/2 # half step
#     for _ in range(num_steps-1):
#         # q += step_size * jnp.linalg.solve(M, p) 
#         q += step_size * p / M
#         p -= step_size * dUdq(q)
#     q += step_size * p / M
#     p -= step_size * dUdq(q)/2  # half step
#     return q, -p # flip memontum for reversability


# def explicit_leapfrog_twostage(q, p, step_size, num_steps, dUdq:callable):
#     """ Two stage symplectic integrator.
#         Only separable hamiltonians.
#         Unit mass.
#         See two-stage in: http://arxiv.org/abs/1608.07048.   
#     """

#     a = (3 - np.sqrt(3)) / 6
#     p -= a * step_size * dUdq(q)  
#     for _ in range(num_steps-1):
#         q += step_size * p / 2  
#         p -= (1 - 2 * a) * step_size * dUdq(q)  
#         q += step_size * p / 2 
#         p -= 2 * a * step_size * dUdq(q) 
#     q += step_size * p / 2  
#     p -= (1 - 2 * a) * step_size * dUdq(q)
#     q += step_size * p / 2  
#     p -= a * step_size * dUdq(q) 

#     return q, -p # flip memontum for reversability


# def implicit_leapfrog(q_0, p_0, step_size, num_steps, M, dUdq:callable, dTdq:callable, dTdp:callable, fp_treshold=1e-6, max_fp_iter=10):
#     """ Sympletic integrator for non-separable Hamiltonians.
#         Unit mass.
#         According to https://arxiv.org/abs/1212.4693
#         c.f. https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/hmc/integrators/impl_leapfrog.hpp

#         p_{n+1/2} = p_n - step_size/2(dUdq(q_n) + dTdq(q_n, p_{n + 1/2}))
#         q_{n+1} = q_n + step_size/2(dTdp(q_n, p_{n+1/2}) + dTdp(q_{n+1}, p_{n+1/2}))
#         p_{n+1} = p_{n+1/2} - step_size/2(dUdq(q_{n+1}) + dTdq(q_{n+1}, p_{n + 1/2}))

#         This implementation fuses

#         Implemented with jax.lax.while_loop which "jitts" automatically
#     """

#     def check_convergence(state) -> bool:      
#         i, x_old, x = state
#         return (i<=max_fp_iter) & (jnp.max(jnp.abs(x_old - x))>fp_treshold) # L1 norm convergence

#     def half_update_p(state):
#         i, _, p = state
#         return (i+1, p, p_0 - step_size*dTdq(q_0,p)/2)

#     def update_p(state):
#         i, _, p = state
#         return (i+1, p, p_n - step_size*dTdq(q,p)/2 )

#     def update_q(state):
#         i, _, q = state
#         return (i+1, q, q_n + step_size*dTdp(q,p)/2)


#     # fixed point for p_{1/2}
#     p_0 -= step_size*dUdq(q_0) / 2 # explicit in dUdq
#     _, _, p = jax.lax.while_loop(check_convergence, half_update_p, half_update_p((0, p_0, p_0)))

#     q = q_0 
#     for n in range(num_steps-1):
#         # fixed point for q_{n+1}    
#         q_n = q + step_size*dTdp(q,p)/2 # explicit in q_n and p_{n+1/2}
#         _, _, q = jax.lax.while_loop(check_convergence, update_q, update_q((0, q_n, q_n)))

#         # fixed point for p_{n+1+1/2}
#         p_n = p - step_size*dUdq(q) - step_size*dTdq(q,p)/ 2 # explicit in q_{n+1} and p_{n+1/2}
#         _, _, p = jax.lax.while_loop(check_convergence, update_p, update_p((0, p_n, p_n)))

#     # fixed point for q_N   
#     q_n = q + step_size*dTdp(q,p)/2 # explicit in q_{N-1} and p_{N-1/2}
#     _, _, q = jax.lax.while_loop(check_convergence, update_q, update_q((0, q_n, q_n)))
    
#     # Last half step to obtain p_N (explicitly)
#     p -= step_size*(dUdq(q) + dTdq(q,p))/2

#     return q, -p # flip memontum for reversability



def explicit_leapfrog(q, p, step_size, num_steps, M, dUdq:callable):
    """ Sympletic integrator of Hamiltonian dynamics. 
        Only for separable Hamiltonians.
        Ensures evolution till divergence due to numerical stability 
    """
    def check_convergence(state):
        i, _, _, nan_flag = state
        return (i<num_steps) & ~nan_flag

    @check_nans
    def explicit_leapfrog_step(state):
        i, q, p, _ = state

        p -= step_size * dUdq(q)/2 # half step
        q += step_size * p / M
        p -= step_size * dUdq(q)/2  # half step
        return i+1, q, p, _


    _, q, p, _ = jax.lax.while_loop(check_convergence, explicit_leapfrog_step, (0,q,p,False))

    return q, -p # flip memontum for reversability


def explicit_leapfrog_twostage(q, p, step_size, num_steps, dUdq:callable):
    """ Two stage symplectic integrator.
        Only separable hamiltonians.
        Unit mass.
        See two-stage in: http://arxiv.org/abs/1608.07048.   
    """
    def check_convergence(state):
        i, _, _, nan_flag = state
        return (i<num_steps) & ~nan_flag

    @check_nans
    def explicit_leapfrog_twostage_step(state):
        i, q, p, _= state
        a = (3 - np.sqrt(3)) / 6
        p -= a * step_size * dUdq(q)
        q += step_size * p / 2  
        p -= (1 - 2 * a) * step_size * dUdq(q)
        q += step_size * p / 2  
        p -= a * step_size * dUdq(q) 

        return i+1, q, p, _

    _, q, p, _ = jax.lax.while_loop(check_convergence, explicit_leapfrog_twostage_step, (0,q,p,False))

    return q, -p # flip memontum for reversability


def implicit_leapfrog(q_0, p_0, step_size, num_steps, M, dUdq:callable, dTdq:callable, dTdp:callable, fp_treshold=1e-8, max_fp_iter=10):
    """ Sympletic integrator for non-separable Hamiltonians.
        Unit mass.
        According to https://arxiv.org/abs/1212.4693
        c.f. https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/hmc/integrators/impl_leapfrog.hpp

        p_{n+1/2} = p_n - step_size/2(dUdq(q_n) + dTdq(q_n, p_{n + 1/2}))
        q_{n+1} = q_n + step_size/2(dTdp(q_n, p_{n+1/2}) + dTdp(q_{n+1}, p_{n+1/2}))
        p_{n+1} = p_{n+1/2} - step_size/2(dUdq(q_{n+1}) + dTdq(q_{n+1}, p_{n + 1/2}))

        Implemented with jax.lax.while_loop which "jitts" automatically
    """
    def check_convergence(state):
        i, _, _, nan_flag = state
        return (i<num_steps) & ~nan_flag

    @check_nans
    def implicit_leapfrog_step(state):
        i, q_0, p_0, _ = state

        def check_convergence(state) -> bool:      
            i, x_old, x = state
            return (i<=max_fp_iter) & ~(jnp.isnan(x).any())  & (jnp.max(jnp.abs(x_old - x))>fp_treshold) # L1 norm convergence

        def half_update_p(state):
            i, _, p = state
            return (i+1, p, p_0 - step_size*dTdq(q_0,p)/2)
            # return (i+1, p, p_0 - step_size*jnp.clip(dTdq(q_0,p), -10.0, 10.0)/2)
            # return (i+1, p, p_0 - step_size*dTdq(q_0,p)/(2*jnp.linalg.norm(dTdq(q_0,p))))

        def update_q(state):
            i, _, q = state
            return (i+1, q, q_0 + step_size*dTdp(q,p)/2)
            # return (i+1, q, q_0 + step_size*jnp.clip(dTdp(q,p), -10.0, 10.0)/2)
            # return (i+1, q, q_0 + step_size*dTdp(q,p)/(2*jnp.linalg.norm(dTdp(q,p))))

        # fixed point for p_{1/2}
        p_0 -= step_size*dUdq(q_0) / 2 # explicit in dUdq
        _, _, p = jax.lax.while_loop(check_convergence, half_update_p, half_update_p((0, p_0, p_0)))

        # fixed point for q_N   
        q_0 += step_size*dTdp(q_0,p)/2 # explicit in q_{N-1} and p_{N-1/2}
        _, _, q = jax.lax.while_loop(check_convergence, update_q, update_q((0, q_0, q_0)))
        
        # Last half step to obtain p_N (explicitly)
        p -= step_size*(dUdq(q) + dTdq(q,p))/2

        return i+1, q, p, _


    _, q, p, _ = jax.lax.while_loop(check_convergence, implicit_leapfrog_step, (0,q_0,p_0,False))

    return q, -p # flip memontum for reversability




