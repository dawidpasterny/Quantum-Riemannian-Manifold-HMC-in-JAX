from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
from jax.nn import logsumexp
import jax.numpy as jnp


def pol2car(r, phi):
    " Returns rectangular x,y coords from polar r,phi"
    return np.cos(phi)*r, np.sin(phi)*r


def car2pol(x, y):
    " Returns polar r,phi from rectangular x,y coords"
    phi = np.arctan2(y,x)
    r = np.sqrt(x**2+y**2)
    return r, phi


def fun(x,y):
    mvrt = multivariate_normal([0,0], [[2, 0], [0, 2]])
    envelope = mvrt.pdf(x,y)
    r,phi = car2pol(x,y)


def basel_fn(x, y):
    return np.sin(x) / x * np.sin(y) / y


def neg_log_normal(mu, sigma):
    """
    -logp(x | mu, sigma) = 0.5 * log(2π) + log(σ) + 0.5 * ((x - μ)/σ)^2
    """
    def nlogp(x):
        return 0.5 * (jnp.log(2 * jnp.pi * sigma * sigma) + ((x - mu) / sigma) ** 2)

    return nlogp


def neg_log_mvnormal(mu, sigma):

    def nlogp(x):
        k = mu.shape[0]
        return (
            k * jnp.log(2 * jnp.pi)
            + jnp.log(jnp.linalg.det(sigma)) # TODO: use a Cholesky decomposition
            + jnp.dot(jnp.dot((x - mu).T, jnp.linalg.inv(sigma)), x - mu)
        ) * 0.5

    return nlogp


def neg_log_mix(neg_log_probs, alphas):
    """ neg_log_probs is an array of functions handles, alphas are coefficients of the mixture"""
    assert sum(alphas)==1

    def logp(x):
        a = jnp.log(alphas) - jnp.array([nlp(x) for nlp in neg_log_probs])
        return -logsumexp(a)

    return logp

def neg_log_funnel():
    """Neal's 1+1D funnel
        p(x,v) = N(v | 0, 3) N(x | 0, exp(v/2) I )
    """
    scale = neg_log_normal(0, 3) # v
    def neg_log_p(x):
        funnel = neg_log_normal(0, jnp.exp(x[1]/2))
        return scale(x[1]) + funnel(x[0])
    return neg_log_p

def neg_log_nD_funnel():
    """Neal's n+1D funnel (takes n+1 dimensional input) Single input
        p(x) = N(x[0] | 0, 3) N(x[1:] | 0, exp(x[0]) I )
        where the second argument in N stands for std
    """
    scale = neg_log_normal(0, 3)
    def neg_log_p(x):
        funnel_dim = x.shape[0] - 1
        funnel = neg_log_mvnormal(jnp.zeros(funnel_dim), jnp.exp(x[0]) * jnp.eye(funnel_dim))
        return scale(x[0]) + funnel(x[1:])
    return neg_log_p

# def neg_log_funnel():
#     """Neal's funnel.
#         p(x) = N(x[0] | 0, 1) N(x[1:] | 0, exp(2 * x[0]) I )
#     """
#     scale = neg_log_normal(0, 1)
#     def neg_log_p(x):
#         funnel_dim = x.shape[0] - 1
#         if funnel_dim == 1:
#             funnel = neg_log_normal(0, jnp.exp(2 * x[0]))
#         else:
#             funnel = neg_log_mvnormal(jnp.zeros(funnel_dim), jnp.exp(2 * x[0]) * jnp.eye(funnel_dim))
#         return scale(x[0]) + funnel(x[1:])
#     return neg_log_p

def neg_log_bananna():
    pass

