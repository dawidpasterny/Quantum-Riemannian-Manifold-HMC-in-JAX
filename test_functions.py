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

def cusp():
    pass
    

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
        p(x,v) = N(v | 0, 9) N(x | 0, exp(v) I )
    """
    scale = neg_log_normal(0, 9) 
    def neg_log_p(x): # v = x[-1]
        funnel = neg_log_normal(0, jnp.exp(x[1]))
        return scale(x[1]) + funnel(x[0])
    return neg_log_p

def neg_log_nD_funnel():
    """Neal's n+1D funnel (takes n+1 dimensional input) Single input
        p(x, v) = N(v | 0, 3) N(x | 0, exp(v) I )
        where the second argument in N stands for std
    """
    scale = neg_log_normal(0, 9)
    def neg_log_p(x):
        funnel_dim = x.shape[0] - 1
        funnel = neg_log_mvnormal(jnp.zeros(funnel_dim), jnp.exp(x[-1]) * jnp.eye(funnel_dim))
        return scale(x[-1]) + funnel(x[:-1])
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

def wave():
    def pwave(x):
        return .5*jnp.exp(-.3*x[0])*jnp.sin(x[0])**2*jnp.cos(1.5*x[1])**2
    return pwave


def neg_log_wave():
    def nlwave(x):
        return jnp.log(2) + .3*x[0] -jnp.log(jnp.sin(x[0])**2+1) + jnp.log(jnp.cos(1.5*x[1])**2+1) # +1 for stability
    return nlwave



if __name__=="__main__":
    import matplotlib.pyplot as plt
    import numpy as np


    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(211, projection='3d')

    # Create the mesh in polar coordinates and compute corresponding Z.
    r = np.linspace(0, 10, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    r, theta = np.meshgrid(r, theta)

    radial = lambda r: .5*np.exp(-.3*r)*np.sin(r)**2
    angular = lambda theta: np.cos(1.5*theta)**2
    z = radial(r)*angular(theta)
    # z = .2*np.exp(-.3*r)*(np.cos(1.5*theta)*np.sin(2*r))

    ax2 = fig.add_subplot(212)
    x = np.linspace(0,10,50)
    ax2.plot(x, -np.log(radial(x)))

    # Express the mesh in the cartesian system.
    x, y = r*np.cos(theta), r*np.sin(theta)

    # Plot the surface.
    ax.plot_surface(x,y,-np.log(z+.5), cmap=plt.cm.YlGnBu_r)

    # Tweak the limits and add latex math labels.
    ax.set_zlim(0, 1)
    ax.set_xlabel(r'$\phi_\mathrm{real}$')
    ax.set_ylabel(r'$\phi_\mathrm{im}$')
    ax.set_zlabel(r'$V(\phi)$')





    r, theta = np.linspace(0, 5, 50), np.linspace(0, 2*np.pi, 50)
    r, theta = np.meshgrid(r, theta)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.contourf(theta, r, wave(r, theta), 50)

    plt.show()