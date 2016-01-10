"""Hamiltonian Monte Carlo (HMC)[Neal+10] with Chainer

Example from section 5.1 of [Welling+11].

[Neal10] [MCMC using Hamiltonian dynamics]
(http://www.cs.utoronto.ca/~radford/ftp/ham-mcmc.pdf)
[Welling+11] [Bayesian Learning via Stochastic Gradient Langevin Dynamics]
(http://www.icml-2011.org/papers/398_icmlpaper.pdf)
"""


import chainer
from chainer import functions as F
import matplotlib.pyplot as plt
import numpy
import six

import model


# sample size
n = 100
batchsize = 100
n_batch = (n + batchsize - 1) // batchsize

# HMC parameter
eps = 0.001
EPOCH = 10000
L = 30

SEED = 0
numpy.random.seed(SEED)


def update(p, q, x):
    """One parmeter-update step with leapfrog

    Args:
        p(numpy.ndarray): generalized momuntum
        q(numpy.ndarray): generalized coordinate
        x(numpy.ndarray): sample data
    Returns:
        pair of numpy.ndarray: updated momentum and coordinate
    """
    def update_p(p, q, x):
        d_q = model.calc_grad(q, x)
        return p + d_q * eps / 2

    def update_q(q, p):
        return q + p * eps

    for l in six.moves.range(L):
        p = update_p(p, q, x)
        q = update_q(q, p)
        p = update_p(p, q, x)
    return p, q


def H(p, q):
    """Calculates Hamiltonian

    Args:
        p(numpy.ndarray): generalized momuntum
        q(numpy.ndarray): generalized coordinate
    Returns:
        float: Hamiltonian calculated from ``p`` and ``q``
    """
    q = chainer.Variable(numpy.array(q, dtype=numpy.float32))
    U = -F.sum(model.calc_log_posterior(q, x)).data
    K = numpy.sum(p ** 2) / 2
    return U + K


theta1_all = numpy.empty((EPOCH * n_batch,), dtype=numpy.float32)
theta2_all = numpy.empty((EPOCH * n_batch,), dtype=numpy.float32)
theta = model.sample_from_prior()
x = model.generate(n)
for epoch in six.moves.range(EPOCH):
    perm = numpy.random.permutation(n)
    for i in six.moves.range(0, n, batchsize):
        p = numpy.random.randn(*x.shape)
        H_prev = H(p, theta)
        p_propose, theta_propose = update(p, theta, x[perm][i: i+batchsize])

        # Because of the conservation law of energy,
        # we can expect H is almost preserved (except numerical and/or
        # discretization error) and hence acc_ratio nearly equals to 1.0.
        # So, this acceptance step has almost no effect.
        acc_ratio = min([1.0, numpy.exp(H_prev - H(p_propose, theta_propose))])
        if numpy.random.randn() < acc_ratio:
            theta = theta_propose

        theta1_all[epoch * n_batch + i // batchsize] = theta[0]
        theta2_all[epoch * n_batch + i // batchsize] = theta[1]
        print(epoch, theta, theta[0] * 2 + theta[1])

H, xedges, yedges = numpy.histogram2d(theta1_all, theta2_all, bins=200)
H = numpy.rot90(H)
H = numpy.flipud(H)
Hmasked = numpy.ma.masked_where(H == 0, H)
plt.pcolormesh(xedges, yedges, Hmasked)
plt.xlabel('x')
plt.ylabel('y')
cbar = plt.colorbar()
plt.savefig('visualize_hmc.png')
