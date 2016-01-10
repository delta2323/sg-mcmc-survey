"""Hamiltonian Monte Carlo (HMC)[Neal+10] with Chainer

Example from section 5.1 of [Welling+11].

[Neal10] [MCMC using Hamiltonian dynamics]
(http://www.cs.utoronto.ca/~radford/ftp/ham-mcmc.pdf)
[Welling+11] [Bayesian Learning via Stochastic Gradient Langevin Dynamics]
(http://www.icml-2011.org/papers/398_icmlpaper.pdf)
"""

from __future__ import print_function
import argparse

import chainer
from chainer import functions as F
import matplotlib.pyplot as plt
import numpy
import six

import model


parser = argparse.ArgumentParser(description='HMC')
# data
parser.add_argument('--N', default=100, type=int, help='training data size')
parser.add_argument('--batchsize', default=100, type=int, help='batchsize')
parser.add_argument('--epoch', default=10000, type=int, help='epoch num')
# HMC parameter
parser.add_argument('--eps', default=0.001, type=float, help='stepsize')
parser.add_argument('--L', default=30, type=int, help='sampling interval')
parser.add_argument('--rejection-sampling', action='store_true',
                    help='If true, rejection phase is introduced')
# others
parser.add_argument('--seed', default=0, type=int, help='random seed')
args = parser.parse_args()

n_batch = (args.N + args.batchsize - 1) // args.batchsize
numpy.random.seed(args.seed)


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
        d_q = model.calc_grad(q, x, args.N)
        return p + d_q * args.eps / 2

    def update_q(q, p):
        return q + p * args.eps

    for l in six.moves.range(args.L):
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


def accept(p, theta, p_propose, theta_propose):
    """Test to accept proposal parameter

    Because of the conservation law of energy,
    we can expect H is almost preserved (except numerical and/or
    discretization error) and hence acc_ratio nearly equals to 1.0.
    So, this acceptance step has almost no effect.
    """

    H_prev = H(p, theta)
    H_propose = H(p_propose, theta_propose)
    acc_ratio = min([1.0, numpy.exp(H_prev - H_propose)])
    return numpy.random.randn() < acc_ratio


theta1_all = numpy.empty((args.epoch * n_batch,), dtype=numpy.float32)
theta2_all = numpy.empty((args.epoch * n_batch,), dtype=numpy.float32)
theta = model.sample_from_prior()
x = model.generate(args.N)
for epoch in six.moves.range(args.epoch):
    perm = numpy.random.permutation(args.N)
    for i in six.moves.range(0, args.N, args.batchsize):
        p = numpy.random.randn(*theta.shape)
        p_propose, theta_propose = update(p, theta, x[perm][i: i + args.batchsize])

        if args.rejection_sampling:
            if accept(p, theta, p_propose, theta_propose):
                theta = theta_propose
        else:
            theta = theta_propose

        theta1_all[epoch * n_batch + i // args.batchsize] = theta[0]
        theta2_all[epoch * n_batch + i // args.batchsize] = theta[1]
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
