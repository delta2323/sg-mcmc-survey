"""Stochastic Langevin Dynamics(SGLD) with Chainer

Reimplementation of toy example in section 5.1 of [Welling+11].

[Welling+11] [Bayesian Learning via Stochastic Gradient Langevin Dynamics]
(http://www.icml-2011.org/papers/398_icmlpaper.pdf)
"""


import matplotlib.pyplot as plt
import numpy
import six

import model


def calc_ab(eps_start, eps_end, gamma, epoch):
    """Returns coefficients that characterize step size

    Args:
        eps_start(float): initial step size
        eps_end(float): initial step size
        gamma(float): decay rate
        epoch(int): # of epoch
    Returns:
        pair of float: (A, B) satisfies ``A / B ** gamma == eps_start``
        and ``A / (B + epoch) ** gamma == eps_end``
    """

    B = 1 / ((eps_start / eps_end) ** (1 / gamma) - 1) * epoch
    A = eps_start * B ** gamma
    eps_start_actual = A / B ** gamma
    eps_end_actual = A / (B + epoch) ** gamma
    assert abs(eps_start - eps_start_actual) < 1e-4
    assert abs(eps_end - eps_end_actual) < 1e-4
    return A, B


# SGLD parameters
EPS_START = 0.01
EPS_END = 0.0001
GAMMA = 0.55
EPOCH = 10000
A, B = calc_ab(EPS_START, EPS_END, GAMMA, EPOCH)

# sample size
n = 100
batchsize = 1
n_batch = (n + batchsize - 1) // batchsize

SEED = 0
numpy.random.seed(SEED)


def update(theta, x, epoch):
    """One parameter-update step of SGLD

    Args:
        theta(numpy.ndarray): model parameeter
        x(numpy.ndarray): sample data
        epoch(int): current epoch index
    Returns:
        numpy.ndarray: updated parameter whose shape is
        same as theta
    """
    d_theta = model.calc_grad(theta, x)
    eps = A / (B + epoch) ** GAMMA
    eta = numpy.random.randn() * numpy.sqrt(eps)
    return theta + d_theta * eps / 2 + eta


theta1_all = numpy.empty((EPOCH * n_batch,), dtype=numpy.float32)
theta2_all = numpy.empty((EPOCH * n_batch,), dtype=numpy.float32)
theta = model.sample_from_prior()
x = model.generate(n)
for epoch in six.moves.range(EPOCH):
    perm = numpy.random.permutation(n)
    for i in six.moves.range(0, n, batchsize):
        theta = update(theta, x[perm][i: i+batchsize], epoch)

        theta1_all[epoch * n_batch + i // batchsize] = theta[0]
        theta2_all[epoch * n_batch + i // batchsize] = theta[1]
        if i == 0:
            print(epoch, theta, theta[0] * 2 + theta[1])

H, xedges, yedges = numpy.histogram2d(theta1_all, theta2_all, bins=200)
H = numpy.rot90(H)
H = numpy.flipud(H)
Hmasked = numpy.ma.masked_where(H == 0, H)
plt.pcolormesh(xedges, yedges, Hmasked)
plt.xlabel('x')
plt.ylabel('y')
cbar = plt.colorbar()
plt.savefig('visualize_sgld.png')
