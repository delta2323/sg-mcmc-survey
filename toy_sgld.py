"""Stochastic Langevin Dynamics(SGLD) with Chainer

Reimplementation of toy example in section 5.1 of [Welling+11].

[Welling+11] [Bayesian Learning via Stochastic Gradient Langevin Dynamics]
(http://www.icml-2011.org/papers/398_icmlpaper.pdf)
"""


import matplotlib.pyplot as plt
import numpy
import six

import model
import stepsize


# sample size
n = 100
batchsize = 1
n_batch = (n + batchsize - 1) // batchsize

# sgld paremter
EPOCH = 1000
SEED = 0
numpy.random.seed(SEED)


def update(theta, x, epoch, eps):
    """One parameter-update step of SGLD

    Args:
        theta(numpy.ndarray): model parameeter
        x(numpy.ndarray): sample data
        epoch(int): current epoch index
    Returns:
        numpy.ndarray: updated parameter whose shape is
        same as theta
    """
    d_theta = model.calc_grad(theta, x, n)
    eta = numpy.random.randn() * numpy.sqrt(eps)
    return theta + d_theta * eps / 2 + eta


theta1_all = numpy.empty((EPOCH * n_batch,), dtype=numpy.float32)
theta2_all = numpy.empty((EPOCH * n_batch,), dtype=numpy.float32)
ssg = stepsize.StepSizeGenerator(EPOCH)
theta = model.sample_from_prior()
x = model.generate(n)
for epoch in six.moves.range(EPOCH):
    perm = numpy.random.permutation(n)
    for i in six.moves.range(0, n, batchsize):
        theta = update(theta, x[perm][i: i+batchsize],
                       epoch, ssg(epoch))

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
