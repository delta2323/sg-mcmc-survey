"""Stochastic Langevin Dynamics(SGLD) with Chainer

Reimplementation of toy example in section 5.1 of [Welling+11].

[Welling+11] [Bayesian Learning via Stochastic Gradient Langevin Dynamics]
(http://www.icml-2011.org/papers/398_icmlpaper.pdf)
"""

from __future__ import print_function
import argparse

from matplotlib import pyplot
import numpy
import six

import model
import plot
import stepsize


parser = argparse.ArgumentParser(description='SGLD')
# data
parser.add_argument('--N', default=100, type=int, help='training data size')
parser.add_argument('--batchsize', default=1, type=int, help='batchsize')
parser.add_argument('--epoch', default=100, type=int, help='epoch num')
# others
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--visualize', default='visualize_hmc.png', type=str,
                    help='path to output file')
args = parser.parse_args()


n_batch = (args.N + args.batchsize - 1) // args.batchsize
numpy.random.seed(args.seed)


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
    d_theta = model.calc_grad(theta, x, args.N)
    eta = numpy.random.randn() * numpy.sqrt(eps)
    return theta + d_theta * eps / 2 + eta


theta1_all = numpy.empty((args.epoch * n_batch,), dtype=numpy.float32)
theta2_all = numpy.empty((args.epoch * n_batch,), dtype=numpy.float32)
ssg = stepsize.StepSizeGenerator(args.epoch)
theta = model.sample_from_prior()
x = model.generate(args.N)
for epoch in six.moves.range(args.epoch):
    perm = numpy.random.permutation(args.N)
    for i in six.moves.range(0, args.N, args.batchsize):
        theta = update(theta, x[perm][i: i + args.batchsize],
                       epoch, ssg(epoch))

        theta1_all[epoch * n_batch + i // args.batchsize] = theta[0]
        theta2_all[epoch * n_batch + i // args.batchsize] = theta[1]
        if i == 0:
            print(epoch, theta, theta[0] * 2 + theta[1])

fig, axes = pyplot.subplots(ncols=1, nrows=1)
plot.visualize2D(fig, axes, theta1_all, theta2_all,
                 xlabel='theta1', ylabel='theta2')
fig.savefig(args.visualize)
