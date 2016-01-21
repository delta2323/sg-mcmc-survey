from __future__ import print_function
import argparse
import math

from matplotlib import pyplot
import numpy
import six

import model
import plot
import stepsize


parser = argparse.ArgumentParser(description='SGHMC')
# true parameter
parser.add_argument('--theta1', default=0, type=float, help='true paremter 1')
parser.add_argument('--theta2', default=1, type=float, help='true paremter 2')
# data
parser.add_argument('--N', default=100, type=int, help='training data size')
parser.add_argument('--batchsize', default=10, type=int, help='batchsize')
parser.add_argument('--epoch', default=1000, type=int, help='epoch num')
# mSGNHT parameter
parser.add_argument('--D', default=10, type=float, help='diffusion parameter')
parser.add_argument('--L', default=10, type=int, help='sampling interval')
parser.add_argument('--initialize-auxiliary', action='store_true',
                    help='If true, initialize auxiliary parameters '
                    'for each sample')
parser.add_argument('--eps-start', default=0.01, type=float,
                    help='start stepsize')
parser.add_argument('--eps-end', default=0.005, type=float,
                    help='end stepsize')
# others
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--visualize', default='visualize_msgnht.png', type=str,
                    help='path to output file')
args = parser.parse_args()


n_batch = (args.N + args.batchsize - 1) // args.batchsize
numpy.random.seed(args.seed)


def update(p, theta, xi, x, eps):
    def update_p(p, theta, xi, x, eps):
        d_theta = model.calc_grad(theta, x, args.N)
        return ((1 - xi * eps) * p + d_theta * eps
                + math.sqrt(2 * args.D * eps)
                * numpy.random.randn(*theta.shape))

    def update_theta(theta, p, xi, eps):
        return theta + p * eps

    def update_xi(xi, p, theta, eps):
        return xi + (p * p - 1) * eps

    p = update_p(p, theta, xi, x, eps)
    theta = update_theta(theta, p, xi, eps)
    xi = update_xi(xi, p, theta, eps)
    return p, theta, xi


theta1_all = numpy.empty((args.epoch * n_batch,), dtype=numpy.float32)
theta2_all = numpy.empty((args.epoch * n_batch,), dtype=numpy.float32)
ssg = stepsize.StepSizeGenerator(args.epoch, args.eps_start, args.eps_end)
theta = model.sample_from_prior()
p = numpy.random.randn(*theta.shape)
xi = numpy.full(theta.shape, args.D, dtype=numpy.float32)
x = model.generate(args.N, args.theta1, args.theta2)
for epoch in six.moves.range(args.epoch):
    perm = numpy.random.permutation(args.N)
    for i in six.moves.range(0, args.N, args.batchsize):
        if args.initialize_auxiliary:
            p = numpy.random.randn(*theta.shape)
            xi = numpy.full(theta.shape, args.D)
        for l in six.moves.range(args.L):
            p, theta, xi = update(
                p, theta, xi, x[perm][i: i + args.batchsize], ssg(epoch))

        theta1_all[epoch * n_batch + i // args.batchsize] = theta[0]
        theta2_all[epoch * n_batch + i // args.batchsize] = theta[1]
        if i == 0:
            print(epoch, theta, theta[0] * 2 + theta[1])

fig, axes = pyplot.subplots(ncols=1, nrows=1)
plot.visualize2D(fig, axes, theta1_all, theta2_all,
                 xlabel='theta1', ylabel='theta2',
                 xlim=(-4, 4), ylim=(-4, 4))
fig.savefig(args.visualize)
