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
# data
parser.add_argument('--N', default=100, type=int, help='training data size')
parser.add_argument('--batchsize', default=1, type=int, help='batchsize')
parser.add_argument('--epoch', default=100, type=int, help='epoch num')
# SGLMC parameter
parser.add_argument('--F', default=30, type=float, help='friction parameter')
parser.add_argument('--D', default=10, type=float, help='diffusion parameter')
parser.add_argument('--initialize-moment', action='store_true',
                    help='If true, initialize moment in each sample')
parser.add_argument('--L', default=10, type=int, help='sampling interval')
# others
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--visualize', default='visualize_sghmc.png', type=str,
                    help='path to output file')
args = parser.parse_args()


n_batch = (args.N + args.batchsize - 1) // args.batchsize
numpy.random.seed(args.seed)


def update(p, theta, x, eps):
    def update_p(p, theta, x, eps):
        d_theta = model.calc_grad(theta, x, args.N)
        return ((1 - args.F * eps) * p + d_theta * eps
                + math.sqrt(2 * args.D * eps)
                * numpy.random.randn(*theta.shape))

    def update_theta(theta, p, eps):
        return theta + p * eps

    p = update_p(p, theta, x, eps)
    theta = update_theta(theta, p, eps)
    return p, theta


theta1_all = numpy.empty((args.epoch * n_batch,), dtype=numpy.float32)
theta2_all = numpy.empty((args.epoch * n_batch,), dtype=numpy.float32)
ssg = stepsize.StepSizeGenerator(args.epoch)
theta = model.sample_from_prior()
p = numpy.random.randn(*theta.shape)
x = model.generate(args.N)
for epoch in six.moves.range(args.epoch):
    perm = numpy.random.permutation(args.N)
    for i in six.moves.range(0, args.N, args.batchsize):
        if args.initialize_moment:
            p = numpy.random.randn(*theta.shape)
        eps = ssg(epoch)
        for l in six.moves.range(args.L):
            p, theta = update(p, theta, x[perm][i: i + args.batchsize], eps)

        theta1_all[epoch * n_batch + i // args.batchsize] = theta[0]
        theta2_all[epoch * n_batch + i // args.batchsize] = theta[1]
        if i == 0:
            print(epoch, theta, theta[0] * 2 + theta[1])

fig, axes = pyplot.subplots(ncols=1, nrows=1)
plot.visualize2D(fig, axes, theta1_all, theta2_all,
                 xlabel='theta1', ylabel='theta2')
fig.savefig(args.visualize)
