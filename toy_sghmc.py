from __future__ import print_function
import argparse
import math

import chainer
import matplotlib.pyplot as plt
import numpy
import six

import model
import stepsize


parser = argparse.ArgumentParser(description='SGHMC')
# training data
parser.add_argument('--N', default=100, type=int, help='training data size')
parser.add_argument('--batchsize', default=10, type=int, help='batchsize')
parser.add_argument('--epoch', default=1000, type=int, help='epoch num')
# SGLMC parameter
parser.add_argument('--D', default=0.1, type=float, help='diffusion parameter')
parser.add_argument('--initialize-moment', action='store_true',
                    help='If true, initialize moment in each sample')
# others
parser.add_argument('--seed', default=0, type=int, help='random seed')
args = parser.parse_args()


n_batch = (args.N + args.batchsize - 1) // args.batchsize
numpy.random.seed(args.seed)


def update_p(p, theta, x, eps):
    d_theta = model.calc_grad(theta, x, args.N)
    return ((1 - args.D * eps) * theta + d_theta
            + math.sqrt(2 * args.D * eps) * numpy.random.randn(*theta.shape))


def update_theta(theta, p, eps):
    return theta + p * eps


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
        for l in six.moves.range(30):
            p = update_p(p, theta, x[perm][i: i+args.batchsize], eps)
            theta = update_theta(theta, p, eps)

        theta1_all[epoch * n_batch + i // args.batchsize] = theta[0]
        theta2_all[epoch * n_batch + i // args.batchsize] = theta[1]
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
plt.savefig('visualize_sghmc.png')
