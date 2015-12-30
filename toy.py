import chainer
from chainer import functions as F
import matplotlib.pyplot as plt
import numpy
import six


def generate(N, theta1, theta2, var_x):
    a = numpy.sqrt(var_x) * numpy.random.randn(N, ) + theta1
    b = numpy.sqrt(var_x) * numpy.random.randn(N, ) + theta1 + theta2
    select = numpy.random.random_integers(0, 1, (N, ))
    return a * select + b * (1 - select)


def gaussian_likelihood(x, mu, var):
    if isinstance(x, numpy.ndarray):
        x = chainer.Variable(x.astype(numpy.float32))
        x, mu = F.broadcast(x, mu)
    return F.exp(-(x - mu) ** 2 / var / 2) / numpy.sqrt(2 * numpy.pi * var)


def calc_ab(eps_start, eps_end, gamma, epoch):
    B = 1 / ((eps_start / eps_end) ** (1 / gamma) - 1) * epoch
    A = eps_start * B ** gamma
    eps_start_actual = A / B ** gamma
    eps_end_actual = A / (B + epoch) ** gamma
    assert abs(eps_start - eps_start_actual) < 1e-4
    assert abs(eps_end - eps_end_actual) < 1e-4
    return A, B


THETA1 = 0
THETA2 = 1
VAR1 = 10
VAR2 = 1
VAR_X = 2
EPS_START = 0.01
EPS_END = 0.0001
GAMMA = 0.55
EPOCH = 100
A, B = calc_ab(EPS_START, EPS_END, GAMMA, EPOCH)
n = 100
batchsize = 1
n_batchsize = (n + batchsize - 1) // batchsize
SEED = 0
numpy.random.seed(SEED)

def update(theta1, theta2, x, epoch):
    theta1 = chainer.Variable(numpy.array(theta1, dtype=numpy.float32))
    theta2 = chainer.Variable(numpy.array(theta2, dtype=numpy.float32))

    log_prior1 = F.sum(F.log(gaussian_likelihood(theta1, 0, VAR1)))
    log_prior2 = F.sum(F.log(gaussian_likelihood(theta2, 0, VAR2)))
    prob1 = gaussian_likelihood(x, theta1, VAR_X)
    prob2 = gaussian_likelihood(x, theta1 + theta2, VAR_X)
    log_likelihood = F.sum(F.log(prob1 / 2 + prob2 / 2))
    log_posterior = log_prior1 + log_prior2 + log_likelihood * n / len(x)

    theta1.zerograd()
    theta2.zerograd()
    log_posterior.backward()

    eps = A / (B + epoch) ** GAMMA
    eta = numpy.random.randn() * numpy.sqrt(eps)
    d_theta1 = theta1.grad[0] * eps / 2 + eta
    d_theta2 = theta2.grad[0] * eps / 2 + eta
    return d_theta1, d_theta2


theta1 = numpy.random.randn(1) * numpy.sqrt(VAR1)
theta2 = numpy.random.randn(1) * numpy.sqrt(VAR2)
theta1_all = numpy.empty((EPOCH * n,), dtype=numpy.float32)
theta2_all = numpy.empty((EPOCH * n,), dtype=numpy.float32)
x = generate(n, THETA1, THETA2, VAR_X)
for epoch in six.moves.range(EPOCH):
    perm = numpy.random.permutation(n)
    for i in six.moves.range(0, n, batchsize):
        d_theta1, d_theta2 = update(
            theta1, theta2, x[perm][i: i+batchsize], epoch)
        theta1 += d_theta1
        theta2 += d_theta2
        theta1_all[epoch * n + i / batchsize] = theta1[0]
        theta2_all[epoch * n + i / batchsize] = theta2[0]
        if i == 0:
            print(epoch, theta1[0], theta2[0], theta1[0] * 2 + theta2[0])

H, xedges, yedges = numpy.histogram2d(theta1_all, theta2_all, bins=200)

Hmasked = numpy.ma.masked_where(H == 0, H)
fig = plt.figure()
plt.pcolormesh(xedges, yedges, Hmasked)
plt.xlabel('x')
plt.ylabel('y')
cbar = plt.colorbar()
cbar.ax.set_ylabel('Counts')
plt.savefig('visualize.png')
