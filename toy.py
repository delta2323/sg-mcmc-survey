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


# data generation
THETA1 = 0
THETA2 = 1
VAR1 = 10
VAR2 = 1
VAR_X = 2

# SGLD parameters
EPS_START = 0.01
EPS_END = 0.0001
GAMMA = 0.55
EPOCH = 100
A, B = calc_ab(EPS_START, EPS_END, GAMMA, EPOCH)

# sample size
n = 100
batchsize = 1

SEED = 0
numpy.random.seed(SEED)


def calc_log_posterior(theta, x):
    theta1, theta2 = F.split_axis(theta, 2, 0)
    log_prior1 = F.sum(F.log(gaussian_likelihood(theta1, 0, VAR1)))
    log_prior2 = F.sum(F.log(gaussian_likelihood(theta2, 0, VAR2)))
    prob1 = gaussian_likelihood(x, theta1, VAR_X)
    prob2 = gaussian_likelihood(x, theta1 + theta2, VAR_X)
    log_likelihood = F.sum(F.log(prob1 / 2 + prob2 / 2))
    return log_prior1 + log_prior2 + log_likelihood * n / len(x)


def calc_grad(theta, x)

def update(theta, epoch):
    eps = A / (B + epoch) ** GAMMA
    eta = numpy.random.randn() * numpy.sqrt(eps)
    d_theta = theta.grad * eps / 2 + eta
    return theta.data + d_theta


theta1_all = numpy.empty((EPOCH * n,), dtype=numpy.float32)
theta2_all = numpy.empty((EPOCH * n,), dtype=numpy.float32)
theta = numpy.random.randn(2) * [numpy.sqrt(VAR1), numpy.sqrt(VAR2)]
x = generate(n, THETA1, THETA2, VAR_X)
for epoch in six.moves.range(EPOCH):
    perm = numpy.random.permutation(n)
    for i in six.moves.range(0, n, batchsize):
        theta = chainer.Variable(numpy.array(theta, dtype=numpy.float32))
        log_posterior = calc_log_posterior(theta, x[perm][i: i+batchsize])

        theta.zerograd()
        log_posterior.backward()

        theta = update(theta, x, epoch)

        theta1_all[epoch * n + i / batchsize] = theta[0]
        theta2_all[epoch * n + i / batchsize] = theta[1]
        if i == 0:
            print(epoch, theta, theta[0] * 2 + theta[1])

H, xedges, yedges = numpy.histogram2d(theta1_all, theta2_all, bins=200)

Hmasked = numpy.ma.masked_where(H == 0, H)
fig = plt.figure()
plt.pcolormesh(xedges, yedges, Hmasked)
plt.xlabel('x')
plt.ylabel('y')
cbar = plt.colorbar()
cbar.ax.set_ylabel('Counts')
plt.savefig('visualize.png')
