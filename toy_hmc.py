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


# data generation
THETA1 = 0
THETA2 = 1
VAR1 = 10
VAR2 = 1
VAR_X = 2

# sample size
n = 100
batchsize = 100
n_batch = (n + batchsize - 1) // batchsize

# HMC parameter
eps = 0.001
EPOCH = 1000
L = 30

SEED = 0
numpy.random.seed(SEED)


def calc_log_posterior(theta, x):
    theta1, theta2 = F.split_axis(theta, 2, 0)
    log_prior1 = F.sum(F.log(gaussian_likelihood(theta1, 0, VAR1)))
    log_prior2 = F.sum(F.log(gaussian_likelihood(theta2, 0, VAR2)))
    prob1 = gaussian_likelihood(x, theta1, VAR_X)
    prob2 = gaussian_likelihood(x, theta1 + theta2, VAR_X)
    log_likelihood = F.sum(F.log(prob1 / 2 + prob2 / 2))
    return log_prior1 + log_prior2 + log_likelihood


def calc_grad(q, x):
    q = chainer.Variable(numpy.array(q, dtype=numpy.float32))
    log_posterior = calc_log_posterior(q, x)
    q.zerograd()
    log_posterior.backward()
    return q.grad


def update_p(p, q, x):
    d_q = calc_grad(q, x)
    return p + d_q * eps / 2


def update_q(q, p):
    return q + p * eps


def update(p, q, x):
    for l in six.moves.range(L):
        p = update_p(p, q, x)
        q = update_q(q, p)
        p = update_p(p, q, x)
    return p, q


def H(p, q):
    q = chainer.Variable(numpy.array(q, dtype=numpy.float32))
    U = -F.sum(calc_log_posterior(q, x)).data
    K = numpy.sum(p ** 2) / 2
    return U + K


theta1_all = numpy.empty((EPOCH * n_batch,), dtype=numpy.float32)
theta2_all = numpy.empty((EPOCH * n_batch,), dtype=numpy.float32)
theta = numpy.random.randn(2) * [numpy.sqrt(VAR1), numpy.sqrt(VAR2)]
x = generate(n, THETA1, THETA2, VAR_X)
for epoch in six.moves.range(EPOCH):
    perm = numpy.random.permutation(n)
    for i in six.moves.range(0, n, batchsize):
        p = numpy.random.randn(2)
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
