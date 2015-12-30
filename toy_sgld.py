"""Stochastic Langevin Dynamics(SGLD) with Chainer

Reimplementation of toy example in section 5.1 of [Welling+11].

[Welling+11] [Bayesian Learning via Stochastic Gradient Langevin Dynamics]
(http://www.icml-2011.org/papers/398_icmlpaper.pdf)
"""


import chainer
from chainer import functions as F
import matplotlib.pyplot as plt
import numpy
import six


def generate(N, theta1, theta2, var_x):
    """Generates sample data from gaussian mixture

    Args:
        N(int): sample size
        theta1(float): mean of one Gaussian
        theta2(float): mean of the other Gaussian
        var_x(float): variance of two Gaussians
    Returns:
        numpy.ndarray: sample data of shape ``(N, )`` drawn from
        ``N(theta1, var_x) / 2 + N(theta2, var_x) / 2``
    """

    a = numpy.sqrt(var_x) * numpy.random.randn(N, ) + theta1
    b = numpy.sqrt(var_x) * numpy.random.randn(N, ) + theta1 + theta2
    select = numpy.random.random_integers(0, 1, (N, ))
    return a * select + b * (1 - select)


def gaussian_likelihood(x, mu, var):
    """Returns likelihood of ``x``, or ``N(x; mu, var)``

    Args:
        x(float, numpy.ndarray or chainer.Variable): sample data
        mu(float or chainer.Variable): mean of Gaussian
        var(float): variance of Gaussian
    Returns:
        chainer.Variable: Variable holding likelihood ``N(x; mu, var)``
        whose shape is same as that of ``x``
    """

    if isinstance(x, numpy.ndarray) or numpy.isscalar(x):
        x = chainer.Variable(x.astype(numpy.float32))
    if isinstance(mu, numpy.ndarray) or numpy.isscalar(mu):
        mu = chainer.Variable(mu.astype(numpy.float32))
    x, mu = F.broadcast(x, mu)
    return F.exp(-(x - mu) ** 2 / var / 2) / numpy.sqrt(2 * numpy.pi * var)


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
EPOCH = 10000
A, B = calc_ab(EPS_START, EPS_END, GAMMA, EPOCH)

# sample size
n = 100
batchsize = 1
n_batch = (n + batchsize - 1) // batchsize

SEED = 0
numpy.random.seed(SEED)


def calc_log_posterior(theta, x):
    """Calculate unnormalized log posterior, ``log p(theta | x) + C``

    theta = (theta1, theta2)
    prior: ``p(theta1) = N(theta1; 0, VAR1)``, ``p(theta2) = N(theta2; 0, VAR2)``
    likelihood: ``p(x | theta) = N(theta1, var_x) / 2 + N(theta2, var_x) / 2``

    Args:
        theta(chainer.Variable): model parameters
        x(numpy.ndarray): sample data
    Returns:
        chainer.Variable: Variable that holding unnormalized log posterior,
        ``log p(theta | x) + C`` of shape ``()``
    """

    theta1, theta2 = F.split_axis(theta, 2, 0)
    log_prior1 = F.sum(F.log(gaussian_likelihood(theta1, 0, VAR1)))
    log_prior2 = F.sum(F.log(gaussian_likelihood(theta2, 0, VAR2)))
    prob1 = gaussian_likelihood(x, theta1, VAR_X)
    prob2 = gaussian_likelihood(x, theta1 + theta2, VAR_X)
    log_likelihood = F.sum(F.log(prob1 / 2 + prob2 / 2))
    return log_prior1 + log_prior2 + log_likelihood * n / len(x)


def calc_grad(theta, x):
    """Computes gradient of log posterior w.r.t. parameter

    Args:
        theta(numpy.ndarray): model parameters
        x(numpy.ndarray): sample data
    Returns:
        numpy.ndarray: ``dp(theta | x) / dtheta``
    """
    theta = chainer.Variable(numpy.array(theta, dtype=numpy.float32))
    log_posterior = calc_log_posterior(theta, x)
    theta.zerograd()
    log_posterior.backward()
    return theta.grad


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
    d_theta = calc_grad(theta, x)
    eps = A / (B + epoch) ** GAMMA
    eta = numpy.random.randn() * numpy.sqrt(eps)
    return theta + d_theta * eps / 2 + eta


theta1_all = numpy.empty((EPOCH * n_batch,), dtype=numpy.float32)
theta2_all = numpy.empty((EPOCH * n_batch,), dtype=numpy.float32)
theta = numpy.random.randn(2) * [numpy.sqrt(VAR1), numpy.sqrt(VAR2)]
x = generate(n, THETA1, THETA2, VAR_X)
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
