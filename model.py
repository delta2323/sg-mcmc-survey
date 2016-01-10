import chainer
from chainer import functions as F
import numpy

import gaussian

# true parameters
THETA1 = 0
THETA2 = 2

# hyper parmeters
VAR_X = 2
VAR1 = 10
VAR2 = 1

"""
Model
* parameter: theta = (theta1, theta2)
* prior: p(theta) = (N(0, VAR1), N(0, VAR2))
* likelihood:
    p(x | theta) ~ N(theta1, VAR_X) / 2 + N(theta1 + theta2, VAR_X) / 2
"""


def sample_from_prior():
    return numpy.random.randn(2) * [numpy.sqrt(VAR1), numpy.sqrt(VAR2)]


def generate(N, theta1=THETA1, theta2=THETA2, var_x=VAR_X):
    """Generates sample data from gaussian mixture

    Args:
        N(int): sample size
        theta1(float): mean of one Gaussian
        theta2(float): mean of the other Gaussian
        var_x(float): variance of two Gaussians
    Returns:
        numpy.ndarray: sample data of shape ``(N, )``
        drawn i.i.d. from p(x | theta)
    """

    a = numpy.sqrt(var_x) * numpy.random.randn(N, ) + theta1
    b = numpy.sqrt(var_x) * numpy.random.randn(N, ) + theta1 + theta2
    select = numpy.random.random_integers(0, 1, (N, ))
    return a * select + b * (1 - select)


def calc_log_posterior(theta, x, n=None):
    """Calculate unnormalized log posterior, ``log p(theta | x) + C``

    Args:
        theta(chainer.Variable): model parameters
        x(numpy.ndarray): sample data
        n(int): total data size
    Returns:
        chainer.Variable: Variable that holding unnormalized log posterior,
        ``log p(theta | x) + C`` of shape ``()``
    """

    theta1, theta2 = F.split_axis(theta, 2, 0)
    log_prior1 = F.sum(F.log(gaussian.gaussian_likelihood(theta1, 0, VAR1)))
    log_prior2 = F.sum(F.log(gaussian.gaussian_likelihood(theta2, 0, VAR2)))
    prob1 = gaussian.gaussian_likelihood(x, theta1, VAR_X)
    prob2 = gaussian.gaussian_likelihood(x, theta1 + theta2, VAR_X)
    log_likelihood = F.sum(F.log(prob1 / 2 + prob2 / 2))
    if n is not None:
        log_likelihood *= n / len(x)
    return log_prior1 + log_prior2 + log_likelihood


def calc_grad(theta, x, n=None):
    """Computes gradient of log posterior w.r.t. parameter

    Args:
        theta(numpy.ndarray): model parameters
        x(numpy.ndarray): sample data
    Returns:
        numpy.ndarray: ``dp(theta | x) / dtheta``
    """
    theta = chainer.Variable(numpy.array(theta, dtype=numpy.float32))
    log_posterior = calc_log_posterior(theta, x, n)
    theta.zerograd()
    log_posterior.backward()
    return theta.grad
