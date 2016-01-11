import chainer
from chainer import functions as F
import numpy


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

    if numpy.isscalar(x):
        x = numpy.array(x)
    if isinstance(x, numpy.ndarray):
        x = chainer.Variable(x.astype(numpy.float32))

    if numpy.isscalar(mu):
        mu = numpy.array(mu)
    if isinstance(mu, numpy.ndarray):
        mu = chainer.Variable(mu.astype(numpy.float32))

    x, mu = F.broadcast(x, mu)
    return F.exp(-(x - mu) ** 2 / var / 2) / numpy.sqrt(2 * numpy.pi * var)
