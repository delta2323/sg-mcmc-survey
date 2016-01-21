from matplotlib import pyplot
import numpy


def visualize2D(fig, ax, xs, ys, bins=200,
                xlabel='x', ylabel='y',
                xlim=None, ylim=None):
    H, xedges, yedges = numpy.histogram2d(xs, ys, bins)
    H = numpy.rot90(H)
    H = numpy.flipud(H)
    Hmasked = numpy.ma.masked_where(H == 0, H)

    ax.pcolormesh(xedges, yedges, Hmasked)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xlim is None:
        xlim = (min(xs), max(xs))
    if ylim is None:
        ylim = (min(ys), max(ys))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    fig.colorbar(pyplot.contourf(Hmasked))
