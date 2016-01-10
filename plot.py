from matplotlib import pyplot
import numpy


def visualize2D(fig, ax, xs, ys, bins=200, xlabel='x', ylabel='y'):
    H, xedges, yedges = numpy.histogram2d(xs, ys, bins)
    H = numpy.rot90(H)
    H = numpy.flipud(H)
    Hmasked = numpy.ma.masked_where(H == 0, H)

    ax.pcolormesh(xedges, yedges, Hmasked)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(min(xs), max(xs))
    ax.set_ylim(min(ys), max(ys))
    fig.colorbar(pyplot.contourf(Hmasked))
