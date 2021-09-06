#!/bin/env python
"""
Create an animation displaying the positions of particles at different points
in time.
"""
import argparse
import collections
import numpy as np
import pandas
import logging
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import pyplot as plt
from matplotlib import animation

AnimationStep = collections.namedtuple('AnimationStep', ['epoch', 'data'])


class data_reader_iterator(object):
    def __init__(self, filename, chunk_size=1000):
        """
        Iterator that efficiently reads data from a file in chunks, in such a way
        that in each iteration the resulting DataFrame object contains all the
        information of an epoch
        """
        dtype = [(n, np.int32) for n in ('epoch', )
                 ] + [(n, np.float32)
                      for n in ('x', 'y', 'z', 'px', 'py', 'pz', 'radius')]
        self.__cache = None
        self.__generator = pandas.read_csv(filename,
                                           dtype=dtype,
                                           sep=' ',
                                           header=None,
                                           names=[name for name, _ in dtype],
                                           chunksize=chunk_size,
                                           iterator=True)
        self.__epoch_counter = 0
        self.__stop_iteration = False

    def __next__(self):
        """
        DataFrame object of the next epoch
        """
        if self.__stop_iteration:
            raise StopIteration()

        if self.__cache is not None and len(self.__cache):

            current = self.__cache[self.__cache['epoch'] ==
                                   self.__epoch_counter]

            if len(current) != len(self.__cache):
                self.__cache = self.__cache[
                    self.__cache['epoch'] != self.__epoch_counter]
                self.__epoch_counter += 1
                return AnimationStep(self.__epoch_counter - 1, current)
            else:
                self.__cache = None
        else:
            current = None

        old_epoch = self.__epoch_counter
        while self.__epoch_counter == old_epoch:

            try:
                data = next(self.__generator)
            except StopIteration:
                self.__stop_iteration = True

            if current is None:
                current = data[data['epoch'] == self.__epoch_counter]
            else:
                current = pandas.concat(
                    [current, data[data['epoch'] == self.__epoch_counter]])

            if self.__cache is None:
                self.__cache = data[data['epoch'] != self.__epoch_counter]
            else:
                self.__cache = pandas.concat([
                    self.__cache, data[data['epoch'] != self.__epoch_counter]
                ])

            if len(self.__cache):
                self.__epoch_counter += 1

        return AnimationStep(self.__epoch_counter - 1, current)


class data_reader(object):
    def __init__(self, *args, **kwargs):
        """
        Create an iterable object that allows to read data in chunks
        """
        self.__args = args
        self.__kwargs = kwargs

    def __iter__(self):
        """
        Return the iterator object
        """
        return data_reader_iterator(*self.__args, **self.__kwargs)


if __name__ == '__main__':

    logging.basicConfig(format='%(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_file', type=str, help='Data file to read')
    parser.add_argument('--marker-sizes',
                        nargs='+',
                        type=int,
                        default=None,
                        help='Marker sizes for the particle points')
    parser.add_argument('--box-view',
                        nargs=2,
                        type=float,
                        default=None,
                        help='Range in X, Y an Z')
    parser.add_argument(
        '--chunk-size',
        type=float,
        default=None,
        help='Number of rows to read from the data file each time')
    parser.add_argument(
        '--show-tracks',
        action='store_true',
        help=
        'Whether to show a faint line with the track followed by the particles'
    )
    parser.add_argument('--track-size',
                        type=float,
                        default=5e-3,
                        help='Size of the tracks')
    parser.add_argument('--figure-size',
                        type=float,
                        default=8,
                        help='Size of the figure')
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help=
        'Name of the output file in which the animation will be saved as a GIF. Either --show or --save must be provided.'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help=
        'Whether to show the animation. Either --show or --save must be provided.'
    )

    args = parser.parse_args()

    if not args.show and not args.save:
        raise RuntimeError(
            'You must tell the script whether you want to show the result, save it or both (via --save/--show)'
        )

    logger.info(f'Input data will be read from "{args.input_file}"')

    # PLOT
    fig = plt.figure(figsize=(args.figure_size, args.figure_size))
    ax = p3.Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    # Setting the axes properties
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    def make_scatter(ax, data):

        if args.marker_sizes is not None:
            df = data[['x', 'y', 'z']]
            x, y, z = df['x'], df['y'], df['z']
            return ax.scatter(x,
                              y,
                              z,
                              marker='o',
                              s=args.marker_sizes,
                              color='k')
        else:
            df = data[['x', 'y', 'z', 'radius']]
            x, y, z, radius = df['x'], df['y'], df['z'], df['radius']
            return ax.scatter(x,
                              y,
                              z,
                              marker='o',
                              s=normalized_sizes(radius),
                              color='k')

    def make_tracks(ax, data):
        df = data[['x', 'y', 'z']]
        x, y, z = df['x'], df['y'], df['z']
        return ax.scatter(x, y, z, marker='.', s=args.track_size, color='k')

    if args.show_tracks:
        logger.info('Tracks of the particles will be shown')

    lines = None

    def update(step, ax):

        global lines

        if step.epoch == 0:  # initialze

            if args.box_view is not None:
                mn, mx = args.box_view
            else:

                logger.info(
                    f'Estimating the view bounds using {bounds_entries} rows of the input data'
                )

                mn = step.data[['x', 'y', 'z']].values.min()
                mx = step.data[['x', 'y', 'z']].values.max()

                if abs(mn) > abs(mx):
                    mn = -1.1 * abs(mn)
                    mx = +1.1 * abs(mn)
                else:
                    mn = -1.1 * abs(mx)
                    mx = +1.1 * abs(mx)

            ax.set_xlim3d((mn, mx))
            ax.set_ylim3d((mn, mx))
            ax.set_zlim3d((mn, mx))

            normalized_sizes = lambda radius: (plt.rcParams[
                'figure.dpi'] * args.figure_size * radius / (mx - mn))**2

            n_initial_points = np.count_nonzero(step.data['epoch'] == 0)

            initial = step.data[step.data['epoch'] == 0]

            if args.marker_sizes is not None:
                assert (len(args.marker_sizes) == n_initial_points)

            if lines is not None:
                lines.remove()

            lines = make_scatter(ax, step.data)

            if args.show_tracks:
                make_tracks(ax, step.data)
        else:
            lines.remove()
            lines = make_scatter(ax, step.data)

            if args.show_tracks:
                make_tracks(ax, step.data)

    logger.info('Start animation')

    dr = data_reader(args.input_file, args.chunk_size)

    ani = animation.FuncAnimation(fig,
                                  update, (d for d in dr),
                                  interval=14,
                                  fargs=(ax, ))

    if args.save:
        logger.info('Saving animation to an output file')
        ani.save(args.save, writer='ffmpeg', fps=30)

    if args.show:
        plt.show()
