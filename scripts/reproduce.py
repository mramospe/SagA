#!/bin/env python
"""
Create an animation displaying the positions of particles at different points
in time.
"""
import argparse
import numpy as np
import pandas
import logging
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import pyplot as plt
from matplotlib import animation


if __name__ == '__main__':

    logging.basicConfig(format='%(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_file', type=str, help='Data file to read')
    parser.add_argument('--marker-sizes', nargs='+', type=int, default=None, help='Marker sizes for the particle points')
    parser.add_argument('--box-view', nargs=2, type=float, default=None, help='Range in X, Y an Z')
    parser.add_argument('--show-tracks', action='store_true', help='Whether to show a faint line with the track followed by the particles')
    parser.add_argument('--track-size', type=float, default=5e-3, help='Size of the tracks')
    parser.add_argument('--figure-size', type=float, default=8, help='Size of the figure')
    parser.add_argument('--save', type=str, default=None, help='Name of the output file in which the animation will be saved as a GIF. Either --show or --save must be provided.')
    parser.add_argument('--show', action='store_true', help='Whether to show the animation. Either --show or --save must be provided.')

    args = parser.parse_args()

    if not args.show and not args.save:
        raise RuntimeError('You must tell the script whether you want to show the result, save it or both (via --save/--show)')

    logger.info(f'Reading input data from "{args.input_file}"')

    df = pandas.DataFrame.from_records(np.loadtxt(args.input_file, dtype=[(n, np.int32) for n in ('epoch',)] + [(n, np.float32) for n in  ('x', 'y', 'z', 'px', 'py', 'pz', 'radius')]))

    # PLOT
    fig = plt.figure(figsize=(args.figure_size, args.figure_size))
    ax = p3.Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    # Setting the axes properties
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if args.box_view is not None:
        mn, mx = args.box_view
    else:
        mn = df[['x', 'y', 'z']].values.min()
        mx = df[['x', 'y', 'z']].values.max()

        if abs(mn) > abs(mx):
            mn = - 1.1 * abs(mn)
            mx = + 1.1 * abs(mn)
        else:
            mn = - 1.1 * abs(mx)
            mx = + 1.1 * abs(mx)

    ax.set_xlim3d((mn, mx))
    ax.set_ylim3d((mn, mx))
    ax.set_zlim3d((mn, mx))

    normalized_sizes = lambda radius: (plt.rcParams['figure.dpi'] * args.figure_size * radius / (mx - mn))**2

    def scatter_for_index(ax, data, index):

        if args.marker_sizes is not None:
            df = data[['x', 'y', 'z']][data['epoch'] == index]
            x, y, z = df['x'], df['y'], df['z']
            return ax.scatter(x, y, z, marker='o', s=args.marker_sizes, color='k')
        else:
            df = data[['x', 'y', 'z', 'radius']][data['epoch'] == index]
            x, y, z, radius = df['x'], df['y'], df['z'], df['radius']
            return ax.scatter(x, y, z, marker='o', s=normalized_sizes(radius), color='k')

    def tracks_for_index(ax, data, index):
        df = data[['x', 'y', 'z']][data['epoch'] <= index]
        x, y, z = df['x'], df['y'], df['z']
        return ax.scatter(x, y, z, marker='.', s=args.track_size, color='k')

    def update(num, ax, data):

        global lines
        global tracks

        if tracks is not None:
            tracks.remove()
            tracks = tracks_for_index(ax, data, num)

        lines.remove()
        lines = scatter_for_index(ax, data, num)

    n_initial_points = np.count_nonzero(df['epoch'] == 0)

    initial = df[df['epoch'] == 0]

    if args.marker_sizes is not None:
        assert(len(args.marker_sizes) == n_initial_points)

    if args.show_tracks:
        logger.info('Tracks of the particles will be shown')
        tracks = tracks_for_index(ax, df, 0)
    else:
        tracks = None

    lines = scatter_for_index(ax, df, 0)

    logger.info('Start animation')

    ani = animation.FuncAnimation(fig, update, np.max(df['epoch']), fargs=(ax, df), interval=14)

    if args.save:
        logger.info('Saving animation to an output file')
        ani.save(args.save, writer='ffmpeg', fps=30)

    if args.show:
        plt.show()
