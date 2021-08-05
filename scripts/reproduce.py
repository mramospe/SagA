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
    parser.add_argument('--colors', nargs='+', type=str, default=None, help='Colors for the particle points')
    parser.add_argument('--marker-sizes', nargs='+', type=int, default=None, help='Marker sizes for the particle points')
    parser.add_argument('--x-range', nargs=2, type=float, default=None, help='Range in X')
    parser.add_argument('--y-range', nargs=2, type=float, default=None, help='Range in Y')
    parser.add_argument('--z-range', nargs=2, type=float, default=None, help='Range in Z')
    parser.add_argument('--show-tracks', action='store_true', help='Whether to show a faint line with the track followed by the particles')
    parser.add_argument('--save', type=str, default=None, help='Name of the output file in which the animation will be saved as a GIF. Either --show or --save must be provided.')
    parser.add_argument('--show', action='store_true', help='Whether to show the animation. Either --show or --save must be provided.')

    args = parser.parse_args()

    if not args.show and not args.save:
        raise RuntimeError('You must tell the script whether you want to show the result, save it or both (via --save/--show)')

    data = np.loadtxt(args.input_file)

    n_particles = data.shape[1] // 3

    logger.info(f'Number of particles: {n_particles}')

    raw_data = pandas.DataFrame(np.zeros((data.shape[0], data.shape[1]), dtype=np.float32), columns=[f'p{i}_{c}' for i in range(n_particles) for c in ('x', 'y', 'z')])
    raw_data.loc[:,:] = data

    particle_data = [raw_data[[f'p{i}_{c}' for c in ('x', 'y', 'z')]] for i in range(n_particles)]

    for df in particle_data:
        df.columns = 'x', 'y', 'z'

    # PLOT
    fig = plt.figure()
    ax = p3.Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    def update(num, data, lines, tracks):

        if tracks is not None:
            for d, l in zip(data, tracks):
                l.set_data(d.values[:num, 0], d.values[:num, 1])
                l.set_3d_properties(d.values[:num, 2])

        for d, l in zip(data, lines):
            l.set_data(d.values[num,:2])
            l.set_3d_properties(d.values[num, 2])

    if args.colors is not None:
        assert(len(args.colors) == len(particle_data))
        colors = args.colors
    else:
        colors = len(particle_data) * ['r']

    if args.marker_sizes is not None:
        assert(len(args.marker_sizes) == len(particle_data))
        marker_sizes = args.marker_sizes
    else:
        marker_sizes = len(particle_data) * [1]

    if args.show_tracks:
        logger.info('Tracks of the particles will be shown')
        tracks = [ax.plot(d['x'][:1], d['y'][:1], d['z'][:1], linestyle='-', linewidth=0.1 * ms, color=c)[0] for d, c, ms in zip(particle_data, colors, marker_sizes)]
    else:
        tracks = None

    lines = [ax.plot(d['x'][0], d['y'][0], d['z'][0], marker='o', color=c, markersize=ms)[0] for d, c, ms in zip(particle_data, colors, marker_sizes)]

    # Setting the axes properties
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    mn = raw_data.values.min()
    mx = raw_data.values.max()

    if abs(mn) > abs(mx):
        mn = - 1.1 * abs(mn)
        mx = + 1.1 * abs(mn)
    else:
        mn = - 1.1 * abs(mx)
        mx = + 1.1 * abs(mx)

    ax.set_xlim3d(args.x_range or (mn, mx))
    ax.set_ylim3d(args.y_range or (mn, mx))
    ax.set_zlim3d(args.z_range or (mn, mx))

    logger.info('Start animation')

    ani = animation.FuncAnimation(fig, update, len(raw_data), fargs=(particle_data, lines, tracks), interval=14)

    if args.save:
        logger.info('Saving animation to an output file')
        ani.save(args.save, writer='ffmpeg', fps=30)

    if args.show:
        plt.show()
