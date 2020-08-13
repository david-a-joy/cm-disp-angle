#!/usr/bin/env python3

""" Optical flow over phase movies

Process all the movies under a specific directory:

.. code-block:: bash

    $ ./calc_optical_flow.py ~/data/new_ana_movie/*

This will load all movies under ``~/data/new_ana_movie/`` and then write the
results to ``~/data/new_ana_movie_optical_flow``. Movies are any file that
ends with '.mov', '.avi', or '.mp4'.

To change the number of clusters (ex: 4 clusters):

.. code-block:: bash

    $ ./calc_optical_flow.py --n-clusters 4 ~/data/new_ana_data/*

For testing, you can also limit the number of frames analyzed (ex: the first 25 frames):

.. code-block:: bash

    $ ./calc_optical_flow.py --max-frames 25 ~/data/new_ana_data/*

To composite all the results, see ``merge_optical_flow.py``

Installing:

pip3 install numpy scipy scikit-image scikit-learn pandas matplotlib seaborn opencv-python av

"""

# Imports
import time
import pathlib
import argparse
import traceback

# Our own imports
from cm_disp_angle import calc_optical_flow
from cm_disp_angle.consts import (
    reFILENAME, SUFFIX, DOWNSAMPLE_RAW, TIME_SCALE, SPACE_SCALE, N_CLUSTERS, SUFFIXES,
)

# Command line interface


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outdir', type=pathlib.Path,
                        help='Directory to write per-movie stats out')
    parser.add_argument('--max-frames', type=int, default=-1,
                        help='Maximum frames to load from the movie (or -1 for all frames)')
    parser.add_argument('--basedir', type=pathlib.Path,
                        help='Directory where movies are stored')
    parser.add_argument('--suffix', default=SUFFIX,
                        help='Suffix to export the plots with')
    parser.add_argument('--n-clusters', default=N_CLUSTERS, type=int,
                        help='Number of clusters to split the movie into')
    parser.add_argument('--downsample-raw', default=DOWNSAMPLE_RAW, type=int,
                        help='Factor to downsample the images by')
    parser.add_argument('--time-scale', default=TIME_SCALE, type=float,
                        help=f'Number of seconds per frame (default {TIME_SCALE:0.3f} seconds/frame)')
    parser.add_argument('--space-scale', default=SPACE_SCALE, type=float,
                        help=f'Number of um per pixel (default {SPACE_SCALE:0.3f} um/pixel')
    parser.add_argument('inpaths', type=pathlib.Path, nargs='+',
                        help='List of paths to load the movies in to process')
    return parser.parse_args(args=args)


def main(args=None):
    args = vars(parse_args(args=args))
    basedir = args.pop('basedir', None)
    inpaths = list(args.pop('inpaths'))

    # Dynamically work out the base directory
    if basedir is None:
        basedir = inpaths[0].parent
        try_inpaths = [p for p in inpaths]
        while try_inpaths:
            # Check for root
            if basedir == basedir.parent:
                break
            p = try_inpaths.pop()
            try:
                _ = p.relative_to(basedir)
            except ValueError:
                try_inpaths.append(p)
                basedir = basedir.parent

    print(f'Searching paths under: {basedir}')
    outdir = args.pop('outdir', None)
    if outdir is None:
        outdir = basedir.parent / f'{basedir.name}_optical_flow'
    print(f'Writing results to: {outdir}')

    t0 = time.time()

    while inpaths:
        p = inpaths.pop()
        if p.name.startswith('.'):
            continue
        if p.is_dir():
            inpaths.extend(p.iterdir())
            continue
        if p.suffix in SUFFIXES:
            relpath = p.relative_to(basedir)
            p_outdir = outdir / relpath.parent / reFILENAME.sub('_', p.stem).strip('_')
            print(f'Loading movie from {p}')
            print(f'Saving data to {p_outdir}')
            try:
                calc_optical_flow(infile=p, outdir=p_outdir, **args)
            except Exception:
                print(f'Error processing "{p}"')
                traceback.print_exc()
    print(f'Script ran in {time.time() - t0} seconds')


if __name__ == '__main__':
    main()
