""" Shared constants across the modules """

# Imports
import re
import pathlib

# Constants
reFILENAME = re.compile(r'[^a-z0-9_-]+', re.IGNORECASE)

SUFFIX = '.png'

BASEDIR = pathlib.Path.home() / 'data' / 'new_ana_movie'

DOWNSAMPLE_RAW = 1  # Downsample factor for the raw image frames

MIN_VEL_MAG = 0.1  # Minimum velocity magnitude (um/second)
MIN_ANG_MAG = 0.25  # Minimum mag for angle calculations (um/second)

MAX_VEL_MAG = 15.0  # Maximum velocity magnitude (um/second)
MAX_DISP_MAG = 5.0  # Maximum cumulative displacement (um)

FIGSIZE = (8, 8)  # Size of the output image plots
PALETTE = 'dark'  # Color palette for the lines

TIME_SCALE = 1.0/20.0  # seconds per frame
SPACE_SCALE = 1.0/3.0769  # um per pixel

SAMPLES_AROUND_PEAK = 3  # Minimum samples around each peak in the signal

SMOOTH_SIGMA = 3  # Sigma of the gaussian to smooth the input images with
SMOOTH_HALFWIDTH = 1  # How many samples forward and back to smooth over (1 = 3 frames of smoothing, 2 = 5 frames, etc)

N_CLUSTERS = 4  # Number of clusters for the labeling process

SUFFIXES = ('.mov', '.avi', '.mp4')  # Suffixes for the files to process
