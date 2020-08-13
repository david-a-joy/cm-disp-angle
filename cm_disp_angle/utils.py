""" Utility functions for calculating flow results

* :py:func:`bootstrap_ci`: Use bootstrap sampling to estimate confidence intervals
* :py:func:`calculate_stats`: Calculate summary stats for the samples
* :py:func:`calc_stats_around_peak`: Run peak calling on the samples
* :py:func:`convert_warp_to_color`: Convert a pair of velocity images to a color image
* :py:func:`fit_kernel_smoothed_bins`: Fit samples into bins with kernel smoothing
* :py:func:`optical_flow`: Actually calculate optical flow between two images
* :py:func:`read_movie`: Read in a movie frame by frame
* :py:func:`refine_signal_peaks`: Refine the peak calling on the samples

"""
# Imports
import pathlib
from typing import Tuple, List, Generator, Dict, Callable, Optional

# 3rd party
import numpy as np

from scipy.integrate import simps
from scipy.stats import gaussian_kde

from skimage.color import hsv2rgb

import cv2

import av

# Our own imports
from .consts import MIN_VEL_MAG, MAX_VEL_MAG, MIN_ANG_MAG

# Functions


def read_movie(infile: pathlib.Path) -> Generator[np.ndarray, None, None]:
    """ Read a movie file in

    :param Path infile:
        The input movie to read
    :returns:
        A generator yielding numpy arrays, one per frame, in order
    """
    container = av.open(str(infile), mode='r')
    for frame in container.decode(video=0):
        yield frame.to_ndarray(format='rgb24')


def optical_flow(img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray]:
    """ Calculate optical flow between two images

    :param ndarray img1:
        First image in the sequence
    :param ndarray img2:
        Second image in the sequence
    :returns:
        Optical flow field as a pair of x, y velocity vectors
    """
    assert img1.ndim == 2
    assert img1.shape == img2.shape
    if img1.dtype != np.uint8:
        min1, max1 = np.min(img1), np.max(img1)
        if min1 < 0 or max1 > 1:
            err = f'Need a floating point image1 between 0 and 1, got {min1} to {max1}'
            raise ValueError(err)
        img1 = np.round(img1*255).astype(np.uint8)

    if img2.dtype != np.uint8:
        min2, max2 = np.min(img2), np.max(img2)
        if min2 < 0 or max2 > 1:
            err = f'Need a floating point image2 between 0 and 1, got {min2} to {max2}'
            raise ValueError(err)
        img2 = np.round(img2*255).astype(np.uint8)

    flow = cv2.calcOpticalFlowFarneback(
        prev=img1,
        next=img2,
        flow=None,
        pyr_scale=0.5,  # Each pyramid is half the size of the previous
        levels=7,  # How many levels of pyramid to calculate
        winsize=13,  # Size of the pixel window - larger is more blurry
        iterations=5,  # Number of iterations at each pyramid level
        poly_n=7,  # Width of the polynomial approximation - typically 5 or 7 - larger is more blurry
        poly_sigma=1.5,  # Sigma of the polynomial approximation - 1.1 to 1.5 is good
        flags=0)
    return flow[:, :, 0], flow[:, :, 1]


def convert_warp_to_color(uu: np.ndarray,
                          vv: np.ndarray,
                          min_vel_mag: float = MIN_VEL_MAG,
                          max_vel_mag: float = MAX_VEL_MAG) -> np.ndarray:
    """ Convert a warp matrix into a color matrix

    :param ndarray uu:
        The x-vectors for the flow image
    :param ndarray vv:
        The y-vectors for the flow image
    :param float min_vel_mag:
        Minimum vector magnitude to plot
    :param float max_vel_mag:
        Maximum vector magnitude to plot
    :returns:
        A color image where angle corresponds to hue and magnitude corresponds
        to value on the HSV color scale
    """
    # Convert to hsv
    mag = np.sqrt(uu**2 + vv**2)
    ang = np.arctan2(vv, uu)

    hue = (ang % np.pi) / np.pi  # hue between 0-1
    sat = np.ones_like(hue)

    # If no bounds passed, clamp by percentile
    if min_vel_mag is None:
        min_vel_mag = np.percentile(mag, 5)
    if max_vel_mag is None:
        max_vel_mag = np.percentile(mag, 95)

    # Rescale magnitude to between 0 and 1
    value = (mag - min_vel_mag) / (max_vel_mag - min_vel_mag)
    value[value < 0] = 0
    value[value > 1] = 1

    return hsv2rgb(np.stack((hue, sat, value), axis=2))


def calculate_stats(values: np.ndarray,
                    axis: int = 1,
                    n_boot: int = 200,
                    n_samples: int = 1000) -> Dict[str, float]:
    """ Calculate all the stats for a single bin

    :param ndarray values:
        The 2D array of n timepoints x k samples
    :param int axis:
        Whiich axis of the matrix corresponds to samples
    :param int n_boot:
        How many iterations of bootstrap to run
    :returns:
        A dictionary of stats for plotting
    """
    # Figure out how many samples to use for bootstrapping
    n_samples = min([values.shape[axis], n_samples])

    # Calculate mean/std/min max
    bin_mean = np.nanmean(values, axis=axis)
    bin_std = np.nanstd(values, axis=axis)
    bin_min = np.nanmin(values, axis=axis)
    bin_max = np.nanmax(values, axis=axis)

    # Calculate quartiles
    bin5, bin25, bin50, bin75, bin95 = np.nanpercentile(values, [5, 25, 50, 75, 95], axis=axis)

    # Estimate confidence intervals using boostrapping
    print(f'Generating mean confidence intervals with {n_samples} samples and {n_boot} rounds...')
    bin_mean_ci0, bin_mean_ci1 = bootstrap_ci(values, func=np.nanmean, axis=axis,
                                              n_boot=n_boot, n_samples=n_samples)
    assert bin_mean_ci0.shape == bin_mean.shape
    assert bin_mean_ci1.shape == bin_mean.shape

    bin_n = np.sum(np.isfinite(values), axis=axis)

    # Stash all the values for later
    return {
        'mean': bin_mean,
        'mean ci low': bin_mean_ci0,
        'mean ci high': bin_mean_ci1,
        'std': bin_std,
        'p5': bin5,
        'p25': bin25,
        'p50': bin50,
        'p75': bin75,
        'p95': bin95,
        'min': bin_min,
        'max': bin_max,
        'n samples': bin_n,
    }


def bootstrap_ci(data: np.ndarray,
                 n_boot: int = 1000,
                 n_samples: Optional[int] = None,
                 random_seed: Optional[int] = None,
                 ci: float = 95,
                 func: Callable = np.mean,
                 axis: int = 0) -> Tuple[np.ndarray]:
    """ Calculate a confidence interval from the input data using bootstrapping

    :param ndarray data:
        The data to bootstrap sample
    :param int n_boot:
        Number of times to sample the frame
    :param int n_samples:
        If not None, the number of samples to draw for each round (default: length of the sample axis)
    :param int random_seed:
        Seed for the random number generator
    :param float ci:
        Confidence interval to calculate (mean +/- ci/2.0)
    :param Callable func:
        Function to calculate the ci around (default: np.mean)
    :param int axis:
        Which axis to sample over
    :returns:
        The upper and lower bounds on the CI
    """
    if n_samples is None:
        n_samples = data.shape[axis]
    rs = np.random.RandomState(random_seed)
    boot_dist = []
    for i in range(n_boot):
        resampler = rs.randint(0, data.shape[axis], n_samples)
        sample = data.take(resampler, axis=axis)
        boot_dist.append(func(sample, axis=axis))
    boot_dist = np.array(boot_dist)
    return np.percentile(boot_dist, [50 - ci/2, 50 + ci/2], axis=0)


def fit_kernel_smoothed_bins(angle: np.ndarray,
                             magnitude: np.ndarray,
                             width: int = 5,
                             min_ang_mag: float = MIN_ANG_MAG) -> Tuple[np.ndarray]:
    """ Fit a bin and kernel smooth approximation to the angle/magnitude calculation

    :param ndarray angle:
        Angle (radians) for the velocity vector
    :param ndarray magnitude:
        Magnitude (pixels) for the velocity vector
    :param int width:
        Bin width (degrees) for the bins on the degree plot
    :param float min_ang_mag:
        Minimum velocity magnitude to use to calculate angles
    :returns:
        x, y coords of the bins, x, y coords of the kernel smoothed fit
    """
    ang_mask = magnitude > min_ang_mag

    # Split the angles up into bins
    ang_bins = np.linspace(0, 180, 180//width+1, endpoint=True) / 180 * np.pi  # rads
    bin_x = ang_bins[:-1]
    kernel_x = np.linspace(0, 180, 180, endpoint=False) / 180 * np.pi  # rads

    # If we didn't get enough measurements, just return the null distribution
    if np.sum(ang_mask) < 10:
        return bin_x, np.zeros_like(bin_x), kernel_x, np.zeros_like(kernel_x)

    # Pull out the magnitudes for large moving pixels
    ang_stack = angle[ang_mask]
    mag_stack = magnitude[ang_mask]

    # Bin all the values
    ang_vals = np.zeros((ang_bins.shape[0] - 1))
    for i, (bin_st, bin_ed) in enumerate(zip(ang_bins[:-1], ang_bins[1:])):
        bin_mask = np.logical_and(ang_stack >= bin_st, ang_stack < bin_ed)
        ang_vals[i] += np.sum(bin_mask)

    # Average the bins over the total stack size
    bin_y = ang_vals / mag_stack.shape[0]

    # Kernel smooth the distribution
    kernel = gaussian_kde(ang_stack)
    kernel_y = kernel(kernel_x)

    # Rescale for total area == 1.0
    rad_width = width / 180 * np.pi  # Radians because width is in degrees
    hist_area = np.sum(rad_width * bin_y)
    kernel_area = simps(kernel_y, kernel_x)

    kernel_y = kernel_y / kernel_area
    bin_y = bin_y / hist_area

    return bin_x, bin_y, kernel_x, kernel_y


def refine_signal_peaks(time: np.ndarray,
                        signal: np.ndarray,
                        peaks: List[int],
                        valley_rel: float = 0.05,
                        min_peak_width: int = 1,
                        min_peak_height: float = 0.0,
                        offset: int = 0) -> List[int]:
    """ Refine the raw peak indicies for the signal

    :param ndarray time:
        The n x 1 time array
    :param ndarray signal:
        The n x 1 signal array
    :param list peaks:
        The list of peak indices called by ``peak_local_max``
    :param float valley_rel:
        What relative fraction of the peak to call the valley floor
    :param min_peak_width:
        Minimum width (in samples) of a peak
    :param min_peak_height:
        Minimum height of a peak (in signal intensity)
    :param int offset:
        Starting temporal index for this dataset
    :returns:
        A new list of peaks, possibly with some maxima combined
    """

    # Augment the peaks with the beginning and end indices of the signal
    peaks = list(peaks)
    if 0 not in peaks:
        peaks.append(0)
    if signal.shape[0]-1 not in peaks:
        peaks.append(signal.shape[0]-1)
    peaks = list(sorted(int(p) for p in peaks))

    # Try to fuse peaks together while we have multiple peaks to fuse
    while len(peaks) >= 3:
        new_peaks = [0]
        final_peaks = []

        for peak_bounds in zip(peaks[:-2], peaks[1:-1], peaks[2:]):
            mid = peak_bounds[1]
            stats = calc_stats_around_peak(time, signal, peak_bounds,
                                           valley_rel=valley_rel)

            peak_start_index = stats['peak_start_index']
            peak_end_index = stats['peak_end_index']

            peak_height = stats['peak_height']

            # If the peak is wide enough and tall enough, keep it
            if all([peak_start_index < mid,
                    peak_end_index > mid,
                    peak_end_index - peak_start_index > min_peak_width,
                    peak_height > min_peak_height]):
                new_peaks.append(mid)

            # Handle mask offset for the indicies
            final_stats = {}
            for key in stats:
                if key.endswith('_index'):
                    final_stats[key] = stats[key] + offset
                else:
                    final_stats[key] = stats[key]

            final_peaks.append(final_stats)

        # Once we converge to a fixed point, return
        new_peaks.append(signal.shape[0]-1)
        if new_peaks == peaks:
            return final_peaks
        peaks = new_peaks

    # Failure
    return []


def calc_stats_around_peak(time: np.ndarray,
                           signal: np.ndarray,
                           peak_bounds: Tuple[int],
                           valley_rel: float = 0.05) -> Dict:
    """ Calculate the stats around a single peak

    :param ndarray time:
        The time vector for the signal
    :param ndarray signal:
        The signal vector
    :param tuple peak_bounds:
        A tuple of start, peak, stop bounds
    :param float valley_rel:
        The relative value of the bottom of a valley
    :param float time_scale:
        The conversion of the time scale to seconds
    :param bool skip_model_fit:
        If True, skip fitting a decay curve model to the data
    :returns:
        The dictionary of stats for the peak
    """

    start_idx, peak_idx, end_idx = peak_bounds

    peak_value = signal[peak_idx]

    before_peak = signal[start_idx:peak_idx+1]
    after_peak = signal[peak_idx:end_idx+1]

    # Work out the indicies of the min point and the 5% threshold
    before_min_index = np.argmin(before_peak)
    after_min_index = np.argmin(after_peak)

    before_min_value = before_peak[before_min_index]
    after_min_value = after_peak[after_min_index]

    before_cutoff = valley_rel * (peak_value - before_min_value) + before_min_value
    after_cutoff = valley_rel * (peak_value - after_min_value) + after_min_value

    before_cutoff = np.max([before_cutoff, before_min_value])
    after_cutoff = np.max([after_cutoff, after_min_value])

    # Find all the indicies below the threshold
    before_peak_locs = np.nonzero(before_peak <= before_cutoff)[0]
    after_peak_locs = np.nonzero(after_peak <= after_cutoff)[0]
    before_peak_locs = np.append(before_peak_locs, before_min_index)
    after_peak_locs = np.append(after_peak_locs, after_min_index)

    peak_start_index = np.max(before_peak_locs)
    peak_start_index += start_idx
    peak_end_index = np.min(after_peak_locs)
    peak_end_index += peak_idx

    # Get stats for total time
    total_wave_time = time[peak_end_index] - time[peak_start_index]

    # Peak height
    peak_height = min([signal[peak_idx] - signal[peak_end_index],
                       signal[peak_idx] - signal[peak_start_index]])
    return {
        'peak_value': peak_value,
        'peak_height': peak_height,
        'peak_index': peak_idx,
        'peak_time': time[peak_idx],
        'peak_start_index': peak_start_index,
        'peak_end_index': peak_end_index,
        'total_wave_time': total_wave_time,
    }
