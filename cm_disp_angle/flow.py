""" Main optical flow calculations

* :py:func:`calc_optical_flow`: Driver script for optical flow calculation and plotting

* :py:class:`AnalyzeFlow`: Optical flow analysis pipeline class

"""

# Imports
import shutil
import pathlib
from typing import Tuple, List, Optional

# 3rd party
import numpy as np

from scipy.integrate import simps

from skimage.feature import peak_local_max
from skimage.transform import downscale_local_mean
from skimage.filters import gaussian

from sklearn.cluster import KMeans

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

# Our own imports
from .utils import (
    calculate_stats, convert_warp_to_color, fit_kernel_smoothed_bins,
    optical_flow, read_movie, refine_signal_peaks
)
from .consts import (
    DOWNSAMPLE_RAW, MIN_VEL_MAG, MIN_ANG_MAG, SUFFIX,
    MAX_VEL_MAG, MAX_DISP_MAG, FIGSIZE, PALETTE, TIME_SCALE, SPACE_SCALE,
    SAMPLES_AROUND_PEAK, SMOOTH_SIGMA, SMOOTH_HALFWIDTH, N_CLUSTERS,
)


# Classes


class AnalyzeFlow(object):
    """ Analyze the optical flow in the data

    :param Path infile:
        The path to the input file to analyze
    :param Path outdir:
        The path to the directory to write the plots and analysis to
    :param int max_frames:
        If >0, maximum number of frames to load
    :param str suffix:
        Suffix to use when saving plots
    :param float smoothing_sigma:
        Standard deviation of the gaussian filter
    :param int half_width:
        Half width of the temporal window to smooth traces over
    :param int n_clusters:
        Number of clusters to split the time series data into
    :param float min_vel_mag:
        Minimum magnitude for a pixel to be considered moving
    :param float max_vel_mag:
        Maximum magnitude for plotting the velocities
    :param float min_ang_mag:
        Minimum velocity magnitude to use to calculate angles
    :param float max_disp_mag:
        Maximum total displacement magnitude to plot
    """

    def __init__(self,
                 infile: pathlib.Path,
                 outdir: pathlib.Path,
                 max_frames: int = -1,
                 suffix: str = SUFFIX,
                 smoothing_sigma: float = SMOOTH_SIGMA,
                 half_width: int = SMOOTH_HALFWIDTH,
                 n_clusters: int = N_CLUSTERS,
                 min_vel_mag: float = MIN_VEL_MAG,
                 max_vel_mag: float = MAX_VEL_MAG,
                 min_ang_mag: float = MIN_ANG_MAG,
                 max_disp_mag: float = MAX_DISP_MAG,
                 time_scale: float = TIME_SCALE,
                 space_scale: float = SPACE_SCALE,
                 downsample_raw: int = DOWNSAMPLE_RAW,
                 samples_around_peak: int = SAMPLES_AROUND_PEAK):

        self.infile = infile
        self.outdir = outdir

        self.suffix = suffix

        self.max_frames = max_frames

        self.smoothing_sigma = smoothing_sigma
        self.half_width = half_width

        self.n_clusters = n_clusters

        # Handle the space and time scales
        self.time_scale = time_scale
        self.space_scale = space_scale * downsample_raw

        self.min_vel_mag = min_vel_mag
        self.max_vel_mag = max_vel_mag
        self.min_ang_mag = min_ang_mag
        self.max_disp_mag = max_disp_mag

        self.downsample_raw = downsample_raw

        self.samples_around_peak = samples_around_peak

        self.figsize = FIGSIZE
        self.palette = sns.color_palette(PALETTE, n_colors=n_clusters)

        self.rows = self.cols = None
        self.xx = self.yy = None

        self.smooth_frames = []
        self.uu_frames = []
        self.vv_frames = []

        self.labels = None
        self.label_image = None

        self.timeline_mag = None
        self.timeline_cumulative_mag = None

        self.timeline_mag_peaks = None
        self.timeline_cumulative_mag_peaks = None

    @property
    def max_frame(self) -> int:
        """ Max frame index """
        return len(self.smooth_frames)

    @property
    def unique_label_ids(self) -> List:
        """ Return a list of unique label ids """
        return list(sorted(np.unique(self.labels)))

    def calc_velocities(self):
        """ Calculate all the velocities for the data """
        self.rows = self.cols = None
        self.xx = self.yy = None

        self.smooth_frames = []
        self.uu_frames = []
        self.vv_frames = []

        print(f'Loading {self.infile}')

        for i, frame in enumerate(read_movie(self.infile)):
            if self.max_frames > 0 and i >= self.max_frames:
                break
            print(f'Loading frame {i:3d} shape: {frame.shape}')
            if frame.ndim == 3:
                frame = np.mean(frame, axis=2)
            assert frame.ndim == 2

            # Smooth and downsample the frames to reduce noise
            frame = frame / 255
            smooth_frame = gaussian(frame, self.smoothing_sigma)
            down_smooth_frame = downscale_local_mean(smooth_frame, (self.downsample_raw, self.downsample_raw))

            # Store off the coordinate system from the downscaled images
            if self.rows is None and self.cols is None:
                self.rows, self.cols = down_smooth_frame.shape
                x = np.arange(0, self.cols) * self.space_scale
                y = np.arange(0, self.rows) * self.space_scale
                self.xx, self.yy = np.meshgrid(x, y)
            else:
                assert down_smooth_frame.shape == (self.rows, self.cols)

            self.smooth_frames.append(down_smooth_frame)

            if len(self.smooth_frames) < 2:
                self.uu_frames.append(np.zeros_like(down_smooth_frame))
                self.vv_frames.append(np.zeros_like(down_smooth_frame))
                continue

            # Finally, calculate optical flow
            frame1 = self.smooth_frames[i-1]
            frame2 = self.smooth_frames[i]

            flow_uu, flow_vv = optical_flow(frame1, frame2)
            self.uu_frames.append(flow_uu * self.space_scale / self.time_scale)
            self.vv_frames.append(flow_vv * self.space_scale / self.time_scale)

        # Make sure we padded all the vectors right
        assert len(self.smooth_frames) == len(self.uu_frames)
        assert len(self.smooth_frames) == len(self.vv_frames)

    def smooth_velocities(self):
        """ Do a simple temporal smoothing to reduce noise """

        print('Generating smoothed velocities...')

        self.smooth_uu_frames = []
        self.smooth_vv_frames = []

        for i in range(self.max_frame):
            i_st = max([0, i - self.half_width])
            i_ed = min([self.max_frame, i + self.half_width + 1])

            mean_uu_frame = np.mean(self.uu_frames[i_st:i_ed], axis=0)
            mean_vv_frame = np.mean(self.vv_frames[i_st:i_ed], axis=0)

            self.smooth_uu_frames.append(mean_uu_frame)
            self.smooth_vv_frames.append(mean_vv_frame)

    def accumulate_displacements(self):
        """ Calculate cumulative displacement over the time series """

        print('Generating cumulative displacements...')

        self.cumulative_uu_frames = []
        self.cumulative_vv_frames = []
        self.cumulative_mag = []

        # Accumulate the values in the frames
        uu_total = np.zeros_like(self.smooth_uu_frames[0])
        vv_total = np.zeros_like(self.smooth_vv_frames[0])

        for i in range(self.max_frame):
            # Total the displacement: ds = dv*dt
            uu_total += self.smooth_uu_frames[i] * self.time_scale
            vv_total += self.smooth_vv_frames[i] * self.time_scale

            mag_total = np.sqrt(uu_total**2, vv_total**2)

            self.cumulative_uu_frames.append(uu_total)
            self.cumulative_vv_frames.append(vv_total)
            self.cumulative_mag.append(mag_total)

    def cluster_timeseries(self):
        """ Cluster the timeseries using the magnitude """

        print(f'Clustering pixels into {self.n_clusters} clusters by magnitude...')
        smooth_mag = np.sqrt(np.array(self.smooth_uu_frames)**2 + np.array(self.smooth_vv_frames)**2)
        # Image coordinates (aka trig is hard when you're standing on your head):
        # x = -vv, y = uu
        # ang = arctan2(y, x) = arctan2(uu, -vv)
        smooth_ang = np.arctan2(np.array(self.smooth_uu_frames),
                                -np.array(self.smooth_vv_frames)) % np.pi
        smooth_mag = np.reshape(smooth_mag, (self.max_frame, -1))
        smooth_ang = np.reshape(smooth_ang, (self.max_frame, -1))

        cumulative_mag = np.reshape(np.array(self.cumulative_mag), (self.max_frame, -1))

        print(f'Got {smooth_mag.shape[1]} total traces')
        assert smooth_mag.shape[1] == smooth_ang.shape[1]
        assert smooth_mag.shape[1] == cumulative_mag.shape[1]

        # Find the peak velocity over all of time for the trace
        peak_vel = np.percentile(smooth_mag, 95, axis=0)

        # If a trace has at least one peak higher than the min, add it to the large traces
        mask = peak_vel > self.min_vel_mag
        large_mag = smooth_mag[:, mask]
        large_ang = smooth_ang[:, mask]
        large_cumulative_mag = cumulative_mag[:, mask]

        assert large_mag.ndim == 2
        assert large_mag.shape[0] == self.max_frame
        assert large_ang.shape[0] == self.max_frame
        assert large_cumulative_mag.shape[0] == self.max_frame

        # Find only the largest traces
        print(f'Got {large_mag.shape[1]} large traces')
        if large_mag.shape[1] > 10:
            tree = KMeans(n_clusters=self.n_clusters)
            self.labels = tree.fit_predict(large_mag.swapaxes(0, 1))

            label_image = np.zeros_like(mask, dtype=np.int)
            label_image[mask] = (self.labels + 1)

            self.label_image = np.reshape(label_image, (self.rows, self.cols))
        else:
            # FIXME: Not sure if this part actually works right...
            self.labels = np.zeros((0, ))
            self.label_image = np.zeros((self.rows, self.cols))

        self.large_mag = large_mag
        self.large_ang = large_ang
        self.large_cumulative_mag = large_cumulative_mag

        print('Binning vectors by angle...')
        bin_x, bin_y, kernel_x, kernel_y = fit_kernel_smoothed_bins(
            large_ang, large_mag, min_ang_mag=self.min_ang_mag)

        self.kernel_bins = bin_x
        self.kernel_bin_density = bin_y
        self.kernel_ang = kernel_x
        self.kernel_density = kernel_y

        self.kernel_label_density = []
        self.kernel_label_bin_density = []

        for label_id in self.unique_label_ids:
            print(f'Binning vectors by angle for label {label_id + 1}...')
            mask = self.labels == label_id
            if np.sum(mask) < 1:
                bin_y = np.zeros_like(self.kernel_bins)
                kernel_y = np.zeros_like(self.kernel_ang)
            else:
                label_mag = (large_mag[:, self.labels == label_id]).flatten()
                label_ang = (large_ang[:, self.labels == label_id]).flatten()

                _, bin_y, _, kernel_y = fit_kernel_smoothed_bins(
                    label_ang, label_mag, min_ang_mag=self.min_ang_mag)

            assert self.kernel_bins.shape == bin_y.shape
            assert self.kernel_ang.shape == kernel_y.shape

            self.kernel_label_bin_density.append(bin_y)
            self.kernel_label_density.append(kernel_y)

    def cluster_mag_timelines(self):
        """ Accumulate timelines for the large magnitutes """

        large_mag = self.large_mag
        large_cumulative_mag = self.large_cumulative_mag

        # Accumulate timelines for the large angles and magnitutes
        print('Averaging displacements over time...')
        self.timeline = np.arange(self.max_frame) * self.time_scale

        timeline_mag = []
        timeline_cumulative_mag = []

        for label_id in [None] + self.unique_label_ids:
            if label_id is None:
                mask = np.ones(self.labels.shape, dtype=np.bool)
                label_name = 'AllClusters'
            else:
                mask = self.labels == label_id
                label_name = f'Cluster{label_id + 1}'

            print(f'Averaging timelines for {label_name}...')
            print(f'{label_name} samples: {np.sum(mask)}')

            # Subset each of the timelines by ROI
            print('Calculating velocity stats...')
            label_mag = pd.DataFrame(calculate_stats(large_mag[:, mask], axis=1))
            assert self.timeline.shape[0] == label_mag.shape[0]

            label_mag['cluster'] = label_name
            label_mag['timepoint'] = self.timeline

            print('Calculating displacement stats...')
            label_cumulative_mag = pd.DataFrame(calculate_stats(large_cumulative_mag[:, mask], axis=1))
            assert self.timeline.shape[0] == label_cumulative_mag.shape[0]

            label_cumulative_mag['cluster'] = label_name
            label_cumulative_mag['timepoint'] = self.timeline

            timeline_mag.append(label_mag)
            timeline_cumulative_mag.append(label_cumulative_mag)

        self.timeline_mag = pd.concat(timeline_mag, ignore_index=True)
        self.timeline_cumulative_mag = pd.concat(timeline_cumulative_mag, ignore_index=True)

    def calc_peak_stats(self,
                        dataset: str = 'velocity',
                        metric: str = 'mean'):
        """ Calculate the peaks and stats for a given set of waveforms

        :param str dataset:
            Which data set to call peaks for
        :param str metric:
            Which measurement to call peaks using
        :param int samples_around_peak:
            How many samples to enforce around a peak
        """
        if dataset == 'velocity':
            data = self.timeline_mag
            if self.timeline_mag_peaks is None:
                all_peaks = []
            else:
                all_peaks = [self.timeline_mag_peaks]
        elif dataset == 'displacement':
            data = self.timeline_cumulative_mag
            if self.timeline_cumulative_mag_peaks is None:
                all_peaks = []
            else:
                all_peaks = [self.timeline_cumulative_mag_peaks]
        else:
            raise KeyError(f'Unknown data set "{dataset}"')

        print(f'Calling peaks for {metric} {dataset}...')
        all_labels = np.unique(data['cluster'])
        for label in all_labels:
            df = data[data['cluster'] == label]

            timeline = df['timepoint'].values
            signal = df[metric].values
            peak_indicies = peak_local_max(signal,
                                           min_distance=self.samples_around_peak,
                                           indices=True)
            peaks = pd.DataFrame(refine_signal_peaks(timeline, signal, peak_indicies))
            peaks['cluster'] = label
            peaks['metric'] = metric

            print(f'Got {peaks.shape[0]} peaks for cluster {label}')

            all_peaks.append(peaks)
        all_peaks = pd.concat(all_peaks, ignore_index=True)
        if dataset == 'velocity':
            self.timeline_mag_peaks = all_peaks
        elif dataset == 'displacement':
            self.timeline_cumulative_mag_peaks = all_peaks
        else:
            raise KeyError(f'Unknown data set "{dataset}"')

    def calc_velocity_peak_stats(self):
        """ Call peaks for the velocity data """
        for metric in ['mean', 'max']:
            self.calc_peak_stats('velocity', metric)

    def calc_displacement_peak_stats(self):
        """ Call peaks for the displacement data """
        for metric in ['mean', 'max']:
            self.calc_peak_stats('displacement', metric)

    def clear_outdir(self):
        """ Clean the output directory """

        if self.outdir.is_dir():
            print(f'Overwriting: {self.outdir}')
            shutil.rmtree(str(self.outdir))
        self.outdir.mkdir(parents=True)

    def extract_peaks(self,
                      peaks: Optional[pd.DataFrame],
                      cluster: str,
                      metric: str) -> Tuple[List[int]]:
        """ Pull out the peak indices for a particular cluster

        :param DataFrame peaks:
            The peak call data frame
        :param str cluster:
            The cluster to load peak calls for
        :param str metric:
            The metric to load peak calls for
        :returns:
            Two lists of low_indices, high_indices
        """
        if peaks is None:
            return [], []
        mask = np.logical_and(peaks['metric'] == metric,
                              peaks['cluster'] == cluster)
        if np.sum(mask) < 1:
            return [], []

        # Pull out the peak indices and de-duplicate
        metric_peaks = peaks[mask]
        low_peaks = set(metric_peaks['peak_start_index'].values)
        low_peaks.update(metric_peaks['peak_end_index'].values)
        low_peaks = [int(i) for i in sorted(low_peaks)]

        high_peaks = [int(i) for i in sorted(set(metric_peaks['peak_index'].values))]
        return low_peaks, high_peaks

    def plot_averaged_timeseries(self,
                                 dataset: str = 'velocity',
                                 metric: str = 'mean',
                                 markersize: float = 10):
        """ Convert the raw collections to time series traces

        :param str dataset:
            Which data set to plot the time series for
        :param str metric:
            Which measurement to plot along the time series
        """
        # Switch based on the selected data set
        if dataset == 'velocity':
            data = self.timeline_mag
            peaks = self.timeline_mag_peaks
            if metric == 'max':
                ylim = [0, self.max_vel_mag*2.5]
            else:
                ylim = [0, self.max_vel_mag*1.1]

            ylabel = 'Velocity Magnitude ($\\mu m/sec$)'
            title = f'{metric.capitalize()} Velocity Magnitude'
        elif dataset == 'displacement':
            data = self.timeline_cumulative_mag
            peaks = self.timeline_cumulative_mag_peaks
            if metric == 'max':
                ylim = [0, self.max_disp_mag*5.0]
            else:
                ylim = [0, self.max_disp_mag*1.1]
            ylabel = 'Total Displacement ($\\mu m$)'
            title = f'{metric.capitalize()} Total Displacement'
        else:
            raise KeyError(f'Unknown data set "{dataset}"')

        has_ci = f'{metric} ci low' in data.columns and f'{metric} ci high' in data.columns

        xlabel = 'Timepoint (sec)'
        xlim = [0, self.max_frame*self.time_scale]

        print(f'Generating {metric} {dataset} timeseries plots...')
        figsize_x, figsize_y = self.figsize

        outdir = self.outdir / f'{dataset}_{metric}_timeseries'
        outdir.mkdir(parents=True, exist_ok=True)

        # Make a velocity timeseries of the composite data
        outfile = outdir / f'{dataset}_{metric}_timeseries_all{self.suffix}'

        df = data[data['cluster'] == 'AllClusters']
        cluster_x = df['timepoint'].values
        cluster_y = df[metric].values
        color = 'black'

        fig, ax = plt.subplots(1, 1, figsize=(figsize_x, figsize_y))
        if has_ci:
            ax.fill_between(cluster_x, df[metric + ' ci low'], df[metric + ' ci high'],
                            color=color, alpha=0.5)
        ax.plot(cluster_x, cluster_y, color=color, linewidth=3, label='AllClusters')

        low_peaks, high_peaks = self.extract_peaks(peaks, cluster='AllClusters', metric=metric)
        for idx in low_peaks:
            ax.plot(cluster_x[idx], cluster_y[idx], 'o', color=color, markersize=markersize)
        for idx in high_peaks:
            ax.plot(cluster_x[idx], cluster_y[idx], '*', color=color, markersize=markersize)

        for label_id in self.unique_label_ids:
            label_name = f'Cluster{label_id + 1}'

            df = data[data['cluster'] == label_name]
            cluster_x = df['timepoint'].values
            cluster_y = df[metric].values
            color = self.palette[label_id]

            if has_ci:
                ax.fill_between(cluster_x, df[metric + ' ci low'], df[metric + ' ci high'],
                                color=color, alpha=0.5)
            ax.plot(cluster_x, cluster_y, color=color, label=label_name)

            low_peaks, high_peaks = self.extract_peaks(peaks, cluster=label_name, metric=metric)
            for idx in low_peaks:
                ax.plot(cluster_x[idx], cluster_y[idx], 'o', color=color, markersize=markersize)
            for idx in high_peaks:
                ax.plot(cluster_x[idx], cluster_y[idx], '*', color=color, markersize=markersize)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.legend()

        ax.set_title(title)

        fig.savefig(outfile, transparent=True)
        plt.close()

        # Make a velocity timeseries for all the clusters only
        outfile = outdir / f'{dataset}_{metric}_timeseries_clusters{self.suffix}'

        fig, ax = plt.subplots(1, 1, figsize=(figsize_x, figsize_y))

        for label_id in self.unique_label_ids:
            label_name = f'Cluster{label_id + 1}'

            df = data[data['cluster'] == label_name]
            cluster_x = df['timepoint'].values
            cluster_y = df[metric].values
            color = self.palette[label_id]

            if has_ci:
                ax.fill_between(cluster_x, df[metric + ' ci low'], df[metric + ' ci high'],
                                color=color, alpha=0.5)
            ax.plot(cluster_x, cluster_y, color=color, label=label_name)

            low_peaks, high_peaks = self.extract_peaks(peaks, cluster=label_name, metric=metric)
            for idx in low_peaks:
                ax.plot(cluster_x[idx], cluster_y[idx], 'o', color=color, markersize=markersize)
            for idx in high_peaks:
                ax.plot(cluster_x[idx], cluster_y[idx], '*', color=color, markersize=markersize)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.legend()

        ax.set_title(title)

        fig.savefig(outfile, transparent=True)
        plt.close()

        # Make plots for the individual clusters
        outfile = outdir / f'{dataset}_{metric}_timeseries_all_clusters{self.suffix}'

        df = data[data['cluster'] == 'AllClusters']
        cluster_x = df['timepoint'].values
        cluster_y = df[metric].values
        color = 'black'

        fig, ax = plt.subplots(1, 1, figsize=(figsize_x, figsize_y))
        if has_ci:
            ax.fill_between(cluster_x, df[metric + ' ci low'], df[metric + ' ci high'],
                            color=color, alpha=0.5)
        ax.plot(cluster_x, cluster_y, color=color, linewidth=3, label='AllClusters')

        low_peaks, high_peaks = self.extract_peaks(peaks, cluster='AllClusters', metric=metric)
        for idx in low_peaks:
            ax.plot(cluster_x[idx], cluster_y[idx], 'o', color=color, markersize=markersize)
        for idx in high_peaks:
            ax.plot(cluster_x[idx], cluster_y[idx], '*', color=color, markersize=markersize)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.set_title('All Clusters ' + title)

        fig.savefig(outfile, transparent=True)
        plt.close()

        for label_id in self.unique_label_ids:
            outfile = outdir / f'{dataset}_{metric}_timeseries_cluster{label_id+1:02d}{self.suffix}'

            label_name = f'Cluster{label_id + 1}'

            fig, ax = plt.subplots(1, 1, figsize=(figsize_x, figsize_y))

            label_name = f'Cluster{label_id + 1}'

            df = data[data['cluster'] == label_name]
            cluster_x = df['timepoint'].values
            cluster_y = df[metric].values
            color = self.palette[label_id]

            if has_ci:
                ax.fill_between(cluster_x, df[metric + ' ci low'], df[metric + ' ci high'],
                                color=color, alpha=0.5)
            ax.plot(cluster_x, cluster_y, color=color, label=label_name)

            low_peaks, high_peaks = self.extract_peaks(peaks, cluster=label_name, metric=metric)
            for idx in low_peaks:
                ax.plot(cluster_x[idx], cluster_y[idx], 'o', color=color, markersize=markersize)
            for idx in high_peaks:
                ax.plot(cluster_x[idx], cluster_y[idx], '*', color=color, markersize=markersize)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            ax.set_title(f'{label_name} {title}')

            fig.savefig(outfile, transparent=True)
            plt.close()

    def plot_velocity_timeseries(self):
        """ Generate velocity timeseries plots """

        for metric in ['mean', 'max']:
            self.plot_averaged_timeseries(dataset='velocity', metric=metric)

    def plot_displacement_timeseries(self):
        """ Plot the timeseries for the displacement data """

        for metric in ['mean', 'max']:
            self.plot_averaged_timeseries(dataset='displacement', metric=metric)

    def plot_cluster_label_image(self):
        """ Plot the label key for the cluster image """

        figsize_x, figsize_y = self.figsize

        # Label the ROIs to match the clusters
        label_colors = np.zeros(self.label_image.shape + (3, ))
        for label_id in self.unique_label_ids:
            label_colors[self.label_image == label_id + 1, :] = self.palette[label_id]

        # Make the image of the cluster labels only
        outfile = self.outdir / f'cluster_image{self.suffix}'

        fig, ax = plt.subplots(1, 1, figsize=(figsize_x, figsize_y))
        ax.imshow(label_colors)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.savefig(outfile, transparent=True)
        plt.close()

    def plot_angle_distribution(self):
        """ Plot the distribution of angles """
        figsize_x, figsize_y = 8, 8

        # Plot everything togethe
        outfile = self.outdir / 'angle' / f'all_bins{self.suffix}'
        outfile.parent.mkdir(parents=True, exist_ok=True)

        fig = plt.figure(figsize=(figsize_x, figsize_y))
        ax = fig.add_subplot(1, 1, 1, projection='polar')
        ax.plot(self.kernel_ang, self.kernel_density, '-', color='black', linewidth=4)
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        ax.set_xticks(np.pi/180. * np.linspace(0,  180, 7, endpoint=True))
        ax.set_yticks([])

        fig.savefig(outfile, transparent=True)
        plt.close()

        # Plot all the clusters on a polar plot
        outfile = self.outdir / 'angle' / f'all_clusters_polar{self.suffix}'
        outfile.parent.mkdir(parents=True, exist_ok=True)

        fig = plt.figure(figsize=(figsize_x, figsize_y))
        ax = fig.add_subplot(1, 1, 1, projection='polar')

        ax.plot(self.kernel_ang, self.kernel_density, '-', color='black', linewidth=4,
                label='All Clusters')
        for label_id, label_density in enumerate(self.kernel_label_density):
            ax.plot(self.kernel_ang, label_density, '-', color=self.palette[label_id],
                    label=f'Cluster {label_id+1}')
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        ax.set_xticks(np.pi/180. * np.linspace(0,  180, 7, endpoint=True))
        ax.set_yticks([])

        ax.legend()

        fig.savefig(outfile, transparent=True)
        plt.close()

        # Plot all the clusters on a linear space, to verify that the areas make sense
        outfile = self.outdir / 'angle' / f'all_clusters_flat{self.suffix}'
        outfile.parent.mkdir(parents=True, exist_ok=True)

        fig = plt.figure(figsize=(figsize_x, figsize_y))
        ax = fig.add_subplot(1, 1, 1)

        label_area = simps(self.kernel_density, self.kernel_ang)
        print(f'Overall Distribution Area: {label_area:0.5f}')
        ax.plot(self.kernel_ang, self.kernel_density, '-', color='black', linewidth=4,
                label='All Clusters')

        for label_id, label_density in enumerate(self.kernel_label_density):
            label_area = simps(label_density, self.kernel_ang)
            print(f'Label {label_id} Area: {label_area:0.5f}')
            ax.plot(self.kernel_ang, label_density, '-', color=self.palette[label_id],
                    label=f'Cluster {label_id+1}')

        ax.set_xlim([-0.1, np.pi+0.1])
        ax.set_xticks(np.pi/180. * np.linspace(0,  180, 7, endpoint=True))
        ax.set_xticklabels([f'{t:0.0f}' for t in np.linspace(0,  180, 7, endpoint=True)])

        ax.set_xlabel('Direction (deg)')
        ax.set_ylabel('Probability Density')

        ax.legend()

        fig.savefig(outfile, transparent=True)
        plt.close()

        # Only plot the main cluster
        outfile = self.outdir / 'angle' / f'main_cluster{self.suffix}'
        outfile.parent.mkdir(parents=True, exist_ok=True)

        fig = plt.figure(figsize=(figsize_x, figsize_y))
        ax = fig.add_subplot(1, 1, 1, projection='polar')

        ax.bar(self.kernel_bins, self.kernel_bin_density)
        ax.plot(self.kernel_ang, self.kernel_density, '-', color='black', linewidth=2)

        ax.set_thetamin(0)
        ax.set_thetamax(180)
        ax.set_xticks(np.pi/180. * np.linspace(0,  180, 7, endpoint=True))
        ax.set_yticks([])

        fig.savefig(outfile, transparent=True)
        plt.close()

        # Plot each cluster individually
        for label_id, label_density in enumerate(self.kernel_label_density):
            outfile = self.outdir / 'angle' / f'cluster{label_id:02d}{self.suffix}'
            outfile.parent.mkdir(parents=True, exist_ok=True)

            fig = plt.figure(figsize=(figsize_x, figsize_y))
            ax = fig.add_subplot(1, 1, 1, projection='polar')

            ax.bar(self.kernel_bins, self.kernel_label_bin_density[label_id])
            ax.plot(self.kernel_ang, label_density, '-', color=self.palette[label_id])

            ax.set_thetamin(0)
            ax.set_thetamax(180)
            ax.set_xticks(np.pi/180. * np.linspace(0,  180, 7, endpoint=True))
            ax.set_yticks([])

            fig.savefig(outfile, transparent=True)
            plt.close()

    def plot_velocities(self):
        """ Make the velocity plots """

        figsize_x, figsize_y = self.figsize

        for i, frame in enumerate(self.smooth_frames):

            flow_uu = self.smooth_uu_frames[i]
            flow_vv = self.smooth_vv_frames[i]

            flow_color = convert_warp_to_color(flow_uu, flow_vv)
            outfile = self.outdir / 'frame_color_split' / f'frame{i:03d}.tif'
            outfile.parent.mkdir(parents=True, exist_ok=True)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize_x*2, figsize_y))
            ax1.imshow(frame, cmap='gray')
            ax1.set_xticks([])
            ax1.set_yticks([])

            ax2.imshow(flow_color)
            ax2.set_xticks([])
            ax2.set_yticks([])
            fig.savefig(outfile, transparent=True)
            plt.close()

            outfile = self.outdir / 'color' / f'frame{i:03d}.tif'
            outfile.parent.mkdir(parents=True, exist_ok=True)

            fig, ax = plt.subplots(1, 1, figsize=(figsize_x, figsize_y))
            ax.imshow(flow_color)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.savefig(outfile, transparent=True)
            plt.close()

            flow_color_alpha = np.sqrt(flow_uu**2 + flow_vv**2) / self.max_vel_mag
            flow_color_comp = flow_color * flow_color_alpha[:, :, np.newaxis]

            composite = np.stack([frame, frame, frame], axis=2)
            composite += flow_color_comp
            composite[composite > 1] = 1
            composite[composite < 0] = 0

            outfile = self.outdir / 'frame_color_merge' / f'frame{i:03d}.tif'
            outfile.parent.mkdir(parents=True, exist_ok=True)

            fig, ax = plt.subplots(1, 1, figsize=(figsize_x, figsize_y))
            ax.imshow(composite)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.savefig(outfile, transparent=True)
            plt.close()

    def save_angle_distribution(self):
        """ Write out the angle distribution table """

        print('Saving angle distribution...')
        outfile = self.outdir / 'angle_dist.xlsx'
        outfile.parent.mkdir(parents=True, exist_ok=True)

        # Write out the angle clusters for each image
        df = {
            'Angle': self.kernel_ang,
            'AllClusters': self.kernel_density,
        }
        for label_id, label_density in enumerate(self.kernel_label_density):
            df[f'Cluster{label_id+1:02d}'] = label_density

        df = pd.DataFrame(df)
        df.to_excel(str(outfile))

    def save_timeline_distribution(self):
        """ Write out the timeline distribution and cumulative distribution """
        print('Saving velocity distribution...')

        # Write out the displacement magnitude clusters for each image
        if self.timeline_mag is not None:
            outfile = self.outdir / 'timeline_mag.xlsx'
            self.timeline_mag.to_excel(str(outfile), index=False)

        if self.timeline_mag_peaks is not None:
            outfile = self.outdir / 'timeline_mag_peaks.xlsx'
            self.timeline_mag_peaks.to_excel(str(outfile), index=False)

        # Write out the cumulative displacement clusters for each image
        print('Saving displacement distribution...')

        outfile = self.outdir / 'timeline_cumulative_mag.xlsx'
        if self.timeline_cumulative_mag is not None:
            self.timeline_cumulative_mag.to_excel(str(outfile), index=False)

        if self.timeline_cumulative_mag_peaks is not None:
            outfile = self.outdir / 'timeline_cumulative_mag_peaks.xlsx'
            self.timeline_cumulative_mag_peaks.to_excel(str(outfile), index=False)

# Main function


def calc_optical_flow(*args, **kwargs):
    """ Calculate the optical flow and plot it for a movie

    :param Path infile:
        The path to the input file to analyze
    :param Path outdir:
        The path to the directory to write the plots and analysis to
    :param int max_frames:
        If >0, maximum number of frames to load
    """
    proc = AnalyzeFlow(*args, **kwargs)
    proc.calc_velocities()
    proc.smooth_velocities()
    proc.accumulate_displacements()
    proc.cluster_timeseries()

    # proc.cluster_mag_timelines()

    # proc.calc_velocity_peak_stats()
    # proc.calc_displacement_peak_stats()

    # Plots
    proc.clear_outdir()

    proc.plot_angle_distribution()
    # proc.plot_velocities()
    # proc.plot_cluster_label_image()
    # proc.plot_velocity_timeseries()
    # proc.plot_displacement_timeseries()

    # Output data
    proc.save_angle_distribution()
    # proc.save_timeline_distribution()
