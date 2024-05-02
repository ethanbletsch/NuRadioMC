from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units, interferometry
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import stationParameters as stp

from radiotools import helper as hp, coordinatesystems
from radiotools.atmosphere import models, refractivity

from NuRadioReco.framework.event import Event
from NuRadioReco.framework.station import Station
from NuRadioReco.framework.base_shower import BaseShower
from NuRadioReco.detector.detector_base import DetectorBase

import numpy as np
import scipy
import sys
import matplotlib.pyplot as plt
from matplotlib import gridspec, colorbar
from scipy.optimize import curve_fit

from typing import Optional
from collections import defaultdict

from tqdm import tqdm
from os import cpu_count

import logging
logger = logging.getLogger(
    'NuRadioReco.efieldRITReconstruction')
"""
This module hosts to classes
    - efieldInterferometricDepthReco
    - efieldInterferometricAxisReco
    - efieldInterferometricLateralReco

The radio-interferometric reconstruction (RIT) was proposed in [1].
The implementation here is based on work published in [2].
It is a rewrite of the original module!

[1]: H. Schoorlemmer, W. R. Carvalho Jr., arXiv:2006.10348
[2]: F. Schlueter, T. Huege, doi:10.1088/1748-0221/16/07/P07048

"""


class efieldInterferometricDepthReco:
    """
    This class reconstructs the depth of the maximum of the longitudinal profile of the
    beam-formed radio emission: X_RIT along a given axis. X_RIT is found to correlate with X_max.
    This correlation may depend on the zenith angle, the frequency band, and detector layout
    (if the detector is very irregular).

    For the reconstruction, a shower axis is needed. The radio emission in the vxB polarisation is used
    (and thus the arrival direction is needed).

    """

    def __init__(self):
        self._debug = None
        self._at = None
        self._tab = None
        self._use_channels = None
        self._signal_kind = None
        self._signal_threshold = None
        self._traces = None
        self._times = None
        self._positions = None
        self._axis = None
        self._core = None
        self._depths = None
        self._binsize = None
        self._tstep = None
        self._mc_jitter = None
        self._cs = None
        self._shower = None
        self._zenith = None
        self._nsampling = None
        self._use_sim_pulses = None
        self._data = {}

    def set_geometry(self, shower: BaseShower, core: Optional[np.ndarray] = None,
                     axis: Optional[np.ndarray] = None, smear_angle_radians: float = 0, smear_core_meter: float = 0):
        assert not (core is not None and smear_core_meter != 0)
        assert not (axis is not None and smear_angle_radians != 0)

        if core is None:
            core = shower[shp.core]
            if len(core) == 3:
                core[2] = 0

        if axis is None:
            axis = hp.spherical_to_cartesian(
                shower[shp.zenith], shower[shp.azimuth])

        if smear_core_meter:
            cs = coordinatesystems.cstrafo(
                *hp.cartesian_to_spherical(*axis), shower[shp.magnetic_field_vector])
            core_vxB = np.random.normal((0, 0), smear_core_meter)
            core = cs.transform_from_vxB_vxvxB_2D(core_vxB, core=core)

        if smear_angle_radians:
            concentration_parameter = 1 / smear_angle_radians**2
            dist = vonmises_fisher()
            axis = dist.rvs(axis, concentration_parameter)

        self._core = core.reshape((3,))
        self._axis = axis.reshape((3,))
        self._zenith = hp.get_angle(np.array([0, 0, 1]), self._axis)
        self._shower = shower
        assert np.round(
            self._core[-1], 8) == np.round(shower[shp.observation_level], 8)
        self._cs = coordinatesystems.cstrafo(
            *hp.cartesian_to_spherical(*self._axis), shower[shp.magnetic_field_vector])

    def begin(self,
              shower: BaseShower,
              core: Optional[np.ndarray] = None,
              axis: Optional[np.ndarray] = None,
              n_sampling: int = 256,
              use_sim_pulses: bool = False,
              use_channels: bool = False,
              debug: bool = False):
        """
        Set module config.

        Parameters
        ----------

        signal_kind : str
            Define which signal "metric" is used on the beamformed traces. Default "power" : sum over the squared amplitudes in a 100 ns window around the peak.
            Other options are "amplitude" or "hilbert_sum"

        relative_signal_treshold: float (default = 0.)
           Fraction of strongest signal necessary for a trace to be used for RIT. Default of 0 includes all channels.

        depths: np.ndarray (default: units.g / units.cm2 * np.arange(400,800,10))
            depths to coarsely sample the axis with, in internal units

        mc_jitter: float (default: 0.)
            Timing jitter to put on traces, in internal units (==nanoseconds)

        debug : bool
            If true, show some debug plots (Default: False).

        """
        self._debug = debug
        self._nsampling = n_sampling
        self._use_channels = use_channels
        self._use_sim_pulses = use_sim_pulses

        self.set_geometry(shower, core, axis)
        self.update_atmospheric_model_and_refractivity_table(shower)

    def sample_longitudinal_profile(self, depths: np.ndarray):
        """
        Returns the longitudinal profile of the interferometic signal sampled along the shower axis.

        Parameters
        ----------

        traces : array(number_of_antennas, samples)
            Electric field traces (one polarisation of it, usually vxB) for all antennas/stations.


        times : array(number_of_antennas, samples)
            Time vectors corresponding to the electric field traces.

        station_positions : array(number_of_antennas, 3)
            Position of each antenna.

        shower_axis : array(3,)
            Axis/direction along which the interferometric signal is sampled. Anchor is "core".

        core : array(3,)
            Shower core. Keep in mind that the altitudes (z-coordinate) matters.

        depths : array (optinal)
            Define the positions (slant depth along the axis) at which the interferometric signal is sampled.
            Instead of "depths" you can provide "distances".

        distances : array (optinal)
            Define the positions (geometrical distance from core along the axis) at which the interferometric signal is sampled.
            Instead of "distances" you can provide "depths".

        Returns
        -------

        signals : array
            Interferometric singals sampled along the given axis
        """

        signals = np.zeros(len(depths))
        for idx, depth in enumerate(depths):
            try:
                dist = self._at.get_distance_xmax_geometric(
                    self._zenith, depth, observation_level=self._shower[shp.observation_level])
            except ValueError:
                logger.info(
                    "ValueError in get_distance_xmax_geometric, setting signal to 0")
                signals[idx] = 0
                continue

            if dist < 0:
                signals[idx] = 0
                continue

            point_on_axis = self._axis * dist + self._core
            sum_trace = interferometry.interfere_traces_interpolation(
                point_on_axis, self._positions, self._traces, self._times, tab=self._tab)

            # plt.title(dod)
            # plt.plot(sum_trace)
            # plt.show()

            signal = interferometry.get_signal(
                sum_trace, self._tstep, kind=self._signal_kind)
            signals[idx] = signal

        return signals

    def reconstruct_interferometric_depth(self, return_profile=False):
        """
        Returns Gauss-parameters fitted to the "peak" of the interferometic
        longitudinal profile along the shower axis.

        A initial samping range and size in defined by "lower_depth", "upper_depth", "bin_size".
        However if the "peak", i.e., maximum signal is found at an edge the sampling range in
        continually increased (with a min/max depth of 0/2000 g/cm^2). The Gauss is fitted around the
        found peak with a refined sampling (use 20 samples in this narrow range).

        Parameters
        ----------

        traces : array(number_of_antennas, samples)
            Electric field traces (one polarisation of it, usually vxB) for all antennas/stations.

        times : array(number_of_antennas, samples)
            Time vectors corresponding to the electric field traces.

        station_positions : array(number_of_antennas, 3)
            Position of each antenna.

        shower_axis : array(3,)
            Axis/direction along which the interferometric signal is sampled. Anchor is "core".

        core : array(3,)
            Shower core. Keep in mind that the altitudes (z-coordinate) matters.

        lower_depth : float
            Define the lower edge for the inital sampling (default: 400 g/cm2).

        upper_depth : float
            Define the upper edge for the inital sampling (default: 800 g/cm2).

        bin_size : float
            Define the step size pf the inital sampling (default: 100 g/cm2).
            The refined sampling around the peak region is / 10 this value.

        return_profile : bool
            If true return the sampled profile in addition to the Gauss parameter (default: False).

        Returns
        -------

        If return_profile is True

            depths_corse : np.array
                Depths along shower axis coarsely sampled

            depths_fine : np.array
                Depths along shower axis finely sampled (used in fitting)

            signals_corese : np.array
                Beamformed signals along shower axis coarsely sampled

            signals_fine : np.array
                Beamformed signals along shower axis finely sampled (used in fitting)

            popt : list
                List of fitted Gauss parameters (amplitude, position, width)

        If return_profile is False:

            popt : list
                List of fitted Gauss parameters (amplitude, position, width)

        """
        signals = self.sample_longitudinal_profile(self._depths)

        # if max signal is at the upper edge add points there
        if np.argmax(signals) == len(self._depths) - 1:
            while True:
                depth_add = np.amax(self._depths) + self._binsize
                signal_add = self.sample_longitudinal_profile([depth_add])
                self._depths = np.hstack((self._depths, depth_add))
                signals = np.append(signals, signal_add)

                if not np.argmax(signals) == len(
                        self._depths) - 1 or depth_add > 2000:
                    break

        # if max signal is at the lower edge add points there
        elif np.argmax(signals) == 0:
            while True:
                depth_add = np.amin(self._depths) - self._binsize
                signal_add = self.sample_longitudinal_profile([depth_add])
                self._depths = np.hstack((depth_add, self. _depths))
                signals = np.append(signal_add, signals)

                if not np.argmax(signals) == 0 or depth_add <= 0:
                    break

        idx_max = np.argmax(signals)
        dtmp_max = self._depths[idx_max]

        depths_fine = np.linspace(
            dtmp_max - 30,
            dtmp_max + 30,
            20)  # 3 g/cm2 bins
        signals_fine = self.sample_longitudinal_profile(depths_fine)

        def normal(x, A, x0, sigma):
            """ Gauss curve """
            return A / np.sqrt(2 * np.pi * sigma ** 2) \
                * np.exp(-1 / 2 * ((x - x0) / sigma) ** 2)

        popt, _ = curve_fit(normal, depths_fine, signals_fine, p0=[np.amax(
            signals_fine), depths_fine[np.argmax(signals_fine)], 100], maxfev=1000)
        xrit = popt[1]

        if return_profile:
            return depths_fine, signals, signals_fine, popt

        return xrit

    @register_run()
    def run(self,
            evt: Event,
            det: Optional[DetectorBase] = None,
            station_ids: Optional[list] = None, signal_kind="power",
            relative_signal_treshold: float = 0.,
            depths: np.ndarray = np.arange(400, 800, 10) * units.g / units.cm2,
            mc_jitter: float = 0 * units.ns,
            ):
        """
        Run interferometric reconstruction of depth of coherent signal.

        Parameters
        ----------

        evt : Event
            Event to run the module on.

        det : Detector
            Detector description

        shower: BaseShower
            Shower to extract geometry and atmospheric information from. Conventional: `evt.get_first_shower()` or `evt.get_first_sim_shower()`

        use_mc_pulses : bool
            if true, take electric field trace from sim_station

        station_ids: Optional[list] (default: None)
            station_ids whose channels will be read out. For all stations, use `evt.get_station_ids()`

        mc_jitter: Optional[float] (with unit of time, default: None)
            Standard deviation of Gaussian noise added to timings, if set.

        shower_axis, core: np.ndarray (3,)
            Geometry to be used to reconstruct XRIT; ignores the geometry of `shower` (but keeps magnetic field vector)
        """
        self._signal_kind = signal_kind
        self._signal_threshold = relative_signal_treshold

        self._depths = depths / units.g * units.cm2
        self._binsize = self._depths[1] - self._depths[0]
        self._mc_jitter = mc_jitter / units.ns

        self.set_station_data(evt, det, station_ids=station_ids)

        if not self._debug:
            xrit = self.reconstruct_interferometric_depth()
        else:
            depths_final, signals_tmp, signals_final, rit_parameters = \
                self.reconstruct_interferometric_depth(return_profile=True)
            xrit = rit_parameters[1]
            ax = plt.figure().add_subplot()
            ax.scatter(self._depths, signals_tmp, color="blue",
                       label="signals_tmp", s=2, zorder=1.1)
            ax.scatter(depths_final, signals_final, color="red",
                       label="signals_final", s=2, zorder=1)
            ax.plot(depths_final, normal(depths_final, *rit_parameters),
                    label="gauss fit", color="black", ls="--")
            ax.axvline(rit_parameters[1])
            ax.set_xlabel("slant depth [g/cm2]")
            ax.set_ylabel(self._signal_kind)
            ax.legend()
            plt.show()

        self._shower.set_parameter(
            shp.interferometric_shower_maximum,
            xrit * units.g / units.cm2)

    def end(self):
        pass

    def update_atmospheric_model_and_refractivity_table(
            self, shower: BaseShower):
        """
        Updates model of the atmosphere and tabulated, integrated refractive index according to shower properties.

        Parameters
        ----------

        shower : BaseShower
        """
        logger.warning(
            "flat earth geometry assumed. default was curved. If issue has been fixed, consider moving back to curved")
        curved = False

        if self._at is None:
            self._at = models.Atmosphere(
                shower[shp.atmospheric_model], curved=curved)
            self._tab = refractivity.RefractivityTable(
                self._at.model, refractivity_at_sea_level=shower[shp.refractive_index_at_ground] - 1, curved=curved)

        elif self._at.model != shower[shp.atmospheric_model]:
            self._at = models.Atmosphere(
                shower[shp.atmospheric_model], curved=curved)
            self._tab = refractivity.RefractivityTable(
                self._at.model, refractivity_at_sea_level=shower[shp.refractive_index_at_ground] - 1, curved=curved)

        elif self._tab._refractivity_at_sea_level != shower[shp.refractive_index_at_ground] - 1:
            self._tab = refractivity.RefractivityTable(
                self._at.model, refractivity_at_sea_level=shower[shp.refractive_index_at_ground] - 1, curved=curved)

    def set_station_data(self, evt: Event, det: Optional[DetectorBase], station_ids: Optional[list] = None):
        """
        Set station data (positions, traces and timing of traces) available to the module. Can handle both electric fields and channels (voltages); for the latter a detector is required.

        Parameters
        ----------

        evt : Event

        det: DetectorBase

        station_ids: Optional[list] (default: None)
            List of requested station_ids. If None, all stations are used.

        """

        traces = []
        times = []
        pos = []
        if station_ids is None:
            station_ids = evt.get_station_ids()

        if self._use_channels:
            dominant_polarisations = [evt.get_station(sid)[stp.cr_dominant_polarisation] for sid in station_ids if self._use_channels]
            unique, counts = np.unique(dominant_polarisations)
            strongest_pol_overall = unique[np.argmax(counts)]

            positions_and_times_and_traces = []
            for sid in station_ids:
                station: Station = evt.get_station(sid)
                station_position = det.get_absolute_position(sid)
                positions_and_times_and_traces += [((station_position
                                                     + det.get_relative_position(sid, cid)),
                                                    station.get_channel(cid).get_times(),
                                                    station.get_channel(cid).get_trace())
                                                   for cid in station.get_channel_ids()
                                                   if det.get_antenna_orientation(sid, cid) == strongest_pol_overall]
        else:
            positions_and_times_and_traces = []
            for sid in station_ids:
                station: Station = evt.get_station(sid)

                if self._use_sim_pulses:
                    station = station.get_sim_station()

                # if det is not None:
                #     station_position = det.get_absolute_position(sid)

                #     positions_and_times_and_traces += [(station_position + np.mean([det.get_relative_position(sid, cid) for cid in electric_field.get_channel_ids()], axis=0),
                #                                         electric_field.get_times(),
                #                                         self._cs.transform_to_vxB_vxvxB(electric_field.get_trace())[0])
                #                                        for electric_field in station.get_electric_fields()]
                positions_and_times_and_traces += [(electric_field.get_position(),
                                                    electric_field.get_times(),
                                                    self._cs.transform_to_vxB_vxvxB(electric_field.get_trace())[0])
                                                   for electric_field in station.get_electric_fields()]

            for position, time, trace in positions_and_times_and_traces:
                if self._use_sim_pulses and self._mc_jitter > 0:
                    time += np.random.normal(scale=self._mc_jitter)
                if self._nsampling is not None:
                    hw = self._nsampling // 2
                    m = np.argmax(np.abs(trace))

                    if m < hw:
                        m = hw
                    if m > len(trace) - hw:
                        m = len(trace) - hw

                    trace = trace[m - hw:m + hw]
                    time = time[m - hw:m + hw]

                traces.append(trace)
                times.append(time)
                pos.append(position)

        traces = np.array(traces)
        times = np.array(times)
        logger.warning(f"Adding observation level to all positions. Makes sense for detector centered coordinate system (eg. LOFAR CS002 at approx. [0,0,0]) when reading the positions (here: {bool(self.det)}). When reading positions from efields ({not bool(det)}), ")
        pos = np.array(pos) + self._shower[shp.observation_level]
        assert not np.all(pos[:, :2] == 0)  # efield positions are set to [0, 0, 0] in voltageToEfieldConverter. This should protect against such behaviour.

        if self._signal_threshold > 0:
            flu = np.sum(traces ** 2, axis=-1)
            mask = (flu >= self._signal_threshold * np.max(flu))
            logger.info(f"{np.round(np.sum(mask) / len(mask) * 100, 3)}% of trace_vector used for RIT with relative fluence above {self._signal_threshold}")

            if self._debug:
                ax = plt.figure().add_subplot()
                import matplotlib as mpl
                cmap = mpl.colormaps.get_cmap("viridis")
                ax.scatter(*(pos[mask].T[:2, :]), c=flu[mask], cmap=cmap, s=1)
                ax.scatter(*(pos[~mask].T[:2, :]), c="red",
                           s=1, marker="x", label="excluded")
                ax.set_aspect("equal")
                ax.legend()
                plt.show()

            traces = traces[mask]
            times = times[mask]
            pos = pos[mask]

        self._traces = traces
        self._times = times
        self._tstep = self._times[0, 1] - self._times[0, 0]
        self._positions = pos

        if self._use_sim_pulses:
            cs_shower = coordinatesystems.cstrafo(
                self._shower[shp.zenith], self._shower[shp.azimuth], magnetic_field_vector=self._shower[shp.magnetic_field_vector])
            logger.debug(f"self._positions shape: {self._positions.shape}")
            pos_showerplane = cs_shower.transform_to_vxB_vxvxB(
                self._positions, core=self._shower[shp.core])
            max_vxB_baseline_proxy = np.amax(np.abs(pos_showerplane[:, 0]))
            max_vxvxB_baseline_proxy = np.amax(np.abs(pos_showerplane[:, 1]))
            self._data["max_vxB_baseline"] = max_vxB_baseline_proxy * units.m
            self._data["max_vxvxB_baseline"] = max_vxvxB_baseline_proxy * units.m


class efieldInterferometricAxisReco(efieldInterferometricDepthReco):
    """
    Class to reconstruct the shower axis with beamforming.
    """

    def __init__(self):
        super().__init__()
        self._multiprocessing = None

    def begin(self,
              shower: BaseShower,
              core: Optional[np.ndarray] = None,
              axis: Optional[np.ndarray] = None,
              n_sampling: int = 256,
              use_sim_pulses: bool = False,
              use_channels: bool = False,
              core_spread: float = 10 * units.m,
              axis_spread: float = 1*units.deg,
              multiprocessing: bool = False,
              sample_angular_resolution: float = 0.005*units.deg,
              initial_grid_spacing: float = 60*units.m,
              cross_section_width: float = 1000*units.m,
              refine_axis: bool = False,
              debug: bool = False
              ):
        """
        Set module config.

        Parameters
        ----------

        signal_kind : str
            Define which signal "metric" is used on the beamformed traces. Default "power" : sum over the squared amplitudes in a 100 ns window around the peak.
            Other options are "amplitude" or "hilbert_sum"

        relative_signal_treshold: float (default = 0.)
           Fraction of strongest signal necessary for a trace to be used for RIT. Default of 0 includes all channels.

        depths: np.ndarray (default: units.g / units.cm2 * np.arange(400,800,10))
            depths to coarsely sample the axis with, in internal units

        mc_jitter: float (default: 0.)
            Timing jitter to put on traces, in internal units (==nanoseconds)

        debug : bool
            If true, show some debug plots (Default: False).

        """
        self._debug = debug
        self._refine_axis = refine_axis
        self._nsampling = n_sampling
        self._use_sim_pulses = use_sim_pulses
        self._use_channels = use_channels
        self._multiprocessing = multiprocessing
        self._angres = sample_angular_resolution / units.radian
        self._initial_grid_spacing = initial_grid_spacing / units.m
        self._cross_section_width = cross_section_width / units.m

        if core is not None:
            logger.warning("Passed core, setting core spread to 0.")
            core_spread = 0
        if axis is not None:
            logger.warning("Passed axis, setting axis spread to 0.")
            axis_spread = 0

        self.set_geometry(shower, core=core, axis=axis, smear_angle_radians=axis_spread *
                          units.radian, smear_core_meter=core_spread / units.m)
        self.update_atmospheric_model_and_refractivity_table(shower)

    def find_maximum_in_plane(self, xs_showerplane, ys_showerplane, p_axis, cs):
        """
        Sample interferometric signals in 2-d plane (vxB-vxvxB) perpendicular to a given axis on a rectangular/quadratic grid.
        The orientation of the plane is defined by the radiotools.coordinatesytem.cstrafo argument.

        Parameters
        ----------

        xs : array
            x-coordinates defining the sampling positions.

        ys : array
            y-coordinates defining the sampling positions.

        p_axis : array(3,)
            Origin of the 2-d plane along the axis.

        station_positions : array(number_of_antennas, 3)
            Position of each antenna.

        traces : array(number_of_antennas, samples)
            Electric field traces (one polarisation of it, usually vxB) for all antennas/stations.

        times : array(number_of_antennas, samples)
            Time vectors corresponding to the electric field traces.

        cs : radiotools.coordinatesytem.cstrafo

        Returns
        -------

        idx : int
            Index of the entry with the largest signal (np.argmax(signals))

        signals : array(len(xs), len(ys))
            Interferometric signal

        """
        def yiteration(xdx, x):
            # for xdx, x in enumerate(tqdm(xs)):
            signals = np.zeros(len(ys_showerplane))
            for ydx, y in enumerate(ys_showerplane):
                p = p_axis + cs.transform_from_vxB_vxvxB(np.array([x, y, 0]))

                sum_trace = interferometry.interfere_traces_interpolation(
                    p, self._positions, self._traces, self._times, tab=self._tab)

                # if False:
                #     from scipy.signal import hilbert
                #     fig = plt.figure()
                #     ax = fig.add_subplot()
                #     ax.plot(np.abs(sum_trace), color="r", label="sum_trace")
                #     ax.plot(np.abs(hilbert(sum_trace)), color="b",
                #             label="hilbert(trace)", ls="--")
                #     # ax.plot((np.abs(sum_trace) - np.abs(hilbert(sum_trace))), color="k", label="|sum_trace| - |hilbert|")
                #     ax.legend()
                #     plt.show()
                #     sys.exit()

                signal = interferometry.get_signal(
                    sum_trace, self._tstep, kind=self._signal_kind)
                signals[ydx] = signal
            return signals

        if self._multiprocessing:
            try:
                from joblib import Parallel, delayed
                signals = Parallel(n_jobs=max(min(cpu_count() // 2, len(xs_showerplane)), 2))(
                    delayed(yiteration)(xdx, x) for xdx, x in enumerate(xs_showerplane))
            except ImportError:
                logger.warning("Could not import joblib, single process instead")
                self._multiprocessing = False

        if not self._multiprocessing:
            signals = []
            for xdx, x in enumerate(xs_showerplane):
                signals.append(yiteration(xdx, x))
        signals = np.vstack(signals)
        idx = np.argmax(signals)
        return idx, signals

    def get_paxis_dr_target(self, depth: float, core: np.ndarray, axis: np.ndarray):
        zenith, azimuth = hp.cartesian_to_spherical(*axis)

        dist = self._at.get_distance_xmax_geometric(
            zenith, depth, observation_level=core[-1])
        dr_ref_target = np.tan(self._angres) * dist
        p_axis = axis * dist + core
        return p_axis, dr_ref_target

    def sample_lateral_cross_section(self,
                                     depth: float,
                                     core: np.ndarray,
                                     axis: np.ndarray,
                                     cross_section_width: float,
                                     initial_grid_spacing: float):

        zenith, azimuth = hp.cartesian_to_spherical(*axis)
        cs = coordinatesystems.cstrafo(
            zenith, azimuth, magnetic_field_vector=self._shower[shp.magnetic_field_vector])

        p_axis, dr_ref_target = self.get_paxis_dr_target(depth, core, axis)
        max_dist = cross_section_width / 2 + initial_grid_spacing

        if self._use_sim_pulses:
            # we use the true core to make sure that it is within the inital search gri
            shower_axis = hp.spherical_to_cartesian(
                self._shower[shp.zenith], self._shower[shp.azimuth])
            mc_at_plane = interferometry.get_intersection_between_line_and_plane(
                axis, p_axis, shower_axis, self._shower[shp.core])
            # gives interserction between a plane normal to the shower axis initial guess (shower_axis_inital)
            # anchored at a point in this vB plane at the requested height/depth along the initial axis (p_axis),
            # with the true/montecarlo shower axis anchored at the true/mc core
            cs_shower = coordinatesystems.cstrafo(
                self._shower[shp.zenith], self._shower[shp.azimuth], self._shower[shp.magnetic_field_vector])
            # could instead use p_axis if no mc available?
            mc_vB = cs_shower.transform_to_vxB_vxvxB(mc_at_plane, core=p_axis)

            max_mc_vB_coordinate = np.max(np.abs(mc_vB))
            if max_dist < max_mc_vB_coordinate:
                logger.warning(f"MC axis does not intersect plane to be sampled around p_axis at {depth} g/cm2! " +
                               "Extending the plane to include MC axis. " +
                               f"Consider increasing cross section size by at least a factor {max_mc_vB_coordinate / max_dist: .2f}, since this warning will not appear for real data;)")
                max_dist = np.max(np.abs(mc_vB)) + initial_grid_spacing

        xlims = np.array([-max_dist, max_dist]) + np.random.uniform(-0.1 *
                                                                    initial_grid_spacing, 0.1 * initial_grid_spacing, 2)
        ylims = np.array([-max_dist, max_dist]) + np.random.uniform(-0.1 *
                                                                    initial_grid_spacing, 0.1 * initial_grid_spacing, 2)
        xs = np.arange(xlims[0], xlims[1] +
                       initial_grid_spacing, initial_grid_spacing)
        ys = np.arange(ylims[0], ylims[1] +
                       initial_grid_spacing, initial_grid_spacing)

        iloop = 0
        xh, yh, sh = [], [], []  # history
        while True:
            idx, signals = self.find_maximum_in_plane(xs, ys, p_axis, cs)

            xh.append(xs)
            yh.append(ys)
            sh.append(signals)

            if self._debug:
                plot_lateral_cross_section(
                    xs, ys, signals, mc_vB, title=r"%.1f$\,$g$\,$cm$^{-2}$" % depth)
            iloop += 1

            # maximum
            x_max = xs[int(idx // len(ys))]
            y_max = ys[int(idx % len(ys))]

            # update range / grid
            dx = xs[1] - xs[0]
            dy = ys[1] - ys[0]

            dr = np.sqrt(dx ** 2 + dy ** 2)
            if iloop == 10 or dr < dr_ref_target:
                break

            if iloop >= 2:
                dx /= 2
                dy /= 2
            xs = np.linspace(x_max - dx, x_max + dx, 5)
            ys = np.linspace(y_max - dy, y_max + dy, 5)

        weight = np.amax(signals)

        xfound = xs[int(idx // len(ys))]
        yfound = ys[int(idx % len(ys))]

        point_found = p_axis + \
            cs.transform_from_vxB_vxvxB(np.array([xfound, yfound, 0]))

        ground_grid_uncertainty = cs.transform_from_vxB_vxvxB_2D(
            np.array([dx, dy])/np.sqrt(12))
        return point_found, weight, ground_grid_uncertainty

    def reconstruct_shower_axis(self):
        """
        Run interferometric reconstruction of the shower axis. Find the maxima of the interferometric signals
        within 2-d plane (slices) along a given axis (initial guess). Through those maxima (their position in the
        atmosphere) a straight line is fitted to reconstruct the shower axis.

        traces : array(number_of_antennas, samples)
            Electric field traces (one polarisation of it, usually vxB) for all antennas/stations.

        times : array(number_of_antennas, samples)
            Time vectors corresponding to the electric field traces.

        station_positions : array(number_of_antennas, 3)
            Position of each antenna.

        shower_axis_inital : array(3,)
            Axis/direction which is used as initial guess for the true shower axis.
            Around this axis the interferometric signals are sample on 2-d planes.

        core : array(3,)
            Shower core which is used as initial guess. Keep in mind that the altitudes (z-coordinate) matters.

        magnetic_field_vector : array(3,)
            Magnetic field vector of the site you are using.

        is_mc : bool
            If true, interprete the provided shower axis as truth and add some gaussian smearing to optain an
            inperfect initial guess for the shower axis (Default: True).

        depths: Optional[list] (default: None)
            slant depths at which to sample lateral profiles. None results in [500, 600, 700, 800, 900, 1000].

        initial_grid_spacing : double
            Spacing of your grid points in meters (Default: 60m)

        cross_section_size : double
            Side length on the 2-d planes (slice) along which the maximum around the initial axis is sampled in meters
            (Default: 1000m).

        """

        found_points = []
        sigma_points = []
        weights = []

        for depth in tqdm(self._depths):
            found_point, weight, ground_grid_uncertainty = self.sample_lateral_cross_section(
                depth, self._core, self._axis, self._cross_section_width, self._initial_grid_spacing)

            found_points.append(found_point)
            sigma_points.append(ground_grid_uncertainty)
            weights.append(weight)

        # extend to new depths if max is found at edges of self._depths
        counter = 0
        while True:
            if np.argmax(weights) != 0 or counter >= 10:
                break

            new_depth = self._depths[0] - self._binsize
            logger.info("extend to", new_depth)
            found_point, weight, ground_grid_uncertainty = self.sample_lateral_cross_section(
                new_depth, self._core, self._axis, self._cross_section_width, self._initial_grid_spacing)

            self._depths = np.hstack(([new_depth], self._depths))
            found_points = [found_point] + found_points
            weights = [weight] + weights
            sigma_points = [np.array(ground_grid_uncertainty)] + sigma_points
            counter += 1

        counter = 0
        while True:
            if np.argmax(weights) != len(weights) or counter >= 10:
                break

            new_depth = self._depths[-1] + self._binsize
            logger.info("extend to", new_depth)
            found_point, weight, ground_grid_uncertainty = self.sample_lateral_cross_section(
                new_depth, self._core, self._axis, self._cross_section_width, self._initial_grid_spacing)

            self._depths = np.hstack((self._depths, [new_depth]))
            found_points.append(found_point)
            weights.append(weight)
            sigma_points.append(ground_grid_uncertainty)
            counter += 1

        direction_rec, core_rec, opening_angle_sph, opening_angle_sph_std, core_std = self.fit_axis(
            found_points, weights, self._axis, self._core, full_output=True)
        logger.info(f"core: {list(np.round(core_rec, 3))} +- {list(np.round(core_std, 3))} m")

        if self._use_sim_pulses:
            logger.info(f"Opening angle with MC: {np.round(opening_angle_sph / units.deg, 3)} +- {np.round(opening_angle_sph_std / units.deg, 3)} deg")

        # add smaller planes sampled along inital rit axis to increase amount of points to fit final rit axis
        if self._refine_axis:
            refinement = 4
            depths2 = np.linspace(
                self._depths[0], self._depths[-1], refinement*len(self._depths))
            for depth in tqdm([d for d in depths2 if d not in self._depths]):
                found_point, weight, ground_grid_uncertainty = self.sample_lateral_cross_section(
                    depth, core_rec, direction_rec, self._cross_section_width / 4, self._cross_section_width / 20)
                found_points.append(found_point)
                weights.append(weight)
                sigma_points.append(ground_grid_uncertainty)

            direction_rec, core_rec, opening_angle_sph, opening_angle_sph_std, core_std = self.fit_axis(
                found_points, weights, direction_rec, core_rec, full_output=True)
            logger.info(f"core (refined): {list(np.round(core_rec, 3))} +- {list(np.round(core_std, 3))} m")

        if self._use_sim_pulses and self._refine_axis:
            logger.info(f"Opening angle with MC (refined): {np.round(opening_angle_sph / units.deg, 3)} +- {np.round(opening_angle_sph_std / units.deg, 3)} deg")
        if self._use_sim_pulses and self._refine_axis and self._debug:
            plot_shower_axis_points(
                np.array(found_points), np.array(weights), self._shower)

        self._data["core"] = {"opt": core_rec *
                              units.m, "std": core_std * units.m}
        self._data["opening_angle_mc"] = {
            "opt": opening_angle_sph * units.rad, "std": opening_angle_sph_std * units.rad}
        self._data["found_points"] = np.array(found_points)
        self._data["weights"] = np.array(weights)
        return direction_rec, core_rec

    def fit_axis(self, points, weights, axis0, core0, full_output: bool = False):
        points = np.array(points)
        weights = np.array(weights)

        # popt, pcov = curve_fit(interferometry.fit_axis, points[:, -1], points.flatten(),
        #                        sigma=sigma_points.flatten(), p0=[*hp.cartesian_to_spherical(*axis0), 0, 0], absolute_sigma=True)
        popt, pcov = curve_fit(interferometry.fit_axis, points[:, -1], points.flatten(),
                               sigma=np.amax(weights) / np.repeat(weights, 3), p0=[*hp.cartesian_to_spherical(*axis0), core0[0], core0[1]])
        direction_rec = hp.spherical_to_cartesian(*popt[:2])
        core_rec = interferometry.fit_axis(np.array([self._core[-1]]), *popt)
        if not full_output:
            return direction_rec, core_rec

        thetavar, phivar, corex_var, corey_var = np.diag(pcov)
        opening_angle, opening_angle_var = opening_angle_spherical(*hp.cartesian_to_spherical(
            *direction_rec), self._shower[shp.zenith], self._shower[shp.azimuth], thetavar, phivar)

        return direction_rec, core_rec, opening_angle, np.sqrt(opening_angle_var), np.sqrt([corex_var, corey_var])

    @register_run()
    def run(self,
            evt: Event,
            det: Optional[DetectorBase] = None,
            station_ids: Optional[list] = None, signal_kind="power",
            relative_signal_treshold: float = 0.,
            depths: np.ndarray = np.arange(
                400, 900, 100) * units.g / units.cm2,
            mc_jitter: float = 0 * units.ns,
            ):
        """
        Run interferometric reconstruction of depth of coherent signal.

        Parameters
        ----------

        evt : Event
            Event to run the module on.

        det : Detector
            Detector description

        shower: BaseShower
            shower to extract geometry from. Conventional: `evt.get_first_shower()` or `evt.get_first_sim_shower()`

        depths: Optional[list] (default: None)
            slant depths in g/cm^2  at which to sample lateral profiles. None results in [500, 600, 700, 800, 900, 1000].

        use_mc_pulses : bool (default: True)
            if true, take electric field trace from sim_station

        station_ids: Optional[list] (default: None)
            station_ids whose channels will be read out. For all stations, use `evt.get_station_ids()`

        mc_jitter: Optional[float] (with unit of time, default: None)
            Standard deviation of Gaussian noise added to timings, if set.

        initial_grid_spoacing: float (default: 60*units.m)
            initial lateral grid spacing to use.

        """
        self._signal_kind = signal_kind
        self._signal_threshold = relative_signal_treshold

        self._depths = depths / units.g * units.cm2
        self._binsize = self._depths[1] - self._depths[0]
        self._mc_jitter = mc_jitter / units.ns

        self.set_station_data(evt, det, station_ids=station_ids)

        direction_rec, core_rec = self.reconstruct_shower_axis()

        self._shower.set_parameter(
            shp.interferometric_shower_axis, direction_rec)
        self._shower.set_parameter(shp.interferometric_core, core_rec)

    def end(self):
        return self._data


class efieldInterferometricLateralReco(efieldInterferometricAxisReco):
    def __init__(self):
        super().__init__()
        self._lateral_sample_count = None
        self._lateral_vxB_width = None

    def begin(self,
              shower: BaseShower,
              core: Optional[np.ndarray] = None,
              axis: Optional[np.ndarray] = None,
              n_sampling: int = 256,
              use_sim_pulses: bool = False,
              use_channels: bool = False,
              multiprocessing: bool = False,
              sample_angular_resolution: float = 0.005*units.deg,
              initial_grid_spacing: float = 40*units.m,
              cross_section_width: float = 200*units.m,
              lateral_vxB_sample_count: int = 80,
              lateral_vxB_width: float = 2000*units.m,
              debug: bool = False
              ):
        """
        Set module config.

        Parameters
        ----------

        signal_kind : str
            Define which signal "metric" is used on the beamformed traces. Default "power" : sum over the squared amplitudes in a 100 ns window around the peak.
            Other options are "amplitude" or "hilbert_sum"

        relative_signal_treshold: float (default = 0.)
           Fraction of strongest signal necessary for a trace to be used for RIT. Default of 0 includes all channels.

        depths: np.ndarray (default: units.g / units.cm2 * np.arange(400,800,10))
            depths to coarsely sample the axis with, in internal units

        mc_jitter: float (default: 0.)
            Timing jitter to put on traces, in internal units (==nanoseconds)

        debug : bool
            If true, show some debug plots (Default: False).

        """
        self._debug = debug
        self._nsampling = n_sampling
        self._use_sim_pulses = use_sim_pulses
        self._use_channels = use_channels
        self._multiprocessing = multiprocessing
        self._angres = sample_angular_resolution / units.radian
        self._cross_section_width = cross_section_width / units.m
        self._initial_grid_spacing = initial_grid_spacing / units.m
        self._lateral_vxB_sample_count = lateral_vxB_sample_count
        self._lateral_vxB_width = lateral_vxB_width

        self.set_geometry(shower, core=core, axis=axis)
        self.update_atmospheric_model_and_refractivity_table(shower)

    def get_lateral_profile(self, p_axis: np.ndarray, xs: np.ndarray):
        signals = np.zeros(len(xs))
        for xdx, x in enumerate(tqdm(xs)):
            p = p_axis + self._cs.transform_from_vxB_vxvxB(np.array([x, 0, 0]))

            sum_trace = interferometry.interfere_traces_interpolation(
                p, self._positions, self._traces, self._times, tab=self._tab)

            signal = interferometry.get_signal(
                sum_trace, self._tstep, kind=self._signal_kind)
            signals[xdx] = signal
        return signals

    def find_fwhm(self, depth: float):
        p_max, weight, ground_grid_uncertainty = self.sample_lateral_cross_section(
            depth, self._core, self._axis, self._cross_section_width, self._initial_grid_spacing)

        _, dr_ref_target = self.get_paxis_dr_target(depth, self._core, self._axis)
        max_dist = self._lateral_vxB_width / 2
        xlims = np.array([-max_dist, max_dist])
        # xs = np.linspace(*xlims, self._lateral_sample_count)

        n = self._lateral_vxB_sample_count // 2
        xs = np.hstack([np.geomspace(xlims[0], -dr_ref_target / 2, n), np.geomspace(dr_ref_target / 2, xlims[1], n)])

        lateral_vxB_profile = self.get_lateral_profile(p_max, xs)

        # def lorentzian(x, gamma):
        #     return weight / (1 + (x / gamma)**2)
        # p, _ = curve_fit(lorentzian, xs, lateral_vxB_profile, p0=[50])
        # fwhm_fit = p[0]

        pk_half = scipy.signal.peak_widths(lateral_vxB_profile,
                                           [np.argmax(lateral_vxB_profile)],
                                           rel_height=.5)

        lips = pk_half[2][0]
        rips = pk_half[3][0]
        fwhm = xs[round(rips)] - xs[round(lips)]

        if True:
            pk_full = scipy.signal.peak_widths(lateral_vxB_profile,
                                               [np.argmax(lateral_vxB_profile)],
                                               rel_height=1)
            fwfm = xs[round(pk_full[3][0])] - xs[round(pk_full[2][0])]
            ax = plt.figure().add_subplot()
            ax.scatter(xs, lateral_vxB_profile, marker="v", s=1.2)
            ax.set_xlabel(r"$x_{\mathbf{v}\times\mathbf{B}}$ [m]")
            ax.set_ylabel(f"Summed {self._signal_kind}")
            ax.hlines(pk_half[1][0], xs[round(pk_half[2][0])], xs[round(pk_half[3][0])], color="pink", label=f"FWHM {fwhm: .2f} m", lw=0.5, ls=":")
            ax.hlines(pk_full[1][0], xs[round(pk_full[2][0])], xs[round(pk_full[3][0])], color="green", label=f"FWFM {fwfm: .2f} m", lw=0.1, ls=":")
            # xs_linspace = np.linspace(xs[0], xs[-1], 1000)
            # ax.plot(xs_linspace, lorentzian(xs_linspace, fwhm_fit), color="m", ls="--", lw=.8, label=f"Fit: FWHM = {fwhm_fit: .2f} m")
            ax.legend()
            plt.show()
            sys.exit()

        return fwhm

    @register_run()
    def run(self,
            evt: Event,
            det: Optional[DetectorBase] = None,
            depth: Optional[float] = None,
            station_ids: Optional[list] = None, signal_kind="power",
            relative_signal_treshold: float = 0.,
            mc_jitter: float = 0 * units.ns,
            ):
        """
        Run interferometric reconstruction of depth of coherent signal.

        Parameters
        ----------

        evt : Event
            Event to run the module on.

        det : Detector
            Detector description

        shower: BaseShower
            shower to extract geometry from. Conventional: `evt.get_first_shower()` or `evt.get_first_sim_shower()`

        depths: Optional[list] (default: None)
            slant depths in g/cm^2  at which to sample lateral profiles. None results in [500, 600, 700, 800, 900, 1000].

        use_mc_pulses : bool (default: True)
            if true, take electric field trace from sim_station

        station_ids: Optional[list] (default: None)
            station_ids whose channels will be read out. For all stations, use `evt.get_station_ids()`

        mc_jitter: Optional[float] (with unit of time, default: None)
            Standard deviation of Gaussian noise added to timings, if set.

        initial_grid_spoacing: float (default: 60*units.m)
            initial lateral grid spacing to use.

        """
        self._signal_kind = signal_kind
        self._signal_threshold = relative_signal_treshold

        self._mc_jitter = mc_jitter / units.ns

        self.set_station_data(evt, det, station_ids=station_ids)

        if depth is None:
            depth = self._shower[shp.interferometric_shower_maximum]

        fwhm = self.find_fwhm(depth / units.g * units.cm2)
        self._shower.set_parameter(shp.interferometric_fwhm, fwhm * units.m)
        logger.info(f"Interferometric vxB FWHM at {depth / units.g * units.cm2: .2f} g/cm2: {fwhm: .2f} m")

    def end(self):
        pass


def plot_shower_axis_points(points: np.ndarray, weights: np.ndarray, shower: Optional[BaseShower] = None):
    if shower is not None:
        cs = coordinatesystems.cstrafo(
            shower[shp.zenith], shower[shp.azimuth], shower[shp.magnetic_field_vector])
    else:
        logger.warning("plot_shower_axis: no shower passed, assuming showerplane points")
        cs = None

    logger.debug(f"points shape: {points.shape}")
    assert points.shape[:-1] == weights.shape
    if len(points.shape) == 3:
        permutation = np.argsort(points[0, :, -1])
        points_showerplane = []
        for axispoints in points:
            if shower is not None:
                axispoints_showerplane = cs.transform_to_vxB_vxvxB(
                    axispoints, shower[shp.core])
            else:
                axispoints_showerplane = axispoints
            points_showerplane.append(axispoints_showerplane)
        points_showerplane = np.array(points_showerplane)
        points_showerplane[..., -1] *= -1  # v-> -v
        quantiles = np.quantile(points_showerplane, [.16, .5, .84], axis=0)[:, permutation]
        assert np.all(quantiles[0] <= quantiles[1]) and np.all(quantiles[1] <= quantiles[2])
        median_showerplane = quantiles[1]
        delta_showerplane = quantiles[[0, 2]]
        logger.debug(f"permutation_shape {permutation.shape}")
        logger.debug(f"permuted shape: {median_showerplane.shape}")
        weights_median = np.median(weights, axis=0)[permutation]
        weights_median /= np.max(weights_median)

    elif len(points.shape) == 2:
        permutation = np.argsort(points[:, -1])
        if shower is not None:
            median_showerplane = cs.transform_to_vxB_vxvxB(
                points[permutation], shower[shp.core])
        else:
            median_showerplane = np.copy(points)[permutation]
        median_showerplane[..., -1] *= -1
        delta_showerplane = None
        weights_median = np.copy(weights)[permutation]
        weights_median /= np.max(weights_median)

    else:
        raise ValueError("points must be 2d or 3d.")

    fig = plt.figure()
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 15], hspace=0.1)
    x, y, z = median_showerplane.T
    ax_z = fig.add_subplot(gs[1, 0])
    ax_w = fig.add_subplot(gs[1, 1])
    weight_modifier = 10
    sm_z = ax_z.scatter(x, y, c=z, s=weight_modifier * weights_median, cmap=plt.cm.viridis, marker="v", zorder=1.1)
    if delta_showerplane is not None:
        assert np.all(delta_showerplane[0] <= median_showerplane) and np.all(median_showerplane <= delta_showerplane[1])
        delta_showerplane[0] = median_showerplane - delta_showerplane[0]
        delta_showerplane[1] -= median_showerplane
        dx, dy, dz = np.moveaxis(delta_showerplane, -1, 0)

        eb_kwargs = {"capsize": 1.5, "zorder": .99,
                     "elinewidth": .1, "capthick": .1, "lw": .1, "ecolor": "k"}

        ax_z.errorbar(x, y, dy, dx, **eb_kwargs)
    cb = colorbar.Colorbar(ax=fig.add_subplot(
        gs[0, 0]), orientation="horizontal", label="RIT -v [m]", ticklocation="top", mappable=sm_z)
    cb.ax.tick_params(labelsize="x-small")
    ax_z.set_xlabel("RIT vxB [m]", fontsize="small")
    ax_z.set_ylabel("RIT vxvxB [m]", fontsize="small")

    sm_w = ax_w.scatter(x, z, c=y, s=weight_modifier * weights_median, cmap=plt.cm.plasma, marker="v", zorder=1.1)
    if delta_showerplane is not None:
        ax_w.errorbar(x, z, dz, dx, **eb_kwargs)
    cb = colorbar.Colorbar(ax=fig.add_subplot(
        gs[0, 1]), orientation="horizontal", label="RIT vxvxB [m]", ticklocation="top", mappable=sm_w)
    cb.ax.tick_params(labelsize="x-small")
    ax_w.set_xlabel("RIT vxB [m]", fontsize="small")
    ax_w.set_ylabel("RIT -v [m]", fontsize="small")
    for ax in [ax_z, ax_w]:
        ax.spines[["top", "right"]].set_visible(True)
        ax.tick_params(labelsize="x-small")
        ax.grid(visible=True, lw=.2)

    if shower is not None:
        fig.suptitle(f"zenith {shower[shp.zenith] / units.deg: .1f} deg, azimuth {shower[shp.azimuth] / units.deg: .1f} deg")
    plt.show()


def plot_lateral_cross_section(
        xs, ys, signals, mc_pos=None, fname=None, title=None):
    """
    Plot the lateral distribution of the beamformed singal (in the vxB, vxvxB directions).

    Parameters
    ----------

    xs : np.array
        Positions on x-axis (vxB) at which the signal is sampled (on a 2d grid)

    ys : np.array
        Positions on y-axis (vxvxB) at which the signal is sampled (on a 2d grid)

    signals : np.array
        Signals sampled on the 2d grid defined by xs and ys.

    mc_pos : np.array(2,)
        Intersection of the (MC-)axis with the "slice" of the lateral distribution plotted.

    fname : str
        Name of the figure. If given the figure is saved, if fname is None the fiture is shown.

    title : str
        Title of the figure (Default: None)
    """

    yy, xx = np.meshgrid(ys, xs)

    ax = plt.figure().add_subplot()
    pcm = ax.pcolormesh(xx, yy, signals, shading='gouraud')
    ax.scatter(xx, yy, s=2, color="k", facecolors="none", lw=0.5)
    cbi = plt.colorbar(pcm, pad=0.02)
    cbi.set_label(r"$f_{B_{j}}$ / eV$\,$m$^{-2}$")

    idx = np.argmax(signals)
    xfound = xs[int(idx // len(ys))]
    yfound = ys[int(idx % len(ys))]

    ax.scatter(xfound, yfound, s=2, color="blue", label="Max")
    if mc_pos is not None:
        ax.scatter(*mc_pos[:2], marker="*", color="red", label=r"intersection with $\hat{a}_\mathrm{MC}$", s=2)

    ax.legend()
    ax.set_ylabel(
        r"$\vec{v} \times \vec{v} \times \vec{B}$ / m")
    ax.set_xlabel(r"$\vec{v} \times \vec{B}$ / m")
    if title is not None:
        ax.set_title(title)  # r"slant depth = %d g / cm$^2$" % depth)

    plt.tight_layout()
    if fname is not None:
        # "rit_xy_%d_%d_%s.png" % (depth, iloop, args.label))
        plt.savefig(fname)
    else:
        plt.show()


def normal(x, A, x0, sigma):
    """ Gauss curve """
    return A / np.sqrt(2 * np.pi * sigma ** 2) \
        * np.exp(-1 / 2 * ((x - x0) / sigma) ** 2)


def gaussian_2d(xy, A, mux, muy, sigmax, sigmay):
    mu = np.array([mux, muy])
    sigma = np.array([sigmax, sigmay])
    return A * np.exp(np.sum(-np.square(xy -
                      mu[:, np.newaxis] / (sigma[:, np.newaxis])) / 2, axis=0))


class vonmises_fisher(scipy.stats._multivariate.multi_rv_generic):
    """copy of scipy.stats.vonmises_fisher"""

    def __init__(self, seed=None):
        super().__init__(seed)

    def _process_parameters(self, mu, kappa):
        """
        Infer dimensionality from mu and ensure that mu is a one-dimensional
        unit vector and kappa positive.
        """
        mu = np.asarray(mu)
        if mu.ndim > 1:
            raise ValueError("'mu' must have one-dimensional shape.")
        if not np.allclose(np.linalg.norm(mu), 1.):
            raise ValueError("'mu' must be a unit vector of norm 1.")
        if not mu.size > 1:
            raise ValueError("'mu' must have at least two entries.")
        kappa_error_msg = "'kappa' must be a positive scalar."
        if not np.isscalar(kappa) or kappa < 0:
            raise ValueError(kappa_error_msg)
        if float(kappa) == 0.:
            raise ValueError("For 'kappa=0' the von Mises-Fisher distribution "
                             "becomes the uniform distribution on the sphere "
                             "surface. Consider using "
                             "'scipy.stats.uniform_direction' instead.")
        dim = mu.size

        return dim, mu, kappa

    def _rvs_3d(self, kappa, size, random_state):
        """
        Generate samples from a von Mises-Fisher distribution
        with mu = [1, 0, 0] and kappa. Samples then have to be
        rotated towards the desired mean direction mu.
        This method is much faster than the general rejection
        sampling based algorithm.
        Reference: https://www.mitsuba-renderer.org/~wenzel/files/vmf.pdf

        """
        if size is None:
            sample_size = 1
        else:
            sample_size = size

        # compute x coordinate acc. to equation from section 3.1
        x = random_state.random(sample_size)
        x = 1. + np.log(x + (1. - x) * np.exp(-2 * kappa)) / kappa

        # (y, z) are random 2D vectors that only have to be
        # normalized accordingly. Then (x, y z) follow a VMF distribution
        temp = np.sqrt(1. - np.square(x))
        dist = scipy.stats.uniform_direction(2)
        uniformcircle = dist.rvs(sample_size, random_state)
        samples = np.stack([x, temp * uniformcircle[..., 0],
                           temp * uniformcircle[..., 1]], axis=-1)
        if size is None:
            samples = np.squeeze(samples)
        return samples

    def _rotate_samples(self, samples, mu, dim):
        """A QR decomposition is used to find the rotation that maps the
        north pole (1, 0,...,0) to the vector mu. This rotation is then
        applied to all samples.

        Parameters
        ----------
        samples: array_like, shape = [..., n]
        mu : array-like, shape=[n, ]
            Point to parametrise the rotation.

        Returns
        -------
        samples : rotated samples

        """
        base_point = np.zeros((dim, ))
        base_point[0] = 1.
        embedded = np.concatenate([mu[None, :], np.zeros((dim - 1, dim))])
        rotmatrix, _ = np.linalg.qr(np.transpose(embedded))
        if np.allclose(np.matmul(rotmatrix, base_point[:, None])[:, 0], mu):
            rotsign = 1
        else:
            rotsign = -1

        # apply rotation
        samples = np.einsum('ij,...j->...i', rotmatrix, samples) * rotsign
        return samples

    def _rvs(self, dim, mu, kappa, size, random_state):
        if dim == 3:
            samples = self._rvs_3d(kappa, size, random_state)
        else:
            print("not implemented! update python to >= 3.9 and scipy to >= 1.11 and use scipy.stats.vonmises_fisher")

        if dim != 2:
            samples = self._rotate_samples(samples, mu, dim)
        return samples

    def rvs(self, mu=None, kappa=1, size=1, random_state=None):
        dim, mu, kappa = self._process_parameters(mu, kappa)
        random_state = self._get_random_state(random_state)
        samples = self._rvs(dim, mu, kappa, size, random_state)
        return samples


def angle_between(v1: np.ndarray, v2: np.ndarray):
    """
    Returns the angle in radians between vectors 'v1' and 'v2': https://stackoverflow.com/a/13849249
    """
    v1_u = v1 / np.linalg.norm(v1, axis=0)
    v2_u = v2 / np.linalg.norm(v2, axis=0)
    return np.arccos(v1_u @ v2_u)


def opening_angle_spherical(theta1, phi1, theta2, phi2, theta1_var, phi1_var):
    """Give the the opening angle and variance on the opening angle between two vectors with spherical coordinates (1, theta, phi), asuming the second vector is known, such that theta2_var, phi2_var are not asked"""
    c1 = np.cos(theta1)
    c2 = np.cos(theta2)
    s1 = np.sin(theta1)
    s2 = np.sin(theta2)
    s12 = np.sin(phi1 - phi2)
    c12 = np.cos(phi1 - phi2)
    arg = s1*s2*c12 + c1*c2
    opening_angle_opt = np.arccos(arg)
    opening_angle_var = (1/(1-arg**2)) * ((c1*s2*c12 - s1*c2)
                                          ** 2 * theta1_var + (s1*s2*s12)**2 * phi1_var)
    return opening_angle_opt, opening_angle_var


def select_channels_per_station(det: DetectorBase, station_id: int,
                                requested_channel_ids: Optional[list] = None) -> defaultdict:
    # TODO: centralize this function which is a copy from readCoREASInterpolator, and replace the usage of this one.
    """

    Returns a defaultdict object containing the requeasted channel ids that are in the given station.
    This dict contains the channel group ids as keys with lists of channel ids as values.

    Parameters
    ----------
    det : DetectorBase
        The detector object that contains the station
    station_id : int
        The station id to select channels from
    requested_channel_ids : list
        List of requested channel ids
    """
    if requested_channel_ids is None:
        requested_channel_ids = det.get_channel_ids(station_id)
    else:
        # ensure ordering as coming from det
        requested_channel_ids = [id for id in det.get_channel_ids(station_id) if id in requested_channel_ids]

    channel_ids = defaultdict(list)
    for channel_id in requested_channel_ids:
        if channel_id in det.get_channel_ids(station_id):
            channel_group_id = det.get_channel_group_id(station_id, channel_id)
            if channel_group_id == -1:
                channel_group_id = channel_id
            channel_ids[channel_group_id].append(channel_id)

    return channel_ids
