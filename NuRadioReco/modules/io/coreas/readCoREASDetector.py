from datetime import timedelta
import logging
import os
import time
import h5py
import numpy as np
from radiotools import coordinatesystems as cstrafo
import cr_pulse_interpolator.signal_interpolation_fourier
import cr_pulse_interpolator.interpolation_fourier
import matplotlib.pyplot as plt
from NuRadioReco.modules.base.module import register_run
import NuRadioReco.framework.event
import NuRadioReco.framework.station
import NuRadioReco.framework.radio_shower
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import stationParameters as stnp
import NuRadioReco.modules.io.coreas
from NuRadioReco.modules.io.coreas import coreas
from NuRadioReco.utilities import units
from NuRadioReco.utilities.signal_processing import half_hann_window
from collections import defaultdict
from scipy.signal.windows import hann

conversion_fieldstrength_cgs_to_SI = 2.99792458e10 * units.micro * units.volt / units.meter

def get_random_core_positions(xmin, xmax, ymin, ymax, n_cores, seed=None):
    random_generator = np.random.RandomState(seed)

    # generate core positions randomly within a rectangle
    cores = np.array([random_generator.uniform(xmin, xmax, n_cores),
                    random_generator.uniform(ymin, ymax, n_cores),
                    np.zeros(n_cores)]).T
    return cores

def get_efield_times(efield, sampling_rate):
    """
    calculate the time axis of the electric field from the sampling rate

    Parameters
    ----------
    efield: array (n_samples, n_polarizations)
    """
    if efield is None:
        return None
    efield_times = np.arange(0, len(efield[:,0])) / sampling_rate
    return efield_times

def apply_hanning(efields):
    """
    Apply a half hann window to the electric field in the time domain

    Parameters
    ----------
    efield in time domain: array (n_samples, n_polarizations)

    Returns
    -------
    smoothed_efield: array (n_samples, n_polarizations)
    """

    if efields is None:
        return None
    else:
        smoothed_trace = np.zeros_like(efields)
        half_hann_window = half_hann_window(efields.shape[0], half_percent=0.1)
        for pol in range(efields.shape[1]):
            smoothed_trace[:,pol] = efields[:,pol] * half_hann_window
        return smoothed_trace

def select_channels_per_station(det, station_id, requested_channel_ids):
    """
    Returns a defaultdict object containing the requested channel ids that are in the given station.
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
    channel_ids = defaultdict(list)
    for channel_id in requested_channel_ids:
        if channel_id in det.get_channel_ids(station_id):
            channel_group_id = det.get_channel_group_id(station_id, channel_id)
            channel_ids[channel_group_id].append(channel_id)
    return channel_ids


class readCoREASDetector():
    """
    Use this as default when reading CoREAS files and combining them with a detector.

    This model reads the electric fields of a CoREAS file with a star shaped pattern and foldes them with it given detector. 
    The electric field of the star shaped pattern is interpolated at the detector positions. If the angle between magnetic field 
    and shower direction are below about 15 deg, the interpolation is no longer reliable and the closest observer is used instead.
    """

    def __init__(self):
        self.__t = 0
        self.__t_event_structure = 0
        self.__t_per_event = 0
        self.__input_file = None
        self.__corsika = None
        self.__interp_lowfreq = None
        self.__interp_highfreq = None
        self.__sampling_rate = None
        self.logger = logging.getLogger('NuRadioReco.readCoREASDetector')

    def begin(self, input_file, interp_lowfreq=30*units.MHz, interp_highfreq=1000*units.MHz, log_level=logging.INFO, debug=False):
        """
        begin method
        initialize readCoREAS module
        Parameters
        ----------
        input_files: input files
            list of coreas hdf5 files
        interp_lowfreq: float (default = 30)
            lower frequency for the bandpass filter in interpolation, should be broader than the sensetivity band of the detector
        interp_highfreq: float (default = 1000)
            higher frequency for the bandpass filter in interpolation,  should be broader than the sensetivity band of the detector
        """
        self.logger.setLevel(log_level)
        filesize = os.path.getsize(input_file)
        if(filesize < 18456 * 2):  # based on the observation that a file with such a small filesize is corrupt
            self.logger.warning("file {} seems to be corrupt".format(input_file))
        self.__corsika = coreas.read_CORSIKA7(input_file)
        self.logger.info(f"using coreas simulation {input_file} with E={self.__corsika.get_first_sim_shower().get_parameter(shp.energy):.2g}eV, zenith angle = {self.__corsika.get_first_sim_shower().get_parameter(shp.zenith) / units.deg:.0f}deg")

        self.__interp_lowfreq = interp_lowfreq
        self.__interp_highfreq = interp_highfreq

        self.debug = debug
        self.coreasInterpolator = NuRadioReco.modules.io.coreas.coreasInterpolator(self.__corsika)
        self.coreasInterpolator.initialize_efield_interpolator(self.__interp_lowfreq, self.__interp_highfreq)

    @register_run()
    def run(self, detector, core_position_list=[], selected_station_ids=[], selected_channel_ids=[]):
        """
        Parameters
        ----------
        detector: Detector object
            Detector description of the detector that shall be simulated
        """
        if len(selected_station_ids) == 0:
            selected_station_ids = detector.get_station_ids()
            logging.info(f"using all station ids in detector description: {selected_station_ids}")
        else:
            logging.info(f"using selected station ids: {selected_station_ids}")


        t = time.time()
        t_per_event = time.time()
        self.__t_per_event += time.time() - t_per_event
        self.__t += time.time() - t

        for iCore, core in enumerate(core_position_list):
            t = time.time()
            evt = NuRadioReco.framework.event.Event(evt.get_run_number(), iCore)  # create empty event
            sim_shower = self.__corsika.get_first_sim_shower()
            sim_shower.set_parameter(shp.core, core)
            evt.add_sim_shower(sim_shower)
            # rd_shower = NuRadioReco.framework.radio_shower.RadioShower(station_ids=selected_station_ids)
            # evt.add_shower(rd_shower)
            corsika_sim_stn = self.__corsika.get_station(0).get_sim_station()
            for station_id in selected_station_ids:
                station = NuRadioReco.framework.station.Station(station_id)
                sim_station = NuRadioReco.framework.sim_station.SimStation(station_id)
                for key, value in corsika_sim_stn.get_parameters().items():
                    sim_station.set_parameter(key, value)  # copy relevant sim_station parameters over
                sim_station.set_magnetic_field_vector(corsika_sim_stn.get_magnetic_field_vector())
                sim_station.set_is_cosmic_ray()

                det_station_position = detector.get_absolute_position(station_id)
                channel_ids_in_station = detector.get_channel_ids(station_id)
                if len(selected_channel_ids) == 0:
                    selected_channel_ids = channel_ids_in_station
                channel_ids_dict = select_channels_per_station(detector, station_id, selected_channel_ids)
                for ch_g_ids in channel_ids_dict.keys():
                    antenna_position_rel = detector.get_relative_position(station_id, ch_g_ids)
                    antenna_position = det_station_position + antenna_position_rel
                    res_efield = self.coreasInterpolator.get_interp_efield_value(antenna_position, core)
                    smooth_res_efield = apply_hanning(res_efield)
                    if smooth_res_efield is None:
                        smooth_res_efield = self.coreasInterpolator.get_empty_efield()
                    efield_times = get_efield_times(smooth_res_efield, self.__sampling_rate)
                    channel_ids_for_group_id = channel_ids_dict[ch_g_ids]
                    coreas.add_electric_field_to_sim_station(sim_station, channel_ids_for_group_id, smooth_res_efield.T, efield_times, self.zenith, self.azimuth, self.magnetic_field_vector, self.__sampling_rate )
                station.set_sim_station(sim_station)
                distance_to_core = np.linalg.norm(det_station_position[:-1] - core[:-1])
                station.set_parameter(stnp.distance_to_core, distance_to_core)
                evt.set_station(station)

            self.__t += time.time() - t
            yield evt

    def end(self):
        self.logger.setLevel(logging.INFO)
        dt = timedelta(seconds=self.__t)
        self.logger.info("total time used by this module is {}".format(dt))
        self.logger.info("\tcreate event structure {}".format(timedelta(seconds=self.__t_event_structure)))
        self.logger.info("per event {}".format(timedelta(seconds=self.__t_per_event)))
        return dt