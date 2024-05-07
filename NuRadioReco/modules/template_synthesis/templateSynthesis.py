from __future__ import annotations

import logging

import h5py
import numpy as np
import radiotools.coordinatesystems

import NuRadioReco.framework.event
import NuRadioReco.framework.station
import NuRadioReco.framework.sim_station
import NuRadioReco.framework.channel
import NuRadioReco.framework.electric_field

from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.modules.template_synthesis.slicedShower import slicedShower
from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units

logger = logging.getLogger("NuRadioReco.templateSynthesis")


def geo_ce_to_e(geo, ce, x, y):
    """
    Combine the geomagnetic and charge-excess back to a 3D electric field, in the (vxB, vxvxB, v)
    coordinate system. The component in v-direction is assumed to be 0.

    Parameters
    ----------
    geo : ndarray
    ce : ndarray
    x : float
        The x-component of the antenna position in the showerplane
    y : float
        The y-component of the antenna position in the showerplane

    Returns
    -------
    e_field : ndarray
        The electric field in (vxB, vxvxB, v) coordinates, shaped as (slices, samples, polarisations)
    """
    e_field = np.zeros((3, *geo.shape[::-1]))  # 3 x SAMPLES x SLICES

    e_field[0] = (-1 * x / np.sqrt(x ** 2 + y ** 2) * ce - geo).T
    e_field[1] = (-1 * y / np.sqrt(x ** 2 + y ** 2) * ce).T

    return e_field.T


def amplitude_function(params, frequencies, d_noise=0.):
    return params[0] * np.exp(params[1] * frequencies + params[2] * frequencies ** 2) + d_noise


class groundElementSynthesis:
    """
    This class is supposed to represent an antenna in the template synthesis framework.
    It holds the position of the antenna and the spectral coefficients for all the slices
    the antenna observes. After it has been equipped with a template shower, it can be
    used to synthesise the electric fields for any Xmax.

    Parameters
    ----------
    n_slices : int
        The number of slices the antenna observes
    pos : array_like
        The x,y,z position of the antenna, in the CORSIKA coordinate system

    Attributes
    ----------
    n_slices
    position
    name
    spectral_coefficients
    has_template : float
        Whether or not a template shower has been set

    Notes
    -----
    The naming scheme is inspired by CoREAS. For this reason, the position is internally
    stored in the CORSIKA coordinate system. When supplying a position to the class
    constructor, it should be in the CORSIKA coordinate system. This setup is chosen
    to more easily interface with the spectral coefficient files. However, the position
    attribute interface assumes the NRR coordinate system to be consistent with the NRR
    framework. So if you wish to set the position using the NRR coordinate system,
    simply leave the position in the constructor to its default value of `None` and
    set the position using the attribute.
    """
    def __init__(self, n_slices, pos=None):
        # Set default values
        self.__x, self.__y, self.__z = 0, 0, 0
        self.__name = None  # antenna name is based on position
        self.__spectral_coefficients = np.ones((3, 3, 3, n_slices)) * -1

        self.has_template = False
        self.__template_spectrum_geo = None
        self.__template_spectrum_ce = None
        self.__template_spectrum_ce_lin = None
        self.__template_frequencies = None

        self.__template_sampling_rate = None
        self.__template_start_time = None

        # Fill in required variables
        if pos is not None:
            logger.status("Provided position is assumed to be in CORSIKA coordinate system \n"
                          "To use the NRR coordinate system use the position attribute")
            self.position = [-pos[1], pos[0], pos[2]]

    @property
    def n_slices(self):
        """
        The number of slices this antenna "sees"
        """
        return self.__spectral_coefficients.shape[0]

    @property
    def position(self):
        """
        Position of the antenna in the ground plane, in meters, in the **NRR** coordinate system
        """
        return np.array([-1 * self.__y, self.__x, self.__z]) / units.m

    @position.setter
    def position(self, pos):
        """
        Set the position of the antenna using the **NRR** coordinate system
        """
        if pos is None:
            logger.warning("No new position provided!")
            return
        if len(pos) != 3:
            raise ValueError("New position array needs to contain exactly 3 elements")

        self.__y, self.__x, self.__z = pos
        self.__y *= -1

        positive_x = 1 if self.__x > 0 else 0
        positive_y = 1 if self.__y > 0 else 0
        self.__name = f'{positive_x}{int(abs(self.__x) / units.m)}_{positive_y}{int(abs(self.__y) / units.m)}'

    @property
    def name(self):
        return self.__name

    @property
    def sampling_rate(self):
        return self.__template_sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, value):
        self.__template_sampling_rate = value

    @property
    def start_time(self):
        return self.__template_start_time

    @start_time.setter
    def start_time(self, value):
        self.__template_start_time = value

    @property
    def spectral_coefficients(self):
        """
        The array of spectral coefficients. This is 3x3x3xSLICES array, with the first dimension referring to the
        the spectral parameters (a/b/c). The second dimension encodes the coefficients of the parabola (p0/p1/p2),
        and the third one is the component, i.e. GEO/CE/CE_LIN (in this order). The last dimension is the number
        of slices this antenna sees.
        """
        return self.__spectral_coefficients

    @spectral_coefficients.setter
    def spectral_coefficients(self, new_coef):
        if new_coef is None:
            logger.warning("No spectral coefficients provided")
            return
        if new_coef.shape != self.__spectral_coefficients.shape:
            raise ValueError("New coefficients array does not have the right shape")

        self.__spectral_coefficients = new_coef

    def spectral_coefficient_per_component(self, new_coef, component):
        """
        Set the 3x3 spectral coefficient array for one component.

        Parameters
        ----------
        new_coef : np.ndarray
        component : {'GEO', 'CE', 'CE_LIN'}
        """
        if new_coef.shape != self.__spectral_coefficients[:, :, 0].shape:  # Select GEO shape for testing
            raise ValueError("New coefficients array does not have the right shape")

        logger.info(f"Setting spectral coefficients of {component}")
        logger.debug(new_coef)
        if component == 'GEO':
            self.__spectral_coefficients[:, :, 0] = new_coef
        elif component == 'CE':
            self.__spectral_coefficients[:, :, 1] = new_coef
        elif component == 'CE_LIN':
            self.__spectral_coefficients[:, :, 2] = new_coef
        else:
            raise ValueError(f"The component {component} is not supported")

    def _calculate_amp_fit(self, x_max, freq):
        a_fit = np.polynomial.polynomial.polyval(x_max, self.spectral_coefficients[0])  # {GEO, CE, CE_LIN} x SLICES
        b_fit = np.polynomial.polynomial.polyval(x_max, self.spectral_coefficients[1])
        c_fit = np.polynomial.polynomial.polyval(x_max, self.spectral_coefficients[2])

        amp_fit = self._amplitude_function(a_fit, b_fit, c_fit, freq)
        amp_fit[np.where(a_fit < 0)] = 0.0  # neg values of a, are points where a should be zero (not fitted)
        amp_fit[np.where(np.abs(amp_fit) <= 1e-20)] = 1.  # remove values which are too small and cause inf

        return amp_fit

    @staticmethod
    def _amplitude_function(a, b, c, frequencies, d_noise=0.):
        return a[:, :, np.newaxis] * np.exp(
            b[:, :, np.newaxis] * frequencies[np.newaxis, np.newaxis, :] +
            c[:, :, np.newaxis] * frequencies[np.newaxis, np.newaxis, :] ** 2
        ) + d_noise

    def set_template(self, shower_xmax, trace_geo, trace_ce, freq):
        """
        Sets the template traces for GEO and CE components. Assumes the traces are already normalised
        with particle numbers and that the central frequency has already been removed from the frequency
        array!
        """

        self.__template_frequencies = freq

        spectrum_geo = np.fft.rfft(trace_geo, norm='ortho', axis=-1)  # SLICES x FREQ
        spectrum_ce = np.fft.rfft(trace_ce, norm='ortho', axis=-1)

        abs_geo = np.abs(spectrum_geo)
        phase_geo = np.angle(spectrum_geo)

        abs_ce = np.abs(spectrum_ce)
        phase_ce = np.angle(spectrum_ce)

        # Normalise
        normalisation = self._calculate_amp_fit(shower_xmax, freq / units.GHz)  # {GEO, CE, CE_LIN} x SLICES x FREQ

        self.__template_spectrum_geo = np.stack((abs_geo / normalisation[0], phase_geo))
        self.__template_spectrum_ce = np.stack((abs_ce / normalisation[1], phase_ce))
        self.__template_spectrum_ce_lin = np.stack((abs_ce / normalisation[2], phase_ce))

        self.has_template = True

    def map_template(self, target_xmax):
        if not self.has_template:
            raise RuntimeError(f"Template has not been set yet for antenna {self.name}")

        normalisation = self._calculate_amp_fit(target_xmax,
                                                self.__template_frequencies / units.GHz)  # COMPONENT x SLICES x FREQ

        synth_geo = self.__template_spectrum_geo[0] * normalisation[0] * np.exp(1j * self.__template_spectrum_geo[1])
        synth_ce = self.__template_spectrum_ce[0] * normalisation[1] * np.exp(1j * self.__template_spectrum_ce[1])
        synth_ce_lin = self.__template_spectrum_ce_lin[0] * normalisation[2] * np.exp(
            1j * self.__template_spectrum_ce_lin[1]
        )

        return np.fft.irfft(np.stack((synth_geo, synth_ce, synth_ce_lin)), norm='ortho', axis=-1)


class templateSynthesis:
    def __init__(self):
        #        low_freq=30.0 * units.MHz, high_freq=500.0 * units.MHz,
        #        phase_method="phasor", sampling_period=2e-10 * units.s):
        # Store variables related to interpolation
        # self.__interpolation_low_freq = low_freq
        # self.__interpolation_high_freq = high_freq
        # self.__interpolation_phase_method = phase_method
        # self.__interpolation_sampling_period = sampling_period

        # Some hardcoded values for checking
        self.slice_axis = 0
        self.antenna_axis = 1

        # Init some variables to filled in later
        self.__spectral_parameters = {}
        self.__antennas = []  # the antennas for which we have spectral parameters
        self.__trace_length = None
        self.__n_slices = None
        self.__g_slices = None  # the grammage of a single slice, usually 5 g/cm2

        # Frequency variables
        self.__freq_interval = (0.0, np.inf)
        self.__freq_center = 0.0

        # Geometry variables
        self.__zenith = None
        self.__azimuth = None
        self.__magnet = None
        self.__core = None

    # Define zenith and azimuth as properties to protect them from getting written
    @property
    def zenith(self):
        return self.__zenith / units.deg

    @property
    def azimuth(self):
        return self.__azimuth / units.deg

    @property
    def antennas(self):
        return self.__antennas

    @property
    def frequency(self):
        return *self.__freq_interval, self.__freq_center

    @frequency.setter
    def frequency(self, new_freq: tuple | list):
        if len(new_freq) != 3:
            raise ValueError("Please provide a list of 3 elements: f_min, f_max and f_0")

        self.__freq_interval = tuple(new_freq[:2])
        self.__freq_center = new_freq[2]

    def __freq_from_hdf5(self, file):
        try:
            self.__freq_interval = (
                file['Metadata']['Fitting metadata']['min_frequency_MHz'] * units.MHz,
                file['Metadata']['Fitting metadata']['max_frequency_MHz'] * units.MHz
            )
            self.__freq_center = file['Metadata']['Fitting metadata']['center_frequency_MHz'] * units.MHz
        except KeyError:
            logger.error("Frequencies are not present in parameter file. Please set them manually before proceeding")

    def begin(self, spectral_parameter_file, slicing_grammage=5):
        spectral_file = h5py.File(spectral_parameter_file)

        self.__freq_from_hdf5(spectral_file)

        self.__g_slices = slicing_grammage
        self.__n_slices = spectral_file['Metadata']['EM profile'].shape[0]

        antenna_positions = spectral_file['Metadata']['Antenna metadata'][:].view((float, 4))

        for ant_ind, ant_pos in enumerate(antenna_positions):
            ant = groundElementSynthesis(self.__n_slices, ant_pos[:3] * units.m)

            for key in spectral_file['SpectralFitParams'].keys():  # loop over GEO/CE/CE_LIN
                component = '_'.join(key.split('_')[1:])  # handle case of CE_LIN

                assert spectral_file['SpectralFitParams'][key].shape[self.slice_axis] == self.__n_slices, \
                    f"Number of slices does not match with metadata for {component}"
                assert spectral_file['SpectralFitParams'][key].shape[self.antenna_axis] == len(antenna_positions), \
                    f"Antenna metadata does not match shape of {component} spectral parameters array"

                ant.spectral_coefficient_per_component(
                    spectral_file['SpectralFitParams'][key][:, ant_ind].T, component
                )

            self.__antennas.append(ant)

        # For later bookkeeping, let's save some variables from the file metadata
        self.__zenith = np.unique(spectral_file['Metadata']['Simulations metadata']['Zenith_deg'])[0] * units.deg
        self.__azimuth = np.unique(spectral_file['Metadata']['Simulations metadata']['Azimuth_deg'])[0] * units.deg
        self.__azimuth -= 90 * units.deg  # rotate to radiotools coordinate system

        spectral_file.close()

    def make_template(self, origin_shower: slicedShower):
        shower_long_profile = origin_shower.get_long_profile()
        shower_sampling_res = origin_shower.get_coreas_settings()['time_resolution']

        # Save magnetic field and core from origin shower
        self.__magnet = origin_shower.magnet
        self.__core = origin_shower.core

        length_saved = False
        for antenna in self.__antennas:
            geo, ce, times = origin_shower.get_trace(antenna.name, return_start_time=True)  # geo, ce : SLICES x SAMPLES

            # Filter traces
            geo_filtered = origin_shower.filter_trace(geo, *self.__freq_interval)
            ce_filtered = origin_shower.filter_trace(ce, *self.__freq_interval)

            if not length_saved:
                # Save trace length only once
                self.__trace_length = geo.shape[1]
                length_saved = True

            # Normalise the traces with the particle numbers in the slice
            # Sometimes the LONG table can contain an entry inside the ground, so index using nr of slices
            geo_filtered /= shower_long_profile[:self.__n_slices, np.newaxis]
            ce_filtered /= shower_long_profile[:self.__n_slices, np.newaxis]

            frequencies = np.fft.rfftfreq(geo_filtered.shape[-1], d=shower_sampling_res / units.s) * units.Hz

            antenna.set_template(origin_shower.xmax, geo_filtered, ce_filtered, frequencies - self.__freq_center)
            antenna.start_time = times[0]  # by CoREAS definition, all slices have the same start time
            antenna.sampling_rate = 1. / shower_sampling_res

    def synthesise(self, target_xmax, long_profile):
        assert len(long_profile) == self.__n_slices, ("Long profile does not have correct number of slices \n"
                                                      f"The bins should be {self.__g_slices} g/cm2")

        traces_synth = np.zeros((len(self.__antennas), 3, self.__n_slices, self.__trace_length))
        for ind, antenna in enumerate(self.__antennas):
            synth = antenna.map_template(target_xmax)

            traces_synth[ind, 0] = synth[0] * long_profile[:, np.newaxis]  # SLICES x SAMPLES
            traces_synth[ind, 1] = synth[1] * long_profile[:, np.newaxis]
            traces_synth[ind, 2] = synth[2] * long_profile[:, np.newaxis]

        return traces_synth  # ANT x {GEO, CE, CE_LIN} x SLICES x SAMPLES

    @register_run()
    def run(self, target_xmax, long_profile, use_ce_lin=False):
        """
        Run the template synthesis module and put the resulting traces in an Event object.
        The Event object has one Station with ID=1, which contains one SimStation with the
        same ID. The SimStation then holds the synthesised electric fields, which are
        equipped with the position of the channel (in the NRR CS) they were synthesised for.

        Parameters
        ----------
        target_xmax : float
            The Xmax of the target shower
        long_profile : array of floats
            The longitudinal profile of the target shower, sliced using the same grammage as the template shower
        use_ce_lin : bool, default=False
            If True, use the linear charge-excess parameters to synthesise (i.e. CE_LIN)

        Returns
        -------
        event : Event
            The Event object with the synthesised electric fields
        """
        traces = self.synthesise(target_xmax, long_profile)
        traces_ground, traces_ground_ce_lin = self.transform_to_ground(traces)

        # Sum over all slices -> ANT x COREAS_POL X SAMPLES
        if use_ce_lin:
            traces_ground = np.sum(traces_ground_ce_lin, axis=2)
        else:
            traces_ground = np.sum(traces_ground, axis=2)

        event = NuRadioReco.framework.event.Event(1, 1)

        # TODO: Add SimShower ?

        # Add electric fields in single SimStation
        station = NuRadioReco.framework.station.Station(1)
        sim_station = NuRadioReco.framework.sim_station.SimStation(1)

        for ind, antenna in enumerate(self.__antennas):
            e_field = NuRadioReco.framework.electric_field.ElectricField(np.array([ind]), position=antenna.position)

            e_field.set_trace(traces_ground[ind], antenna.sampling_rate)
            e_field.set_trace_start_time(antenna.start_time)

            e_field.set_parameter(efp.zenith, self.__zenith)
            e_field.set_parameter(efp.azimuth, self.__azimuth)

            sim_station.add_electric_field(e_field)

        sim_station.set_parameter(stnp.zenith, self.__zenith)
        sim_station.set_parameter(stnp.azimuth, self.__azimuth)
        sim_station.set_parameter(stnp.cr_xmax, target_xmax)

        sim_station.set_magnetic_field_vector(self.__magnet)
        sim_station.set_is_cosmic_ray()

        station.set_sim_station(sim_station)
        event.set_station(station)

        yield event

    def get_transformer(self):
        transformer = radiotools.coordinatesystems.cstrafo(
            self.__zenith / units.rad, self.__azimuth / units.rad,
            magnetic_field_vector=self.__magnet
        )

        return transformer

    def transform_to_ground(self, traces):
        assert traces.shape == (len(self.__antennas), 3, self.__n_slices, self.__trace_length), \
            "Please provide the traces shaped as (ant, component, slices, samples)"

        transformer = self.get_transformer()

        traces_ground = np.zeros_like(traces)
        traces_ground_lin = np.zeros_like(traces)
        for ind, antenna in enumerate(self.__antennas):
            # Antenna position in vvB
            antenna_vvB = transformer.transform_to_vxB_vxvxB(
                np.array([-antenna.position[1], antenna.position[0], antenna.position[2]]),
                core=self.__core
            )

            # Sum all slices
            e_field = geo_ce_to_e(traces[ind, 0], traces[ind, 1], *antenna_vvB[:2])  # SLICES x SAMPLES x 3
            e_field_lin = geo_ce_to_e(traces[ind, 0], traces[ind, 2], *antenna_vvB[:2])

            # Save traces on ground
            for ind_slice in range(self.__n_slices):
                traces_ground[ind, :, ind_slice, :] = transformer.transform_from_vxB_vxvxB(e_field[ind_slice].T)
                traces_ground_lin[ind, :, ind_slice, :] = transformer.transform_from_vxB_vxvxB(e_field_lin[ind_slice].T)

        return traces_ground, traces_ground_lin  # ANT x COREAS_POL x SLICES x SAMPLES

    def end(self):
        pass
