from __future__ import annotations

import logging

import h5py
import numpy as np
import radiotools.coordinatesystems

from scipy.constants import c as c_vacuum

from NuRadioReco.modules.base.module import register_run
from NuRadioReco.utilities import units

logger = logging.getLogger("NuRadioReco.templateSynthesis")


def e_geo(traces, x, y):
    return traces[:, 1] * x / y - traces[:, 0]


def e_ce(traces, x, y):
    return -traces[:, 1] * np.sqrt(x ** 2 + y ** 2) / y


def geo_ce_to_e(geo, ce, x, y):
    e_field = np.zeros((3, *geo.shape[::-1]))

    e_field[0] = -1 * x / np.sqrt(x ** 2 + y ** 2) * ce - geo
    e_field[1] = -1 * y / np.sqrt(x ** 2 + y ** 2) * ce

    return e_field.T


def amplitude_function(params, frequencies, d_noise=0.):
    return params[0] * np.exp(params[1] * frequencies + params[2] * frequencies ** 2) + d_noise


class groundElementSynthesis:
    def __init__(self, n_slices, pos=None):
        # Set default values
        self.__x, self.__y, self.__z = 0, 0, 0
        self.__name = None  # antenna name is based on position
        self.__spectral_coefficients = np.ones((3, 3, 3, n_slices)) * -1

        self.has_template = False
        self.__template_spectrum_geo = None
        self.__template_spectrum_ce = None
        self.__template_spectrum_ce_lin = None
        self.__template_sampling = None

        # Fill in required variables
        self.position = pos

    @property
    def n_slices(self):
        """
        The number of slices this antenna "sees"
        """
        return self.__spectral_coefficients.shape[0]

    @property
    def position(self):
        """
        Position of the antenna in the ground plane, in meters, in the **CORSIKA** coordinate system
        """
        return np.array([self.__x, self.__y, self.__z]) / units.m

    @position.setter
    def position(self, pos):
        if pos is None:
            logger.warning("No new position provided!")
            return
        if len(pos) != 3:
            raise ValueError("New position array needs to contain exactly 3 elements")

        self.__x, self.__y, self.__z = pos

        positive_x = 1 if self.__x > 0 else 0
        positive_y = 1 if self.__y > 0 else 0
        self.__name = f'{positive_x}{int(abs(self.__x) / units.m)}_{positive_y}{int(abs(self.__y) / units.m)}'

    @property
    def name(self):
        return self.__name

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

        return self._amplitude_function(a_fit, b_fit, c_fit, freq)

    @staticmethod
    def _amplitude_function(a, b, c, frequencies, d_noise=0.):
        return a[:, :, np.newaxis] * np.exp(
            b[:, :, np.newaxis] * frequencies[np.newaxis, np.newaxis, :] +
            c[:, :, np.newaxis] * frequencies[np.newaxis, np.newaxis, :] ** 2
        ) + d_noise

    def set_template(self, shower_xmax, trace_geo, trace_ce, sampling_d=2e-10 * units.s):
        """
        Sets the template traces for GEO and CE components. Assumes the traces are already normalised
        with particle numbers!
        """
        self.__template_sampling = sampling_d

        freq_geo = np.fft.rfftfreq(trace_geo.shape[-1], sampling_d / units.s) * units.Hz
        freq_ce = np.fft.rfftfreq(trace_ce.shape[-1], sampling_d / units.s) * units.Hz

        # Frequency arrays should be the same
        assert np.all(freq_geo == freq_ce), "Frequencies differ for GEO and CE"

        spectrum_geo = np.fft.rfft(trace_geo, norm='ortho', axis=-1)  # SLICES x FREQ
        spectrum_ce = np.fft.rfft(trace_ce, norm='ortho', axis=-1)

        abs_geo = np.abs(spectrum_geo)
        phase_geo = np.angle(spectrum_geo)

        abs_ce = np.abs(spectrum_ce)
        phase_ce = np.angle(spectrum_ce)

        # Normalise
        normalisation = self._calculate_amp_fit(shower_xmax, freq_geo)  # {GEO, CE, CE_LIN} x SLICES x FREQ
        # TODO: round values which are too small to avoid numerical imprecisions

        self.__template_spectrum_geo = np.stack((abs_geo / normalisation[0], phase_geo))
        self.__template_spectrum_ce = np.stack((abs_ce / normalisation[1], phase_ce))
        self.__template_spectrum_ce_lin = np.stack((abs_ce / normalisation[2], phase_ce))

        self.has_template = True

    def map_template(self, target_xmax):
        if not self.has_template:
            raise RuntimeError(f"Template has not been set yet for antenna {self.name}")

        trace_len = len(self.__template_spectrum_geo) // 2 * 2
        freq = np.fft.rfftfreq(trace_len, self.__template_sampling)

        normalisation = self._calculate_amp_fit(target_xmax, freq)  # {GEO, CE, CE_LIN} x SLICES x FREQ

        synth_geo = self.__template_spectrum_geo[0] * normalisation[0] * np.exp(1j * self.__template_spectrum_geo[1])
        synth_ce = self.__template_spectrum_ce[0] * normalisation[1] * np.exp(1j * self.__template_spectrum_ce[1])
        synth_ce_lin = self.__template_spectrum_ce_lin[0] * normalisation[2] * np.exp(
            1j * self.__template_spectrum_ce_lin[1]
        )

        return np.fft.irfft(np.stack((synth_geo, synth_ce, synth_ce_lin)), norm='ortho', axis=-1)


class slicedShower:
    def __init__(self, file_path, slicing_grammage=5):
        self.__slice_gram = slicing_grammage  # g/cm2
        self.__file = file_path

        self._trace_length = None
        self._GH_parameters = None
        self._magnetic_field = None

        self.ant_names = None
        self.nr_slices = None

        self.__parse_hdf5()

    def __parse_hdf5(self):
        file = h5py.File(self.__file)

        self.ant_names = set([key.split('x')[0] for key in file['CoREAS']['observers'].keys()])
        self.nr_slices = len(file['CoREAS']['observers'].keys()) // len(self.ant_names)

        self._trace_length = len(file['CoREAS']['observers'][f'{next(iter(self.ant_names))}x{self.__slice_gram}'])
        self._GH_parameters = file['atmosphere'].attrs['Gaisser-Hillas-Fit']
        self._magnetic_field = np.array([0, file['inputs'].attrs['MAGNET'][0], -1 * file['inputs'].attrs['MAGNET'][1]])

        file.close()

    @property
    def xmax(self):
        if self._GH_parameters is not None:
            return self._GH_parameters[2]

    @property
    def magnet(self):
        if self._magnetic_field is not None:
            return self._magnetic_field

    def get_trace(self, ant_name):
        if ant_name not in self.ant_names:
            raise ValueError(f"Antenna name {ant_name} is not present in shower")

        file = h5py.File(self.__file)

        zenith = file['inputs'].attrs['THETAP'][0] * units.deg
        azimuth = file['inputs'].attrs['PHIP'][0] * units.deg - 90 * units.deg  # transform to radiotools coord

        transformer = radiotools.coordinatesystems.cstrafo(
            zenith / units.rad, azimuth / units.rad,
            magnetic_field_vector=self._magnetic_field
        )

        antenna_ground = file['CoREAS']['observers'][f'{ant_name}x{self.__slice_gram}'].attrs['position'] * units.cm
        antenna_vvB = transformer.transform_to_vxB_vxvxB(
            np.array([-antenna_ground[1], antenna_ground[0], antenna_ground[2]])
        )

        traces_geo = np.zeros((self.nr_slices, self._trace_length))
        traces_ce = np.zeros((self.nr_slices, self._trace_length))
        for i_slice in range(self.nr_slices):
            g_slice = (i_slice + 1) * self.__slice_gram

            trace_slice = file['CoREAS']['observers'][f'{ant_name}x{g_slice}'][:] * c_vacuum * 1e2  # samples x 4
            trace_slice *= units.microvolt / units.m

            trace_slice_ground = np.matrix(
                [-trace_slice[:, 2], trace_slice[:, 1], trace_slice[:, 3]]
            )
            trace_slice_vvB = transformer.transform_to_vxB_vxvxB(trace_slice_ground).T

            # unit of pos does not matter, this is divided away
            traces_geo[i_slice] = e_geo(trace_slice_vvB, *antenna_vvB[:2])
            traces_ce[i_slice] = e_ce(trace_slice_vvB, *antenna_vvB[:2])

        file.close()

        return traces_geo, traces_ce

    def get_long_profile(self):
        file = h5py.File(self.__file)

        long_table = file['atmosphere']['NumberOfParticles'][:]
        long_profile = np.sum(long_table[:, 2:4], axis=1)

        file.close()

        return long_profile

    def get_coreas_settings(self):
        file = h5py.File(self.__file)

        time_resolution = file['CoREAS'].attrs['TimeResolution'] * units.s

        return {
            'time_resolution': time_resolution
        }


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

        self.__zenith = None
        self.__azimuth = None
        self.__magnet = None

    # Define zenith and azimuth as properties to protect them from getting written
    @property
    def zenith(self):
        return self.__zenith / units.deg

    @property
    def azimuth(self):
        return self.__azimuth / units.deg

    def begin(self, spectral_parameter_file, slicing_grammage=5):
        spectral_file = h5py.File(spectral_parameter_file)

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

        spectral_file.close()

    def make_template(self, origin_shower: slicedShower):
        shower_long_profile = origin_shower.get_long_profile()
        shower_sampling_res = origin_shower.get_coreas_settings()['time_resolution']

        # Save magnetic field from origin shower
        self.__magnet = origin_shower.magnet

        length_saved = False
        for antenna in self.__antennas:
            geo, ce = origin_shower.get_trace(antenna.name)

            if not length_saved:
                # Save trace length only once
                self.__trace_length = len(geo)
                length_saved = True

            # Normalise the traces with the particle numbers in the slice
            # Sometimes the LONG table can contain an entry inside the ground, so index using nr of slices
            geo /= shower_long_profile[:self.__n_slices, np.newaxis]
            ce /= shower_long_profile[:self.__n_slices, np.newaxis]

            antenna.set_template(origin_shower.xmax, geo, ce, sampling_d=shower_sampling_res)

    @register_run()
    def run(self, target_xmax, long_profile):
        assert len(long_profile) == self.__n_slices, "Long profile does not have correct number of slices"

        transformer = radiotools.coordinatesystems.cstrafo(
            self.__zenith / units.rad, self.__azimuth / units.rad,
            magnetic_field_vector=self.__magnet
        )

        traces_synth = np.zeros((len(self.__antennas), self.__trace_length, 2))
        for ind, antenna in enumerate(self.__antennas):
            synth = antenna.map_template(target_xmax)

            geo = synth[0] * long_profile[:, np.newaxis]  # SLICES x SAMPLES
            ce = synth[1] * long_profile[:, np.newaxis]
            ce_lin = synth[2] * long_profile[:, np.newaxis]

            # Antenna position in vvB
            antenna_vvB = transformer.transform_to_vxB_vxvxB(
                np.array([-antenna.position[1], antenna.position[0], antenna.position[2]])
            )

            # Sum all slices
            traces_synth[ind, :, 0] = np.sum(geo_ce_to_e(geo, ce, *antenna_vvB[:2]), axis=0)
            traces_synth[ind, :, 1] = np.sum(geo_ce_to_e(geo, ce_lin, *antenna_vvB[:2]), axis=0)

    def end(self):
        pass
