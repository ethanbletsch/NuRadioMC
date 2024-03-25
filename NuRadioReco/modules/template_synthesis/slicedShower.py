import logging

import h5py
import numpy as np
import radiotools.coordinatesystems

from scipy.constants import c as c_vacuum

from NuRadioReco.utilities import units

logger = logging.getLogger("NuRadioReco.slicedShower")


def e_geo(traces, x, y):
    return traces[:, 1] * x / y - traces[:, 0]


def e_ce(traces, x, y):
    return -traces[:, 1] * np.sqrt(x ** 2 + y ** 2) / y


def filter_trace(trace, trace_sampling, f_min, f_max, sample_axis=0):
    # Assuming `trace_sampling` has the correct internal unit, freq is already in the internal unit system
    freq = np.fft.rfftfreq(trace.shape[sample_axis], trace_sampling)
    freq_range = np.logical_and(freq > f_min, freq < f_max)

    # Find the median maximum sample number of the traces
    max_index = np.median(np.argmax(trace, axis=sample_axis))
    to_roll = int(trace.shape[sample_axis] / 2 - max_index)

    # Roll all traces such that max is in the middle
    roll_pulse = np.roll(trace, to_roll)

    # FFT, filter, IFFT
    spectrum = np.fft.rfft(roll_pulse, axis=sample_axis)
    spectrum = np.apply_along_axis(lambda ax: ax * freq_range.astype('int'), sample_axis, spectrum)
    filtered = np.fft.irfft(spectrum, axis=sample_axis)

    return np.roll(filtered, -to_roll)


class slicedShower:
    """
    This class can be used to read in an HDF5 file of sliced CoREAS simulation. It can read in the traces
    of all slices for a given antenna and return them in a Numpy array.
    """
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

    @property
    def slice_grammage(self):
        return self.__slice_gram

    def filter_trace(self, trace, f_min, f_max):
        trace_axis = -1  # based on self.get_trace()
        if trace.shape[trace_axis] != self._trace_length:
            logger.warning("Trace shape does not match recorded trace length along the last axis")
            logger.status("Attempting to find the trace axis...")
            for shape_i in range(len(trace.shape)):
                if trace.shape[shape_i] == self._trace_length:
                    logger.status(f"Found axis {shape_i} which matches trace length!")
                    trace_axis = shape_i
                    break
        return filter_trace(trace, self.get_coreas_settings()['time_resolution'], f_min, f_max, sample_axis=trace_axis)

    def get_trace(self, ant_name):
        """
        Get the traces from all slices for a given antenna. The traces are converted to GEO/CE components.

        Parameters
        ----------
        ant_name : str
            The name of the antenna. Must be the same as the key in the HDF5!

        Returns
        -------
        traces_geo : ndarray
            The geomagnetic traces, shaped as (slices, samples)
        traces_ce : ndarray
            The charge-excess traces, shaped as (slices, samples)
        """
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

            trace_slice_ground = np.array(
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
