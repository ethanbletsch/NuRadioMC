import numpy as np
from NuRadioReco.utilities import units
import scipy.signal


class channelBandPassFilter:
    """
    Band pass filters the channels using different band-pass filters.
    """

    def begin(self):
        pass

    def run(self, evt, station, det, passband=[55 * units.MHz, 1000 * units.MHz],
            filter_type='rectangular'):
        """
        Run the filter

        Parameters
        ---------

        evt, station, det
            Event, Station, Detector
        passband: list
            passband[0]: lower boundary of filter, passband[1]: upper boundary of filter
        filter_type: string
            'rectangular': perfect straight line filter
            'butter10': butterworth filter from scipy
            'butter10abs': absolute of butterworth filter from scipy

        """
        channels = station.get_channels()
        for channel in channels:
            frequencies = channel.get_frequencies()
            trace_fft = channel.get_frequency_spectrum()

            if(filter_type == 'rectangular'):
                trace_fft[np.where(frequencies < passband[0])] = 0.
                trace_fft[np.where(frequencies > passband[1])] = 0.
            elif(filter_type == 'butter10'):
                b, a = scipy.signal.butter(10, passband, 'bandpass', analog=True)
                w, h = scipy.signal.freqs(b, a, frequencies)
                trace_fft *= h
            elif(filter_type == 'butter10abs'):
                b, a = scipy.signal.butter(10, passband, 'bandpass', analog=True)
                w, h = scipy.signal.freqs(b, a, frequencies)
                trace_fft *= np.abs(h)
            channel.set_frequency_spectrum(trace_fft, channel.get_sampling_rate())

    def end(self):
        pass
