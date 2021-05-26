import numpy as np
import os, scipy, sys
import bz2
import _pickle as cPickle
import yaml
import helper_cr_eff as hcr
import scipy.constants
import datetime
import matplotlib.pyplot as plt
import pickle
import pygdsm
import astropy
from NuRadioReco.utilities import units, io_utilities
from NuRadioReco.detector.generic_detector import GenericDetector
import NuRadioReco.modules.io.coreas.readCoREASStation
import NuRadioReco.modules.io.coreas.simulationSelector
import NuRadioReco.modules.efieldToVoltageConverter
import NuRadioReco.modules.RNO_G.hardwareResponseIncorporator
import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.modules.channelGalacticNoiseAdder
import NuRadioReco.modules.trigger.envelopeTrigger
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.eventTypeIdentifier
import NuRadioReco.modules.channelStopFilter
import NuRadioReco.modules.channelSignalReconstructor
import NuRadioReco.modules.correlationDirectionFitter
import NuRadioReco.modules.voltageToEfieldConverter
import NuRadioReco.modules.electricFieldSignalReconstructor
import NuRadioReco.modules.voltageToAnalyticEfieldConverter
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.electricFieldResampler
import NuRadioReco.framework.event
import NuRadioReco.modules.io.eventWriter
import logging
import argparse
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('air_shower_reco')

'''
This script reconstructs the air shower (stored in air_shower_sim as hdf5 files) with 
the trigger parameters calculated in step 1-3. Please set triggered channels manually in l.96'''

parser = argparse.ArgumentParser(description='Run air shower Reconstruction')
parser.add_argument('config_file', type=str, nargs='?', default = 'config_file_air_shower_reco.yml', help = 'config file with eventlist')
parser.add_argument('result_dict', type=str, nargs='?', default = 'results/ntr/dict_ntr_high_low_pb_80_180.pbz2', help = 'settings from the ntr results')
parser.add_argument('number', type=int, nargs='?', default = 0, help = 'number of element in eventlist')

args = parser.parse_args()
config_file = args.config_file
result_dict = args.result_dict
number = args.number

with open(config_file, 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)

eventlist = cfg['eventlist']
output_filename = cfg['output_filename']
os.makedirs(output_filename, exist_ok=True)

bz2 = bz2.BZ2File(result_dict, 'rb')
data = cPickle.load(bz2)

detector_file = data['detector_file']
default_station = data['default_station']
sampling_rate = data['sampling_rate'] * units.gigahertz
station_time = data['station_time']
station_time_random = data['station_time_random']

Vrms_thermal_noise = data['Vrms_thermal_noise'] * units.volt
T_noise = data['T_noise'] * units.kelvin
T_noise_min_freq = data['T_noise_min_freq'] * units.megahertz
T_noise_max_freq = data['T_noise_max_freq '] * units.megahertz

galactic_noise_n_side = data['galactic_noise_n_side']
galactic_noise_interpolation_frequencies_start = data['galactic_noise_interpolation_frequencies_start']
galactic_noise_interpolation_frequencies_stop = data['galactic_noise_interpolation_frequencies_stop']
galactic_noise_interpolation_frequencies_step = data['galactic_noise_interpolation_frequencies_step']

trigger_name = data['trigger_name']
passband_trigger = data['passband_trigger']
number_coincidences = data['number_coincidences']
coinc_window = data['coinc_window'] * units.ns
order_trigger = data['order_trigger']
trigger_thresholds = data['threshold']
n_iterations = data['iteration']
hardware_response = data['hardware_response']

trigger_rate = data['trigger_rate']
threshold_tested = data['threshold']
zeros = np.where(trigger_rate == 0)[0]
first_zero = zeros[0]  # gives the index of the element where the trigger rate is zero for the first time
trigger_threshold = threshold_tested[first_zero] * units.volt

input_files = eventlist[number]
if(default_station == 101):
    triggered_channels = [16, 19, 22]
    used_channels_efield = [16, 19, 22]
    used_channels_fit = [16, 19, 22]
    channel_pairs = ((16, 19), (16, 22), (19, 22))

elif(default_station == 32):
    triggered_channels = [0, 1, 2, 3]
    used_channels_efield = [0, 1, 2, 3]
    used_channels_fit = [0, 1, 2, 3]
    channel_pairs = ((0, 1), (1, 2), (3, 1))

else:
    logger.info("Default channels not defined for station_id != 101")

det = GenericDetector(json_filename=detector_file, default_station=default_station) # detector file
det.update(datetime.datetime(2019, 10, 1))

station_ids = det.get_station_ids()
station_id = station_ids[0]
channel_ids = det.get_channel_ids(station_id)

dir_path = os.path.dirname(os.path.realpath(__file__)) # get the directory of this file

# initialize all modules that are needed for processing
# provide input parameters that are to remain constant during processung
readCoREASStation = NuRadioReco.modules.io.coreas.readCoREASStation.readCoREASStation()
readCoREASStation.begin([input_files], default_station, debug=False)
simulationSelector = NuRadioReco.modules.io.coreas.simulationSelector.simulationSelector()
simulationSelector.begin()
efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
efieldToVoltageConverter.begin(debug=False)
hardwareResponseIncorporator = NuRadioReco.modules.RNO_G.hardwareResponseIncorporator.hardwareResponseIncorporator()
channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
channelGenericNoiseAdder.begin()
channelGalacticNoiseAdder = NuRadioReco.modules.channelGalacticNoiseAdder.channelGalacticNoiseAdder()
channelGalacticNoiseAdder.begin(n_side=galactic_noise_n_side,
                                interpolation_frequencies=
                                np.arange(galactic_noise_interpolation_frequencies_start,
                                          galactic_noise_interpolation_frequencies_stop,
                                          galactic_noise_interpolation_frequencies_step) * units.MHz)
if trigger_name == 'high_low':
    triggerSimulator = NuRadioReco.modules.trigger.highLowThreshold.triggerSimulator()
    triggerSimulator.begin()

if trigger_name == 'envelope':
    triggerSimulator = NuRadioReco.modules.trigger.envelopeTrigger.triggerSimulator()
    triggerSimulator.begin()

channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelBandPassFilter.begin()
eventTypeIdentifier = NuRadioReco.modules.eventTypeIdentifier.eventTypeIdentifier()
channelStopFilter = NuRadioReco.modules.channelStopFilter.channelStopFilter()
channelSignalReconstructor = NuRadioReco.modules.channelSignalReconstructor.channelSignalReconstructor()
channelSignalReconstructor.begin()
correlationDirectionFitter = NuRadioReco.modules.correlationDirectionFitter.correlationDirectionFitter()
voltageToEfieldConverter = NuRadioReco.modules.voltageToEfieldConverter.voltageToEfieldConverter()
electricFieldSignalReconstructor = NuRadioReco.modules.electricFieldSignalReconstructor.electricFieldSignalReconstructor()
electricFieldSignalReconstructor.begin()
voltageToAnalyticEfieldConverter = NuRadioReco.modules.voltageToAnalyticEfieldConverter.voltageToAnalyticEfieldConverter()
voltageToAnalyticEfieldConverter.begin()
electricFieldResampler = NuRadioReco.modules.electricFieldResampler.electricFieldResampler()
electricFieldResampler.begin()
channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
channelResampler.begin()
eventWriter = NuRadioReco.modules.io.eventWriter.eventWriter()
eventWriter.begin(output_filename + 'pb_' + str(int(passband_trigger[0]/units.MHz)) + '_' + str(int(passband_trigger[1]/units.MHz)) + '_tt_' + str(trigger_threshold.round(6)) + '.nur')


# Loop over all events in file as initialized in readCoRREAS and perform analysis
i = 0
for evt in readCoREASStation.run(det):
    for sta in evt.get_stations():
        #if i == 30: # use this if you want to test something or if you want only 30 position
        #   break
        logger.info("processing event {:d} with id {:d}".format(i, evt.get_id()))

        station = evt.get_station(default_station)
        if station_time_random == True:
            station = hcr.set_random_station_time(station)

        efieldToVoltageConverter.run(evt, sta, det)
        eventTypeIdentifier.run(evt, sta, "forced", 'cosmic_ray')
        channelGenericNoiseAdder.run(evt, sta, det, amplitude=Vrms_thermal_noise, min_freq=T_noise_min_freq,
                                 max_freq=T_noise_max_freq, type='rayleigh', bandwidth=None)

        channelGalacticNoiseAdder.run(evt, sta, det)

        if hardware_response == True:
            hardwareResponseIncorporator.run(evt, sta, det, sim_to_data=True)

        # The bandpass for the envelope trigger is included in the trigger module,
        # in the high low the filter is applied externally
        if trigger_name == 'high_low':
            channelBandPassFilter.run(evt, sta, det, passband=passband_trigger,
                                      filter_type='butter', order=order_trigger)

            triggerSimulator.run(evt, sta, det, threshold_high=trigger_threshold, threshold_low=-trigger_threshold,
                                 coinc_window=coinc_window, number_concidences=number_coincidences,
                                 triggered_channels=triggered_channels, trigger_name='{}_pb_{:.0f}_{:.0f}_tt_{:.2f}'.format(trigger_name ,passband_trigger[0]/units.MHz, passband_trigger[1]/units.MHz, trigger_threshold/units.mV))

        if trigger_name == 'envelope':
            triggerSimulator.run(evt, sta, det, passband=passband_trigger, order=order_trigger,
                             number_coincidences=number_coincidences, threshold=trigger_threshold,
                             coinc_window=coinc_window, triggered_channels=triggered_channels,
                trigger_name='{}_pb_{:.0f}_{:.0f}_tt_{:.2f}'.format(trigger_name, passband_trigger[0]/units.MHz, passband_trigger[1]/units.MHz, trigger_threshold/units.mV))

        ##channelSignalReconstructor.run(evt, sta, det)

        ##correlationDirectionFitter.run(evt, sta, det, n_index=1., channel_pairs=channel_pairs)

        #voltageToEfieldConverter.run(evt, sta, det, use_channels=used_channels_efield)

        ##electricFieldSignalReconstructor.run(evt, sta, det)

        #voltageToAnalyticEfieldConverter.run(evt, sta, det, use_channels=used_channels_efield, bandpass=[80*units.MHz, 500*units.MHz], useMCdirection=False)

        channelResampler.run(evt, sta, det, sampling_rate=1 * units.GHz)

        electricFieldResampler.run(evt, sta, det, sampling_rate=1 * units.GHz)
        i += 1

    #eventWriter.run(evt)
    eventWriter.run(evt, mode='micro')  # here you can change what should be stored in the nur files

nevents = eventWriter.end()
