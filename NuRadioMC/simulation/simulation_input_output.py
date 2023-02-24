import h5py
import numpy as np
import os.path
from six import iteritems
import six
import logging
import yaml
import time
import NuRadioMC.simulation.simulation_base
import NuRadioReco.framework.particle
from NuRadioReco.framework.parameters import particleParameters as simp
from NuRadioReco.framework.parameters import showerParameters as shp


logger = logging.getLogger('NuRadioMC')


class simulation_input_output(NuRadioMC.simulation.simulation_base.simulation_base):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super(simulation_input_output, self).__init__(*args, **kwargs)

    def _read_input_hdf5(self):
        """
        reads input file into memory
        """
        fin = h5py.File(self._inputfilename, 'r')
        self._fin = {}
        self._fin_stations = {}
        self._fin_attrs = {}
        for key, value in iteritems(fin):
            if isinstance(value, h5py._hl.group.Group):
                self._fin_stations[key] = {}
                for key2, value2 in iteritems(value):
                    self._fin_stations[key][key2] = np.array(value2)
            else:
                if len(value) and type(value[0]) == bytes:
                    self._fin[key] = np.array(value).astype('U')
                else:
                    self._fin[key] = np.array(value)
        for key, value in iteritems(fin.attrs):
            self._fin_attrs[key] = value

        fin.close()

    def _write_output_file(self, empty=False):
        folder = os.path.dirname(self._outputfilename)
        if not os.path.exists(folder) and folder != '':
            logger.warning(f"output folder {folder} does not exist, creating folder...")
            os.makedirs(folder)
        fout = h5py.File(self._outputfilename, 'w')

        if not empty:
            # here we add the first interaction to the saved events
            # if any of its children triggered

            # Careful! saved should be a copy of the triggered array, and not
            # a reference! saved indicates the interactions to be saved, while
            # triggered should indicate if an interaction has produced a trigger
            saved = np.copy(self._mout['triggered'])
            if 'n_interactions' in self._fin:  # if n_interactions is not specified, there are not parents
                parent_mask = self._fin['n_interaction'] == 1
                for event_id in np.unique(self._fin['event_group_ids']):
                    event_mask = self._fin['event_group_ids'] == event_id
                    if True in self._mout['triggered'][event_mask]:
                        saved[parent_mask & event_mask] = True

            logger.status("start saving events")
            # save data sets
            for (key, value) in iteritems(self._mout):
                fout[key] = value[saved]

            # save all data sets of the station groups
            for (key, value) in iteritems(self._mout_groups):
                sg = fout.create_group("station_{:d}".format(key))
                for (key2, value2) in iteritems(value):
                    sg[key2] = np.array(value2)[np.array(value['triggered'])]

            # save "per event" quantities
            if 'trigger_names' in self._mout_attrs:
                n_triggers = len(self._mout_attrs['trigger_names'])
                for station_id in self._mout_groups:
                    n_events_for_station = len(self._output_triggered_station[station_id])
                    if n_events_for_station > 0:
                        n_channels = self._det.get_number_of_channels(station_id)
                        sg = fout["station_{:d}".format(station_id)]
                        sg['event_group_ids'] = np.array(self._output_event_group_ids[station_id])
                        sg['event_ids'] = np.array(self._output_sub_event_ids[station_id])
                        sg['maximum_amplitudes'] = np.array(self._output_maximum_amplitudes[station_id])
                        sg['maximum_amplitudes_envelope'] = np.array(self._output_maximum_amplitudes_envelope[station_id])
                        sg['triggered_per_event'] = np.array(self._output_triggered_station[station_id])

                        # the multiple triggeres 2d array might have different number of entries per event
                        # because the number of different triggers can increase dynamically
                        # therefore we first create an array with the right size and then fill it
                        tmp = np.zeros((n_events_for_station, n_triggers), dtype=np.bool)
                        for iE, values in enumerate(self._output_multiple_triggers_station[station_id]):
                            tmp[iE] = values
                        sg['multiple_triggers_per_event'] = tmp
                        tmp_t = np.nan * np.zeros_like(tmp, dtype=float)
                        for iE, values in enumerate(self._output_trigger_times_station[station_id]):
                            tmp_t[iE] = values
                        sg['trigger_times_per_event'] = tmp_t

        # save meta arguments
        for (key, value) in iteritems(self._mout_attrs):
            fout.attrs[key] = value

        with open(self._detectorfile, 'r') as fdet:
            fout.attrs['detector'] = fdet.read()

        if not empty:
            # save antenna position separately to hdf5 output
            for station_id in self._mout_groups:
                n_channels = self._det.get_number_of_channels(station_id)
                positions = np.zeros((n_channels, 3))
                for channel_id in self._det.get_channel_ids(station_id):
                    positions[self._get_channel_index(channel_id)] = self._det.get_relative_position(station_id, channel_id) + self._det.get_absolute_position(station_id)
                fout["station_{:d}".format(station_id)].attrs['antenna_positions'] = positions
                fout["station_{:d}".format(station_id)].attrs['Vrms'] = list(self._Vrms_per_channel[station_id].values())
                fout["station_{:d}".format(station_id)].attrs['bandwidth'] = list(self._bandwidth_per_channel[station_id].values())

            fout.attrs.create("Tnoise", self._noise_temp, dtype=np.float)
            fout.attrs.create("Vrms", self._Vrms, dtype=np.float)
            fout.attrs.create("dt", self._dt, dtype=np.float)
            fout.attrs.create("bandwidth", self._bandwidth, dtype=np.float)
            fout.attrs['n_samples'] = self._n_samples
        fout.attrs['config'] = yaml.dump(self._cfg)

        # save NuRadioMC and NuRadioReco versions
        from NuRadioReco.utilities import version
        import NuRadioMC
        fout.attrs['NuRadioMC_version'] = NuRadioMC.__version__
        fout.attrs['NuRadioMC_version_hash'] = version.get_NuRadioMC_commit_hash()

        if not empty:
            # now we also save all input parameters back into the out file
            for key in self._fin.keys():
                if key.startswith("station_"):
                    continue
                if not key in fout.keys():  # only save data sets that havn't been recomputed and saved already
                    if np.array(self._fin[key]).dtype.char == 'U':
                        fout[key] = np.array(self._fin[key], dtype=h5py.string_dtype(encoding='utf-8'))[saved]

                    else:
                        fout[key] = np.array(self._fin[key])[saved]

        for key in self._fin_attrs.keys():
            if not key in fout.attrs.keys():  # only save atrributes sets that havn't been recomputed and saved already
                if key not in ["trigger_names", "Tnoise", "Vrms", "bandwidth", "n_samples", "dt", "detector", "config"]:  # don't write trigger names from input to output file, this will lead to problems with incompatible trigger names when merging output files
                    fout.attrs[key] = self._fin_attrs[key]
        fout.close()

    def _create_meta_output_datastructures(self):
        """
        creates the data structures of the parameters that will be saved into the hdf5 output file
        """
        self._mout = {}
        self._mout_attributes = {}
        self._mout['weights'] = np.zeros(self._n_showers)
        self._mout['triggered'] = np.zeros(self._n_showers, dtype=np.bool)
#         self._mout['multiple_triggers'] = np.zeros((self._n_showers, self._number_of_triggers), dtype=np.bool)
        self._mout_attributes['trigger_names'] = None
        self._amplitudes = {}
        self._amplitudes_envelope = {}
        self._output_triggered_station = {}
        self._output_event_group_ids = {}
        self._output_sub_event_ids = {}
        self._output_multiple_triggers_station = {}
        self._output_maximum_amplitudes = {}
        self._output_maximum_amplitudes_envelope = {}
        self._output_trigger_times_station = {}
        for station_id in self._station_ids:
            self._mout_groups[station_id] = {}
            self._output_event_group_ids[station_id] = []
            self._output_sub_event_ids[station_id] = []
            self._output_triggered_station[station_id] = []
            self._output_multiple_triggers_station[station_id] = []
            self._output_maximum_amplitudes[station_id] = []
            self._output_maximum_amplitudes_envelope[station_id] = []
            self._output_trigger_times_station[station_id] = []

    def _create_station_output_structure(self, n_showers, n_antennas):
        nS = self._raytracer.get_number_of_raytracing_solutions()  # number of possible ray-tracing solutions
        station_output_structure = {}
        station_output_structure['triggered'] = np.zeros(n_showers, dtype=np.bool)
        # we need the reference to the shower id to be able to find the correct shower in the upper level hdf5 file
        station_output_structure['shower_id'] = np.zeros(n_showers, dtype=int) * -1
        station_output_structure['event_id_per_shower'] = np.zeros(n_showers, dtype=int) * -1
        station_output_structure['event_group_id_per_shower'] = np.zeros(n_showers, dtype=int) * -1
        station_output_structure['launch_vectors'] = np.zeros((n_showers, n_antennas, nS, 3)) * np.nan
        station_output_structure['receive_vectors'] = np.zeros((n_showers, n_antennas, nS, 3)) * np.nan
        station_output_structure['polarization'] = np.zeros((n_showers, n_antennas, nS, 3)) * np.nan
        station_output_structure['travel_times'] = np.zeros((n_showers, n_antennas, nS)) * np.nan
        station_output_structure['travel_distances'] = np.zeros((n_showers, n_antennas, nS)) * np.nan
        if self._cfg['speedup']['amp_per_ray_solution']:
            station_output_structure['max_amp_shower_and_ray'] = np.zeros((n_showers, n_antennas, nS))
            station_output_structure['time_shower_and_ray'] = np.zeros((n_showers, n_antennas, nS))
        for parameter_entry in self._raytracer.get_output_parameters():
            if parameter_entry['ndim'] == 1:
                station_output_structure[parameter_entry['name']] = np.zeros((n_showers, n_antennas, nS)) * np.nan
            else:
                station_output_structure[parameter_entry['name']] = np.zeros((n_showers, n_antennas, nS, parameter_entry['ndim'])) * np.nan
        return station_output_structure

    def _read_input_particle_properties(self, idx=None):
        if idx is None:
            idx = self._primary_index
        self._event_group_id = self._fin['event_group_ids'][idx]

        self.input_particle = NuRadioReco.framework.particle.Particle(0)
        self.input_particle[simp.flavor] = self._fin['flavors'][idx]
        self.input_particle[simp.energy] = self._fin['energies'][idx]
        self.input_particle[simp.interaction_type] = self._fin['interaction_type'][idx]
        self.input_particle[simp.inelasticity] = self._fin['inelasticity'][idx]
        self.input_particle[simp.vertex] = np.array([self._fin['xx'][idx],
                                                     self._fin['yy'][idx],
                                                     self._fin['zz'][idx]])
        self.input_particle[simp.zenith] = self._fin['zeniths'][idx]
        self.input_particle[simp.azimuth] = self._fin['azimuths'][idx]
        self.input_particle[simp.inelasticity] = self._fin['inelasticity'][idx]
        self.input_particle[simp.n_interaction] = self._fin['n_interaction'][idx]
        if self._fin['n_interaction'][idx] <= 1:
            # parents before the neutrino and outgoing daughters without shower are currently not
            # simulated. The parent_id is therefore at the moment only rudimentarily populated.
            self.input_particle[simp.parent_id] = None  # primary does not have a parent

        self.input_particle[simp.vertex_time] = 0
        if 'vertex_times' in self._fin:
            self.input_particle[simp.vertex_time] = self._fin['vertex_times'][idx]

    def _read_input_shower_properties(self):
        """ read in the properties of the shower with index _shower_index from input """
        self._event_group_id = self._fin['event_group_ids'][self._shower_index]

        self._shower_vertex = np.array([self._fin['xx'][self._shower_index],
                                        self._fin['yy'][self._shower_index],
                                        self._fin['zz'][self._shower_index]])

        self._vertex_time = 0
        if 'vertex_times' in self._fin:
            self._vertex_time = self._fin['vertex_times'][self._shower_index]

    def _save_triggers_to_hdf5(self, event, station, output_data, local_shower_index, global_shower_index):
        extend_array = self._create_trigger_structures(station)
        # now we also need to create the trigger structure also in the sg (station group) dictionary that contains
        # the information fo the current station and event group
        n_showers = output_data['launch_vectors'].shape[0]
        if 'multiple_triggers' not in output_data:
            output_data['multiple_triggers'] = np.zeros((n_showers, len(self._mout_attrs['trigger_names'])), dtype=np.bool)
            output_data['trigger_times'] = np.nan * np.zeros_like(output_data['multiple_triggers'], dtype=float)
        elif extend_array:
            tmp = np.zeros((n_showers, len(self._mout_attrs['trigger_names'])), dtype=np.bool)
            nx, ny = output_data['multiple_triggers'].shape
            tmp[:, 0:ny] = output_data['multiple_triggers']
            output_data['multiple_triggers'] = tmp
            # repeat for trigger times
            tmp_t = np.nan * np.zeros_like(tmp, dtype=float)
            tmp_t[:, :ny] = output_data['trigger_times']
            output_data['trigger_times'] = tmp_t
        self._output_event_group_ids[self._station_id].append(event.get_run_number())
        self._output_sub_event_ids[self._station_id].append(event.get_id())
        multiple_triggers = np.zeros(len(self._mout_attrs['trigger_names']), dtype=np.bool)
        trigger_times = np.nan * np.zeros_like(multiple_triggers)
        for iT, trigger_name in enumerate(self._mout_attrs['trigger_names']):
            if station.has_trigger(trigger_name):
                multiple_triggers[iT] = station.get_trigger(trigger_name).has_triggered()
                trigger_times[iT] = station.get_trigger(trigger_name).get_trigger_time()
                for iSh in local_shower_index:  # now save trigger information per shower of the current station
                    output_data['multiple_triggers'][iSh][iT] = station.get_trigger(trigger_name).has_triggered()
                    output_data['trigger_times'][iSh][iT] = trigger_times[iT]
        for iSh, iSh2 in zip(local_shower_index, global_shower_index):  # now save trigger information per shower of the current station
            output_data['triggered'][iSh] = np.any(output_data['multiple_triggers'][iSh])
            self._mout['triggered'][iSh2] |= output_data['triggered'][iSh]
            self._mout['multiple_triggers'][iSh2] |= output_data['multiple_triggers'][iSh]
            self._mout['trigger_times'][iSh2] = np.fmin(self._mout['trigger_times'][iSh2], output_data['trigger_times'][iSh])
        output_data['event_id_per_shower'][local_shower_index] = event.get_id()
        output_data['event_group_id_per_shower'][local_shower_index] = event.get_run_number()
        self._output_multiple_triggers_station[self._station_id].append(multiple_triggers)
        self._output_trigger_times_station[self._station_id].append(trigger_times)
        self._output_triggered_station[self._station_id].append(np.any(multiple_triggers))

    def _create_empty_multiple_triggers(self):
        if 'trigger_names' not in self._mout_attrs:
            self._mout_attrs['trigger_names'] = np.array([])
            self._mout['multiple_triggers'] = np.zeros((self._n_showers, 1), dtype=np.bool)
            for station_id in self._station_ids:
                n_showers = self._mout_groups[station_id]['launch_vectors'].shape[0]
                self._mout_groups[station_id]['multiple_triggers'] = np.zeros((n_showers, 1), dtype=np.bool)
                self._mout_groups[station_id]['triggered'] = np.zeros(n_showers, dtype=np.bool)

    def _create_trigger_structures(
            self,
            station
    ):

        if 'trigger_names' not in self._mout_attrs:
            self._mout_attrs['trigger_names'] = []
        extend_array = False
        for trigger in six.itervalues(station.get_triggers()):
            if trigger.get_name() not in self._mout_attrs['trigger_names']:
                self._mout_attrs['trigger_names'].append((trigger.get_name()))
                extend_array = True
        # the 'multiple_triggers' output array is not initialized in the constructor because the number of
        # simulated triggers is unknown at the beginning. So we check if the key already exists and if not,
        # we first create this data structure
        if 'multiple_triggers' not in self._mout:
            self._mout['multiple_triggers'] = np.zeros((self._n_showers, len(self._mout_attrs['trigger_names'])), dtype=np.bool)
            self._mout['trigger_times'] = np.nan * np.zeros_like(self._mout['multiple_triggers'], dtype=float)
        elif extend_array:
            tmp = np.zeros((self._n_showers, len(self._mout_attrs['trigger_names'])), dtype=np.bool)
            nx, ny = self._mout['multiple_triggers'].shape
            tmp[:, 0:ny] = self._mout['multiple_triggers']
            self._mout['multiple_triggers'] = tmp
        return extend_array

    def _create_event_structure(
            self,
            iEvent,
            indices,
            channel_identifiers
    ):
        evt = NuRadioReco.framework.event.Event(self._event_group_id, iEvent)  # create new event

        if self._particle_mode:
            # add MC particles that belong to this (sub) event to event structure
            # add only primary for now, since full interaction chain is not typically in the input hdf5s
            evt.add_particle(self.primary)
        # copy over generator information from temporary event to event
        evt._generator_info = self._generator_info

        new_station = NuRadioReco.framework.station.Station(self._station_id)
        sim_station = NuRadioReco.framework.sim_station.SimStation(self._station_id)
        sim_station.set_is_neutrino()
        self._shower_ids_of_sub_event = []
        for iCh in indices:
            ch_uid = channel_identifiers[iCh]
            shower_id = ch_uid[1]
            if shower_id not in self._shower_ids_of_sub_event:
                self._shower_ids_of_sub_event.append(shower_id)
            sim_station.add_channel(self._station.get_sim_station().get_channel(ch_uid))
            efield_uid = ([ch_uid[0]], ch_uid[1], ch_uid[
                2])  # the efield unique identifier has as first parameter an array of the channels it is valid for
            for efield in self._station.get_sim_station().get_electric_fields():
                if efield.get_unique_identifier() == efield_uid:
                    sim_station.add_electric_field(efield)
        if self._particle_mode:
            # add showers that contribute to this (sub) event to event structure
            for shower_id in self._shower_ids_of_sub_event:
                evt.add_sim_shower(self._evt_tmp.get_sim_shower(shower_id))
        new_station.set_sim_station(sim_station)
        new_station.set_station_time(self._evt_time)
        evt.set_station(new_station)
        return evt, new_station
    def _write_nur_file(
            self,
            event,
            station
    ):
        # downsample traces to detector sampling rate to save file size
        self._channelResampler.run(event, station, self._det, sampling_rate=self._sampling_rate_detector)
        self._channelResampler.run(event, station.get_sim_station(), self._det,
                                   sampling_rate=self._sampling_rate_detector)
        self._electricFieldResampler.run(event, station.get_sim_station(), self._det,
                                         sampling_rate=self._sampling_rate_detector)

        output_mode = {'Channels': self._cfg['output']['channel_traces'],
                       'ElectricFields': self._cfg['output']['electric_field_traces'],
                       'SimChannels': self._cfg['output']['sim_channel_traces'],
                       'SimElectricFields': self._cfg['output']['sim_electric_field_traces']}
        if self._write_detector:
            self._eventWriter.run(event, self._det, mode=output_mode)
        else:
            self._eventWriter.run(event, mode=output_mode)

    def _write_progress_output(
            self,
            iCounter,
            i_event_group_id,
            unique_event_group_ids
    ):
        n_shower_station = len(self._station_ids) * self._n_showers
        eta = NuRadioMC.simulation.simulation_base.pretty_time_delta((time.time() - self._t_start) * (n_shower_station - iCounter) / iCounter)
        total_time_sum = self._input_time + self._rayTracingTime + self._detSimTime + self._outputTime + self._weightTime + self._distance_cut_time  # askaryan time is part of the ray tracing time, so it is not counted here.
        total_time = time.time() - self._t_start
        if total_time > 0:
            logger.status(
                "processing event group {}/{} and shower {}/{} ({} showers triggered) = {:.1f}%, ETA {}, time consumption: ray tracing = {:.0f}%, askaryan = {:.0f}%, detector simulation = {:.0f}% reading input = {:.0f}%, calculating weights = {:.0f}%, distance cut {:.0f}%, unaccounted = {:.0f}% ".format(
                    i_event_group_id,
                    len(unique_event_group_ids),
                    iCounter,
                    n_shower_station,
                    np.sum(self._mout['triggered']),
                    100. * iCounter / n_shower_station,
                    eta,
                    100. * (self._rayTracingTime - self._askaryan_time) / total_time,
                    100. * self._askaryan_time / total_time,
                    100. * self._detSimTime / total_time,
                    100. * self._input_time / total_time,
                    100. * self._weightTime / total_time,
                    100 * self._distance_cut_time / total_time,
                    100 * (total_time - total_time_sum) / total_time))