import numpy as np
from NuRadioReco.utilities import units

class stationSimulator:
    def __init__(
            self,
            detector,
            channel_ids,
            config,
            input_data,
            input_attributes,
            input_stations,
            shower_simulator,
            raytracer,
            was_pre_simulated
    ):
        self.__detector = detector
        self.__channel_ids = channel_ids
        self.__config = config
        self.__input_data = input_data
        self.__input_attributes = input_attributes
        self.__input_stations = input_stations
        self.__shower_simulator = shower_simulator
        self.__raytracer = raytracer
        self.__was_pre_simulated = was_pre_simulated
        self.__event_group_vertices = None
        self.__i_event_group = None
        self.__event_group_id = None
        self.__shower_indices = None
        if self.__config['speedup']['distance_cut']:
            self.__distance_cut_polynomial = np.polynomial.polynomial.Polynomial(self.__config['speedup']['distance_cut_coefficients'])
        else:
            self.__distance_cut_polynomial = None


    def set_event_group(
            self,
            i_event_group,
            event_group_id,
            shower_indices
    ):
        self.__shower_simulator.set_event_group(
            i_event_group,
            event_group_id,
            shower_indices
        )
        self.__i_event_group = i_event_group
        self.__event_group_id = event_group_id
        self.__shower_indices = shower_indices
        self.__event_group_vertices = np.array([
            np.array(self.__input_data['xx'])[shower_indices],
            np.array(self.__input_data['yy'])[shower_indices],
            np.array(self.__input_data['zz'])[shower_indices]
        ]).T

    def simulate_station(
            self,
            station_id
    ):
        if not self.__distance_cut(
            station_id
        ):
            return
        if self.__was_pre_simulated:
            ray_tracing_performed = False
        else:
            if 'station_{:d}'.format(station_id) in self.__input_stations:
                ray_tracing_performed = self.__raytracer.get_output_parameters()[0]['name'] in self.__input_stations['station_{:d}'.format(station_id)]
            else:
                ray_tracing_performed= False
        self.__shower_simulator.set_station(station_id)
        output_structure = self.__get_output_structure(station_id)
        efield_array = []
        n_showers = len(self.__shower_indices)
        n_antennas = len(self.__detector.get_channel_ids(station_id))
        n_raytracing_solutions = self.__raytracer.get_number_of_raytracing_solutions()  # number of possible ray-tracing solutions

        for i_shower, shower_index in enumerate(self.__shower_indices):
            efield_objects, launch_vectors, receive_vectors, travel_times, path_lengths, polarization_directions, \
                efield_amplitudes, raytracing_output = self.__shower_simulator.simulate_shower(
                self.__input_data['shower_ids'][shower_index],
                shower_index,
                self.__was_pre_simulated,
                ray_tracing_performed
            )
            efield_array.append(efield_objects)
            output_structure['launch_vectors'][i_shower] = launch_vectors
            output_structure['receive_vectors'][i_shower] = receive_vectors
            output_structure['travel_times'][i_shower] = travel_times
            output_structure['travel_distances'][i_shower] = path_lengths
            output_structure['polarization'][i_shower] = polarization_directions
            for key in raytracing_output:
                if key not in output_structure:
                    output_structure[key] = np.full((n_showers, n_antennas, n_raytracing_solutions), np.nan)
                output_structure[key][i_shower] = raytracing_output[key]
        return output_structure, efield_array
    def __distance_cut(
            self,
            station_id
    ):
        if not self.__config['speedup']['distance_cut']:
            return True
        channel_ids = self.__detector.get_channel_ids(station_id)
        channel_positions = np.zeros((len(channel_ids), 3))
        for i_channel, channel_id in enumerate(channel_ids):
            channel_positions[i_channel] = self.__detector.get_relative_position(station_id, channel_id)
        station_barycenter = np.mean(channel_positions, axis=0) + self.__detector.get_absolute_position(station_id)
        vertex_distances_to_station = np.linalg.norm(self.__event_group_vertices - station_barycenter, axis=1)
        if 'shower_energies' in self.__input_data.keys():
            shower_energies = np.array(self.__input_data['shower_energies'])[self.__shower_indices]
        else:
            shower_energies = np.zeros(self.__shower_indices.shape)
        shower_energy_sum = np.sum(shower_energies)
        if shower_energy_sum <= 0:
            distance_cut_value = 200. * units.m
        else:
            distance_cut_value = np.max([
                100. * units.m,
                10. ** self.__distance_cut_polynomial(np.log10(shower_energy_sum))
            ]) + 100. * units.m
        return vertex_distances_to_station.min() <= distance_cut_value

    def __get_output_structure(self, station_id):
        n_showers = len(self.__shower_indices)
        n_antennas = len(self.__detector.get_channel_ids(station_id))
        n_raytracing_solutions = self.__raytracer.get_number_of_raytracing_solutions()  # number of possible ray-tracing solutions
        station_output_structure = {}
        station_output_structure['triggered'] = np.zeros(n_showers, dtype=np.bool)
        # we need the reference to the shower id to be able to find the correct shower in the upper level hdf5 file
        station_output_structure['shower_id'] = np.zeros(n_showers, dtype=int)
        station_output_structure['event_id_per_shower'] = np.zeros(n_showers, dtype=int)
        station_output_structure['launch_vectors'] = np.zeros((n_showers, n_antennas, n_raytracing_solutions, 3)) * np.nan
        station_output_structure['receive_vectors'] = np.zeros((n_showers, n_antennas, n_raytracing_solutions, 3)) * np.nan
        station_output_structure['polarization'] = np.zeros((n_showers, n_antennas, n_raytracing_solutions, 3)) * np.nan
        station_output_structure['travel_times'] = np.zeros((n_showers, n_antennas, n_raytracing_solutions)) * np.nan
        station_output_structure['travel_distances'] = np.zeros((n_showers, n_antennas, n_raytracing_solutions)) * np.nan
        for parameter_entry in self.__raytracer.get_output_parameters():
            if parameter_entry['ndim'] == 1:
                station_output_structure[parameter_entry['name']] = np.zeros((n_showers, n_antennas, n_raytracing_solutions)) * np.nan
            else:
                station_output_structure[parameter_entry['name']] = np.zeros((n_showers, n_antennas, n_raytracing_solutions, parameter_entry['ndim'])) * np.nan
        return station_output_structure