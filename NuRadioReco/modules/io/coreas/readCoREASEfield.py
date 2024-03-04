from NuRadioReco.modules.base.module import register_run
from NuRadioReco.framework.event import Event
from NuRadioReco.framework.station import Station
from NuRadioReco.modules.io.coreas import coreas
from NuRadioReco.framework.sim_station import SimStation
from NuRadioReco.framework.electric_field import ElectricField
from NuRadioReco.detector.generic_detector import GenericDetector
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from radiotools.coordinatesystems import cstrafo
from typing import Optional

import numpy as np
from NuRadioReco.utilities import units
import logging
import h5py

logger = logging.getLogger("NuRadioReco.readCoREASEfield")

ref_stnp = [
    'pos_site',
    # 'pos_measurement_time',
    # 'pos_position',
    # 'position',
    'station_type',
    # 'MAC_address',
    # 'MBED_type',
    # 'board_number',
    # 'commission_time',
    # 'decommission_time',
    'pos_altitude',
    'pos_easting',
    'pos_northing',
    'station_id'
]

ref_chnp = [
    # 'adc_id',
    'adc_n_samples',
    # 'adc_nbits',
    'adc_sampling_frequency',
    # 'adc_time_delay',
    # 'amp_reference_measurement',
    'amp_type',
    'ant_type',
    # 'ant_comment',
    # 'ant_deployment_time',
    # 'commission_time',
    # 'decommission_time',
    # 'cab_id',
    # 'cab_length',
    # 'cab_reference_measurement',
    'cab_time_delay',
    # 'cab_type',
    'ant_position_x',
    'ant_position_y',
    'ant_position_z',
    'ant_orientation_phi',
    'ant_orientation_theta',
    'ant_rotation_phi',
    'ant_rotation_theta',
    'channel_id',
    'station_id'
]


class readCoREASEfield():
    """A CoREAS reader that does not require a detector description. The latter is taken from the CoREAS observers, set as different channels within a single station."""
    def __init__(self) -> None:
        pass

    def begin(self, evt_id: int, hdf5_path: str, logger_level=logging.WARNING):
        logger.setLevel(logger_level)
        self._event_id = evt_id
        self._filename = hdf5_path
        self.corsika = h5py.File(self._filename)

    @register_run()
    def run(self, det: Optional[GenericDetector] = None) -> Event:
        if det is None:
            det = GenericDetector(json_filename=None, source="dict", dictionary=gen_detector_dict(self.corsika))
        evt = Event(1, self._event_id)
        for station_id in det.get_station_ids():
            station = Station(station_id)
            sim_station = SimStation(station_id)
            zenith, azimuth, magnetic_field_vector = coreas.get_angles(
                self.corsika)
            cs = cstrafo(zenith, azimuth, magnetic_field_vector)
            sampling_rate = 1. / \
                (self.corsika['CoREAS'].attrs['TimeResolution']
                 * units.second)
            obsnames = [key for key in self.corsika["CoREAS/observers"].keys()]
            for channel_id in det.get_channel_ids(station_id):
                obsname = obsnames[channel_id]
                pos = det.get_relative_position(station_id, channel_id)
                electric_field = ElectricField([channel_id], position=pos)
                data = coreas.observer_to_si_geomagnetic(
                    self.corsika[f"CoREAS/observers/{obsname}"])
                efield = cs.transform_from_magnetic_to_geographic(
                    data[:, 1:].T)
                efield = cs.transform_from_ground_to_onsky(efield)

                electric_field.set_trace(efield, sampling_rate)
                electric_field.set_trace_start_time(data[0, 0])
                electric_field.set_parameter(efp.ray_path_type, 'direct')
                electric_field.set_parameter(efp.zenith, zenith)
                electric_field.set_parameter(efp.azimuth, azimuth)
                sim_station.add_electric_field(electric_field)
            sim_station.set_parameter(stnp.azimuth, azimuth)
            sim_station.set_parameter(stnp.zenith, zenith)
            energy = self.corsika['inputs'].attrs["ERANGE"][0] * units.GeV
            sim_station.set_parameter(stnp.cr_energy, energy)
            sim_station.set_magnetic_field_vector(magnetic_field_vector)
            sim_station.set_parameter(
                stnp.cr_xmax, self.corsika['CoREAS'].attrs['DepthOfShowerMaximum'])
            sim_station.set_is_cosmic_ray()
            station.set_sim_station(sim_station)

            evt.set_station(station)
        sim_shower = coreas.make_sim_shower(self.corsika)
        sim_shower.set_parameter(shp.core, np.array([0.,0.,sim_shower[shp.observation_level]]))
        evt.add_sim_shower(sim_shower)
        return evt

    def end(self):
        self.corsika.close()


def gen_detector_dict(corsika: h5py.File):
    """Generate a dictionary describing a GenericDetector, from a corsika hdf5 file. Written to be used only when using electric fields are needed, since antenna description is arbitrary except for positions. All observers are described as channels within one station."""

    station = {"station_id": 1, "pos_altitude": 0,
               "pos_easting": 0, "pos_northing": 0}
    for key in ref_stnp:
        if key not in station.keys():
            station[key] = None
    channels = {}
    for i, observer_name in enumerate(corsika["CoREAS/observers"]):
        position = corsika[f"CoREAS/observers/{
            observer_name}"].attrs["position"]
        position_keys = [f"ant_position_{dir}" for dir in ["x", "y", "z"]]
        channels[f"{i}"] = {key: val for key,
                            val in zip(position_keys, position)}
        channels[f"{i}"]["station_id"] = 1
        channels[f"{i}"]["channel_id"] = i
        channels[f"{i}"]["reference_channel"] = 0
    
    channels["0"]["ant_orientation_phi"] = 225.
    channels["0"]["ant_orientation_theta"] = 90.
    channels["0"]["ant_rotation_phi"] = 0.
    channels["0"]["ant_rotation_theta"] = 0.
    channels["0"]["ant_type"] = ""
    del channels["0"]["reference_channel"]
    for key in ref_chnp:
        if key not in channels["0"]:
            channels["0"][key]  = None
    
    detector = {"stations": {"1": station}, "channels": channels}

    return detector
    