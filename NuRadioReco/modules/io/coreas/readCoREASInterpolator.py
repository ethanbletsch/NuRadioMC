from NuRadioReco.modules.base.module import register_run
from NuRadioReco.detector.detector_base import DetectorBase
from NuRadioReco.framework import event, station, radio_shower, sim_station, electric_field
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.modules.io.coreas import coreas
import numpy as np
# import NuRadioReco.framework.station
from radiotools.coordinatesystems import cstrafo
from NuRadioReco.utilities import units
import cr_pulse_interpolator.signal_interpolation_fourier as sigF  # traces
import cr_pulse_interpolator.interpolation_fourier as fluF  # fluence
from typing import Optional
import h5py
import logging


logger = logging.getLogger("NuRadioReco.readCoREASInterpolator")


class readCoREASInterpolator():
    def __init__(self, 
    logger_level=logging.WARNING,
    lowfreq=30.0,
    highfreq=500.0,
    sampling_period=0.1e-9,
    interpolator_kwargs:dict = None,
    ):
        logger.setLevel(logger_level)
        
        self.lowfreq = lowfreq
        self.highfreq = highfreq
        self.sampling_period = sampling_period
        if interpolator_kwargs:
            self.interpolator_kwargs == interpolator_kwargs
        else:
            self.interpolator_kwargs == {}
        
        self.corsika = None
        self.signal_interpolator = None

    def begin(self, filename):
        # TODO:
        # After for interpolation we should convert positions to vB_vvB and project out z,
        # and turn efields to onsky (cstrafo.transform_from_magnetic_to_geographic and cstrafo.transform_from_ground_to_onsky)
        self.corsika = h5py.File(filename)

        footprint_positions = np.array(demo_file.get('footprint_positions'))

        (footprint_pos_x_coreas, footprint_pos_y_coreas) = (footprint_positions[:, 0], footprint_positions[:, 1])
        footprint_antenna_data_coreas = np.array(demo_file.get('footprint_antennas'))

        # convert to NRR coordinates and units
        footprint_pos_x = - footprint_pos_y_coreas * coreas.conversion_fieldstrength_cgs_to_SI
        footprint_pos_y = footprint_pos_x_coreas * coreas.conversion_fieldstrength_cgs_to_SI

        self.signal_interpolator = sigF.interp2d_signal(
            footprint_pos_x, footprint_pos_y, footprint_antenna_data,
            lowfreq = self.lowfreq,
            highfreq = self.highfreq,
            sampling_period = self.sampling_period,
            **self.interpolator_kwargs)

    def begin(self, filename):
        self.corsika = h5py.File(filename)
        self.coreas_shower = coreas.make_sim_shower(self.corsika)
        self.coreas_shower.set_parameter(shp.core, [0.,0.,self.coreas_shower[shp.observation_level]])
        self.cs = cstrafo(*coreas.get_angles(self.corsika))

        self.starshape_showerplane = get_showerplane_observer_positions(self.corsika, self.cs)
        starshape_onsky = self.cs.
        traces = []


    @register_run()
    def run(self, det: DetectorBase, requested_channel_ids: Optional[dict] = None, core_shift: Optional[np.ndarray] = None):
        evt = event.Event(0, 0)
        evt.add_sim_shower(self.coreas_shower)
        station_ids = det.get_station_ids()
        rd_shower = radio_shower.RadioShower(station_ids=station_ids)
        evt.add_shower(rd_shower)

        for station_id in station_ids:
            stat = station.Station(station_id)

            channel_ids = select_channels(
                requested_channel_ids, det, station_id)

            station_position = det.get_absolute_position(station_id)
            ground_channel_positions = np.array(
                [station_position + det.get_relative_position(station_id, id) for id in channel_ids])
            
            if core_shift is not None:
                ground_channel_positions -= core_shift

            channels_pos_showerplane = self.cs.transform_to_vxB_vxvxB(
                ground_channel_positions, self.coreas_shower[shp.core])[:, :-1]

            if not position_contained_in_starshape(channels_pos_showerplane, self.starshape_showerplane):
                logger.warn(
                    "Channel positions are not all contained in the starshape! Will extrapolate.")

            efields = self.signal_interpolator(
                *channels_pos_showerplane[:, :-1])
            efields = np.array([efields[:, i] for i in range(3)])

            efields_interp = self.signal_interpolator()
            sim_stat = coreas.make_sim_station(
                station_id, self.corsika, efields_interp, channel_ids, coreas_observer_format=False)
            stat.set_sim_station(sim_stat)
            evt.set_station(stat)

    def end(self):
        self.corsika.close()
        pass



def get_showerplane_observer_positions(corsika: h5py.File, cs: cstrafo):
    starpos = []
    for observer in corsika['CoREAS']['observers'].values():
        position = observer.attrs['position']
        starpos.append(
            np.array([-position[1], position[0], 0]) * units.cm)
        logger.debug("({:.0f}, {:.0f})".format(
            position[0], position[1]))
    starpos = np.array(starpos)

    starpos_vBvvB = cs.transform_from_magnetic_to_geographic(starpos.T)
    starpos_vBvvB = cs.transform_to_vxB_vxvxB(starpos_vBvvB).T
    dd = (starpos_vBvvB[:, 0] ** 2 + starpos_vBvvB[:, 1] ** 2) ** 0.5
    logger.info(
        "assumed star shape from: {} - {}".format(-dd.max(), dd.max()))
    return starpos_vBvvB


def select_channels(requested_channel_ids: list, det: DetectorBase, station_id: int):
    station_channel_ids = det.get_channel_ids(station_id)

    # select channels
    if requested_channel_ids is not None and station_id in requested_channel_ids.keys():
        requested_set = set(requested_channel_ids[station_id])
        if not requested_set.issubset(set(station_channel_ids)):
            # keep as raise ValueError or send to logger.warning?
            raise ValueError(
                f"`requested_channel_ids` at station {station_id} is not a subset of available channel ids; {requested_set.difference(set(station_channel_ids))} not found.")
        channel_ids = requested_channel_ids[station_id]
    else:
        channel_ids = station_channel_ids
    return channel_ids


def position_contained_in_starshape(station_positions: np.ndarray, starhape_positions: np.ndarray):
    """
    Verify if `station_positions` lie within the starshape defined by `starshape_positions`. Ensures interpolation. Projects out z-component.    

    station_positions: np.ndarray (n, 3)

    starshape_positions: np.ndarray (m, 3)
    """
    star_radius = np.max(np.linalg.norm(starhape_positions[:, :-1], axis=-1))
    contained = np.linalg.norm(
        station_positions[:, :-1], axis=-1) <= star_radius
    return bool(np.sum(~contained))


def main():
    filename = "/home/tiepolo/Documents/VUB/MA2/nrr-dev/hdf5/SIM000013.hdf5"
    corsika = h5py.File(filename, "r")
    zenith, azimuth, B = coreas.get_angles(corsika)
    cs = cstrafo(zenith, azimuth, magnetic_field_vector=B)

    get_showerplane_observer_positions(corsika, cs)


if __name__ == '__main__':
    main()
