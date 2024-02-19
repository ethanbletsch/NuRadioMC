from NuRadioReco.modules.base.module import register_run
from NuRadioReco.detector.detector_base import DetectorBase
from NuRadioReco.framework.event import Event
from NuRadioReco.framework.base_shower import BaseShower
from NuRadioReco.framework.parameters import showerParameters as shp
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
    def __init__(self) -> None:
        self.signal_interpolator = sigF.interp2d_signal()

    def begin(self, filename, logger_level=logging.WARNING):
        self.corsika = h5py.File(filename)
        logger.setLevel(logger_level)
        pass

    @register_run()
    def run(self, det: DetectorBase, requested_channel_ids: Optional[dict] = None, core_shift: Optional[np.ndarray] = None):
        for station_id in det.get_station_ids():

            channel_ids = select_channels(
                requested_channel_ids, det, station_id)

            station_position = det.get_absolute_position(station_id)
            ground_channel_positions = np.array(
                [station_position + det.get_relative_position(station_id, id) for id in channel_ids])
            if core_shift is not None:
                ground_channel_positions -= core_shift

            zenith, azimuth, magnetic_field_vector = coreas.get_angles(
                self.corsika)
            cs = cstrafo(zenith, azimuth,
                         magnetic_field_vector=magnetic_field_vector)
            showerplane_channel_positions = project_to_showerplane(
                ground_channel_positions, cs)

            # if not position_contained_in_starshape(showerplane_channel_positions, self.sim_event.get_)

    def end(self):
        self.corsika.close()
        pass


def get_corsika_pos_showerplane(corsika: h5py.File, cs: cstrafo):
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
        "star shape from: {} - {}".format(-dd.max(), dd.max()))
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


def project_to_showerplane(station_positions: np.ndarray, cs: cstrafo, simshower):
    """
    transform `station_positions` (ground coordinates) to shower plane coordinates, and project to this plane (drop z-component)

    station_positions: np.ndarray (n, 3)

    Returns
    -------

    projected: np.ndarray (n, 2)
    """
    station_positions_vxB_vxvxB = cs.transform_to_vxB_vxvxB(
        station_positions, core=simshower[shp.core])
    projected = station_positions_vxB_vxvxB[:, :-1]
    return projected


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

    get_corsika_pos_showerplane(corsika, cs)


if __name__ == '__main__':
    main()
