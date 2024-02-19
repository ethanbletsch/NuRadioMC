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
import cr_pulse_interpolator.signal_interpolation_fourier as sigF
from typing import Optional
import h5py
import logging



class readCoREASInterpolator():
    def __init__(self) -> None:
        self.signal_interpolator = sigF.interp2d_signal()
        self.logger = logging.getLogger("NuRadioReco.readCoREASInterpolator")

    def begin(self, filename, logger_level = logging.WARNING):
        self.corsika = hdf5.File(filename)
        logger.setLevel(logger_level)
        pass

    @register_run()
    def run(self, det: DetectorBase, requested_channel_ids: Optional[dict] = None, core_shift: Optional[np.ndarray] = None):
        for station_id in det.get_station_ids():
            # station = det.get_station(station_id)
            station_channel_ids = det.get_channel_ids(station_id)
            if requested_channel_ids is not None and station_id in requested_channel_ids.keys():
                requested_set = set(requested_channel_ids[station_id])
                if not requested_set.issubset(set(station_channel_ids)):
                    # keep as raise ValueError or send to logger.warning?
                    raise ValueError(
                        f"`requested_channel_ids` at station {station_id} is not a subset of available channel ids; {requested_set.difference(station_set)} not found.")
                channel_ids = requested_channel_ids[station_id]
            else:
                channel_ids = station_channel_ids

            simpos = []
            for i, observer in enumerate(corsika['CoREAS']['observers'].values()):
                position = observer.attrs['position']
                simpos.append(np.array([-position[1], position[0], 0]) * units.cm)
                self.logger.debug("({:.0f}, {:.0f})".format(position[0], position[1]))
            simpos = np.array(simpos)

            simpos_vBvvB = cs.transform_from_magnetic_to_geographic(simpos.T)
            simpos_vBvvB = cs.transform_to_vxB_vxvxB(simpos_vBvvB).T
            dd = (simpos_vBvvB[:, 0] ** 2 + simpos_vBvvB[:, 1] ** 2) ** 0.5
            ddmax = dd.max()
            self.logger.info("star shape from: {} - {}".format(-dd.max(), dd.max()))

            station_position = det.get_absolute_position(station_id)
            ground_channel_positions = np.array([station_position + det.get_relative_position(station_id, id) for id in channel_ids])
            if core_shift is not None:
                ground_channel_positions -= core_shift

            zenith, azimuth, magnetic_field_vector = coreas.get_angles(self.corsika)
            cs = cstrafo(zenith, azimuth, magnetic_field_vector=magnetic_field_vector)
            showerplane_channel_positions = project_to_showerplane(ground_channel_positions, cs)

            # if not position_contained_in_starshape(showerplane_channel_positions, self.sim_event.get_)


            

    def end():
        self.file.close()
        pass


def project_to_showerplane(station_positions: np.ndarray, cs: cstrafo):
    """
    transform `station_positions` (ground coordinates) to shower plane coordinates, and project to this plane (drop z-component)

    station_positions: np.ndarray (n, 3)

    returns
    -------

    projected: np.ndarray (n, 2)
    """
    station_positions_vxB_vxvxB = cs.transform_to_vxB_vxvxB(
        station_positions, core=shower[shp.core])
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
