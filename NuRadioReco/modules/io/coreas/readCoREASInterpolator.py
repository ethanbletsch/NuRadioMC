from NuRadioReco.modules.base.module import register_run
from NuRadioReco.detector.detector_base import DetectorBase
from NuRadioReco.framework import event, station, radio_shower, sim_station, electric_field
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.modules.io.coreas import coreas
import numpy as np
from radiotools.coordinatesystems import cstrafo
from NuRadioReco.utilities import units
import cr_pulse_interpolator.signal_interpolation_fourier as sigF  # traces
import cr_pulse_interpolator.interpolation_fourier as fluF  # fluence
from typing import Optional
import h5py
import logging
from collections import defaultdict

logger = logging.getLogger("NuRadioReco.readCoREASInterpolator")

warning_printed_coreas_py = False


class readCoREASInterpolator():
    def __init__(self,
                 logger_level=logging.WARNING,
                 lowfreq: float = 30.0,
                 highfreq: float = 500.0,
                 interpolator_kwargs: dict = None,
                 ):
        logger.setLevel(logger_level)

        self.lowfreq = lowfreq
        self.highfreq = highfreq
        if interpolator_kwargs:
            self.interpolator_kwargs = interpolator_kwargs
        else:
            self.interpolator_kwargs = {}

        self.corsika = None
        self.signal_interpolator = None

    def begin(self, filename):
        """
        Initializes the interpolator from a star-shape CoREAS simulation file
        """
        self.corsika = h5py.File(filename)
        self.coreas_shower = coreas.make_sim_shower(self.corsika)
        self.coreas_shower.set_parameter(
            shp.core, [0., 0., self.coreas_shower[shp.observation_level]])
        self.cs = cstrafo(*coreas.get_angles(self.corsika))

        self._set_showerplane_positions_and_signals()

        self.signal_interpolator = sigF.interp2d_signal(
            self.starshape_showerplane[..., 0],
            self.starshape_showerplane[..., 1],
            self.signals,
            lowfreq=self.lowfreq,
            highfreq=self.highfreq,
            sampling_period=self.corsika["CoREAS"].attrs["TimeResolution"],
            **self.interpolator_kwargs
        )

    def _set_showerplane_positions_and_signals(self):
        """
        Reads the positions and signals from the star-shape CoREAS simulation,
        then converts them to the correct coordinate system and units.
        """
        assert self.corsika != None and self.cs != None

        starpos = []
        signals = []

        for observer in self.corsika['CoREAS/observers'].values():
            position_coreas = observer.attrs['position']
            position_nr = np.array(
                [-position_coreas[1], position_coreas[0], 0]) * units.cm
            starpos.append(position_nr)

            signal = self.cs.transform_from_magnetic_to_geographic(
                coreas.observer_to_si_geomagnetic(observer)[:, 1:].T)
            signal = self.cs.transform_to_vxB_vxvxB(signal).T
            signals.append(signal)

            logger.debug(
                f"parsed starshape detector at position {position_nr}")

        starpos = np.array(starpos)
        signals = np.array(signals)
        starpos_vBvvB = self.cs.transform_from_magnetic_to_geographic(
            starpos.T)
        starpos_vBvvB = self.cs.transform_to_vxB_vxvxB(starpos_vBvvB.T)

        dd = (starpos_vBvvB[:, 0] ** 2 + starpos_vBvvB[:, 1] ** 2) ** 0.5
        logger.info(f"assumed star shape from: {-dd.max()} - {dd.max()}")

        self.starshape_showerplane = starpos_vBvvB
        self.signals = signals

    @register_run()
    def run(self, det: DetectorBase, station_ids: Optional[list] = None, requested_channel_ids: Optional[list] = None, core_shift: np.ndarray = np.zeros(3), multiprocess: bool = False) -> event.Event:
        evt = event.Event(0, 0)
        evt.add_sim_shower(self.coreas_shower)

        if station_ids is None:
            station_ids = det.get_station_ids()

        for station_id in station_ids:
            stat = station.Station(station_id)

            channel_ids = select_channels_per_station(
                det=det,
                station_id=station_id,
                requested_channel_ids=requested_channel_ids)

            if len(channel_ids.keys()) == 0:
                logger.info(f"station {station_id} did not contain any requested channel_ids")
                continue            

            station_position = det.get_absolute_position(station_id)

            channels_pos_ground = defaultdict(list)
            for group_id, assoc_channel_ids in channel_ids.items():
                channels_pos_ground[group_id] = station_position + det.get_relative_position(
                    station_id, assoc_channel_ids[0]) - core_shift
            channels_ground_flat = np.vstack(
                [pos for pos in channels_pos_ground.values()])

            channels_vxB = self.cs.transform_to_vxB_vxvxB(
                channels_ground_flat, self.coreas_shower[shp.core])
            channels_pos_showerplane = defaultdict(list)
            for group_id, pos in zip(channel_ids.keys(), channels_vxB):
                channels_pos_showerplane[group_id] = pos

            flattened_positions = np.vstack(
                [pos for pos in channels_pos_showerplane.values()])
            if np.any(~position_contained_in_starshape(flattened_positions, self.starshape_showerplane)):
                logger.warn(
                    "Channel positions are not all contained in the starshape! Will extrapolate.")

            # proper shapes? expect (channel, polarization, trace length)
            efields = defaultdict(list)
            for group_id, position in channels_pos_showerplane.items():
                interpolated = self.signal_interpolator(
                    *position[:-1], lowfreq=self.lowfreq, highfreq=self.highfreq)
                efields[group_id] = self.cs.transform_from_vxB_vxvxB(
                    interpolated.T)

            # channel_ids are those associated to efield!
            # should get channels by group
            # need different efields per station, in the sense of different positions, not reflection, direct efield like in ice
            sim_stat = make_sim_station(
                station_id, self.corsika, efields, channel_ids, channels_pos_ground)
            stat.set_sim_station(sim_stat)
            evt.set_station(stat)
        return evt

    def end(self):
        self.corsika.close()
        self.corsika = None

    def __del__(self):
        if self.corsika:
            self.corsika.close()


def make_sim_station(station_id, corsika, efields: dict, channel_ids: dict, positions: dict, weight=None):
    """
    creates an NuRadioReco sim station from the (interpolated) observer object of the coreas hdf5 file

    Parameters
    ----------
    station_id : station id
        the id of the station to create

    corsika : hdf5 file object
        the open hdf5 file object of the corsika hdf5 file

    observer : hdf5 observer object

    channel_ids :

    weight : weight of individual station
        weight corresponds to area covered by station

    Returns
    -------
    sim_station: sim station
        simulated station object
    """

    zenith, azimuth, magnetic_field_vector = coreas.get_angles(corsika)
    cs = cstrafo(zenith, azimuth, magnetic_field_vector=magnetic_field_vector)

    # prepend trace with zeros to not have the pulse directly at the start
    sim_station_ = sim_station.SimStation(station_id)
    for group_id, assoc_channel_ids in channel_ids.items():
        # expect time, Ex, Ey, Ez (ground coordinates)
        data = efields[group_id]
        efield = cs.transform_from_ground_to_onsky(data)

        # prepending zeros to not have pulse at start
        n_samples_prepend = efield.shape[1]
        efield2 = np.zeros((3, n_samples_prepend + efield.shape[1]))
        efield2[0] = np.append(np.zeros(n_samples_prepend), efield[0])
        efield2[1] = np.append(np.zeros(n_samples_prepend), efield[1])
        efield2[2] = np.append(np.zeros(n_samples_prepend), efield[2])

        sampling_rate = 1. / \
            (corsika['CoREAS'].attrs['TimeResolution'] * units.second)
        electric_field_ = electric_field.ElectricField(
            assoc_channel_ids, position=positions[group_id])
        electric_field_.set_trace(efield2, sampling_rate)
        electric_field_.set_trace_start_time(data[0, 0])
        electric_field_.set_parameter(efp.ray_path_type, 'direct')
        electric_field_.set_parameter(efp.zenith, zenith)
        electric_field_.set_parameter(efp.azimuth, azimuth)
        sim_station_.add_electric_field(electric_field_)
    sim_station_.set_parameter(stnp.azimuth, azimuth)
    sim_station_.set_parameter(stnp.zenith, zenith)
    energy = corsika['inputs'].attrs["ERANGE"][0] * units.GeV
    sim_station_.set_parameter(stnp.cr_energy, energy)
    sim_station_.set_magnetic_field_vector(magnetic_field_vector)
    sim_station_.set_parameter(
        stnp.cr_xmax, corsika['CoREAS'].attrs['DepthOfShowerMaximum'])
    try:
        sim_station_.set_parameter(
            stnp.cr_energy_em, corsika["highlevel"].attrs["Eem"])
    except:
        global warning_printed_coreas_py
        if (not warning_printed_coreas_py):
            logger.warning(
                "No high-level quantities in HDF5 file, not setting EM energy, this warning will be only printed once")
            warning_printed_coreas_py = True
    sim_station_.set_is_cosmic_ray()
    sim_station_.set_simulation_weight(weight)
    return sim_station_


def select_channels_per_station(det: DetectorBase, station_id: int, requested_channel_ids: Optional[list]) -> defaultdict:
    """
    Returns a defaultdict object containing the requeasted channel ids that are in the given station.
    This dict contains the channel group ids as keys with lists of channel ids as values.

    Parameters
    ----------
    det : DetectorBase
        The detector object that contains the station
    station_id : int
        The station id to select channels from
    requested_channel_ids : list
        List of requested channel ids
    """
    if requested_channel_ids is None:
        requested_channel_ids = det.get_channel_ids(station_id)

    channel_ids = defaultdict(list)
    for channel_id in requested_channel_ids:
        if channel_id in det.get_channel_ids(station_id):
            channel_group_id = det.get_channel_group_id(station_id, channel_id)
            channel_ids[channel_group_id].append(channel_id)

    return channel_ids


def position_contained_in_starshape(channel_positions: np.ndarray, starhape_positions: np.ndarray):
    """
    Verify if `station_positions` lie within the starshape defined by `starshape_positions`. Ensures interpolation. Projects out z-component.    

    station_positions: np.ndarray (n, 3)

    starshape_positions: np.ndarray (m, 3)
    """
    # scatter_stations(channel_positions, starhape_positions)
    star_radius = np.max(np.linalg.norm(starhape_positions[:, :-1], axis=-1))
    contained = np.linalg.norm(
        channel_positions[:, :-1], axis=-1) <= star_radius
    return contained

def scatter_stations(true, star):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(*(true[:,:-1].T), color="k",s=2, label="true")
    ax.scatter(*(star[:,:-1].T), marker="*", s=2,color="r", label="star")
    ax.legend()
    ax.set_aspect("equal")
    plt.show()
    plt.close()


def main():
    import matplotlib.pyplot as plt
    from datetime import datetime

    detector = DetectorBase(
        json_filename='LOFAR.json',
        source='json',
        antenna_by_depth=False
    )
    detector.update(datetime.now())

    interpolator = readCoREASInterpolator()
    interpolator.begin("SIM000013.hdf5")

    logger.debug(
        f"starshape position array shape: {interpolator.starshape_showerplane.shape} (antenna, polarization)")
    logger.debug(f"interpolator signal input shape {interpolator.signals.shape}")

    output = interpolator.run(detector)

    # show showerplane positions
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    cm = plt.colormaps['RdYlBu_r']
    amp = np.sqrt(np.max(np.sum(interpolator.signals**2, axis=-1), axis=1))
    sc = ax.scatter(*interpolator.starshape_showerplane.T, s=2, c=amp,
                    vmin=amp.min(), vmax=amp.max(), cmap=cm)
    plt.colorbar(sc)

    plt.show()

    interpolator.end()


if __name__ == '__main__':
    main()
