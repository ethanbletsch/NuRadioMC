import numpy as np
import matplotlib.pyplot as plt
from radiotools import helper as hp
from radiotools import coordinatesystems
from NuRadioReco.utilities import units
import NuRadioReco.framework.sim_station
import NuRadioReco.framework.base_trace
import NuRadioReco.framework.electric_field
import NuRadioReco.framework.radio_shower
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.framework.parameters import showerParameters as shp
import cr_pulse_interpolator.signal_interpolation_fourier
import cr_pulse_interpolator.interpolation_fourier
import logging
logger = logging.getLogger('coreas')

warning_printed_coreas_py = False

conversion_fieldstrength_cgs_to_SI = 2.99792458e10 * units.micro * units.volt / units.meter

def get_angles(corsika):
    """
    Converting angles in corsika coordinates to local coordinates. 
    
    Corsika positiv x-axis points to the magnetic north, NuRadio coordinates positiv x-axis points to the geographic east.
    Corsika positiv y-axis points to the west, NuRadio coordinates positiv y-axis points to the geographic north.
    Corsika z-axis points upwards, NuRadio z-axis points upwards.

    Corsikas zenith angle of a particle trajectory is defined between the particle momentum vector and the negativ z-axis, meaning that
    the particle is described in the direction where it is going to. The azimuthal angle is described between the positive x-axis and 
    the horizontal component of the particle momentum vector (i.e. with respect to the magnetic north) proceeding counterclockwise. 
    
    NuRadio describes the particle zenith and azimuthal angle in the direction where the particle is coming from. 
    Therefore the zenith angle is the same, but the azimuthal angle has to be shifted by 180 + 90 degrees. The north has to be shifted by 90 degrees plus difference in geomagetic and magnetic north.
    """
    #TODO difference between magnetic and geographic north, sofar 90deg is assumed
    zenith = np.deg2rad(corsika['inputs'].attrs["THETAP"][0])
    azimuth = hp.get_normalized_angle(3 * np.pi / 2. + np.deg2rad(corsika['inputs'].attrs["PHIP"][0]))

    Bx, Bz = corsika['inputs'].attrs["MAGNET"]
    B_inclination = np.arctan2(Bz, Bx)

    B_strength = (Bx ** 2 + Bz ** 2) ** 0.5 * units.micro * units.tesla

    # in local coordinates north is + 90 deg
    magnetic_field_vector = B_strength * hp.spherical_to_cartesian(np.pi * 0.5 + B_inclination, 0 + np.pi * 0.5)

    return zenith, azimuth, magnetic_field_vector

def get_geomagnetic_angle(zenith, azimuth, magnetic_field_vector):
    """
    Calculates the angle between the geomagnetic field and the shower axis.
    Parameters
    ----------
    zenith : float
        zenith angle in radians
    azimuth : float
        azimuth angle in radians
    magnetic_field_vector : np.array (3)
        magnetic field vector in cartesian coordinates
    Returns
    -------
    geomagnetic_angle : float
        geomagnetic angle in radians
    """
    shower_axis_vector = hp.spherical_to_cartesian(zenith, azimuth)
    geomagnetic_angle = hp.get_angle(magnetic_field_vector, shower_axis_vector)
    return geomagnetic_angle

def convert_obs_to_nuradio_efield(observer, zenith, azimuth, magnetic_field_vector, prepend_zeros=False):
    """
    converts the electric field from the coreas observer to NuRadio units and spherical coordinated eR, eTheta, ePhi (on sky)

    Parameters
    ----------
    observer : value
        observer as in the hdf5 file object, e.g. corsika['CoREAS']['observers'].values()
    
    Returns
    -------
    efield: np.array (n_samples, 3)
        efield with three polarizations (r, theta, phi)
    efield_times: np.array (n_samples)

    """
    cs = coordinatesystems.cstrafo(zenith, azimuth, magnetic_field_vector)

    efield = np.array([observer[()][:,0]*units.second,
                        -observer[()][:,2]*conversion_fieldstrength_cgs_to_SI,
                        observer[()][:,1]*conversion_fieldstrength_cgs_to_SI,
                        observer[()][:,3]*conversion_fieldstrength_cgs_to_SI])
    
    efield_times = efield[0,:]
    efield_geo = cs.transform_from_magnetic_to_geographic(efield[1:,:])
    # convert coreas efield to NuRadio spherical coordinated eR, eTheta, ePhi (on sky)
    efield_on_sky = cs.transform_from_ground_to_onsky(efield_geo)
    
    if prepend_zeros:
        # prepend trace with zeros to not have the pulse directly at the start
        n_samples_prepend = efield_on_sky.shape[1]
        efield = np.zeros((3, n_samples_prepend + efield_on_sky.shape[1]))
        efield[0] = np.append(np.zeros(n_samples_prepend), efield_on_sky[0])
        efield[1] = np.append(np.zeros(n_samples_prepend), efield_on_sky[1])
        efield[2] = np.append(np.zeros(n_samples_prepend), efield_on_sky[2])
    else:
        efield = efield_on_sky
    
    return efield.T, efield_times

def convert_obs_positions_to_nuradio_on_ground(observer, zenith, azimuth, magnetic_field_vector):
    """
    Converts observer positions from the corsika observer to NuRadio units on ground.
    coreas: x-axis pointing to the magnetic north, the positive y-axis to the west, and the z-axis upwards.
    NuRadio: x-axis pointing to the east, the positive y-axis geographical north, and the z-axis upwards.
    NuRadio_x = -coreas_y, NuRadio_y = coreas_x, NuRadio_z = coreas_z and then correct for mag north

    Parameters
    ----------
    observer : value
        observer as in the hdf5 file object, e.g. corsika['CoREAS']['observers'].values()
  
    Returns
    -------
    obs_positions_geo: np.array (3)
        observer positions in geographic coordinates
    
    """
    cs = coordinatesystems.cstrafo(zenith, azimuth, magnetic_field_vector)

    obs_positions = np.array([-observer.attrs['position'][1], observer.attrs['position'][0], 0]) * units.cm

    # second to last dimension has to be 3 for the transformation
    obs_positions_geo = cs.transform_from_magnetic_to_geographic(obs_positions.T)

    return obs_positions_geo
            
def convert_obs_positions_to_vxB_vxvxB(observer, zenith, azimuth, magnetic_field_vector):
    """
    Converts observer positions from the corsika file to NuRadio units in the (vxB, vxvxB) shower plane.
    coreas: x-axis pointing to the magnetic north, the positive y-axis to the west, and the z-axis upwards.
    NuRadio: x-axis pointing to the east, the positive y-axis geographical north, and the z-axis upwards.
    NuRadio_x = -coreas_y, NuRadio_y = coreas_x, NuRadio_z = coreas_z and then correct for mag north

    Parameters
    ----------
    observer : value
        observer as in the hdf5 file object, e.g. corsika['CoREAS']['observers'].values()
  
    Returns
    -------
    obs_positions_vxB_vxvxB: np.array (3)
        observer positions in (vxB, vxvxB) shower plane
    
    """
    cs = coordinatesystems.cstrafo(zenith, azimuth, magnetic_field_vector)
    obs_positions_geo = convert_obs_positions_to_nuradio_on_ground(observer, zenith, azimuth, magnetic_field_vector)
    # transforms the coreas observer positions into the vxB, vxvxB shower plane
    obs_positions_vxB_vxvxB = cs.transform_to_vxB_vxvxB(obs_positions_geo).T

    return obs_positions_vxB_vxvxB


def make_sim_station(station_id, corsika, weight=None):
    """
    creates an NuRadioReco sim station from the (interpolated) observer object of the coreas hdf5 file

    Parameters
    ----------
    station_id : station id
        the id of the station to create
    corsika : hdf5 file object
        the open hdf5 file object of the corsika hdf5 file
    weight : weight of individual station
        weight corresponds to area covered by station

    Returns
    -------
    sim_station: sim station
        simulated station object
    """

    zenith, azimuth, magnetic_field_vector = get_angles(corsika)
    sim_station = NuRadioReco.framework.sim_station.SimStation(station_id)
    sim_station.set_parameter(stnp.azimuth, azimuth)
    sim_station.set_parameter(stnp.zenith, zenith)
    energy = corsika['inputs'].attrs["ERANGE"][0] * units.GeV
    sim_station.set_parameter(stnp.cr_energy, energy)
    sim_station.set_magnetic_field_vector(magnetic_field_vector)
    sim_station.set_parameter(stnp.cr_xmax, corsika['CoREAS'].attrs['DepthOfShowerMaximum'])
    try:
        sim_station.set_parameter(stnp.cr_energy_em, corsika["highlevel"].attrs["Eem"])
    except:
        global warning_printed_coreas_py
        if(not warning_printed_coreas_py):
            logger.warning("No high-level quantities in HDF5 file, not setting EM energy, this warning will be only printed once")
            warning_printed_coreas_py = True
    sim_station.set_is_cosmic_ray()
    sim_station.set_simulation_weight(weight)
    return sim_station

def make_sim_shower(corsika, observer=None, detector=None, station_id=None):
    """
    creates an NuRadioReco sim shower from the coreas hdf5 file, the core positions are set such that the detector station is on top of
    each coreas observer position

    Parameters
    ----------
    corsika : hdf5 file object
        the open hdf5 file object of the corsika hdf5 file
    observer : hdf5 observer object
    detector : detector object
    station_id : station id of the station relativ to which the shower core is given
    
    Returns
    -------
    sim_shower: sim shower
        simulated shower object
    """
    sim_shower = NuRadioReco.framework.radio_shower.RadioShower()

    zenith, azimuth, magnetic_field_vector = get_angles(corsika)
    sim_shower.set_parameter(shp.zenith, zenith)
    sim_shower.set_parameter(shp.azimuth, azimuth)
    sim_shower.set_parameter(shp.magnetic_field_vector, magnetic_field_vector)

    energy = corsika['inputs'].attrs["ERANGE"][0] * units.GeV
    sim_shower.set_parameter(shp.energy, energy)
    # We can only set the shower core relative to the station if we know its position
    if observer is not None and detector is not None and station_id is not None:
        station_position = detector.get_absolute_position(station_id)
        position = observer.attrs['position']
        observer_position = convert_obs_positions_to_nuradio_on_ground(position, zenith, azimuth, magnetic_field_vector)
        core_position = (-observer_position + station_position)
        core_position[2] = 0
        sim_shower.set_parameter(shp.core, core_position)

    sim_shower.set_parameter(shp.shower_maximum, corsika['CoREAS'].attrs['DepthOfShowerMaximum'] * units.g / units.cm2)
    sim_shower.set_parameter(shp.refractive_index_at_ground, corsika['CoREAS'].attrs["GroundLevelRefractiveIndex"])
    sim_shower.set_parameter(shp.magnetic_field_rotation,
                             corsika['CoREAS'].attrs["RotationAngleForMagfieldDeclination"] * units.degree)
    sim_shower.set_parameter(shp.distance_shower_maximum_geometric,
                             corsika['CoREAS'].attrs["DistanceOfShowerMaximum"] * units.cm)

    sim_shower.set_parameter(shp.observation_level, corsika["inputs"].attrs["OBSLEV"] * units.cm)
    sim_shower.set_parameter(shp.primary_particle, corsika["inputs"].attrs["PRMPAR"])
    if 'ATMOD' in corsika['inputs'].attrs.keys():
        sim_shower.set_parameter(shp.atmospheric_model, corsika["inputs"].attrs["ATMOD"])

    try:
        sim_shower.set_parameter(shp.electromagnetic_energy, corsika["highlevel"].attrs["Eem"] * units.eV)
    except:
        global warning_printed_coreas_py
        if(not warning_printed_coreas_py):
            logger.warning("No high-level quantities in HDF5 file, not setting EM energy, this warning will be only printed once")
            warning_printed_coreas_py = True

    return sim_shower

def add_electric_field_to_sim_station(sim_station, channel_ids, efield, efield_times, zenith, azimuth, sampling_rate, fluence=None):
    """
    adds an electric field trace to an existing sim station

    Parameters
    ----------
    sim_station : sim station object
         simulated station object, e.g. from make_empty_sim_station()
    channel_ids : list or int
        channel_ids for which the efield is to be used
    efield : 3d array (3, n_samples)
        efield with three polarizations (r, theta, phi) for channel
    efield_times : 1d array (n_samples)
        time array for efield
    corsika : hdf5 file object
        the open hdf5 file object of the corsika hdf5 file
    """
    if type(channel_ids) is not list:
        channel_ids = [channel_ids]
    
    electric_field = NuRadioReco.framework.electric_field.ElectricField(channel_ids)
    electric_field.set_trace(efield, sampling_rate)
    electric_field.set_trace_start_time(efield_times[0])
    electric_field.set_parameter(efp.ray_path_type, 'direct')
    electric_field.set_parameter(efp.zenith, zenith)
    electric_field.set_parameter(efp.azimuth, azimuth)
    if fluence is not None:
        electric_field.set_parameter(efp.signal_energy_fluence, fluence)
    sim_station.add_electric_field(electric_field)

def calculate_simulation_weights(positions, zenith, azimuth, site='summit', debug=False):
    """
    Calculate weights according to the area that one simulated position in readCoreasStation represents.
    Weights are therefore given in units of area.
    Note: The volume of a 2d convex hull is the area.
    
    Parameters
    ----------
    positions : list
        station position with [x, y, z]
    zenith : float
        zenith angle of the shower
    azimuth : float
        azimuth angle of the shower
    site : str
        site of the simulation, default is 'summit'
    debug : bool
        if true, plots are created for debugging
    Returns
    -------
    weights : np.array
        weights of the observer position
    """

    import scipy.spatial as spatial

    positions = np.array(positions)
    cs = coordinatesystems.cstrafo(zenith=zenith, azimuth=azimuth, magnetic_field_vector=None,
                                                   site=site)
    x_trafo_from_shower = cs.transform_from_vxB_vxvxB(station_position=np.array([1, 0, 0]))
    y_trafo_from_shower = cs.transform_from_vxB_vxvxB(station_position=np.array([0, 1, 0]))
    z_trafo_from_shower = cs.transform_from_vxB_vxvxB(station_position=np.array([0, 0, 1]))

    # voronoi has to be calculated in the shower plane due to symmetry reasons
    shower = cs.transform_to_vxB_vxvxB(station_position=positions)
    vor = spatial.Voronoi(shower[:, :2])  # algorithm will find no solution if flat simulation is given in 3d.

    if debug:
        fig1 = plt.figure(figsize=[12, 4])
        ax1 = fig1.add_subplot(121)
        ax2 = fig1.add_subplot(122)
        spatial.voronoi_plot_2d(vor, ax1)
        ax1.set_aspect('equal')
        ax1.set_title('In shower plane, zenith = {:.2f}'.format(zenith / units.degree))
        ax1.set_xlabel(r'Position in $\vec{v} \times \vec{B}$ - direction [m]')
        ax1.set_ylabel(r'Position in $\vec{v} \times \vec{v} \times \vec{B}$ - direction [m]')

    weights = np.zeros_like(positions[:, 0])
    for p in range(0, weights.shape[0]):  # loop over all observer positions
        vertices_shower_2d = vor.vertices[vor.regions[vor.point_region[p]]]

        # x_vertice_ground = x_trafo_from_shower[0] * x_vertice_shower + y_trafo_from_shower[0] * y_vertice_shower + z_trafo_from_shower[0] * z_vertice_shower
        # y_vertice_ground = x_trafo_from_shower[1] * x_vertice_shower + y_trafo_from_shower[1] * y_vertice_shower + z_trafo_from_shower[1] * z_vertice_shower
        # z_vertice_ground = x_trafo_from_shower[2] * x_vertice_shower + y_trafo_from_shower[2] * y_vertice_shower + z_trafo_from_shower[2] * z_vertice_shower

        x_vertice_shower = vertices_shower_2d[:, 0]
        y_vertice_shower = vertices_shower_2d[:, 1]
        z_vertice_shower = -(x_trafo_from_shower[2] * x_vertice_shower + y_trafo_from_shower[2] * y_vertice_shower) / z_trafo_from_shower[2]

        vertices_shower_3d = []
        for iter in range(len(x_vertice_shower)):
            vertices_shower_3d.append([x_vertice_shower[iter], y_vertice_shower[iter], z_vertice_shower[iter]])
        vertices_shower_3d = np.array(vertices_shower_3d)
        vertices_ground = cs.transform_from_vxB_vxvxB(station_position=vertices_shower_3d)

        n_arms = 8  # mask last observer position of each arm
        length_shower = np.sqrt(shower[:, 0] ** 2 + shower[:, 1] ** 2)
        ind = np.argpartition(length_shower, -n_arms)[-n_arms:]

        weight = spatial.ConvexHull(vertices_ground[:, :2])
        weights[p] = weight.volume  # volume of a 2d dataset is the area, area of a 2d data set is the perimeter
        weights[ind] = 0

        if debug:
            ax2.plot(vertices_ground[:, 0], vertices_ground[:, 1], c='grey', zorder=1)
            ax2.scatter(vertices_ground[:, 0], vertices_ground[:, 1], c='tab:orange', zorder=2)
    if debug:
        ax2.scatter(positions[:, 0], positions[:, 1], c='tab:blue', s=10, label='Position of observer')
        ax2.scatter(vertices_ground[:, 0], vertices_ground[:, 1], c='tab:orange', label='Vertices of cell')
        ax2.set_aspect('equal')
        ax2.set_title('On ground, total area {:.2f} $km^2$'.format(sum(weights) / units.km ** 2))
        ax2.set_xlabel('East [m]')
        ax2.set_ylabel('West [m]')
        ax2.set_xlim(-5000, 5000)
        ax2.set_ylim(-5000, 5000)
        plt.legend()
        plt.show()

        fig3 = plt.figure(figsize=[12, 4])
        ax4 = fig3.add_subplot(121)
        ax5 = fig3.add_subplot(122)

        ax4.hist(weights)
        ax4.set_title('Weight distribution')
        ax4.set_xlabel(r'Weights (here area) $[m^2]$')
        ax4.set_ylabel(r'Number of observer')

        ax5.scatter(length_shower ** 2, weights)
        ax5.set_xlabel(r'$Length^2 [m^2]$')
        ax5.set_ylabel('Weight $[m^2]$')
        plt.show()

    return weights

class coreasInterpolator:
    """
    The functions in this class are used to interpolate the fluence and signal shape from coreas files
    """

    def __init__(self, corsika):
        self.zenith = None
        self.azimuth = None
        self.magnetic_field_vector = None
        self.electric_field_on_sky = None
        self.efield_times = None
        self.obs_positions_geo = None
        self.obs_positions_vxB_vxvxB = None
        self.empty_efield = None
        self.max_coreas_efield = None
        self.star_radius = None
        self.geo_star_radius = None

        self.corsika = corsika
        self.star_shape_initilized = False
        self.efield_interpolator_initilized = False
        self.fluence_interpolator_initilized = False

        self.initialize_star_shape()

    def initialize_star_shape(self):
        self.zenith, self.azimuth, self.magnetic_field_vector = get_angles(self.corsika)
        cs = coordinatesystems.cstrafo(self.zenith, self.azimuth, self.magnetic_field_vector)

        obs_positions_geo = []
        electric_field_on_sky = []
        efield_times = []
        for j_obs, observer in enumerate(self.corsika['CoREAS']['observers'].values()):
            obs_positions_geo.append(convert_obs_positions_to_nuradio_on_ground(observer, self.zenith, self.azimuth, self.magnetic_field_vector))
            efield, efield_time = convert_obs_to_nuradio_efield(observer, self.zenith, self.azimuth, self.magnetic_field_vector)
            electric_field_on_sky.append(efield)
            efield_times.append(efield_time)

        # shape: (n_observers, n_samples, (eR, eTheta, ePhi))
        self.electric_field_on_sky = np.array(electric_field_on_sky)
        self.efield_times = np.array(efield_times)
        self.obs_positions_geo = np.array(obs_positions_geo)
        self.obs_positions_vxB_vxvxB = cs.transform_to_vxB_vxvxB(self.obs_positions_geo).T

        self.max_coreas_efield = np.max(np.abs(self.electric_field_on_sky))
        self.empty_efield = np.zeros_like(self.electric_field_on_sky[0,:,:])

        self.star_radius = np.max(np.linalg.norm(self.obs_positions_vxB_vxvxB[:, :-1], axis=-1))
        self.geo_star_radius = np.max(np.linalg.norm(self.obs_positions_geo[:-1, :], axis=0))
        logging.info(f'Initialize star shape pattern for interpolation. The shower arrives at zenith={self.zenith/units.deg:.0f}deg, azimuth={self.azimuth/units.deg:.0f}deg with radius {self.star_radius:.0f}m in the shower plane and {self.geo_star_radius:.0f}m on ground. ')
        self.star_shape_initilized = True

    def get_empty_efield(self):
        if not self.star_shape_initilized:
            logging.error('interpolator not initialized, call initialize_star_shape first')
            return None
        else:
            return self.empty_efield
    
    def get_max_efield(self):
        if not self.star_shape_initilized:
            logging.error('interpolator not initialized, call initialize_star_shape first')
            return None
        else:
            return self.max_coreas_efield

    def get_star_radius(self):
        if not self.star_shape_initilized:
            logging.error('interpolator not initialized, call initialize_star_shape first')
            return None
        else:
            return self.star_radius

    def get_geo_star_radius(self):
        if not self.star_shape_initilized:
            logging.error('interpolator not initialized, call initialize_star_shape first')
            return None
        else:
            return self.geo_star_radius

    def initialize_efield_interpolator(self, interp_lowfreq, interp_highfreq):
        self.efield_interpolator_initilized = True
        self.interp_lowfreq = interp_lowfreq
        self.interp_highfreq = interp_highfreq

        coreas_dt = (self.corsika['CoREAS'].attrs['TimeResolution'] * units.second)            
        geomagnetic_angle = get_geomagnetic_angle(self.zenith, self.azimuth, self.magnetic_field_vector)

        if geomagnetic_angle < 15*units.deg:
            logging.warning(f'geomagnetic angle is {geomagnetic_angle/units.deg:.2f} deg, which is smaller than 15 deg, which is the lower limit for the signal interpolation. The closest obersever is used instead.')
            self.efield_interpolator = -1
        else:
            logging.info(f'electric field interpolation with lowfreq {interp_lowfreq/units.MHz} MHz and highfreq {interp_highfreq/units.MHz} MHz')
            self.efield_interpolator = cr_pulse_interpolator.signal_interpolation_fourier.interp2d_signal(
                self.obs_positions_vxB_vxvxB[:, 0],
                self.obs_positions_vxB_vxvxB[:, 1],
                self.electric_field_on_sky,
                lowfreq=(interp_lowfreq-0.01)/units.MHz,
                highfreq=(interp_highfreq+0.01)/units.MHz,
                sampling_period=coreas_dt/units.s,
                phase_method="phasor",
                radial_method='cubic',
                upsample_factor=5,
                coherency_cutoff_threshold=0.9,
                ignore_cutoff_freq_in_timing=False,
                verbose=False
            )
        return self.efield_interpolator
                
    def initialize_fluence_interpolator(self, debug=False):
        self.fluence_interpolator_initilized = True
        if debug:
            max_efield = []
            for i in range(len(self.electric_field_on_sky[:,0,1])):
                max_efield.append(np.max(np.abs(self.electric_field_on_sky[i,:,:])))
            plt.scatter(self.obs_positions_vxB_vxvxB[:,0], self.obs_positions_vxB_vxvxB[:,1], c=max_efield, cmap='viridis', marker='o', edgecolors='k')
            cbar = plt.colorbar()
            cbar.set_label('max amplitude')
            plt.xlabel('v x B [m]')
            plt.ylabel('v x v x B [m]')
            plt.show()
            plt.close()

        #TODO use 10ns around the maximum for the fluence interpolation, then the sum of the square of the efield, get_electric_field_energy_fluence in utilities
        logging.info(f'fluence interpolation')
        self.fluence_interpolator = cr_pulse_interpolator.interpolation_fourier.interp2d_fourier(
            self.obs_positions_vxB_vxvxB[:, 0],
            self.obs_positions_vxB_vxvxB[:, 1],
            np.max(np.max(np.abs(self.electric_field_on_sky), axis=1),axis=1),
            fill_value="extrapolate"  # THIS OPTION IS UNUSED  IN fluinterp.interp2d_fourier.__init__
        )
        return self.fluence_interpolator

    def plot_footprint_fluence(self, dist_scale=300, save_file_path=None):
        from matplotlib import cm
        print("plotting footprint")

        # Make color plot of f(x, y), using a meshgrid
        ti = np.linspace(-dist_scale, dist_scale, 500)
        XI, YI = np.meshgrid(ti, ti)

        ### Get interpolated values at each grid point, calling the instance of interp2d_fourier
        ZI =  self.fluence_interpolator(XI, YI)
        ZI_mV = ZI/units.mV
        ###
        # And plot it
        maxp = np.max(ZI_mV)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pcolor(XI, YI, ZI_mV, vmax=maxp, vmin=0, cmap=cm.gnuplot2_r)
        mm = cm.ScalarMappable(cmap=cm.gnuplot2_r)
        mm.set_array([0.0, maxp])
        cbar = plt.colorbar(mm, ax=ax)
        cbar.set_label(r'Efield strength [mV/m]', fontsize=14)
        ax.set_xlabel(r'$\vec{v} \times \vec{B} [m]', fontsize=16)
        ax.set_ylabel(r'$\vec{v} \times (\vec{v} \times \vec{B})$ [m]', fontsize=16)
        ax.set_xlim(-dist_scale, dist_scale)
        ax.set_ylim(-dist_scale, dist_scale)
        ax.set_aspect('equal')
        if save_file_path is not None:
            plt.savefig(save_file_path)
        plt.close()

    def get_interp_efield_value(self, position_on_ground, core, kind):
        """
        Accesses the interpolated electric field given the position of the detector on ground. For the interpolation, the pulse will be
        projected in the shower plane for geometrical resonst. Set pulse_centered to True to
        shift all pulses to the center of the trace and account for the physical time delay of the signal.
        """
        antenna_position = position_on_ground
        z_plane = core[2]
        #core and antenna need to be in the same z plane
        antenna_position[2] = z_plane
        # transform antenna position into shower plane with respect to core position, core position is set to 0,0 in shower plane
        antenna_pos_vBvvB = self.cs.transform_to_vxB_vxvxB(antenna_position, core=core)

        # calculate distance between core position at(0,0) and antenna positions in shower plane
        dcore_vBvvB = np.linalg.norm(antenna_pos_vBvvB[:-1])
        # interpolate electric field at antenna position in shower plane which are inside star pattern
        if dcore_vBvvB > self.star_radius:
            efield_interp = self.empty_efield
            fluence_interp = 0
            logging.info(f'antenna position with distance {dcore_vBvvB:.2f} to core is outside of star pattern with radius {self.star_radius:.2f} on ground {self.geo_star_radius:.2f}, set efield and fluence to zero')
        else:
            if kind == 'efield':
                if self.efield_interpolator == -1:
                    efield = self.get_closest_observer_efield(antenna_pos_vBvvB)
                    efield_interp = efield
                else:
                    efield_interp = self.efield_interpolator(antenna_pos_vBvvB[0], antenna_pos_vBvvB[1],
                                                    lowfreq=self.interp_lowfreq/units.MHz,
                                                    highfreq=self.interp_highfreq/units.MHz,
                                                    filter_up_to_cutoff=False,
                                                    account_for_timing=False,
                                                    pulse_centered=True,
                                                    const_time_offset=20.0e-9,
                                                    full_output=False)
            elif kind == 'fluence':
                fluence_interp = self.fluence_interpolator(antenna_pos_vBvvB[0], antenna_pos_vBvvB[1])
            else:
                raise ValueError(f'kind {kind} not supported, please choose between efield and fluence')
        
        #check if interpolation is within expected range
        if kind == 'efield':
            if np.max(np.abs(efield_interp)) > self.max_coreas_efield:
                logging.warning(f'interpolated efield {np.max(np.abs(efield_interp)):.2f} is larger than the maximum coreas efield {self.max_coreas_efield:.2f}')
            return efield_interp
        elif kind == 'fluence':
            if np.max(np.abs(fluence_interp)) > self.max_coreas_efield:
                logging.warning(f'interpolated fluence {np.max(np.abs(fluence_interp)):.2f} is larger than the maximum coreas efield {self.max_coreas_efield:.2f}')
            return fluence_interp

    def get_closest_observer_efield(self, antenna_pos_vBvvB):
        distances = np.linalg.norm(antenna_pos_vBvvB[:2] - self.obs_positions_vBvvB[:,:2], axis=1)
        index = np.argmin(distances)
        distance = distances[index]
        efield = self.electric_field_on_sky[index,:,:]
        logging.info(f'antenna position with distance {distance:.2f} to closest observer is used')
        return efield