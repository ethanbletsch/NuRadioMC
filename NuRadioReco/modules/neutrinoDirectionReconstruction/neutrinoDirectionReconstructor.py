import scipy
from scipy import constants
import scipy.stats as stats 
import NuRadioReco.modules.io.eventReader
from radiotools import helper as hp
import matplotlib.pyplot as plt
import numpy as np
from NuRadioReco.utilities import fft
from NuRadioReco.framework.parameters import stationParameters as stnp
import h5py
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.modules.neutrinoDirectionReconstruction import analytic_pulse
import matplotlib
from scipy import signal
from scipy import optimize as opt
from matplotlib import rc
import datetime
import math
from NuRadioReco.utilities import units
import datetime


class neutrinoDirectionReconstructor:
    
    
    def __init__(self):
        pass

    def begin(self, station, det, event, shower_ids, use_channels=[6, 14], ch_Vpol = 6, sim = True):
        """
        begin method. This function is executed before the event loop.
        We do not use this function for the reconsturctions. But only to determining uncertainties.
        """

        self._station = station
        self._use_channels = use_channels
        self._det = det
        self._sampling_rate = station.get_channel(0).get_sampling_rate()
        simulated_energy = 0
        for i in np.unique(shower_ids):
            simulated_energy += event.get_sim_shower(i)[shp.energy]
            
        shower_id = shower_ids
        self._simulated_azimuth = event.get_sim_shower(shower_id)[shp.azimuth]
        self._simulated_zenith = event.get_sim_shower(shower_id)[shp.zenith]
        if sim:
            vertex =event.get_sim_shower(shower_id)[shp.vertex]
        else:
            vertex = station[stnp.nu_vertex]
        simulation = analytic_pulse.simulation(False, vertex)
        rt = ['direct', 'refracted', 'reflected'].index(self._station[stnp.raytype]) + 1
        simulation.begin(det, station, use_channels, raytypesolution = rt, ch_Vpol = ch_Vpol, ch_Hpol = ch_Hpol)#[1, 2, 3] [direct, refracted, reflected]
        a, b, self._launch_vector_sim, c, d, e =  simulation.simulation(det, station, vertex[0],vertex[1], vertex[2], self._simulated_zenith, self._simulated_azimuth, simulated_energy, use_channels, first_iter = True)
        self._simulation = simulation
        return self._launch_vector_sim
    
    def run(self, event, station, det, shower_ids = None,
            use_channels=[6, 14], filenumber = 1, PA_channels = [0,1,2,3], single_pulse = False, debug_plots = False, template = False, sim_vertex = True, Vrms_Vpol = 0.0114, Vrms_Hpol = 0.0114, only_simulation = False, ch_Vpol = 6, ch_Hpol = 13, full_station = True, brute_force = True, fixed_timing = False, restricted_input = True, starting_values = False, debugplots_path = None, PA_cluster_channels = [0,1,2,3, 7,8], Hpol_channels = [7,8], window_Hpol = [10, +40], window_Vpol = [-10, +50], single_pulse_fit = False):

        """
        Module to reconstruct the direction of the event.
        event: Event
            The event to reconstruct the direction
        station: Station
            The station used to reconstruct the direction
        shower_ids: list
            list of shower ids for the event, only used if simulated vertex is used for input. Default shower_ids = None.
        use_channels: list
            list of channel ids used for the reconstruction
        filenumber: int
            This is only to link the debug plots to correct file. Default filenumber = 1.
        single_pulse: Boolean
            if True,
        debug_plots: Boolean
            if True, debug plots are produced. Default debug_plots = False.
        debugplots_path: str
            Path to store the debug plots. Default = None.
        template: Boolean
            If True, ARZ templates are used for the reconstruction. If False, a parametrization is used. 'Alvarez2009' and 'ARZ average' is available. Default template = False.
        sim_vertex: Boolean
            If True, the simulated vertex is used. This is for debugging purposes. Default sim_vertex = False.
        Vrms: float
            Noise root mean squared. Default = 0.0114 V
        only_simulation: Boolean
            if True, the fit is not performed but only the simulated values are compared with the data. This is just for debugging purposes. Default only_simulation = False.
        ch_Vpol: int
            channel id of the Vpol used to determine reference timing. Should be the top Vpol of the phased array. Default ch_Vpol = 6.
        ch_Hpol: int
            channel id of Hpol nearest to the Vpol. Timing of the Hpol is determined using the vertex position, because difference is only 1 m. Default ch_Vpol = 13.
        full_station: Boolean
            If True, all the raytypes in the list use_channels are used. If False, only the triggered pulse is used. Default full_station = True.
        brute_force: Boolean
            If True, brute force method is used. If False, minimization is used. Default brute_force = True.
        fixed_timing: Boolean
            If True, the known positions of the pulses are used calculated using the vertex position. Only allowed when sim_vertex is True. If False, an extra correlation is used to find the exact pulse position. Default fixed_timing = False.
        restricted_input: Boolean
            If True, a reconstruction is performed a few degrees around the MC values. This is (of course) only for simulations. Default restricted_input = False.
        starting_values: Boolean
            if True, first the channels of the phased array are used to get starting values for the viewing angle and the energy.
        debugplots_Path: str
            path to store plots.
        PA_cluster_channels:
        single_pulse_fit: Boolean
            if True, the viewing angle and energy are fitted with a PA Vpol and the polarization is fitted using an Hpol. Default single_pulse_fit = False.
        
        """

        station.set_is_neutrino()
        self._Vrms = Vrms_Vpol
        self._Vrms_Hpol = Vrms_Hpol
        self._station = station
        self._use_channels = use_channels
        self._det = det
        self._model_sys = 0.0 ## test amplitude effect of systematics on the model
        self._PA_cluster_channels = PA_cluster_channels
        self._Hpol_channels = Hpol_channels
        self._window_Hpol = window_Hpol
        self._window_Vpol = window_Vpol
        self._single_pulse_fit = False#single_pulse_fit
        self._PA_channels = PA_channels
        if single_pulse_fit:
            starting_values = True
        
            
        channl = station.get_channel(use_channels[0])
        n_samples = channl.get_number_of_samples()
        self._sampling_rate = int(channl.get_sampling_rate())
        sampling_rate = self._sampling_rate

        if sim_vertex:
            shower_id = shower_ids[0]
            reconstructed_vertex = event.get_sim_shower(shower_id)[shp.vertex]
            print("simulated vertex direction reco", event.get_sim_shower(shower_id)[shp.vertex])
        else:
            reconstructed_vertex = station[stnp.nu_vertex]
        
            print("reconstructed vertex direction reco", reconstructed_vertex)
      
        simulation = analytic_pulse.simulation(template, reconstructed_vertex) ### if the templates are used, than the templates for the correct distance are loaded
        rt = ['direct', 'refracted', 'reflected'].index(self._station[stnp.raytype]) + 1 ## raytype from the triggered pulse
        simulation.begin(det, station, use_channels, raytypesolution = rt, ch_Vpol = ch_Vpol, ch_Hpol = ch_Hpol)
        self._simulation = simulation

        if self._station.has_sim_station():
           
            sim_station = True
            simulated_zenith = event.get_sim_shower(shower_id)[shp.zenith]
            simulated_azimuth = event.get_sim_shower(shower_id)[shp.azimuth]
            self._simulated_azimuth = simulated_azimuth
            simulated_energy = 0
            for i, shower_id in enumerate(np.unique(shower_ids)):
                if (event.get_sim_shower(shower_id)[shp.type] != "em"):
                    simulated_energy += event.get_sim_shower(shower_id)[shp.energy]
                    print("simulated energy", simulated_energy)
            self.__simulated_energy =simulated_energy
            simulated_vertex = event.get_sim_shower(shower_id)[shp.vertex]
            ### values for simulated vertex and simulated direction
            tracsim, timsim, lv_sim, vw_sim, a, pol_sim = simulation.simulation(det, station, event.get_sim_shower(shower_id)[shp.vertex][0], event.get_sim_shower(shower_id)[shp.vertex][1], event.get_sim_shower(shower_id)[shp.vertex][2], simulated_zenith, simulated_azimuth, simulated_energy, use_channels, first_iter = True) 
       
            ## check SNR of channels
            SNR = []
            for ich, channel in enumerate(station.iter_channels()):
                print("channel {}, SNR {}".format(channel.get_id(),(abs(min(channel.get_trace())) + max(channel.get_trace())) / (2*self._Vrms) ))
                if channel.get_id() in use_channels:
                    SNR.append((abs(abs(min(channel.get_trace()))) + max(channel.get_trace())) / (2*self._Vrms))
        
       
    
        
        if 1:
        
            print("simulated vertex", simulated_vertex)
            print('reconstructed', reconstructed_vertex)
           
            
            #### values for reconstructed vertex and simulated direction
            if sim_station:
                traces_sim, timing_sim, self._launch_vector, viewingangles_sim, rayptypes, a = simulation.simulation( det, station, event.get_sim_shower(shower_id)[shp.vertex][0], event.get_sim_shower(shower_id)[shp.vertex][1], event.get_sim_shower(shower_id)[shp.vertex][2], simulated_zenith, simulated_azimuth, simulated_energy, use_channels, first_iter = True)

         
                fsimsim = self.minimizer([simulated_zenith,simulated_azimuth, np.log10(simulated_energy)], event.get_sim_shower(shower_id)[shp.vertex][0], event.get_sim_shower(shower_id)[shp.vertex][1], event.get_sim_shower(shower_id)[shp.vertex][2], minimize =  True, first_iter = True, ch_Vpol = ch_Vpol, ch_Hpol = ch_Hpol, full_station = full_station)
                all_fsimsim = self.minimizer([simulated_zenith,simulated_azimuth, np.log10(simulated_energy)], event.get_sim_shower(shower_id)[shp.vertex][0], event.get_sim_shower(shower_id)[shp.vertex][1], event.get_sim_shower(shower_id)[shp.vertex][2], minimize =  False, first_iter = True, ch_Vpol = ch_Vpol, ch_Hpol = ch_Hpol, full_station = full_station)[3]
                tracsim = self.minimizer([simulated_zenith,simulated_azimuth, np.log10(simulated_energy)], event.get_sim_shower(shower_id)[shp.vertex][0], event.get_sim_shower(shower_id)[shp.vertex][1], event.get_sim_shower(shower_id)[shp.vertex][2], minimize =  False, first_iter = True, ch_Vpol = ch_Vpol, ch_Hpol = ch_Hpol, full_station = full_station)[0]
                tracsim_recvertex = self.minimizer([simulated_zenith,simulated_azimuth, np.log10(simulated_energy)], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], minimize =  False, first_iter = True,ch_Vpol = ch_Vpol, ch_Hpol = ch_Hpol, full_station = full_station)[0]
 
                fsim = self.minimizer([simulated_zenith,simulated_azimuth, np.log10(simulated_energy)], event.get_sim_shower(shower_id)[shp.vertex][0], event.get_sim_shower(shower_id)[shp.vertex][1], event.get_sim_shower(shower_id)[shp.vertex][2], minimize =  True, first_iter = True, ch_Vpol = ch_Vpol, ch_Hpol = ch_Hpol, full_station = full_station)
              
                all_fsim = self.minimizer([simulated_zenith,simulated_azimuth, np.log10(simulated_energy)], event.get_sim_shower(shower_id)[shp.vertex][0], event.get_sim_shower(shower_id)[shp.vertex][1], event.get_sim_shower(shower_id)[shp.vertex][2], minimize =  False, first_iter = True, ch_Vpol = ch_Vpol, ch_Hpol = ch_Hpol, full_station = full_station)[3]
                print("Chi2 values for simulated direction and with/out simulated vertex are {}/{}".format(fsimsim, fsim))
            
                sim_reduced_chi2_Vpol = self.minimizer([simulated_zenith,simulated_azimuth, np.log10(simulated_energy)], event.get_sim_shower(shower_id)[shp.vertex][0], event.get_sim_shower(shower_id)[shp.vertex][1], event.get_sim_shower(shower_id)[shp.vertex][2], minimize =  False, ch_Vpol = ch_Vpol, ch_Hpol = ch_Hpol, full_station = full_station)[4][0]
               

                sim_reduced_chi2_Hpol = self.minimizer([simulated_zenith,simulated_azimuth, np.log10(simulated_energy)], event.get_sim_shower(shower_id)[shp.vertex][0], event.get_sim_shower(shower_id)[shp.vertex][1], event.get_sim_shower(shower_id)[shp.vertex][2], minimize =  False, first_iter = True, ch_Vpol = ch_Vpol, ch_Hpol = ch_Hpol, full_station = full_station)[4][1]
                self.minimizer([simulated_zenith,simulated_azimuth, np.log10(simulated_energy)], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], minimize =  False, first_iter = True, ch_Vpol = ch_Vpol, ch_Hpol = ch_Hpol, full_station = full_station)

            signal_zenith, signal_azimuth = hp.cartesian_to_spherical(*self._launch_vector) ## due to
            sig_dir = hp.spherical_to_cartesian(signal_zenith, signal_azimuth)
            self._vertex_azimuth = hp.cartesian_to_spherical(*reconstructed_vertex)[1]

            
            
           
            
            
                ###### START MINIMIZATION ################
                ### find starting values viewing angle and energy for brute force method ####
                ## for range of viewing angles, fit best energy.Brute force because otherwise side of cherenkov cone cannot be determined
            if starting_values or single_pulse_fit: ## if you include more channels,  it does not take the exact same viewing angle  for all channels. Same for adding the polarization. It takes the viewing angles wrt the used neutrino direction,  which is not what we want.  TODO: for starting values and single_pulse fit, it should take a single viewing angle! and result from signal direction fitter as input. Also, I still need to check if the polarization angle is well reconstructed with this approach.   
                for deg_angle in np.arange(-180, 180, 30):
                    angle = np.deg2rad(deg_angle)
                    cherenkov_angle = np.deg2rad(55.6)
                    signal_zenith, signal_azimuth = hp.cartesian_to_spherical(*self._launch_vector)

                    sig_dir = hp.spherical_to_cartesian(signal_zenith, signal_azimuth)
                    rotation_matrix = hp.get_rotation(sig_dir, np.array([0, 0,1]))


                    p3 = np.array([np.sin(cherenkov_angle)*np.cos(angle), np.sin(cherenkov_angle)*np.sin(angle), np.cos(cherenkov_angle)])
                    p3 = rotation_matrix.dot(p3)
                    azimuth = hp.cartesian_to_spherical(p3[0], p3[1], p3[2])[1]
                    zenith = hp.cartesian_to_spherical(p3[0], p3[1], p3[2])[0]
                    zenith = np.deg2rad(180) - zenith
                    if (np.rad2deg(zenith) < 100) and (np.rad2deg(zenith) > 20):
                        self._angle = angle#np.deg2rad(-100)
                        break;


                #print("zenith", np.rad2deg(zenith))
                #print(stop)
                method = 'Nelder-Mead'
                viewangles = []
                L = []
                energies = []
                v_angles = np.deg2rad([50,60])#np.deg2rad(np.arange(44, 65, .2))
                bounds = [((None, None), (None, None)), ((np.deg2rad(40), np.deg2rad(56)), (np.deg2rad(56), np.deg2rad(70)))]
                for iv, view in enumerate(v_angles):
                    
                    res = opt.minimize(self.minimizer, x0=(view, 16),args = (reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], True, False, False, True, False, ch_Vpol, ch_Hpol, False,False, False, True), bounds = bounds[iv],method = method)
                    viewangles.append(res.x[0])
                    energies.append(res.x[1])
                    L.append(res.fun)
                    self._viewing_angle = viewangles[np.argmin(L)]
                    self._log_energy = energies[np.argmin(L)]
                    
        
                if single_pulse_fit:
                    self._single_pulse_fit = True
                    res = opt.minimize(self.minimizer, x0=(.1),args = (reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], True, False, False, True, False, ch_Vpol, ch_Hpol, False,False, False, True), bounds = bounds[iv],method = method)
                    # minimizer with fitted viewing angle and energy for polarization
                    station.set_parameter(stnp.viewing_angle, [viewangles[np.argmin(L)], viewingangles_sim])
                    station.set_parameter(stnp.nu_energy, 10**energies[np.argmin(L)])
             #       print("single pulse fit")
              
                    station.set_parameter(stnp.polarization, res.x[0])
                
               #     exit()
        
                print("L", L)
                print("min L", min(L))
                check_starting_values =True  #
                if check_starting_values:
                    station.set_parameter(stnp.viewing_angle, [viewangles[np.argmin(L)], viewingangles_sim])
                    station.set_parameter(stnp.nu_energy, 10**energies[np.argmin(L)])
                    #exit()
                    if sim_station:
                        print("simulated viewing angle: {}, reconstructed viewing angle {}".format(np.rad2deg(viewingangles_sim), np.rad2deg(viewangles[np.argmin(L)])))
                        print("simulated energy: {}, reconstructed energy {}".format(simulated_energy, 10**energies[np.argmin(L)]))
                
                    #print("view", viewangles[np.argmin(L)])
                    #print("energies[np.argmin(L)]", energies[np.argmin(L)])    

                    print("debug plots...")
                    timingsim = self.minimizer([simulated_zenith, simulated_azimuth, np.log10(simulated_energy)], event.get_sim_shower(shower_id)[shp.vertex][0], event.get_sim_shower(shower_id)[shp.vertex][1], event.get_sim_shower(shower_id)[shp.vertex][2], first_iter = True, minimize = False, ch_Vpol = ch_Vpol, ch_Hpol = ch_Hpol, full_station = full_station)[2]
                    tracrec = self.minimizer([station[stnp.polarization]], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], minimize = False, starting_values = True, banana = True, ch_Vpol = ch_Vpol, ch_Hpol = ch_Hpol, full_station = False, single_pulse = True)[0]
                    #exit()
                    timingdata = self.minimizer([viewangles[np.argmin(L)], energies[np.argmin(L)]], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], minimize = False, banana = True, ch_Vpol = ch_Vpol, ch_Hpol = ch_Hpol, full_station = False, single_pulse = True)[2]
                    L_sim = self.minimizer([simulated_zenith, simulated_azimuth, np.log10(simulated_energy)], event.get_sim_shower(shower_id)[shp.vertex][0], event.get_sim_shower(shower_id)[shp.vertex][1], event.get_sim_shower(shower_id)[shp.vertex][2], first_iter = True, minimize = True, ch_Vpol = ch_Vpol, ch_Hpol = ch_Hpol, full_station = False, single_pulse = True, starting_values = True)
                    print("L for simulated:", L_sim)
                    L_rec = self.minimizer([viewangles[np.argmin(L)], energies[np.argmin(L)]], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], minimize = True, banana = True, ch_Vpol = ch_Vpol, ch_Hpol = ch_Hpol, full_station = False, single_pulse = True, starting_values = True)
                    print("L for rec:", L_rec)
                    debug_figure = True
                    if debug_figure:
                         linewidth = 1
                         fig, ax = plt.subplots(4, 1, sharex = True)
                         ich = 0
                         for channel in station.iter_channels():
                             if channel.get_id() in self._use_channels:#self._PA_cluster_channels:
                                 ax[ich].plot(channel.get_times(), channel.get_trace(), label = 'data', lw = linewidth, color = 'black')
                               
                                 ax[ich].plot(timingdata[channel.get_id()][0], tracrec[channel.get_id()][0], label = 'reconstruction', color = 'green')
                                 ax[ich].plot(timingsim[channel.get_id()][0], tracsim[channel.get_id()][0],'--', label = 'simulation', color = 'orange', lw = linewidth)
                                 ax[ich].legend(loc = 1)
                                 ax[ich].grid()
                                 ax[ich].set_xlim((9920, 9980))#timingsim[channel.get_id()][0][0], timingsim[channel.get_id()][0][-1])
                                 ich += 1
                         fig.savefig("{}/startingvalues.pdf".format(debugplots_path))
                         exit()          	
               

            if 1:#
            
                cherenkov = 55.6 ## cherenov angle
                viewing_start = vw_sim - np.deg2rad(2)
                viewing_end = vw_sim + np.deg2rad(2)
                energy_start = simulated_energy
                if starting_values:
                    viewing_start = viewangles[np.argmin(L)] - np.deg2rad(2)
                    viewing_end = viewangles[np.argmin(L)] + np.deg2rad(2)
                    energy_start = 10**energies[np.argmin(L)]
                #viewing_start = np.deg2rad(cherenkov) - np.deg2rad(15)
                #viewing_end = np.deg2rad(cherenkov) + np.deg2rad(15)
                theta_start = np.deg2rad(-180)
                theta_end =  np.deg2rad(180)

                cop = datetime.datetime.now()
                print("SIMULATED DIRECTION {} {}".format(np.rad2deg(simulated_zenith), np.rad2deg(simulated_azimuth)))

                if only_simulation:
                    print("no reconstructed is performed. The script is tested..")
                elif brute_force and not restricted_input:# restricted_input:
                    if starting_values:
                        results2 = opt.brute(self.minimizer, ranges=(slice(viewing_start, viewing_end, np.deg2rad(.5)), slice(theta_start, theta_end, np.deg2rad(1)), slice(np.log10(energy_start) - .15, np.log10(energy_start) + .15, .1)), full_output = True, finish = opt.fmin , args = (reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], True, False, False, True, False, ch_Vpol, ch_Hpol, full_station))
                        results1 = opt.brute(self.minimizer, ranges=(slice(viewing_start, viewing_end, np.deg2rad(.5)), slice(theta_start, theta_end, np.deg2rad(1)), slice(np.log10(energy_start) - .15, np.log10(energy_start) + .15, .1)), full_output = True, finish = opt.fmin , args = (reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], True, False, False, True, False, ch_Vpol, ch_Hpol, full_station))
                        if results2[1] < results1[1]:
                            results = results2
                        else:
                            results = results1
                    else:
                        results = opt.brute(self.minimizer, ranges=(slice(viewing_start, viewing_end, np.deg2rad(.5)), slice(theta_start, theta_end, np.deg2rad(1)), slice(np.log10(energy_start) - .1, np.log10(energy_start) + .1, .1)), full_output = True, finish = opt.fmin , args = (reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], True, False, False, True, False, ch_Vpol, ch_Hpol, full_station))
                        
                elif restricted_input:
                    zenith_start =  simulated_zenith - np.deg2rad(2)
                    zenith_end =simulated_zenith +  np.deg2rad(2)
                    azimuth_start =simulated_azimuth - np.deg2rad(2)
                    azimuth_end = simulated_azimuth + np.deg2rad(2)
                    energy_start = np.log10(simulated_energy) - 1
                    energy_end = np.log10(simulated_energy) + 1
                    results = opt.brute(self.minimizer, ranges=(slice(zenith_start, zenith_end, np.deg2rad(1)), slice(azimuth_start, azimuth_end, np.deg2rad(1)), slice(energy_start, energy_end, .1)), finish = opt.fmin, full_output = True, args = (reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], True, False, False, False, False, ch_Vpol, ch_Hpol, full_station))
                    
                print('start datetime', cop)
                print("end datetime", datetime.datetime.now() - cop)
                    
        
                ###### GET PARAMETERS #########
                
                if only_simulation:
                    rec_zenith = simulated_zenith
                    rec_azimuth = simulated_azimuth
                    rec_energy = simulated_energy
                
                elif brute_force and not restricted_input:
                    rotation_matrix = hp.get_rotation(sig_dir, np.array([0, 0,1]))
                    cherenkov_angle = results[0][0]
                    angle = results[0][1]

                    p3 = np.array([np.sin(cherenkov_angle)*np.cos(angle), np.sin(cherenkov_angle)*np.sin(angle), np.cos(cherenkov_angle)])
                    p3 = rotation_matrix.dot(p3)
                    global_az = hp.cartesian_to_spherical(p3[0], p3[1], p3[2])[1]
                    global_zen = hp.cartesian_to_spherical(p3[0], p3[1], p3[2])[0]
                    global_zen = np.deg2rad(180) - global_zen
                    

                    rec_zenith = global_zen
                    rec_azimuth = global_az
                    rec_energy = 10**results[0][2]
                    
                elif restricted_input:
                    rec_zenith = results[0][0]
                    rec_azimuth = results[0][1]
                    rec_energy = 10**results[0][2]

                ###### PRINT RESULTS ###############
                print("reconstructed energy {}".format(rec_energy))
                print("reconstructed zenith {} and reconstructed azimuth {}".format(np.rad2deg(rec_zenith), np.rad2deg(self.transform_azimuth(rec_azimuth))))
                print("         simualted zenith {}".format(np.rad2deg(simulated_zenith)))
                print("         simualted azimuth {}".format(np.rad2deg(simulated_azimuth)))
                
                print("     reconstructed zenith = {}".format(np.rad2deg(rec_zenith)))
                print("     reconstructed azimuth = {}".format(np.rad2deg(self.transform_azimuth(rec_azimuth))))
                
                print("###### seperate fit reconstructed valus")
                print("         zenith = {}".format(np.rad2deg(rec_zenith)))
                print("         azimuth = {}".format(np.rad2deg(self.transform_azimuth(rec_azimuth))))
                print("        energy = {}".format(rec_energy))
                print("         simualted zenith {}".format(np.rad2deg(simulated_zenith)))
                print("         simualted azimuth {}".format(np.rad2deg(simulated_azimuth)))
                print("         simulated energy {}".format(simulated_energy))
            
     
                
                print("RECONSTRUCTED DIRECTION ZENITH {} AZIMUTH {}".format(np.rad2deg(rec_zenith), np.rad2deg(self.transform_azimuth(rec_azimuth))))
                print("RECONSTRUCTED ENERGY", rec_energy)
                
               
                
                ## get the traces for the reconstructed energy and direction
                tracrec = self.minimizer([rec_zenith, rec_azimuth, np.log10(rec_energy)], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], minimize = False, ch_Vpol = ch_Vpol, ch_Hpol = ch_Hpol, full_station = full_station)[0]
                fit_reduced_chi2_Vpol = self.minimizer([rec_zenith, rec_azimuth, np.log10(rec_energy)], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], minimize = False, ch_Vpol = ch_Vpol, ch_Hpol = ch_Hpol, full_station = full_station)[4][0]
                fit_reduced_chi2_Hpol = self.minimizer([rec_zenith, rec_azimuth, np.log10(rec_energy)], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], minimize = False, ch_Vpol = ch_Vpol, ch_Hpol = ch_Hpol, full_station = full_station)[4][1]
                channels_overreconstructed = self.minimizer([rec_zenith, rec_azimuth, np.log10(rec_energy)], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], minimize = False, ch_Vpol = ch_Vpol, ch_Hpol = ch_Hpol, full_station = full_station)[5]
                extra_channel = self.minimizer([rec_zenith, rec_azimuth, np.log10(rec_energy)], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], minimize = False, ch_Vpol = ch_Vpol, ch_Hpol = ch_Hpol, full_station = full_station)[6]

                fminfit = self.minimizer([rec_zenith, rec_azimuth, np.log10(rec_energy)], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], minimize =  True, ch_Vpol = ch_Vpol, ch_Hpol = ch_Hpol, full_station = full_station)
               
                all_fminfit = self.minimizer([rec_zenith, rec_azimuth, np.log10(rec_energy)], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], minimize =  False, ch_Vpol = ch_Vpol, ch_Hpol = ch_Hpol, full_station = full_station)[3]
                bounds = ((14, 20))
                method = 'BFGS'
                results = scipy.optimize.minimize(self.minimizer, [14],method = method, args=(reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], True, False, False,False, [simulated_zenith, simulated_azimuth], ch_Vpol, ch_Hpol, True, False), bounds= bounds)
                fmin_simdir_recvertex = self.minimizer([simulated_zenith, simulated_azimuth, results.x[0]], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], minimize = True, ch_Vpol = ch_Vpol, ch_Hpol = ch_Hpol, full_station = full_station)
                print("energy reconstructed for simulated direction", 10**results.x[0])
                print("FMIN SIMULATED direction with simulated vertex", fsim)
                print("FMIN RECONSTRUCTED VALUE FIT", fminfit)
                

               # print("debug plots")
                if debug_plots:
                    linewidth = 5
                    tracdata = self.minimizer([rec_zenith, rec_azimuth, np.log10(rec_energy)], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], minimize = False, ch_Vpol = ch_Vpol, ch_Hpol = ch_Hpol, full_station = full_station)[1]
                    timingdata = self.minimizer([rec_zenith, rec_azimuth, np.log10(rec_energy)], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], minimize = False, ch_Vpol = ch_Vpol, ch_Hpol = ch_Hpol, full_station = full_station)[2]
                    timingsim = self.minimizer([simulated_zenith, simulated_azimuth, np.log10(simulated_energy)], event.get_sim_shower(shower_id)[shp.vertex][0], event.get_sim_shower(shower_id)[shp.vertex][1], event.get_sim_shower(shower_id)[shp.vertex][2], first_iter = True, minimize = False, ch_Vpol = ch_Vpol, ch_Hpol = ch_Hpol, full_station = full_station)[2]
                                  
                    timingsim_recvertex = self.minimizer([simulated_zenith, simulated_azimuth, np.log10(simulated_energy)], reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], first_iter = True, minimize = False, ch_Vpol = ch_Vpol, ch_Hpol = ch_Hpol, full_station = full_station)[2]
                    plt.rc('xtick', labelsize = 25)
                    plt.rc('ytick', labelsize = 25)
                    fig, ax = plt.subplots(len(use_channels), 3, sharex=False, figsize=(40, 10*len(use_channels)))

                    ich = 0
                    SNRs = np.zeros((len(use_channels), 2))

                    for channel in station.iter_channels():
                        if channel.get_id() in use_channels: # use channels needs to be sorted
                            isch = 0
                            for sim_channel in self._station.get_sim_station().get_channels_by_channel_id(channel.get_id()):
                                if isch == 0:
                                    sim_trace = sim_channel
                                if isch == 1:
                                    sim_trace += sim_channel
                                isch += 1

                            
                            if len(tracdata[channel.get_id()]) > 0:
                                ax[ich][0].grid()
                                ax[ich][2].grid()
                                ax[ich][0].set_xlabel("timing [ns]", fontsize = 30)
                                ax[ich][0].plot(channel.get_times(), channel.get_trace(), lw = linewidth, label = 'data', color = 'black')
                           
                                ax[ich][0].fill_between(timingdata[channel.get_id()][0], tracrec[channel.get_id()][0] - self._model_sys*tracrec[channel.get_id()][0], tracrec[channel.get_id()][0] + self._model_sys * tracrec[channel.get_id()][0], color = 'green', alpha = 0.2)
                                ax[ich][2].plot( np.fft.rfftfreq(len(tracdata[channel.get_id()][0]), 1/sampling_rate), abs(fft.time2freq( tracdata[channel.get_id()][0], sampling_rate)), color = 'black', lw = linewidth)
                                ax[ich][0].plot(timingsim[channel.get_id()][0], tracsim[channel.get_id()][0], label = 'simulation', color = 'orange', lw = linewidth)
                                ax[ich][0].plot(sim_trace.get_times(), sim_trace.get_trace(), label = 'sim channel', color = 'red', lw = linewidth)

                                #ax[ich][0].plot(timingsim_recvertex[channel.get_id()][0], tracsim_recvertex[channel.get_id()][0], label = 'simulation rec vertex', color = 'lightblue' , lw = linewidth, ls = '--')

                                ax[ich][0].set_xlim((timingsim[channel.get_id()][0][0], timingsim[channel.get_id()][0][-1]))
                
                                if 1:
                                     ax[ich][0].plot(timingdata[channel.get_id()][0], tracrec[channel.get_id()][0], label = 'reconstruction', lw = linewidth, color = 'green')
                                     #ax[ich][0].plot(timingdata[channel.get_id()][0], tracrec[channel.get_id()][0], label = 'reconstruction', color = 'green')

                                ax[ich][2].plot( np.fft.rfftfreq(len(sim_trace.get_trace()), 1/sampling_rate), abs(fft.time2freq(sim_trace.get_trace(), sampling_rate)), lw = linewidth, color = 'red')
                                ax[ich][2].plot( np.fft.rfftfreq(len(tracsim[channel.get_id()][0]), 1/sampling_rate), abs(fft.time2freq(tracsim[channel.get_id()][0], sampling_rate)), lw = linewidth, color = 'orange')
                                if 1:
                                     ax[ich][2].plot( np.fft.rfftfreq(len(tracrec[channel.get_id()][0]), 1/sampling_rate), abs(fft.time2freq(tracrec[channel.get_id()][0], sampling_rate)), color = 'green', lw = linewidth)
                                     ax[ich][2].set_xlim((0, 1))
                                     ax[ich][2].set_xlabel("frequency [GHz]", fontsize = 30)
                                ax[ich][0].legend(fontsize = 30)
                               
                            if len(tracdata[channel.get_id()]) > 1:
                                ax[ich][1].grid()
                                ax[ich][1].set_xlabel("timing [ns]", fontsize = 30)
                                ax[ich][1].plot(channel.get_times(), channel.get_trace(), label = 'data', lw = linewidth, color = 'black')
                                ax[ich][2].plot(np.fft.rfftfreq(len(timingsim[channel.get_id()][1]), 1/sampling_rate), abs(fft.time2freq(tracsim[channel.get_id()][1], sampling_rate)), lw = linewidth, color = 'red')
                                ax[ich][2].plot( np.fft.rfftfreq(len(tracdata[channel.get_id()][1]), 1/sampling_rate), abs(fft.time2freq(tracdata[channel.get_id()][1], sampling_rate)), color = 'black', lw = linewidth)
                                ax[ich][1].plot(timingsim[channel.get_id()][1], tracsim[channel.get_id()][1], label = 'simulation', color = 'orange', lw = linewidth)
                                ax[ich][1].plot(sim_trace.get_times(), sim_trace.get_trace(), label = 'sim channel', color = 'red', lw = linewidth)
                                if 1:#channel.get_id() in [6]:#,7,8,9]:
                                    ax[ich][1].plot(timingdata[channel.get_id()][1], tracrec[channel.get_id()][1], label = 'reconstruction', color = 'green', lw = linewidth)
                                    ax[ich][1].fill_between(timingdata[channel.get_id()][1], tracrec[channel.get_id()][1] - self._model_sys*tracrec[channel.get_id()][1], tracrec[channel.get_id()][1] + self._model_sys * tracrec[channel.get_id()][1], color = 'green', alpha = 0.2)
                            
                                ax[ich][2].plot( np.fft.rfftfreq(len(tracsim[channel.get_id()][1]), 1/sampling_rate), abs(fft.time2freq(tracsim[channel.get_id()][1], sampling_rate)), lw = linewidth, color = 'orange')
                                #ax[ich][1].plot(timingsim_recvertex[channel.get_id()][1], tracsim_recvertex[channel.get_id()][1], label = 'simulation rec vertex', color = 'lightblue', lw = linewidth, ls = '--')
                                ax[ich][1].set_xlim((timingsim[channel.get_id()][1][0], timingsim[channel.get_id()][1][-1]))
                                if 1:#channel.get_id() in [6]:
                                     ax[ich][2].plot( np.fft.rfftfreq(len(tracrec[channel.get_id()][1]), 1/sampling_rate), abs(fft.time2freq(tracrec[channel.get_id()][1], sampling_rate)), color = 'green', lw = linewidth)
                            
                            
                            ich += 1
                    ax[0][0].legend(fontsize = 30)

                    
                    fig.tight_layout()
                    print("output path for stored figure","{}/fit_{}.pdf".format(debugplots_path, filenumber))
                    fig.savefig("{}/fit_{}.pdf".format(debugplots_path, filenumber, shower_id))
                    #exit()

                 ### values for reconstructed vertex and reconstructed direction
                traces_rec, timing_rec, launch_vector_rec, viewingangle_rec, a, pol_rec =  simulation.simulation( det, station, reconstructed_vertex[0], reconstructed_vertex[1], reconstructed_vertex[2], rec_zenith, rec_azimuth, rec_energy, use_channels, first_iter = True)

                ###### STORE PARAMTERS AND PRINT PARAMTERS #########
                station.set_parameter(stnp.extra_channels, extra_channel)
                station.set_parameter(stnp.over_rec, channels_overreconstructed)
                station.set_parameter(stnp.nu_zenith, rec_zenith)
                station.set_parameter(stnp.nu_azimuth, self.transform_azimuth(rec_azimuth))
                station.set_parameter(stnp.nu_energy, rec_energy)
                station.set_parameter(stnp.chi2, [fsim, fminfit, fsimsim, self.__dof, sim_reduced_chi2_Vpol, sim_reduced_chi2_Hpol, fit_reduced_chi2_Vpol, fit_reduced_chi2_Hpol, fmin_simdir_recvertex])
                station.set_parameter(stnp.launch_vector, [lv_sim, launch_vector_rec])
                station.set_parameter(stnp.polarization, [pol_sim, pol_rec])
                station.set_parameter(stnp.viewing_angle, [vw_sim, viewingangle_rec])
                print("chi2 for simulated rec vertex {}, simulated sim vertex {} and fit {}".format(fsim, fsimsim, fminfit))#reconstructed vertex
                print("chi2 for all channels simulated rec vertex {}, simulated sim vertex {} and fit {}".format(all_fsim, all_fsimsim, all_fminfit))#reconstructed vertex
                print("launch vector for simulated {} and fit {}".format(lv_sim, launch_vector_rec))
                zen_sim = hp.cartesian_to_spherical(*lv_sim)[0]
                zen_rec = hp.cartesian_to_spherical(*launch_vector_rec)[0]
                print("launch zenith for simulated {} and fit {}".format(np.rad2deg(zen_sim), np.rad2deg(zen_rec)))
                print("polarization for simulated {} and fit {}".format(pol_sim, pol_rec))
                print("polarization angle for simulated {} and fit{}".format(np.rad2deg(np.arctan2(pol_sim[2], pol_sim[1])), np.rad2deg(np.arctan2(pol_rec[2], pol_rec[1]))))
                print("viewing angle for simulated {} and fit {}".format(np.rad2deg(vw_sim), np.rad2deg(viewingangle_rec)))
                print("reduced chi2 Vpol for simulated {} and fit {}".format(sim_reduced_chi2_Vpol, fit_reduced_chi2_Vpol))
                print("reduced chi2 Hpol for simulated {} and fit {}".format(sim_reduced_chi2_Hpol, fit_reduced_chi2_Hpol))
                print("over reconstructed channels", channels_overreconstructed)
                print("extra channels", extra_channel)
                print("L for rec vertex sim direction rec energy:", fmin_simdir_recvertex)
                print("L for reconstructed vertexy directin and energy:", fminfit)

    def transform_azimuth(self, azimuth): ## from [-180, 180] to [0, 360]
        azimuth = np.rad2deg(azimuth)
        if azimuth < 0:
            azimuth = 360 + azimuth
        return np.deg2rad(azimuth)


                  
    def minimizer(self, params, vertex_x, vertex_y, vertex_z, minimize = True, timing_k = False, first_iter = False, banana = False,  direction = [0, 0], ch_Vpol = 6, ch_Hpol = False, full_station = False, single_pulse =False, fixed_timing = False, starting_values = False, penalty = False ):
            """""""""""
            params: list
                input paramters for viewing angle / direction
            vertex_x, vertex_y, vertex_z: float
                input vertex
            minimize: Boolean
                If true, minimization output is given (chi2). If False, parameters are returned. Default minimize = True.
            first_iter: Boolean
                If true, raytracing is performed. If false, raytracing is not perfomred. Default first_iter = False.
            banana: Boolean
                If true, input values are viewing angle and energy. If false, input values should be theta and phi. Default banana = False.
            direction: list
                List with phi and theta direction. This is only for determining contours. Default direction = [0,0].
            ch_Vpol: int
                channel id for the Vpol of the reference pulse. Must be upper Vpol in phased array. Default ch_Vpol = 6.
            ch_Hpol: int
    		channel id for the Hpol which is closest by the ch_Vpol
            full_station:       
                if True, all raytype solutions for all channels are used, regardless of SNR of pulse. Default full_station = True.
            single_pulse: Boolean
                if True, only 1 pulse is used from the reference Vpol. Default single_pulse = False.
            fixed_timing: Boolean
                if True, the positions of the pulses using the simulated timing is used. This only works for the simulated vertex and for Alvarez2009 reconstruction and simulation. Default fixed_timing = False.
            starting_values: Boolean
                if True, the phased array cluster is used to obtain starting values for the viewing angle and the energy to limit the timing for the brute force approach. Default starting_values = False.
            penalty: Boolean
                if True, a penalty is included such that the reconstruction is not allowed to overshoot the traces with snr< 3.5. Default penalty = False.
                
                
            """""""""""""""
    
    
            model_sys = 0
            
            ### add filter
            #model_sys = 0
            #ff = np.fft.rfftfreq(600, .1)
            #mask = ff > 0
            #order = 8
            #passband = [300* units.MHz, 400* units.MHz]
            #b, a = signal.butter(order, passband, 'bandpass', analog=True)
            #w, ha = signal.freqs(b, a, ff[mask])
            #fa = np.zeros_like(ff, dtype=np.complex)
            #fa[mask] = ha
            #pol_filt = fa
            ######

            
            if banana: ## if input is viewing angle and energy, they need to be transformed to zenith and azimuth
                if len(params) ==3:
                    cherenkov_angle, angle, log_energy = params 
                    print("viewing angle and energy and angle ", [np.rad2deg(cherenkov_angle), log_energy, np.rad2deg(angle)])
                if len(params) == 2:
                    cherenkov_angle, log_energy = params
                    angle = self._angle
                    print("viewing angle and energy and angle ", [np.rad2deg(cherenkov_angle), log_energy, np.rad2deg(angle)])
                if len(params) == 1:
                    cherenkov_angle = self._viewing_angle
                    self._pol_angle = params
                    print("pol angle", self._pol_angle)
                    log_energy = self._log_energy
                    angle = self._angle
                energy = 10**log_energy
                

                #print("viewing angle and energy and angle ", [np.rad2deg(cherenkov_angle), log_energy, np.rad2deg(angle)])
                signal_zenith, signal_azimuth = hp.cartesian_to_spherical(*self._launch_vector)

                sig_dir = hp.spherical_to_cartesian(signal_zenith, signal_azimuth)
                rotation_matrix = hp.get_rotation(sig_dir, np.array([0, 0,1]))


                p3 = np.array([np.sin(cherenkov_angle)*np.cos(angle), np.sin(cherenkov_angle)*np.sin(angle), np.cos(cherenkov_angle)])
                p3 = rotation_matrix.dot(p3)
                azimuth = hp.cartesian_to_spherical(p3[0], p3[1], p3[2])[1]
                zenith = hp.cartesian_to_spherical(p3[0], p3[1], p3[2])[0]
                zenith = np.deg2rad(180) - zenith
               


                if np.rad2deg(zenith) > 100:
                    return np.inf ## not in field of view
                if np.rad2deg(zenith) < 20:  ## not in field of view
                    return np.inf
             
            else: 
                if len(params) ==3:
                    zenith, azimuth, log_energy = params 
                    energy = 10**log_energy
             #       print("parameters zen {} az {} energy {}".format(np.rad2deg(zenith), np.rad2deg(azimuth), energy))
                if len(params) == 1:
                    log_energy = params
                 
                    energy = 10**log_energy[0]
                    zenith, azimuth = direction
            
            azimuth = self.transform_azimuth(azimuth)
            
            pol_angle = 0
            if self._single_pulse_fit:
                pol_angle = self._pol_angle
           # print('starting values', starting_values) 
            traces, timing, launch_vector, viewingangles, raytypes, pol = self._simulation.simulation(self._det, self._station, vertex_x, vertex_y, vertex_z, zenith, azimuth, energy, self._use_channels, first_iter = first_iter, starting_values = starting_values, pol_angle = pol_angle) ## get traces due to neutrino direction and vertex position
         
            chi2 = 0
            all_chi2 = []
            over_reconstructed = [] ## list for channel ids where reconstruction is larger than data
            extra_channel = 0 ## count number of pulses besides triggering pulse in Vpol + Hpol


            rec_traces = {} ## to store reconstructed traces
            data_traces = {} ## to store data traces
            data_timing = {} ## to store timing
        

            #get timing and pulse position for raytype of triggered pulse
            for iS in raytypes[ch_Vpol]:
                #print("iS", raytypes[ch_Vpol][iS])
                if raytypes[ch_Vpol][iS] == ['direct', 'refracted', 'reflected'].index(self._station[stnp.raytype]) + 1:
                  #  print("station raytype", raytypes[ch_Vpol][iS])
                    solution_number = iS#for reconstructed vertex it can happen that the ray solution does not exist 
            T_ref = timing[ch_Vpol][solution_number]

            k_ref = self._station[stnp.pulse_position]# get pulse position for triggered pulse
          
            ks = {}
            
            ich = -1
            reduced_chi2_Vpol = 0
            reduced_chi2_Hpol = 0
            channels_Vpol = self._use_channels
            dict_dt = {}
            for ch in self._use_channels:    
                dict_dt[ch] = {}
            for channel in self._station.iter_channels(): ### FIRST SET TIMINGS
                channel_id = channel.get_id()
                if (channel_id in self._use_channels):

                    ich += 1 ## number of channel
                    data_trace = np.copy(channel.get_trace())
                    rec_traces[channel_id] = {}
                    data_traces[channel_id] = {}
                    data_timing[channel_id] = {}

                    ### if no solution exist, than analytic voltage is zero
                    rec_trace = np.zeros(len(data_trace))# if there is no raytracing solution, the trace is only zeros

                    delta_k = [] ## if no solution type exist then channel is not included
                    num = 0
                    chi2s = np.zeros(2)
                    for i_trace, key in enumerate(traces[channel_id]):#get dt for phased array pulse
                        #print("I_trace", i_trace)
                        rec_trace = traces[channel_id][key]

                        delta_T =  timing[channel_id][key] - T_ref
                        if int(delta_T) == 0:
                            trace_ref = i_trace
                           # print("I trace", i_trace)
   
                        ## before correlating, set values around maximum voltage trace data to zero
                        delta_toffset = delta_T * self._sampling_rate

                        ### figuring out the time offset for specfic trace
                        dk = int(k_ref + delta_toffset )# where do we expect the pulse to be wrt channel 6 main pulse and rec vertex position
                       
                        if 1:
                            data_trace_timing = np.copy(data_trace) ## cut data around timing
                            ## DETERMIINE PULSE REGION DUE TO REFERENCE TIMING

                            data_timing_timing = np.copy(channel.get_times())#np.arange(0, len(channel.get_trace()), 1)#
                            dk_1 = data_timing_timing[dk]
             #               print("channel id {} dk1 {}".format(channel_id, dk_1))
                            data_timing_timing = data_timing_timing[dk - self._sampling_rate*30 : dk + self._sampling_rate*50] ## 800 samples, like the simulation
                            data_trace_timing = data_trace_timing[dk - self._sampling_rate*30 : dk + self._sampling_rate*50]
                            data_trace_timing_1 = np.copy(data_trace_timing)
                            ### cut data trace timing to make window to search for pulse smaller
                            data_trace_timing_1[data_timing_timing < (dk_1 - 50)] = 0
                            data_trace_timing_1[data_timing_timing > (dk_1 + 50)] = 0
                    
                            library_channels ={}
                            for i_ch in self._use_channels:
                                library_channels[i_ch] = [i_ch]

                            corr = signal.hilbert(signal.correlate(rec_trace, data_trace_timing_1))
                            dt1 = np.argmax(corr) - (len(corr)/2) + 1
              
                            chi2_dt1 = np.sum((np.roll(rec_trace, math.ceil(-1*dt1)) - data_trace_timing_1)**2 / ((self._Vrms)**2))/len(rec_trace)
                            dt2 = np.argmax(corr) - (len(corr)/2)
                            chi2_dt2 = np.sum((np.roll(rec_trace, math.ceil(-1*dt2)) - data_trace_timing_1)**2 / ((self._Vrms)**2))/len(rec_trace)
                            if chi2_dt2 < chi2_dt1:
                                dt = dt2
                            else:
                                dt = dt1

                            corresponding_channels = library_channels[channel_id]
                            for ch in corresponding_channels:
                                dict_dt[ch][i_trace] = dt

            for channel in self._station.iter_channels(): # if SNR is low, get timings due to relative positions of channels for PA cluster
                if channel_id in self._PA_cluster_channels:
                    if (((max(channel.get_trace()) - min(channel.get_trace())) / (2*self._Vrms)) < 4):
                        dict_dt[channel_id][trace_ref] = dict_dt[ch_Vpol][trace_ref]
        
            fixed_timing = True
            if fixed_timing:
                for i_ch in self._use_channels:
                    dict_dt[i_ch][0] = dict_dt[ch_Vpol][trace_ref]
                    dict_dt[i_ch][1] = dict_dt[ch_Vpol][trace_ref]

            dof = 0
            for channel in self._station.iter_channels():
                channel_id = channel.get_id()
              #  print("channel id!", channel_id)
                if channel_id in self._use_channels:
             #       print("use chanenls", self._use_channels) 
       	            chi2s = np.zeros(2)
                    echannel = np.zeros(2)
                    dof_channel = 0
                    rec_traces[channel_id] = {}
                    data_traces[channel_id] = {}
                    data_timing[channel_id] = {}
                    data_trace = np.copy(channel.get_trace())
                    if traces[channel_id]:
                        for i_trace, key in enumerate(traces[channel_id]): ## iterate over ray type solutions
                            rec_trace = traces[channel_id][key]
                            delta_T =  timing[channel_id][key] - T_ref
            #                print("channel id", channel_id)
                            ## before correlating, set values around maximum voltage trace data to zero
                            delta_toffset = delta_T * self._sampling_rate

                            ### figuring out the time offset for specfic trace
                            dk = int(k_ref + delta_toffset )
                            if 1:#
                                data_trace_timing = np.copy(data_trace) ## cut data around timing
                                
                                ## DETERMINE PULSE REGION DUE TO REFERENCE TIMING
                                data_timing_timing = np.copy(channel.get_times())#np.arange(0, len(channel.get_trace()), 1)#
                                dk_1 = data_timing_timing[dk]

                                data_timing_timing = data_timing_timing[dk - self._sampling_rate*30 : dk + self._sampling_rate*50] ## 800 samples, like the simulation
                                data_trace_timing = data_trace_timing[dk - self._sampling_rate*30 : dk + self._sampling_rate*50]
                                dt = dict_dt[channel_id][i_trace]
                                rec_trace = np.roll(rec_trace, math.ceil(-1*dt))
                                 
                                #### select fitting time-window ####
                                if channel_id in self._Hpol_channels:
                                    indices = [i for i, x in enumerate(data_timing_timing) if (x > (dk_1 + self._window_Hpol[0])  and (x < (dk_1 + self._window_Hpol[1]) ))]
                                else:
                                    indices = [i for i, x in enumerate(data_timing_timing) if (x > (dk_1 + self._window_Vpol[0])  and (x < (dk_1 + self._window_Vpol[1]) ))]
                                rec_trace = rec_trace[indices]
                                data_trace_timing = data_trace_timing[indices]
                                data_timing_timing = data_timing_timing[indices]
           
                           
                                ks[channel_id] = delta_k
                                rec_traces[channel_id][i_trace] = rec_trace
                                data_traces[channel_id][i_trace] = data_trace_timing
                                data_timing[channel_id][i_trace] = data_timing_timing
                                SNR = abs(max(data_trace_timing) - min(data_trace_timing) ) / (2*self._Vrms)
                
                                
                                ### set vrms and time_window for channel
                                if channel_id in self._Hpol_channels:
                                    Vrms = self._Vrms_Hpol
                                else:
                                    Vrms = self._Vrms
                                 
                                if fixed_timing:
                                    if SNR > 3.5:
                                        echannel[i_trace] = 1
                                
                                if (single_pulse):
                                    if ((channel_id == ch_Vpol) and (i_trace == trace_ref)):
                                        chi2s[i_trace] = np.sum((rec_trace - data_trace_timing)**2 / ((Vrms+model_sys*abs(data_trace_timing))**2))
                                        reduced_chi2_Vpol = np.sum((rec_trace - data_trace_timing)**2 / ((self._Vrms+model_sys*abs(data_trace_timing))**2))/len(rec_trace)
                                        Vpol_ref = np.sum((rec_trace - data_trace_timing)**2 / ((self._Vrms+model_sys*abs(data_trace_timing))**2))/len(rec_trace)
                                    #print("single pulse", single_pulse)
                                    dof_channel += 1
                                    
                                elif (self._single_pulse_fit) and (i_trace == trace_ref): #use only 1 Vpol and 1 Hpol as input channels! 
                                    if ((channel_id == ch_Vpol) and (i_trace == trace_ref)):
                                        chi2s[i_trace] = np.sum((rec_trace - data_trace_timing)**2 / ((Vrms+model_sys*abs(data_trace_timing))**2))
                                        reduced_chi2_Vpol = np.sum((rec_trace - data_trace_timing)**2 / ((self._Vrms+model_sys*abs(data_trace_timing))**2))/len(rec_trace)
                                        Vpol_ref = np.sum((rec_trace - data_trace_timing)**2 / ((self._Vrms+model_sys*abs(data_trace_timing))**2))/len(rec_trace)
                                    if ((channel_id == ch_Hpol) and (i_trace == trace_ref)):
                                        chi2s[i_trace] = np.sum((rec_trace - data_trace_timing)**2 / ((Vrms+model_sys*abs(data_trace_timing))**2))
                                        reduced_chi2_Hpol = np.sum((rec_trace - data_trace_timing)**2 / ((self._Vrms+model_sys*abs(data_trace_timing))**2))/len(rec_trace)
                                        Hpol_ref = np.sum((rec_trace - data_trace_timing)**2 / ((self._Vrms+model_sys*abs(data_trace_timing))**2))/len(rec_trace)
                                    dof_channel += 1
                    #                print("single pulse fit")                  
                                elif ((channel_id in self._PA_channels) and (i_trace == trace_ref) and starting_values) and not self._single_pulse_fit: #PA_cluster_channels contains all channels that are definitely included in the fit and for which the timings are fixed.
                                   
                                #    print("starting values")
                                    if channel_id == ch_Vpol:
                                        reduced_chi2_Vpol = np.sum((rec_trace - data_trace_timing)**2 / ((self._Vrms+model_sys*abs(data_trace_timing))**2))/len(rec_trace)
                                        Vpol_ref = np.sum((rec_trace - data_trace_timing)**2 / ((self._Vrms+model_sys*abs(data_trace_timing))**2))/len(rec_trace)
                                        #print("ch_Vpol", ch_Vpol)    
                                    #print("rec trace", rec_trace)
                                    #print('data trace timing', data_trace_timing) 
                                   
                                    chi2s[i_trace] = np.sum((rec_trace - data_trace_timing)**2 / ((Vrms+model_sys*abs(data_trace_timing))**2))
                                    dof_channel += 1
                                    echannel[i_trace] = 1
                                    #print("chi2s i trace", chi2s[i_trace]) 
                                elif ((channel_id in self._PA_cluster_channels) and (i_trace == trace_ref) and not starting_values and not self._single_pulse_fit):
                                    if channel_id == ch_Vpol:
                                        reduced_chi2_Vpol = np.sum((rec_trace - data_trace_timing)**2 / ((self._Vrms+model_sys*abs(data_trace_timing))**2))/len(rec_trace)
                                        Vpol_ref = np.sum((rec_trace - data_trace_timing)**2 / ((self._Vrms+model_sys*abs(data_trace_timing))**2))/len(rec_trace)
                                    if channel_id == ch_Hpol:
                                        reduced_chi2_Hpol = np.sum((rec_trace - data_trace_timing)**2 / ((self._Vrms_Hpol)**2))/len(rec_trace)
                                        Hpol_ref = np.sum((rec_trace - data_trace_timing)**2 / ((self._Vrms+model_sys*abs(data_trace_timing))**2))/len(rec_trace)
                                        
                                    chi2s[i_trace] = np.sum((rec_trace - data_trace_timing)**2 / ((Vrms+model_sys*abs(data_trace_timing))**2))
                                    dof_channel += 1
                                    echannel[i_trace] = 1
                                    #print("extra", channel_id)
                                elif 1:#((channel_id in self._use_channels) and (full_station) and (SNR > 3.5) and not starting_values and not self._single_pulse_fit):
                                    chi2s[i_trace] = np.sum((rec_trace - data_trace_timing)**2 / ((Vrms+model_sys*abs(data_trace_timing))**2))
                                    dof_channel += 1
                  #                  print("SNR > 3.5 channel id", channel_id)
                                    echannel[i_trace] = 1
                                elif penalty:
                                    #print("penalty", penalty)
                                    if abs(max(rec_trace) - min(rec_trace))/(2*Vrms) > 4.0:
                                        chi2s[i_trace] = np.inf
            
                    else:#if no raytracing solution exist
                        rec_traces[channel_id][0] = np.zeros(80 * int(self._sampling_rate))
                        data_traces[channel_id][0] = np.zeros(80 * int(self._sampling_rate))
                        data_timing[channel_id][0] = np.zeros(80 * int(self._sampling_rate))
                        rec_traces[channel_id][1] = np.zeros(80 * int(self._sampling_rate))
                        data_traces[channel_id][1] = np.zeros(80 * int(self._sampling_rate))
                        data_timing[channel_id][1] = np.zeros(80 * int(self._sampling_rate))

                    #### if the pulses are overlapping, than we don't include them in the fit because the timing is not exactly known.
                    if max(data_timing[channel_id][0]) > min(data_timing[channel_id][1]):
                        if int(min(data_timing[channel_id][1])) != 0:
        #
                            if (channel_id == ch_Vpol):
                                chi2 += chi2s[trace_ref]# = np.sum((rec_trace1 - data_trace_timing)**2 / ((sigma+model_sys*abs(data_trace_timing))**2))/len(rec_trace1)
                #                reduced_chi2_Vpol = Vpol_ref
                                dof += 1
                                #combine fit
            #                    print("1 pulse")
                            if (channel_id == ch_Hpol):
                                if 'Hpol_ref' in locals(): #Hpol_ref is only defined when this is supposed to be included in the fit
                                    chi2 += chi2s[trace_ref]
             #                       print("Hpol")#                   reduced_chi2_Hpol = Hpol_ref
                         
        
                    else:
                            extra_channel += echannel[0]
                            extra_channel += echannel[1]
                            #print("chi2s {} for channel {}".format( chi2s,channel_id))
                            chi2 += chi2s[0]
                            chi2 += chi2s[1]
                            dof += dof_channel
                            all_chi2.append(chi2s[0])
                            all_chi2.append(chi2s[1])
                
            self.__dof = dof
            #print("extra channel", extra_channel)
            if timing_k:
                return ks
            if not minimize:
                return [rec_traces, data_traces, data_timing, all_chi2, [reduced_chi2_Vpol, reduced_chi2_Hpol], over_reconstructed, extra_channel]
        #    print("parameters zen {} az {} energy {} viewing angle {} chi2 {}".format(np.rad2deg(zenith), np.rad2deg(azimuth), np.log10(energy), np.rad2deg(viewingangles), chi2))
           
            #print("chi2", chi2)
            return chi2 ### sum of reduced chi2 per time window
            
            """
                    ### helper functions for plotting
            def mollweide_azimuth(az):
                az -= (simulated_azimuth - np.deg2rad(180)) ## put simulated azimuth at 180 degrees
                az = np.remainder(az, np.deg2rad(360)) ## rotate values such that they are between 0 and 360
                az -= np.deg2rad(180)
                return az
        
            def mollweide_zenith(zen):
                zen -= (simulated_zenith  - np.deg2rad(90)) ## put simulated azimuth at 90 degrees
                zen = np.remainder(zen, np.deg2rad(180)) ## rotate values such that they are between 0 and 180
                zen -= np.deg2rad(90) ## hisft to mollweide projection
                return zen


            def get_normalized_angle(angle, degree=False, interval=np.deg2rad([0, 360])):
                import collections
                if degree:
                    interval = np.rad2deg(interval)
                delta = interval[1] - interval[0]
                if(isinstance(angle, (collections.Sequence, np.ndarray))):
                    angle[angle >= interval[1]] -= delta
                    angle[angle < interval[0]] += delta
                else:
                    while (angle >= interval[1]):
                        angle -= delta
                    while (angle < interval[0]):
                        angle += delta
                return angle
            """
    def end(self):
        pass
