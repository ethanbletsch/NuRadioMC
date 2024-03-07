import os
import yaml
import numpy as np
from NuRadioMC.simulation import simulation2 as sim
from NuRadioReco.detector import detector
from NuRadioMC.SignalProp import propagation
from NuRadioMC.utilities import medium
import NuRadioReco.framework.radio_shower
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.utilities import units
from datetime import datetime

"""
This script is an example of how to calculate the efield at observer positions
for a list of a showers using the `claculate_sim_efield` function.
The observer positions are defined in the detector object.
The showers are defined in the shower objects.
General config settings are defined in the NuRadioMC yaml config file.
The user also needs to specify the medium model (i.e. ice model) and the
propagation module to use (e.g. the analytic ray tracer).
"""

# set the ice model
ice = medium.get_ice_model('southpole_simple')
# set the propagation module
propagator = propagation.get_propagation_module("analytic")(ice)

# set the station id and channel id
sid = 101
cid = 0

# get the general config settings
cfg = sim.get_config("config.yaml")

# initialize the detector description (from the json file)
kwargs = dict(json_filename="surface_station_1GHz.json", antenna_by_depth=False)
det = detector.Detector(**kwargs)
det.update(datetime.now())

# define the showers that should be simulated
showers = []
shower = NuRadioReco.framework.radio_shower.RadioShower(0)
# according to our convention, the shower direction is the direction of
# where the shower is coming from.
shower[shp.zenith] = 70 * units.deg # propagation downwards
shower[shp.azimuth] = 180 * units.deg # propagation into the positive x direction
shower[shp.energy] = 1e17 * units.eV
shower[shp.vertex] = np.array([-500*units.m, 0, -1*units.km])
shower[shp.type] = 'had'
showers.append(shower)

# calculate the electric fields at the observer positions from the showers
sim_station = sim.calculate_sim_efield(showers, sid, cid,
                         det, propagator, ice, cfg)


# Plot the resulting electric fields in the time and frequency domain
# We only simulated one shower and one channel (i.e. oberver position)
# but we get two ray tracing solutions.
# The unique identifier of the efield object (plotted in the legend)
# is the channel id, shower id and ray tracing solution id.
import matplotlib.pyplot as plt
fig, (ax, ax2) = plt.subplots(1,2)

for i, efield in enumerate(sim_station.get_electric_fields()):
    trace = efield.get_trace()
    ax.plot(efield.get_times(), trace[1]/units.V*units.m, f"-C{i}", label=f'efield id {efield.get_unique_identifier()}')
    ax.plot(efield.get_times(), trace[2]/units.V*units.m, f"--C{i}")
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Electric Field (V/m)')
    ax2.plot(efield.get_frequencies()/units.MHz, np.abs(efield.get_frequency_spectrum()[1]/units.V*units.m*units.MHz),
             f"-C{i}",label=f'efield id {efield.get_unique_identifier()}')
    ax2.plot(efield.get_frequencies()/units.MHz, np.abs(efield.get_frequency_spectrum()[2]/units.V*units.m*units.MHz),
             f"--C{i}")
    ax2.set_xlabel('Frequency (MHz)')
    ax2.set_ylabel('Electric Field (V/m/MHz)')
ax.legend()
ax2.legend()
fig.tight_layout()
plt.show()