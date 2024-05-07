import unittest

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colormaps as cm

from NuRadioReco.modules.template_synthesis.slicedShower import slicedShower
from NuRadioReco.modules.template_synthesis.templateSynthesis import templateSynthesis, geo_ce_to_e
from NuRadioReco.utilities import units

from cr_pulse_interpolator.interpolation_fourier import interp2d_fourier as interpF


F_MIN, F_MAX, F_0 = 50, 350, 50


def compare_trace(trace1, trace2, plot=False):
    assertion = np.sum(np.abs(trace1 - trace2)) < 1e-14

    if not plot:
        return assertion

    if not assertion:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        ax.plot(np.sum(trace1, axis=0))
        ax.plot(np.sum(trace2, axis=0))

        ax.set_xlim([0, 500])

        return fig


class MyTestCase(unittest.TestCase):
    def self_synthesis(self):
        origin = slicedShower(
            '/home/mitjadesmet/Data/ShowersSKA/OriginShowers/SIM130165.hdf5'
        )

        synthesis = templateSynthesis()
        synthesis.begin(
            f'/home/mitjadesmet/Data/ShowersSKA/SpectralFitParameters_{F_MIN}_{F_MAX}_{F_0}/Showers30_1130.hdf5'
        )
        synthesis.frequency = np.array([F_MIN, F_MAX, F_0]) * units.MHz

        synthesis.make_template(origin)

        traces_synth = synthesis.synthesise(
            origin.xmax,
            origin.get_long_profile()[:-1]
        )

        for ind, ant in enumerate(synthesis.antennas):
            target_geo, target_ce = origin.get_trace(ant.name)

            target_geo_filtered = origin.filter_trace(
                target_geo, F_MIN * units.MHz, F_MAX * units.MHz
            )
            target_ce_filtered = origin.filter_trace(
                target_ce, F_MIN * units.MHz, F_MAX * units.MHz
            )

            traces_geo, traces_ce, traces_ce_lin = traces_synth[ind]

            assert np.sum(np.abs(target_geo_filtered - traces_geo)) < 1e-14
            assert np.sum(np.abs(target_ce_filtered - traces_ce)) < 1e-14
            assert np.sum(np.abs(target_ce_filtered - traces_ce_lin)) < 1e-14

    def test_synthesis(self):
        origin = slicedShower(
            '/home/mitjadesmet/Data/ShowersSKA/OriginShowers/SIM130165.hdf5'
        )

        target = slicedShower(
            '/home/mitjadesmet/Data/ShowersSKA/OriginShowers/SIM130188.hdf5'
        )

        synthesis = templateSynthesis()
        synthesis.begin(
            f'/home/mitjadesmet/Data/ShowersSKA/SpectralFitParameters_{F_MIN}_{F_MAX}_{F_0}/Showers30_1130.hdf5'
        )
        synthesis.frequency = np.array([F_MIN, F_MAX, F_0]) * units.MHz

        synthesis.make_template(origin)

        traces_synth = synthesis.synthesise(
            target.xmax,
            target.get_long_profile()[:-1]
        )

        for ind, ant in enumerate(synthesis.antennas):
            _, _, origin_times = origin.get_trace(ant.name, return_start_time=True)
            target_geo, target_ce, target_times = target.get_trace(ant.name, return_start_time=True)

            sample_diff = np.median(origin_times - target_times) / target.get_coreas_settings()['time_resolution']

            target_geo_filtered = target.filter_trace(
                target_geo, F_MIN * units.MHz, F_MAX * units.MHz
            )
            target_ce_filtered = target.filter_trace(
                target_ce, F_MIN * units.MHz, F_MAX * units.MHz
            )

            traces_geo, traces_ce, traces_ce_lin = traces_synth[ind]

            assert_geo = compare_trace(target_geo_filtered, np.roll(traces_geo, int(sample_diff)), plot=True)
            assert_ce = compare_trace(target_ce_filtered, np.roll(traces_ce, int(sample_diff)), plot=True)
            assert_ce_lin = compare_trace(target_ce_filtered, np.roll(traces_ce_lin, int(sample_diff)), plot=True)

            for component, assertion in zip(
                    ["GEO", "CE", "CE_LIN"],
                    [assert_geo, assert_ce, assert_ce_lin]
            ):
                if assertion is not None:
                    if isinstance(assertion, bool):
                        assert assertion, f"{component} trace does not fall within bounds"
                    else:
                        assertion.savefig(
                            f'{ant.name}_{component}_error.png'
                        )

        # Plot the result
        select_ant = 20

        _, _, origin_times = origin.get_trace(synthesis.antennas[select_ant].name,
                                              return_start_time=True)
        select_geo, select_ce, select_times = target.get_trace(synthesis.antennas[select_ant].name,
                                                               return_start_time=True)

        sample_diff = np.median(select_times - origin_times) / target.get_coreas_settings()['time_resolution']

        select_geo_filtered = target.filter_trace(
            select_geo, F_MIN * units.MHz, F_MAX * units.MHz
        )
        select_ce_filtered = target.filter_trace(
            select_ce, F_MIN * units.MHz, F_MAX * units.MHz
        )

        fig, ax = plt.subplots(1, 1)

        ax.plot(np.roll(np.sum(select_geo_filtered, axis=0), int(sample_diff)))
        ax.plot(np.sum(traces_synth[select_ant, 0], axis=0), '--')

        ax.plot(np.roll(np.sum(select_ce_filtered, axis=0), int(sample_diff)))
        ax.plot(np.sum(traces_synth[select_ant, 1], axis=0), '--')

        ax.set_xlim([0, 1000])

        plt.show()

    def fluence_interpolation(self):
        origin = slicedShower(
            '/home/mitjadesmet/Data/ShowersSKA/OriginShowers/SIM130165.hdf5'
        )

        synthesis = templateSynthesis()
        synthesis.begin(
            f'/home/mitjadesmet/Data/ShowersSKA/SpectralFitParameters_{F_MIN}_{F_MAX}_{F_0}/Showers30_1130.hdf5'
        )
        synthesis.frequency = np.array([F_MIN, F_MAX, F_0]) * units.MHz

        synthesis.make_template(origin)

        traces_synth = synthesis.synthesise(
            origin.xmax,
            origin.get_long_profile()[:-1]
        )

        traces_synth_ground, traces_synth_ground_lin = synthesis.transform_to_ground(traces_synth)

        traces_fluences = np.sum(traces_synth_ground ** 2, axis=(1, 2, 3))  # sum (x,y,z), slices and time samples
        # traces_fluences_lin = np.sum(traces_synth_ground_lin ** 2, axis=(1, 2, 3))

        transformer = synthesis.get_transformer()

        x_vB = []
        y_vvB = []
        origin_fluences = []
        for ind, ant in enumerate(synthesis.antennas):
            ant_pos = transformer.transform_to_vxB_vxvxB(
                np.array([-ant.position[1], ant.position[0], ant.position[2]]), core=[0, 0, 460]
            )
            x_vB.append(ant_pos[0])
            y_vvB.append(ant_pos[1])

            trace_geo, trace_ce = origin.get_trace(ant.name)
            trace_geo_filtered = origin.filter_trace(
                trace_geo, F_MIN * units.MHz, F_MAX * units.MHz
            )
            trace_ce_filtered = origin.filter_trace(
                trace_ce, F_MIN * units.MHz, F_MAX * units.MHz
            )
            trace_ground = geo_ce_to_e(trace_geo_filtered, trace_ce_filtered, *ant_pos[:2])

            origin_fluences.append(np.sum(trace_ground ** 2))

        # Make interpolator object
        interpolator = interpF(np.array(x_vB), np.array(y_vvB), traces_fluences)
        interpolator_origin = interpF(np.array(x_vB), np.array(y_vvB), origin_fluences)

        # Calculate footprint
        dist_scale = 200.0
        ti = np.linspace(-dist_scale, dist_scale, 1000)
        XI, YI = np.meshgrid(ti, ti)

        ZI = interpolator(XI, YI)
        maxp = np.max(ZI)

        ZI_origin = interpolator_origin(XI, YI)
        maxp_origin = np.max(ZI_origin)

        # Plot footprint
        norm = mcolors.Normalize(
            vmin=0,
            vmax=max(maxp, maxp_origin)
        )
        cmap = cm.get_cmap('viridis')

        fig, ax = plt.subplots(1, 2, figsize=(20, 10))

        ax[0].pcolor(XI, YI, ZI_origin, cmap=cmap, norm=norm)
        ax[1].pcolor(XI, YI, ZI, cmap=cmap, norm=norm)

        plt.show()


if __name__ == '__main__':
    unittest.main()
