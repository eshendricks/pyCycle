import unittest

import numpy as np

import openmdao.api as om

from pycycle.thermo.cea import species_data
from pycycle.constants import AIR_ELEMENTS, AIR_FUEL_ELEMENTS
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials

from pycycle.elements.mix_ratio import MixRatio

class MixRatioTestCase(unittest.TestCase):
    def test_mix_1fuel(self): 

        thermo_spec = species_data.janaf

        air_thermo = species_data.Properties(thermo_spec, init_elements=AIR_ELEMENTS)

        p = om.Problem()

        fuel = 'JP-7'
        p.model = MixRatio(inflow_thermo_data=thermo_spec, mix_thermo_data=thermo_spec,
                           inflow_elements=AIR_ELEMENTS, mix_mode='reactant', 
                           mix_elements=fuel, mix_names='fuel')


        p.setup(force_alloc_complex=True)

        # p['Fl_I:stat:P'] = 158.428
        p['Fl_I:stat:W'] = 38.8
        p['Fl_I:tot:h'] = 181.381769
        p['Fl_I:tot:b0'] = air_thermo.b0

        p['fuel:ratio'] = 0.02673


        p.run_model()

        tol = 5e-7
        assert_near_equal(p['mass_avg_h'], 176.65965638, tolerance=tol)
        assert_near_equal(p['Wout'], 39.837124, tolerance=tol)
        assert_near_equal(p['fuel:W'], p['Fl_I:stat:W']*p['fuel:ratio'], tolerance=tol)
        assert_near_equal(p['b0_out'], np.array([0.0003149, 0.00186566, 0.00371394, 0.05251212, 0.01410888]), tolerance=tol)

        data = p.check_partials(out_stream=None, method='cs')
        # data = p.check_partials(method='cs')
        assert_check_partials(data, atol=1.e-6, rtol=1.e-6)

    def test_mix_2fuel(self): 

        thermo_spec = species_data.janaf

        air_thermo = species_data.Properties(thermo_spec, init_elements=AIR_ELEMENTS)

        p = om.Problem()

        fuel = 'JP-7'
        p.model = MixRatio(inflow_thermo_data=thermo_spec, mix_thermo_data=thermo_spec,
                           inflow_elements=AIR_ELEMENTS, mix_mode='reactant', 
                           mix_elements=[fuel, fuel], mix_names=['fuel1','fuel2'])


        p.setup(force_alloc_complex=True)

        # p['Fl_I:stat:P'] = 158.428
        p['Fl_I:stat:W'] = 38.8
        p['Fl_I:tot:h'] = 181.381769
        p['Fl_I:tot:b0'] = air_thermo.b0

        # half the ratio from the 1 fuel test
        ratio = 0.02673/2.
        p['fuel1:ratio'] = ratio
        p['fuel2:ratio'] = ratio


        p.run_model()

        tol = 5e-7
        assert_near_equal(p['mass_avg_h'], 176.65965638, tolerance=tol)
        assert_near_equal(p['Wout'], 39.837124, tolerance=tol)
        assert_near_equal(p['fuel1:W'], p['Fl_I:stat:W']*ratio, tolerance=tol)
        assert_near_equal(p['fuel2:W'], p['Fl_I:stat:W']*ratio, tolerance=tol)
        assert_near_equal(p['b0_out'], np.array([0.0003149, 0.00186566, 0.00371394, 0.05251212, 0.01410888]), tolerance=tol)

        data = p.check_partials(out_stream=None, method='cs')
        # data = p.check_partials(method='cs')
        assert_check_partials(data, atol=1.e-6, rtol=1.e-6)

    def test_mix_1flow(self): 

        thermo_spec = species_data.janaf 

        p = om.Problem()

        p.model = MixRatio(inflow_thermo_data=thermo_spec, mix_thermo_data=thermo_spec,
                           inflow_elements=AIR_FUEL_ELEMENTS, mix_mode='flow', 
                           mix_elements=AIR_ELEMENTS, mix_names='mix')

        p.setup(force_alloc_complex=True)

        p['Fl_I:stat:W'] = 62.15
        # p['Fl_I:tot:h'] = 181.381769
        # p['Fl_I:tot:b0'] = [0.00031378, 0.00211278, 0.00420881, 0.05232509, 0.01405863]
        p['Fl_I:tot:b0'] = [0.000313780313538, 0.0021127831122, 0.004208814234964, 0.052325087161902, 0.014058631311261]

        p['mix:W'] = 4.44635
        p['mix:b0'] = [3.23319258e-04, 1.10132241e-05, 5.39157736e-02, 1.44860147e-02]

        p.run_model()

        tol = 5e-7
        # assert_near_equal(p['b0_out'], np.array([0.0003149, 0.00186566, 0.00371394, 0.05251212, 0.01410888]), tolerance=tol)
        assert_near_equal(p['b0_out'], np.array([0.00031442, 0.00197246, 0.00392781, 0.05243129, 0.01408717]), tolerance=tol)

    def test_mix_2flow(self): 

        thermo_spec = species_data.janaf 

        p = om.Problem()

        p.model = MixRatio(inflow_thermo_data=thermo_spec, mix_thermo_data=thermo_spec,
                           inflow_elements=AIR_FUEL_ELEMENTS, mix_mode='flow', 
                           mix_elements=[AIR_ELEMENTS, AIR_ELEMENTS], mix_names=['mix1', 'mix2'])

        p.setup(force_alloc_complex=True)

        p['Fl_I:stat:W'] = 62.15
        # p['Fl_I:tot:h'] = 181.381769
        # p['Fl_I:tot:b0'] = [0.00031378, 0.00211278, 0.00420881, 0.05232509, 0.01405863]
        p['Fl_I:tot:b0'] = [0.000313780313538, 0.0021127831122, 0.004208814234964, 0.052325087161902, 0.014058631311261]

        p['mix1:W'] = 4.44635/2
        p['mix1:b0'] = [3.23319258e-04, 1.10132241e-05, 5.39157736e-02, 1.44860147e-02]

        p['mix2:W'] = 4.44635/2
        p['mix2:b0'] = [3.23319258e-04, 1.10132241e-05, 5.39157736e-02, 1.44860147e-02]

        p.run_model()

        tol = 5e-7
        # assert_near_equal(p['b0_out'], np.array([0.0003149, 0.00186566, 0.00371394, 0.05251212, 0.01410888]), tolerance=tol)
        assert_near_equal(p['b0_out'], np.array([0.00031442, 0.00197246, 0.00392781, 0.05243129, 0.01408717]), tolerance=tol)





if __name__ == "__main__": 

    unittest.main()