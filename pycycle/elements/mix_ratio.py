import numpy as np

import openmdao.api as om

from pycycle.constants import AIR_ELEMENTS
from pycycle.thermo.cea.species_data import Properties, janaf

class MixRatio(om.ExplicitComponent):
    """
    MixRatio calculates a new b0 given inflow, a reactant to add, and a mix ratio.
    """

    def initialize(self):
        self.options.declare('inflow_thermo_data', default=None,
                             desc=('Thermodynamic data set for incoming flow. This only needs to be set if different '
                                   'thermo data is used for incoming flow and outgoing flow.'), 
                             recordable=False)
        self.options.declare('mix_thermo_data', default=janaf,
                             desc=('Thermodynamic data set for flow. This is used for incoming and '
                                   'outgoing flow unless inflow_thermo_data is set, in which case it '
                                   'is used only for outgoing flow.'), 
                             recordable=False)
        self.options.declare('inflow_elements', default=AIR_ELEMENTS,
                             desc='set of elements present in the flow')

        self.options.declare('mix_mode', values=['reactant', 'flow'], default='reactant')

        self.options.declare('mix_elements', default="JP-7",
                             desc='Type of fuel.', types=(dict, str, list, tuple))
        self.options.declare('mix_names', default='mix', types=(str, list, tuple))

    def setup(self):
        thermo_data = self.options['mix_thermo_data']
        if self.options['inflow_thermo_data'] is not None:
            # Set the inflow thermodynamic data package if it is different from the outflow one
            inflow_thermo_data = self.options['inflow_thermo_data']

        else:
            # Set the inflow thermodynamic data package if it is the same as the outflow one
            inflow_thermo_data = thermo_data

        mix_mode = self.options['mix_mode']

        mix_elements = self.options['mix_elements']
        if isinstance(mix_elements, (str, dict)): # cast it to tuple
            mix_elements = (mix_elements,)
        self.mix_elements = mix_elements

        mix_names = self.options['mix_names']
        if isinstance(mix_names, str): # cast it to tuple 
            mix_names = (mix_names,)    
        self.mix_names = mix_names

        inflow_elements = self.options['inflow_elements']


        self.mixed_elements = inflow_elements.copy()
        if mix_mode == "reactant": # get the elements from the reactant dict in the thermo_data
            for reactant in mix_elements: 
                self.mixed_elements.update(thermo_data.reactants[reactant]) #adds the fuel elements to the mix outflow
        else: # flow mode 
            for flow_elements in mix_elements: 
                self.mixed_elements.update(flow_elements)

        inflow_thermo = Properties(inflow_thermo_data, init_elements=inflow_elements)
        self.inflow_elements = inflow_thermo.elements
        self.num_inflow_elements = len(self.inflow_elements)

        mixed_thermo = Properties(thermo_data, init_elements=self.mixed_elements)
        self.air_fuel_elements = mixed_thermo.elements
        self.air_fuel_wt_mole = mixed_thermo.element_wt


        self.init_fuel_amounts_1kg = {}

        if mix_mode == 'reactant': 
            for reactant in mix_elements: 
                self.init_fuel_amounts_1kg[reactant] = np.zeros(mixed_thermo.num_element)
                ifa_1kg = self.init_fuel_amounts_1kg[reactant]
                for i, e in enumerate(self.air_fuel_elements): 
                    ifa_1kg[i] = thermo_data.reactants[reactant].get(e, 0) * thermo_data.element_wts[e]

                ifa_1kg[:] = ifa_1kg/sum(ifa_1kg) # make it 1 kg of fuel

        else: # flow 
            mix_b0 = {}
            self.mix_wt_mole = {}
            self.mix_out_flow_idx_maps = {}
            for name, elements in zip(mix_names, mix_elements): 
                thermo = Properties(thermo_data, init_elements=elements)
                mix_b0[name] = thermo.b0
                self.mix_wt_mole[name] = thermo.element_wt

                # mapping matrix to convert mix to outflow
                self.mix_out_flow_idx_maps[name] = mix_map = np.zeros((mixed_thermo.num_element, thermo.num_element))
                for i,e in enumerate(thermo.elements): 
                    j = self.air_fuel_elements.index(e)
                    mix_map[j,i] = 1.


        # inputs
        self.add_input('Fl_I:stat:W', val=0.0, desc='weight flow', units='lbm/s')
        self.add_input('Fl_I:tot:h', val=0.0, desc='total enthalpy', units='Btu/lbm')
        self.add_input('Fl_I:tot:b0', val=inflow_thermo.b0, desc='incoming flow composition')
        
        for name in mix_names: 
            self.add_input(f'{name}:h', val=0.0, units='Btu/lbm', desc="reactant enthalpy")

            if mix_mode == 'reactant': 
                self.add_input(f'{name}:ratio', val=0.0, desc='reactant to air mass ratio')
                self.add_output(f'{name}:W', shape=1, units="lbm/s", desc="mix input massflow")

            else: 
                self.add_input(f'{name}:b0', val=mix_b0[name], desc='mix flow composition' )
                self.add_input(f'{name}:W', shape=1, units="lbm/s", desc="mix input massflow")


        # outputs
        self.add_output('mass_avg_h', shape=1, units='Btu/lbm',
                        desc="mass flow rate averaged specific enthalpy")
        self.add_output('Wout', shape=1, units="lbm/s", desc="total massflow out")
        self.add_output('b0_out', val=mixed_thermo.b0)

       
        # create a mapping between the composition indices of the inflow and outflow arrays
        # which is basically a permutation matrix of ones resize the input to the output

        self.in_out_flow_idx_map = np.zeros((mixed_thermo.num_element, inflow_thermo.num_element))
        for i,e in enumerate(self.inflow_elements): 
            j = self.air_fuel_elements.index(e)
            self.in_out_flow_idx_map[j,i] = 1.

        # self.declare_partials('mass_avg_h', 'Fl_I:tot:h')
        # self.declare_partials('Wout', 'Fl_I:stat:W')
        # self.declare_partials('Wfuel', 'Fl_I:stat:W')
        # self.declare_partials('b0_out', 'Fl_I:tot:b0')


        # for name in mix_names: 
        #     ratio_name = f'{name}:ratio'
        #     self.declare_partials('mass_avg_h', ratio_name)
        #     self.declare_partials('Wout', ratio_name)
        #     self.declare_partials('Wfuel', ratio_name)
        #     self.declare_partials('b0_out', ratio_name)

        self.declare_partials('*', '*', method='cs')


    def compute(self, inputs, outputs):
        W = inputs['Fl_I:stat:W']
        Fl_I_tot_b0 = inputs['Fl_I:tot:b0']


        # copy the incoming flow into a correctly sized array for the outflow composition
        init_stuff = self.in_out_flow_idx_map.dot(Fl_I_tot_b0)

        init_stuff *= self.air_fuel_wt_mole # convert to mass units
        init_stuff /= np.sum(init_stuff) # scale to 1 kg
        init_stuff *= W  # scale to full mass flow

        mass_avg_h = inputs['Fl_I:tot:h'] * W
        W_out = W.copy()


        mix_mode = self.options['mix_mode']
        if mix_mode == 'reactant': 
            for name, reactant in zip(self.mix_names, self.mix_elements): 
                ratio = inputs[f'{name}:ratio']
                # compute the amount of fuel-flow rate in terms of the incoming mass-flow rate
                outputs[f'{name}:W'] = W_mix = W*ratio
                init_stuff += self.init_fuel_amounts_1kg[reactant]*W_mix

                mass_avg_h += inputs[f'{name}:h'] * W_mix
                W_out += W_mix

        else: # inflow mixing
            for name in self.mix_names: 
                mix_stuff = inputs[f'{name}:b0']
                mix_stuff *= self.mix_wt_mole[name]
                mix_stuff /= np.sum(mix_stuff) # normalize to 1kg 
                mix_stuff *= inputs[f'{name}:W'] # scale to actual mass flow of that mix stream
                
                init_stuff += self.mix_out_flow_idx_maps[name].dot(mix_stuff)

        init_stuff /= np.sum(init_stuff) # scale back to 1 kg
        outputs['b0_out'] = init_stuff/self.air_fuel_wt_mole
        
        mass_avg_h /= W_out
        outputs['mass_avg_h'] = mass_avg_h
        outputs['Wout'] = W_out

        self.fuel_ht = 0  # makes ht happy

    # def compute_partials(self, inputs, J):
    #     FAR = inputs['mix_ratio']
    #     W = inputs['Fl_I:stat:W']
    #     ht = inputs['Fl_I:tot:h']
    #     Fl_I_tot_b0 = inputs['Fl_I:tot:b0']

    #     # AssertionError: 4.2991138611171866e-05 not less than or equal to 1e-05 : DESIGN.burner.mix_fuel: init_prod_amounts  w.r.t Fl_I:tot:n
    #     J['mass_avg_h', 'mix_ratio'] = -ht/(1+FAR)**2 + self.fuel_ht/(1+FAR)**2  # - self.fuel_ht*FAR/(1+FAR)**2
    #     J['mass_avg_h', 'Fl_I:tot:h'] = 1.0/(1.0 + FAR)

    #     J['Wout', 'Fl_I:stat:W'] = (1.0 + FAR)
    #     J['Wout', 'mix_ratio'] = W

    #     J['Wfuel', 'Fl_I:stat:W'] = FAR
    #     J['Wfuel', 'mix_ratio'] = W

    #     # for i, j in enumerate(self.in_out_flow_idx_map):
    #     #     self.init_air_amounts[j] = Fl_I_tot_b0[i]
    #     self.init_air_amounts = self.in_out_flow_idx_map.dot(Fl_I_tot_b0)

    #     self.init_air_amounts *= self.air_fuel_wt_mole
    #     # iam => init_air_amounts
    #     sum_iam = np.sum(self.init_air_amounts)
    #     d_iam0__db0 = self.in_out_flow_idx_map.dot(np.ones(4))*self.air_fuel_wt_mole

    #     term1 = -self.init_air_amounts/sum_iam**2
    #     d_iam1__db0 = np.einsum('i,j->ij', term1, d_iam0__db0).dot(self.in_out_flow_idx_map)
    #     d_iam0__db0 = np.einsum('i,ij->ij', d_iam0__db0, self.in_out_flow_idx_map)
        
    #     self.init_air_amounts /= sum_iam

    #     self.init_air_amounts *= W  # convert to kg and scale with mass flow
    #     d_iam__db0 = (d_iam0__db0 + d_iam1__db0)*W

    #     init_fuel_amounts = self.init_fuel_amounts_1kg * W * FAR

    #     init_stuff = (self.init_air_amounts + init_fuel_amounts)
    #     sum_is = np.sum(init_stuff)
   
    #     dinit_fuel__dFAR = self.init_fuel_amounts_1kg * W # check
    #     J['b0_out', 'mix_ratio'] = (-(self.init_air_amounts + init_fuel_amounts)/sum_is**2 *np.sum(dinit_fuel__dFAR)
    #                                + dinit_fuel__dFAR/sum_is) / self.air_fuel_wt_mole

    #     for i in range(self.num_inflow_elements): 
    #         # print('bar', d_iam__db0[:,0]/sum_is - init_stuff[0]/sum_is**2 * np.sum(d_iam__db0[:,0]))
    #         J['b0_out', 'Fl_I:tot:b0'][:,i] = (d_iam__db0[:,i]/sum_is - init_stuff[i]/sum_is**2 * np.sum(d_iam__db0[:,i]))/self.air_fuel_wt_mole
