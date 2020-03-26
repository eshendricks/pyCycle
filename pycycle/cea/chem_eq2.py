import numpy as np

import openmdao.api as om
 
from pycycle.constants import P_REF, R_UNIVERSAL_ENG, MIN_VALID_CONCENTRATION

class ChemEq2(om.ImplicitComponent):
    """ Find the equilibirum composition for a given gaseous mixture """

    def initialize(self):
        self.options.declare('thermo_data', default=None)
        self.options.declare('num_nodes', default=1)

    def setup(self):

        thermo_data = self.options['thermo_data']
        nn = self.options['num_nodes']

        num_prods = len(list(thermo_data.products.keys()))
        num_elements = len(list(thermo_data.elements.keys()))

        # self.add_input('P', val=1.0*np.ones(nn), units="bar", desc="Pressure")
        # self.add_input('T', val=400.*np.ones(nn), units="degK", desc="Temperature")

        self.add_input('b0', val=np.ones((num_elements,nn)),
                        desc='assigned kg-atoms of element i per total kg of reactant '
                             'for the initial prod amounts')
        
        self.add_input('mu', val=np.ones((num_prods,nn)), units=None) #Fix units

        self.add_output('pi', val=np.ones((num_elements,nn)),
                        desc="modified lagrange multipliers from the Gibbs lagrangian")

        self.add_output('n', val=np.ones((num_prods,nn)),
                        desc="mole fractions of the mixture",
                        lower=1e-10,
                        res_ref=10000.
                        )

        # self.declare_partials('*','*', method='cs')
        self.declare_partials('pi','n', method='cs')
        self.declare_partials('pi','b0', method='cs')
        self.declare_partials('n','mu', method='cs')
        self.declare_partials('n','pi', method='cs')

        ar_elems = np.arange(num_elements)
        ar_prods = np.arange(num_prods)
        # self.declare_partials('pi','n', val=thermo_data.aij)
        # self.declare_partials('pi','b0', rows=ar_elems, cols=ar_elems, val=-1.)
        # # self.declare_partials('pi','pi')
        # self.declare_partials('n','mu', rows=ar_prods, cols=ar_prods, val=1.)
        # self.declare_partials('n','pi', val=-thermo_data.aij.T)

    def apply_nonlinear(self, inputs, outputs, resids):
        thermo_data = self.options['thermo_data']

        # print('aij', thermo_data.aij)
        # print('n_c', outputs['n'])
        # print('b0', inputs['b0'])
        # print('pi_err', np.matmul(thermo_data.aij, outputs['n']) - inputs['b0'])
        resids['pi'] = np.matmul(thermo_data.aij, outputs['n']) - inputs['b0']
        # print()


        # print('mu',inputs['mu'])
        # print('pi', outputs['pi'])
        # print(thermo_data.aij.T)
        # print('n_err', inputs['mu'] - np.matmul(thermo_data.aij.T,outputs['pi']))
        # print('n_err', np.multiply((inputs['mu'] - np.matmul(thermo_data.aij.T,outputs['pi'])).T,np.array([0.0093051,1.,0.00465265])))
        resids['n'] = inputs['mu'] - np.matmul(thermo_data.aij.T,outputs['pi'])

        # print('----------------------')
        # print('pi',outputs['pi'],resids['pi'])
        # print('n',outputs['n'],resids['n'])


    # def linearize(self,inputs,outputs,J):
