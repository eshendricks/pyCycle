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

        self.add_input('b0', val=np.ones((nn, num_elements)),
                        desc='assigned kg-atoms of element i per total kg of reactant '
                             'for the initial prod amounts')

        self.add_input('mu', val=np.ones((nn, num_prods)), units=None) #Fix units

        self.add_output('pi', val=np.ones((nn, num_elements)),
                        desc="modified lagrange multipliers from the Gibbs lagrangian")

        self.add_output('n', val=np.ones((nn, num_prods)),
                        desc="mole fractions of the mixture",
                        lower=0.,
                        res_ref=10000.
                        )

        rows = np.repeat(np.arange(num_elements * nn), num_prods)
        col = np.tile(np.arange(num_prods), num_elements)
        cols = np.tile(col, nn) + np.repeat(num_prods * np.arange(nn), num_elements * num_prods)
        val = np.tile(thermo_data.aij.flatten(), nn)

        self.declare_partials('pi', 'n', rows=rows, cols=cols, val=val)

        row_col = np.arange(nn * num_elements)
        self.declare_partials('pi', 'b0', rows=row_col, cols=row_col, val=-1.0)

        row_col = np.arange(nn * num_prods)
        self.declare_partials('n', 'mu', rows=row_col, cols=row_col, val=1.0)

        rows = np.repeat(np.arange(num_prods * nn), num_elements)
        col = np.tile(np.arange(num_elements), num_prods)
        cols = np.tile(col, nn) + np.repeat(num_elements * np.arange(nn), num_elements * num_prods)
        val = -np.tile(thermo_data.aij.T.flatten(), nn)

        self.declare_partials('n', 'pi', rows=rows, cols=cols, val=val)

    def guess_nonlinear(self, inputs, outputs, residuals):
        thermo_data = self.options['thermo_data']
        aij = thermo_data.aij

        aij_inv = np.linalg.inv(np.matmul(aij,aij.T))
        # Compute initial guesses using left and right inverses of aij
        inv_right = np.matmul(aij.T, aij_inv)
        outputs['n'] = np.einsum('ij,kj->ki', inv_right, inputs['b0'])
        inv_left = np.matmul(aij_inv, aij)
        outputs['pi'] = np.einsum('ij,kj->ki', inv_left, inputs['mu'])

        # print(outputs['n'])
        # print(outputs['pi'])

    def apply_nonlinear(self, inputs, outputs, resids):
        thermo_data = self.options['thermo_data']

        resids['pi'] = np.einsum('ij,kj->ki', thermo_data.aij, outputs['n']) - inputs['b0']
        resids['n'] = inputs['mu'] - np.einsum('ji,kj->ki', thermo_data.aij, outputs['pi'])

