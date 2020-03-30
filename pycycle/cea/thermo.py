import numpy as np

import openmdao.api as om

from pycycle.cea.chem_potential import Chem_Potential
from pycycle.cea.chem_eq2 import ChemEq2

class Thermo(om.Group):

    def initialize(self):
        self.options.declare('thermo_data', default=None)
        self.options.declare('num_nodes', default=1)

    def setup(self):
        thermo_data = self.options['thermo_data']
        nn = self.options['num_nodes']

        self.add_subsystem('chem_pot', Chem_Potential(thermo_data=thermo_data, num_nodes=nn, interp_method='lagrange3'))

        self.add_subsystem('chem_eq', ChemEq2(thermo_data=thermo_data, num_nodes=nn))

        self.connect('chem_pot.mu','chem_eq.mu')

        self.promotes('chem_pot', inputs=['T','P','n'])
        self.promotes('chem_eq', inputs=['b0'], outputs=['n'])

        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options['maxiter'] = 100
        newton.options['iprint'] = 2
        newton.options['atol'] = 1e-8
        newton.options['rtol'] = 1e-8
        newton.options['solve_subsystems'] = True
        newton.options['reraise_child_analysiserror'] = False

        self.options['assembled_jac_type'] = 'dense'
        self.linear_solver = om.DirectSolver(assemble_jac=True)

        ln_bt = newton.linesearch = om.ArmijoGoldsteinLS()
        ln_bt.options['maxiter'] = 2
        ln_bt.options['bound_enforcement'] = 'scalar'
        ln_bt.options['iprint'] = -1


if __name__ == "__main__":

    from thermo_data.test_composition2 import test

    prob = om.Problem()
    des_vars = prob.model.add_subsystem('des_vars', om.IndepVarComp(), promotes=["*"])
    # des_vars.add_output('T', np.array([1500,1700]), units='degK')
    # des_vars.add_output('P', np.array([1.034210,1.3]), units='bar')
    # des_vars.add_output('b0', np.array([[0.0227221,0.04544422],[0.0227221,0.04544422]]).T, units=None)

    #des_vars.add_output('T', np.array([1500]), units='degK')
    #des_vars.add_output('P', np.array([1.034210]), units='bar')
    #des_vars.add_output('b0', np.array([0.0227221,0.04544422,0.00]).T, units=None)

    # for num_nodes = 2
    des_vars.add_output('T', np.array([[1500], [1500]]), units='degK')
    des_vars.add_output('P', np.array([[1.034210], [1.034210]]), units='bar')
    des_vars.add_output('b0', np.array([[0.0227221,0.04544422,0.00], [0.0227221,0.04544422,0.00]]), units=None)

    prob.model.add_subsystem('thermo', Thermo(thermo_data=test, num_nodes=2), promotes=['*'])

    prob.setup()

    # prob['n'] = np.array([[8.15344263e-06, 2.27139552e-02, 4.07672148e-06]]).T
    # prob['chem_eq.pi'] = np.array([[-25.34234058, -18.19254736]]).T
    prob.run_model()
    prob.check_partials()

    print(prob['n'])
    print(prob['chem_eq.pi'])