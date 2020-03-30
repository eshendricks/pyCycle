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

    from thermo_data.co2_co_o2_table import co2_co_o2
    from thermo_data.test_composition2 import test
    from thermo_data.janaf_table import janaf

    prob = om.Problem()
    des_vars = prob.model.add_subsystem('des_vars', om.IndepVarComp(), promotes=["*"])

    # Basic co2_co_o2 with num_nodes=2
    # des_vars.add_output('T', np.array([[1500],[1700]]), units='degK')
    # des_vars.add_output('P', np.array([[1.034210],[1.3]]), units='bar')
    # des_vars.add_output('b0', np.array([[0.0227221,0.04544422],[0.0227221,0.04544422]]), units=None)
    # prob.model.add_subsystem('thermo', Thermo(thermo_data=co2_co_o2, num_nodes=2), promotes=['*'])

    # COH system for num_nodes = 2
    # des_vars.add_output('T', np.array([[1500], [1700]]), units='degK')
    # des_vars.add_output('P', np.array([[1.034210], [1.034210]]), units='bar')
    # des_vars.add_output('b0', np.array([[0.0227221,0.04544422,0.03], [0.0227221,0.04544422,0.00]]), units=None)
    # prob.model.add_subsystem('thermo', Thermo(thermo_data=test, num_nodes=2), promotes=['*'])

    # Janaf table with air_fuel_mix for num_nodes = 2
    des_vars.add_output('T', np.array([[2297.500], [2297.500]]), units='degR')
    des_vars.add_output('P', np.array([[183.047], [183.047]]), units='psi')
    des_vars.add_output('b0', np.array([[0.00031774,0.0012403,0.00246166,0.05298583,0.01423616],
                                        [0.00031774,0.0012403,0.00246166,0.05298583,0.01423616]]), units=None)
    prob.model.add_subsystem('thermo', Thermo(thermo_data=janaf, num_nodes=2), promotes=['*'])



    prob.setup(force_alloc_complex=True)

    # prob['n'] = np.array([[3.17742628e-04,1.00000000e-10,1.00000000e-10,1.86729035e-10,1.24029826e-03,1.00000000e-10,1.35553156e-09,1.01209848e-10,1.23076287e-03,1.08987358e-10,1.00000000e-10,1.00000000e-10,1.27786394e-05,4.01276597e-07,1.00000000e-10,2.64863254e-02,8.86705169e-10,1.27098861e-07,5.25554351e-03],
    #     [3.17742628e-04,1.00000000e-10,1.00000000e-10,1.86729035e-10,1.24029826e-03,1.00000000e-10,1.35553156e-09,1.01209848e-10,1.23076287e-03,1.08987358e-10,1.00000000e-10,1.00000000e-10,1.27786394e-05,4.01276597e-07,1.00000000e-10,2.64863254e-02,8.86705169e-10,1.27098861e-07,5.25554351e-03]])
    # prob['chem_eq.pi'] = np.array([[-22.51944872,-39.58823671,-17.65277801,-11.65681692,-13.34005802],[-22.51944872,-39.58823671,-17.65277801,-11.65681692,-13.34005802]])

    prob.run_model()
    # prob.check_partials(method='cs')

    print(prob['n'])
    print(prob['chem_eq.pi'])
