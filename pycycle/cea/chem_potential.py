import openmdao.api as om

import numpy as np

from pycycle.constants import P_REF

class Chem_Potential_Calcs(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', default=1)
        self.options.declare('prod_list')

    def setup(self):
        nn = self.options['num_nodes']
        prod_list = self.options['prod_list']
        num_prods = len(prod_list)

        self.add_input('P', val=np.ones(nn), units="bar", desc="Pressure")
        self.add_input('n', val=np.ones((nn, num_prods)), desc="Mass fractions of products")
        self.add_input('H0', val=np.ones((nn, num_prods)), units=None) #Fix units
        self.add_input('S0', val=np.ones((nn, num_prods)), units=None) #Fix units

        self.add_output('mu', val=np.ones((nn, num_prods)), units=None) #Fix units

        # Replace the partial derivatives with analytic calculations
        ar = np.arange(nn*num_prods)
        self.declare_partials('mu',['P','n'], method='cs')
        self.declare_partials('mu','H0', val=1, rows=ar, cols=ar)
        self.declare_partials('mu','S0', val=-1, rows=ar, cols=ar)

    def compute(self, inputs, outputs):

        n_moles = np.sum(inputs['n'], axis=0)

        # For cases where concentrations get small or go to zero,
        # set a lower value the concentration for computing mu
        n = np.where(inputs['n']>1e-10, inputs['n'], 1e-10)

        outputs['mu'] = inputs['H0'] - inputs['S0'] + np.log(n) + np.log(inputs['P']/P_REF)[..., np.newaxis] - np.log(n_moles)


    # def compute_partials(self, inputs, J):
    #     nn = self.options['num_nodes']
    #     prod_list = self.options['prod_list']
    #     num_prods = len(prod_list)

    #     print(np.ones((num_prods,nn)))
    #     print(1./inputs['P'])
    #     temp = np.ones((num_prods,nn))/inputs['P']
    #     print(temp.shape)

    #     J['mu', 'P'] = np.ones((num_prods,nn))
    #     # J['mu', 'P'] = np.ones((num_prods,nn))/inputs['P']
    #     J['mu', 'n'] = 1./inputs['n'] - 1./np.sum(n_moles)



class Chem_Potential(om.Group):
    """Runs design and off-design mode compressor map calculations"""

    def initialize(self):
        self.options.declare('thermo_data', default=None)
        self.options.declare('num_nodes', default=1)
        self.options.declare('interp_method', default='slinear')
        self.options.declare('extrap', default=False)


    def setup(self):
        thermo_data = self.options['thermo_data']
        nn = self.options['num_nodes']
        method = self.options['interp_method']
        extrap = self.options['extrap']
        prod_dict = thermo_data.products
        prod_list = list(thermo_data.products.keys())

        for prod in prod_list:
            table = om.MetaModelStructuredComp(method=method, extrapolate=extrap, vec_size=nn)
            table.add_input('T', val=400., units='K', training_data=thermo_data.products[prod]['T_range'])
            table.add_output('H0_T', val=0., units=None, training_data=thermo_data.products[prod]['H0_T']) #Fix units
            table.add_output('S0_T', val=0., units=None, training_data=thermo_data.products[prod]['S0_T']) #Fix units

            self.add_subsystem('HS_'+prod, table, promotes_inputs=['T'])

        mux_comp = self.add_subsystem(name='mux', subsys=om.MuxComp(vec_size=len(prod_list)))
        mux_comp.add_var('H0', shape=(nn,), axis=1, units=None)
        mux_comp.add_var('S0', shape=(nn,), axis=1, units=None)

        for num, prod in enumerate(prod_list):
            self.connect('HS_%s.H0_T'%prod,'mux.H0_%i'%num)
            self.connect('HS_%s.S0_T'%prod,'mux.S0_%i'%num)


        self.add_subsystem('chem_pot_calcs', Chem_Potential_Calcs(num_nodes=nn,prod_list=prod_list),
                            promotes_inputs=['P', 'n'], promotes_outputs=['mu'])

        self.connect('mux.H0','chem_pot_calcs.H0')
        self.connect('mux.S0','chem_pot_calcs.S0')

if __name__ == "__main__":

    from thermo_data.test_composition import test

    prob = om.Problem()
    des_vars = prob.model.add_subsystem('des_vars', om.IndepVarComp(), promotes=["*"])
    des_vars.add_output('T', np.array([518.67, 600]), units='degR')
    des_vars.add_output('P', np.array([14.697, 16.]), units='psi')
    des_vars.add_output('n', np.array([[3,1,5],[4,3,7]]).T, units=None)

    prob.model.add_subsystem('chem_pot', Chem_Potential(thermo_data=test, num_nodes=2, interp_method='lagrange3'), promotes=['*'])

    # des_vars.add_output('T', np.array([518.67, 600, 700]), units='degR')
    # des_vars.add_output('P', np.array([14.697, 16., 20.]), units='psi')
    # des_vars.add_output('n', np.array([[3,1,5],[4,3,7],[6,2,4]]).T, units=None)

    # prob.model.add_subsystem('chem_pot', Chem_Potential(thermo_data=test,num_nodes=3,interp_method='lagrange3'), promotes=['*'])


    prob.setup()
    prob.run_model()

    print('T', prob['T'])
    print('HS_CO.H0_T', prob['HS_CO.H0_T'])
    print('HS_CO.S0_T', prob['HS_CO.S0_T'])
    print('H0', prob['mux.H0'])
    print('S0', prob['mux.S0'])
    print('mu', prob['mu'])

    prob.check_partials()

