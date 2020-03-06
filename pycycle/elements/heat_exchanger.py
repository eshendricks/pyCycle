""" Class definition for a Heat Exchanger."""

import openmdao.api as om 

class HeatExchanger(om.ExplicitComponent):
    """
    Computes heat transfer based on effectiveness
    """

    def setup(self):
        # inputs
        self.add_input('Tt1', val=518.67, units='degR', desc='Total temperature of flow 1')
        self.add_input('Tt2', val=518.67, units='degR', desc='Total temperature of flow 2')
        self.add_input('Cp1', val=1.0, units='Btu/(lbm*degR)', desc='Total specific heat at constant pressure of flow 1')
        self.add_input('Cp2', val=1.0, units='Btu/(lbm*degR)', desc='Total specific heat at constant pressure of flow 2')
        self.add_input('W1', val= 1.0, units='lbm/s', desc='Mass flow rate of flow 1')
        self.add_input('W2', val= 1.0, units='lbm/s', desc='Mass flow rate of flow 2')
        self.add_input('e', val=1.0, units=None, desc='Heat exchanger effectiveness')

        # outputs
        self.add_output('Q1', val=0.0, units='Btu/s', desc='Heat flow rate into (positive) or out of (negative) flow 1')
        self.add_output('Q2', val=0.0, units='Btu/s', desc='Heat flow rate into (positive) or out of (negative) flow 2')

        # partials
        self.declare_partials('Q1', '*')
        self.declare_partials('Q2', '*')

    def compute(self, inputs, outputs):
        C1 = inputs['W1'] * inputs['Cp1']
        C2 = inputs['W2'] * inputs['Cp2']
        Cmin = min(C1,C2)

        outputs['Q1'] = inputs['e']*Cmin*(inputs['Tt2']-inputs['Tt1'])
        outputs['Q2'] = inputs['e']*Cmin*(inputs['Tt1']-inputs['Tt2'])

        # print(outputs['Q1'], outputs['Q2'])

    def compute_partials(self, inputs, J):
        C1 = inputs['W1'] * inputs['Cp1']
        C2 = inputs['W2'] * inputs['Cp2']
        Cmin = min(C1,C2)

        J['Q1','Tt1'] = -inputs['e']*Cmin
        J['Q1','Tt2'] = inputs['e']*Cmin

        J['Q2','Tt1'] = inputs['e']*Cmin
        J['Q2','Tt2'] = -inputs['e']*Cmin

        J['Q1','e'] = Cmin*(inputs['Tt2']-inputs['Tt1'])
        J['Q2','e'] = Cmin*(inputs['Tt1']-inputs['Tt2'])

        delta_T_1 = inputs['Tt2']-inputs['Tt1']
        delta_T_2 = inputs['Tt1']-inputs['Tt2']

        if C1<=C2:
            J['Q1','W1'] = inputs['e']*inputs['Cp1']*delta_T_1
            J['Q1','Cp1'] = inputs['e']*inputs['W1']*delta_T_1
            J['Q1','W2'] = 0.
            J['Q1','Cp2'] = 0.

            J['Q2','W1'] = inputs['e']*inputs['Cp1']*delta_T_2
            J['Q2','Cp1'] = inputs['e']*inputs['W1']*delta_T_2
            J['Q2','W2'] = 0.
            J['Q2','Cp2'] = 0.
        else:
            J['Q1','W2'] = 0.
            J['Q1','Cp2'] = 0.
            J['Q1','W2'] = inputs['e']*inputs['Cp2']*delta_T_1
            J['Q1','Cp2'] = inputs['e']*inputs['W2']*delta_T_1

            J['Q2','W1'] = 0.
            J['Q2','Cp1'] = 0.
            J['Q2','W2'] = inputs['e']*inputs['Cp1']*delta_T_2
            J['Q2','Cp2'] = inputs['e']*inputs['W1']*delta_T_2



if __name__ == "__main__":

    p = om.Problem()
    p.model = om.Group()

    des_vars = p.model.add_subsystem('des_vars', om.IndepVarComp(), promotes=["*"])
    des_vars.add_output('Tt1', 1190.178, units='degR')
    des_vars.add_output('Tt2', 1437.376, units='degR')
    des_vars.add_output('Cp1', 0.25450022, units='Btu/(lbm*degR)')
    des_vars.add_output('Cp2', 0.26919304, units='Btu/(lbm*degR)')
    des_vars.add_output('W1', 27.337, units='lbm/s')
    des_vars.add_output('W2', 27.798, units='lbm/s')
    des_vars.add_output('e', 0.2, units=None)

    p.model.add_subsystem('comp', HeatExchanger(), promotes=['*'])

    
    p.setup(check=False)
    p.run_model()

    print(p['Q1'])
    print(p['Q2'])

    p.check_partials()

