from openmdao.api import Group, IndepVarComp
import numpy as np

from pycycle.cea import species_data
from pycycle.cea.set_total import SetTotal
from pycycle.cea.set_static import SetStatic
from pycycle.constants import AIR_MIX

class FlowStart(Group):

    def initialize(self):

        self.options.declare('thermo_data', default=species_data.janaf,
                              desc='thermodynamic data set', recordable=False)
        self.options.declare('elements', default=AIR_MIX,
                              desc='set of elements present in the flow')
        self.options.declare('initial_b0_provided', default=False)
        self.options.declare('statics', default=True,
                              desc='If True, calculate static properties.')

    def setup(self):
        thermo_data = self.options['thermo_data']
        elements = self.options['elements']
        init_b0 = self.options['initial_b0_provided']

        thermo = species_data.Thermo(thermo_data, init_reacts=elements)
        self.air_prods = thermo.products
        self.num_prod = len(self.air_prods)

        # set initial mass compowtition based on init_prod_amounts of selected thermo_data unless specified by user
        if not init_b0:
            self.add_subsystem('b0_init', IndepVarComp('b0',np.sum(thermo.aij*thermo.init_prod_amounts, axis=1)), promotes_outputs=['b0'])

        set_TP = SetTotal(mode="T", fl_name="Fl_O:tot",
                          thermo_data=thermo_data,
                          init_reacts=elements)

        params = ('T','P', 'b0')

        self.add_subsystem('totals', set_TP, promotes_inputs=params,
                           promotes_outputs=('Fl_O:tot:*',))


        # if self.options['statics']: 
        set_stat_MN = SetStatic(mode="MN", thermo_data=thermo_data,
                                init_reacts=elements, fl_name="Fl_O:stat")

        self.add_subsystem('exit_static', set_stat_MN, promotes_inputs=('MN', 'W', 'b0'),
                           promotes_outputs=('Fl_O:stat:*', ))

        self.connect('totals.h','exit_static.ht')
        self.connect('totals.S','exit_static.S')
        self.connect('Fl_O:tot:P','exit_static.guess:Pt')
        self.connect('totals.gamma', 'exit_static.guess:gamt')


if __name__ == "__main__": 
    from collections import OrderedDict

    from openmdao.api import Problem, IndepVarComp

    # np.seterr(all='raise')

    p = Problem()
    p.model = FlowStart()
    # p.root.add('init_prod', IndepVarComp('init_prod_amounts', p.root.totals.thermo.init_prod_amounts), promotes=['*'])
    p.model.add_subsystem('temp', IndepVarComp('T', 4000., units="degR"), promotes=["*"])
    p.model.add_subsystem('pressure', IndepVarComp('P', 1.0342, units="bar"), promotes=["*"])
    # p.root.add('MN', IndepVarComp('MN_target', .3), promotes=["*"])
    p.model.add_subsystem('W', IndepVarComp('W', 100.0), promotes=['*'])

    p.setup()

    def find_order(group):
        subs = OrderedDict()

        for s in group.subsystems():
            if isinstance(s, Group):
                subs[s.name] = find_order(s)
            else:
                subs[s.name] = {}
        return subs

    # order = find_order(p.root)
    # import json
    # print(json.dumps(order, indent=4))
    # exit()

    # p['exit_static.mach_calc.Ps_guess'] = .97
    import time
    st = time.time()
    p.run_model()
    print("time", time.time() - st)

    print("Temp", p['T'], p['Fl_O:tot:T'])
    print("Pressure", p['P'], p['Fl_O:tot:P'])
    print("h", p['totals.h'], p['Fl_O:tot:h'])
    print("S", p['totals.S'])
    print("actual Ps", p['exit_static.Ps'],p['Fl_O:stat:P'])
    print("Mach", p['Fl_O:stat:MN'])
    print("n tot", p['Fl_O:tot:n'])
    print("n stat", p['Fl_O:stat:n'])
