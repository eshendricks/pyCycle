from __future__ import print_function

import sys
import numpy as np

import openmdao.api as om

import pycycle.api as pyc

class Turboshaft(om.Group):

    def initialize(self):
        self.options.declare('design', default=True,
                              desc='Switch between on-design and off-design calculation.')

    def setup(self):

        thermo_spec = pyc.species_data.janaf
        design = self.options['design']

        self.add_subsystem('fc', pyc.FlightConditions(thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        self.add_subsystem('inlet', pyc.Inlet(design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        self.add_subsystem('duct1', pyc.Duct(design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        self.add_subsystem('lpc', pyc.Compressor(map_data=pyc.LPCMap, design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX), 
                           promotes_inputs=[('Nmech','IP_Nmech')])
        self.add_subsystem('icduct', pyc.Duct(design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        self.add_subsystem('hpc_axi', pyc.Compressor(map_data=pyc.HPCMap, design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX),
                           promotes_inputs=[('Nmech','HP_Nmech')])
        self.add_subsystem('bld25', pyc.BleedOut(design=design, bleed_names=['cool1','cool2']))
        self.add_subsystem('hpc_centri', pyc.Compressor(map_data=pyc.HPCMap, design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX),
                           promotes_inputs=[('Nmech','HP_Nmech')])
        self.add_subsystem('bld3', pyc.BleedOut(design=design, bleed_names=['cool3','cool4']))
        self.add_subsystem('duct6', pyc.Duct(design=design, thermo_data=thermo_spec, elements=pyc.AIR_MIX))
        self.add_subsystem('burner', pyc.Combustor(design=design,thermo_data=thermo_spec,
                                                   inflow_elements=pyc.AIR_MIX,
                                                   air_fuel_elements=pyc.AIR_FUEL_MIX,
                                                   fuel_type='Jet-A(g)'))
        self.add_subsystem('hpt', pyc.Turbine(map_data=pyc.HPTMap, design=design, thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX,
                                              bleed_names=['cool3','cool4']),
                           promotes_inputs=[('Nmech','HP_Nmech')])
        self.add_subsystem('duct43', pyc.Duct(design=design, thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX))
        self.add_subsystem('lpt', pyc.Turbine(map_data=pyc.LPTMap, design=design, thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX,
                                              bleed_names=['cool1','cool2']),
                           promotes_inputs=[('Nmech','IP_Nmech')])
        self.add_subsystem('itduct', pyc.Duct(design=design, thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX))
        self.add_subsystem('pt', pyc.Turbine(map_data=pyc.LPTMap, design=design, thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX),
                           promotes_inputs=[('Nmech','LP_Nmech')])
        self.add_subsystem('duct12', pyc.Duct(design=design, thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX))
        self.add_subsystem('nozzle', pyc.Nozzle(nozzType='CV', lossCoef='Cv', thermo_data=thermo_spec, elements=pyc.AIR_FUEL_MIX))

        self.add_subsystem('lp_shaft', pyc.Shaft(num_ports=1),promotes_inputs=[('Nmech','LP_Nmech')])
        self.add_subsystem('ip_shaft', pyc.Shaft(num_ports=2),promotes_inputs=[('Nmech','IP_Nmech')])
        self.add_subsystem('hp_shaft', pyc.Shaft(num_ports=3),promotes_inputs=[('Nmech','HP_Nmech')])
        self.add_subsystem('perf', pyc.Performance(num_nozzles=1, num_burners=1))

        self.connect('duct1.Fl_O:tot:P', 'perf.Pt2')
        self.connect('hpc_centri.Fl_O:tot:P', 'perf.Pt3')
        self.connect('burner.Wfuel', 'perf.Wfuel_0')
        self.connect('inlet.F_ram', 'perf.ram_drag')
        self.connect('nozzle.Fg', 'perf.Fg_0')
        self.connect('lp_shaft.pwr_in', 'perf.power')

        self.connect('pt.trq', 'lp_shaft.trq_0')
        self.connect('lpc.trq', 'ip_shaft.trq_0')
        self.connect('lpt.trq', 'ip_shaft.trq_1')
        self.connect('hpc_axi.trq', 'hp_shaft.trq_0')
        self.connect('hpc_centri.trq', 'hp_shaft.trq_1')
        self.connect('hpt.trq', 'hp_shaft.trq_2')
        self.connect('fc.Fl_O:stat:P', 'nozzle.Ps_exhaust')

        balance = self.add_subsystem('balance', om.BalanceComp())
        if design:

            balance.add_balance('W', units='lbm/s', eq_units=None)
            self.connect('balance.W', 'inlet.Fl_I:stat:W')
            self.connect('nozzle.PR', 'balance.lhs:W')

            balance.add_balance('FAR', eq_units='degR', lower=1e-4, val=.017)
            self.connect('balance.FAR', 'burner.Fl_I:FAR')
            self.connect('burner.Fl_O:tot:T', 'balance.lhs:FAR')

            balance.add_balance('lpt_PR', val=1.5, lower=1.001, upper=8, eq_units='hp', rhs_val=0.)
            self.connect('balance.lpt_PR', 'lpt.PR')
            self.connect('ip_shaft.pwr_net', 'balance.lhs:lpt_PR')

            balance.add_balance('hpt_PR', val=1.5, lower=1.001, upper=8, eq_units='hp', rhs_val=0.)
            self.connect('balance.hpt_PR', 'hpt.PR')
            self.connect('hp_shaft.pwr_net', 'balance.lhs:hpt_PR')

            balance.add_balance('pt_PR', val=1.5, lower=1.001, upper=8, eq_units='hp', rhs_val=0.)
            self.connect('balance.pt_PR', 'pt.PR')
            self.connect('lp_shaft.pwr_net', 'balance.lhs:pt_PR')

           
        else:
            # Need to check all these balances once power turbine map is updated
            balance.add_balance('FAR', eq_units='lbf', lower=1e-4, val=.017)
            self.connect('balance.FAR', 'burner.Fl_I:FAR')
            self.connect('perf.Fn', 'balance.lhs:FAR')

            balance.add_balance('W', units='lbm/s', eq_units=None)
            self.connect('balance.W', 'inlet.Fl_I:stat:W')
            self.connect('nozzle.Throat:stat:area', 'balance.lhs:W')

            balance.add_balance('IP_Nmech', val=12000.0, units='rpm', lower=1.001, eq_units='hp', rhs_val=0.)
            self.connect('balance.IP_Nmech', 'IP_Nmech')
            self.connect('ip_shaft.pwr_net', 'balance.lhs:IP_Nmech')

            balance.add_balance('HP_Nmech', val=14800.0, units='rpm', lower=1.001, eq_units='hp', rhs_val=0.)
            self.connect('balance.HP_Nmech', 'HP_Nmech')
            self.connect('hp_shaft.pwr_net', 'balance.lhs:HP_Nmech')

            balance.add_balance('LP_Nmech', val=1800.0, units='rpm', lower=1.001, eq_units='hp', rhs_val=0.)
            self.connect('balance.LP_Nmech', 'LP_Nmech')
            self.connect('lp_shaft.pwr_net', 'balance.lhs:LP_Nmech')

        pyc.connect_flow(self, 'fc.Fl_O', 'inlet.Fl_I', connect_w=False)
        pyc.connect_flow(self, 'inlet.Fl_O', 'duct1.Fl_I')
        pyc.connect_flow(self, 'duct1.Fl_O', 'lpc.Fl_I')
        pyc.connect_flow(self, 'lpc.Fl_O', 'icduct.Fl_I')
        pyc.connect_flow(self, 'icduct.Fl_O', 'hpc_axi.Fl_I')
        pyc.connect_flow(self, 'hpc_axi.Fl_O', 'bld25.Fl_I')
        pyc.connect_flow(self, 'bld25.Fl_O', 'hpc_centri.Fl_I')
        pyc.connect_flow(self, 'hpc_centri.Fl_O', 'bld3.Fl_I')
        pyc.connect_flow(self, 'bld3.Fl_O', 'duct6.Fl_I')
        pyc.connect_flow(self, 'duct6.Fl_O', 'burner.Fl_I')
        pyc.connect_flow(self, 'burner.Fl_O', 'hpt.Fl_I')
        pyc.connect_flow(self, 'hpt.Fl_O', 'duct43.Fl_I')
        pyc.connect_flow(self, 'duct43.Fl_O', 'lpt.Fl_I')
        pyc.connect_flow(self, 'lpt.Fl_O', 'itduct.Fl_I')
        pyc.connect_flow(self, 'itduct.Fl_O', 'pt.Fl_I')
        pyc.connect_flow(self, 'pt.Fl_O', 'duct12.Fl_I')
        pyc.connect_flow(self, 'duct12.Fl_O', 'nozzle.Fl_I')
        
        pyc.connect_flow(self, 'bld25.cool1', 'lpt.cool1', connect_stat=False)
        pyc.connect_flow(self, 'bld25.cool2', 'lpt.cool2', connect_stat=False)
        pyc.connect_flow(self, 'bld3.cool3', 'hpt.cool3', connect_stat=False)
        pyc.connect_flow(self, 'bld3.cool4', 'hpt.cool4', connect_stat=False)

        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options['atol'] = 1e-6
        newton.options['rtol'] = 1e-6
        newton.options['iprint'] = 2
        newton.options['maxiter'] = 10
        newton.options['solve_subsystems'] = True
        newton.options['max_sub_solves'] = 100
        newton.options['reraise_child_analysiserror'] = False
        newton.linesearch = om.BoundsEnforceLS()
        # newton.linesearch = ArmijoGoldsteinLS()
        # newton.linesearch.options['c'] = .0001
        newton.linesearch.options['bound_enforcement'] = 'scalar'
        newton.linesearch.options['iprint'] = -1

        self.linear_solver = om.DirectSolver()

def viewer(prob, pt, file=sys.stdout): 
    """
    print a report of all the relevant cycle properties
    """

    print(file=file, flush=True)
    print(file=file, flush=True)
    print(file=file, flush=True)
    print("----------------------------------------------------------------------------", file=file, flush=True)
    print("                              POINT:", pt, file=file, flush=True)
    print("----------------------------------------------------------------------------", file=file, flush=True)
    print("                       PERFORMANCE CHARACTERISTICS", file=file, flush=True)
    print("    Mach      Alt       W      Fn      Fg    Fram     OPR     PSFC ")
    print(" %7.5f  %7.1f %7.3f %7.1f %7.1f %7.1f %7.3f  %7.5f" \
                %(prob[pt+'.fc.Fl_O:stat:MN'], prob[pt+'.fc.alt'],prob[pt+'.inlet.Fl_O:stat:W'], \
                prob[pt+'.perf.Fn'],prob[pt+'.perf.Fg'],prob[pt+'.inlet.F_ram'],prob[pt+'.perf.OPR'],prob[pt+'.perf.PSFC']))    


    fs_names = ['fc.Fl_O','inlet.Fl_O','duct1.Fl_O','lpc.Fl_O',
                'icduct.Fl_O','hpc_axi.Fl_O','bld25.Fl_O',
                'hpc_centri.Fl_O','bld3.Fl_O','duct6.Fl_O',
                'burner.Fl_O','hpt.Fl_O','duct43.Fl_O','lpt.Fl_O',
                'itduct.Fl_O','pt.Fl_O','duct12.Fl_O','nozzle.Fl_O']
    fs_full_names = [f'{pt}.{fs}' for fs in fs_names]
    pyc.print_flow_station(prob, fs_full_names, file=file)

    comp_names = ['lpc','hpc_axi','hpc_centri']
    comp_full_names = [f'{pt}.{c}' for c in comp_names]
    pyc.print_compressor(prob, comp_full_names, file=file)

    pyc.print_burner(prob, [f'{pt}.burner'])

    turb_names = ['hpt','lpt']
    turb_full_names = [f'{pt}.{t}' for t in turb_names]
    pyc.print_turbine(prob, turb_full_names, file=file)

    noz_names = ['nozzle']
    noz_full_names = [f'{pt}.{n}' for n in noz_names]
    pyc.print_nozzle(prob, noz_full_names, file=file)

    shaft_names = ['hp_shaft','lp_shaft']
    shaft_full_names = [f'{pt}.{s}' for s in shaft_names]
    pyc.print_shaft(prob, shaft_full_names, file=file)

    bleed_names = ['bld25', 'bld3']
    bleed_full_names = [f'{pt}.{b}' for b in bleed_names]
    pyc.print_bleed(prob, bleed_full_names, file=file)


if __name__ == "__main__":

    import time
    from openmdao.api import Problem
    from openmdao.utils.units import convert_units as cu

    prob = om.Problem()

    des_vars = prob.model.add_subsystem('des_vars', om.IndepVarComp(), promotes=["*"])

    # FOR DESIGN
    des_vars.add_output('alt', 28000., units='ft'),
    des_vars.add_output('MN', 0.5),
    des_vars.add_output('T4max', 2740.0, units='degR'),
    des_vars.add_output('nozz_PR_des', 1.1)
    des_vars.add_output('inlet:ram_recovery', 1.0),
    des_vars.add_output('inlet:MN_out', 0.4),
    des_vars.add_output('duct1:dPqP', 0.0),
    des_vars.add_output('duct1:MN_out', 0.4),
    des_vars.add_output('lpc:PRdes', 5.000),
    des_vars.add_output('lpc:effDes', 0.8900),
    des_vars.add_output('lpc:MN_out', 0.3),
    des_vars.add_output('icduct:dPqP', 0.002),
    des_vars.add_output('icduct:MN_out', 0.3),
    des_vars.add_output('hpc_axi:PRdes', 3.0),
    des_vars.add_output('hpc_axi:effDes', 0.8900),
    des_vars.add_output('hpc_axi:MN_out', 0.25),
    des_vars.add_output('bld25:cool1:frac_W', 0.024),
    des_vars.add_output('bld25:cool2:frac_W', 0.0146),
    des_vars.add_output('bld25:MN_out', 0.3000),
    des_vars.add_output('hpc_centri:PRdes', 2.7),
    des_vars.add_output('hpc_centri:effDes', 0.8800),
    des_vars.add_output('hpc_centri:MN_out', 0.20),
    des_vars.add_output('bld3:cool3:frac_W', 0.1705),
    des_vars.add_output('bld3:cool4:frac_W', 0.1209),
    des_vars.add_output('bld3:MN_out', 0.2000),
    des_vars.add_output('duct6:dPqP', 0.00),
    des_vars.add_output('duct6:MN_out', 0.2000),
    des_vars.add_output('burner:dPqP', 0.050),
    des_vars.add_output('burner:MN_out', 0.15),
    des_vars.add_output('hpt:effDes', 0.89),
    des_vars.add_output('hpt:cool3:frac_P', 1.0),
    des_vars.add_output('hpt:cool4:frac_P', 0.0),
    des_vars.add_output('hpt:MN_out', 0.30),
    des_vars.add_output('duct43:dPqP', 0.0051),
    des_vars.add_output('duct43:MN_out', 0.30),
    des_vars.add_output('lpt:effDes', 0.9),
    des_vars.add_output('lpt:cool1:frac_P', 1.0),
    des_vars.add_output('lpt:cool2:frac_P', 0.0),
    des_vars.add_output('lpt:MN_out', 0.4),
    des_vars.add_output('itduct:dPqP', 0.00),
    des_vars.add_output('itduct:MN_out', 0.4),
    des_vars.add_output('pt:effDes', 0.85),
    des_vars.add_output('pt:MN_out', 0.4),
    des_vars.add_output('duct12:dPqP', 0.00),
    des_vars.add_output('duct12:MN_out', 0.4),
    des_vars.add_output('nozzle:Cv', 0.99),
    des_vars.add_output('lp_shaft:Nmech', 12750., units='rpm'),
    des_vars.add_output('lp_shaft:HPX', 1800.0, units='hp'),
    des_vars.add_output('ip_shaft:Nmech', 12000., units='rpm'),
    des_vars.add_output('hp_shaft:Nmech', 14800., units='rpm'),

    # des_vars.add_output('FAR', 0.02261)
    # des_vars.add_output('W', 10.76, units='lbm/s')

    # OFF DESIGN 1
    des_vars.add_output('OD1_MN', 0.5),
    des_vars.add_output('OD1_alt', 28000.0, units='ft'),
    # des_vars.add_output('OD1_Fn_target', 5497.0, units='lbf'),
    des_vars.add_output('OD1_P_target', 7500.0, units='hp')

    # DESIGN CASE
    prob.model.add_subsystem('DESIGN', Turboshaft())

    prob.model.connect('alt', 'DESIGN.fc.alt')
    prob.model.connect('MN', 'DESIGN.fc.MN')
    prob.model.connect('T4max', 'DESIGN.balance.rhs:FAR')
    prob.model.connect('nozz_PR_des', 'DESIGN.balance.rhs:W')

    prob.model.connect('inlet:ram_recovery', 'DESIGN.inlet.ram_recovery')
    prob.model.connect('inlet:MN_out', 'DESIGN.inlet.MN')
    prob.model.connect('duct1:dPqP', 'DESIGN.duct1.dPqP')
    prob.model.connect('duct1:MN_out', 'DESIGN.duct1.MN')
    prob.model.connect('lpc:PRdes', 'DESIGN.lpc.PR')
    prob.model.connect('lpc:effDes', 'DESIGN.lpc.eff')
    prob.model.connect('lpc:MN_out', 'DESIGN.lpc.MN')
    prob.model.connect('icduct:dPqP', 'DESIGN.icduct.dPqP')
    prob.model.connect('icduct:MN_out', 'DESIGN.icduct.MN')
    prob.model.connect('hpc_axi:PRdes', 'DESIGN.hpc_axi.PR')
    prob.model.connect('hpc_axi:effDes', 'DESIGN.hpc_axi.eff')
    prob.model.connect('hpc_axi:MN_out', 'DESIGN.hpc_axi.MN')
    prob.model.connect('bld25:cool1:frac_W', 'DESIGN.bld25.cool1:frac_W')
    prob.model.connect('bld25:cool2:frac_W', 'DESIGN.bld25.cool2:frac_W')
    prob.model.connect('bld25:MN_out', 'DESIGN.bld25.MN')
    prob.model.connect('hpc_centri:PRdes', 'DESIGN.hpc_centri.PR')
    prob.model.connect('hpc_centri:effDes', 'DESIGN.hpc_centri.eff')
    prob.model.connect('hpc_centri:MN_out', 'DESIGN.hpc_centri.MN')
    prob.model.connect('bld3:cool3:frac_W', 'DESIGN.bld3.cool3:frac_W')
    prob.model.connect('bld3:cool4:frac_W', 'DESIGN.bld3.cool4:frac_W')
    prob.model.connect('bld3:MN_out', 'DESIGN.bld3.MN')
    prob.model.connect('duct6:dPqP', 'DESIGN.duct6.dPqP')
    prob.model.connect('duct6:MN_out', 'DESIGN.duct6.MN')
    prob.model.connect('burner:dPqP', 'DESIGN.burner.dPqP')
    prob.model.connect('burner:MN_out', 'DESIGN.burner.MN')
    prob.model.connect('hpt:effDes', 'DESIGN.hpt.eff')
    prob.model.connect('hpt:cool3:frac_P', 'DESIGN.hpt.cool3:frac_P')
    prob.model.connect('hpt:cool4:frac_P', 'DESIGN.hpt.cool4:frac_P')
    prob.model.connect('hpt:MN_out', 'DESIGN.hpt.MN')
    prob.model.connect('duct43:dPqP', 'DESIGN.duct43.dPqP')
    prob.model.connect('duct43:MN_out', 'DESIGN.duct43.MN')
    prob.model.connect('lpt:effDes', 'DESIGN.lpt.eff')
    prob.model.connect('lpt:cool1:frac_P', 'DESIGN.lpt.cool1:frac_P')
    prob.model.connect('lpt:cool2:frac_P', 'DESIGN.lpt.cool2:frac_P')
    prob.model.connect('lpt:MN_out', 'DESIGN.lpt.MN')
    prob.model.connect('itduct:dPqP', 'DESIGN.itduct.dPqP')
    prob.model.connect('itduct:MN_out', 'DESIGN.itduct.MN')
    prob.model.connect('pt:effDes', 'DESIGN.pt.eff')
    prob.model.connect('pt:MN_out', 'DESIGN.pt.MN')
    prob.model.connect('duct12:dPqP', 'DESIGN.duct12.dPqP')
    prob.model.connect('duct12:MN_out', 'DESIGN.duct12.MN')
    prob.model.connect('nozzle:Cv', 'DESIGN.nozzle.Cv')
    prob.model.connect('lp_shaft:Nmech', 'DESIGN.LP_Nmech')
    prob.model.connect('lp_shaft:HPX', 'DESIGN.lp_shaft.HPX')
    prob.model.connect('ip_shaft:Nmech', 'DESIGN.IP_Nmech')
    prob.model.connect('hp_shaft:Nmech', 'DESIGN.HP_Nmech')

    # prob.model.connect('FAR', 'DESIGN.burner.Fl_I:FAR')
    # prob.model.connect('W', 'DESIGN.inlet.Fl_I:stat:W')
    
    # OFF DESIGN CASES
    pts = [] # 'OD1','OD2','OD3','OD4','OD5','OD6','OD7','OD8']

    # for pt in pts:
    #     ODpt = prob.model.add_subsystem(pt, Turboshaft(design=False))

    #     prob.model.connect(pt+'_alt', pt+'.fc.alt')
    #     prob.model.connect(pt+'_MN', pt+'.fc.MN')
    #     # prob.model.connect(pt+'_Fn_target', pt+'.thrust_balance.rhs')

    #     prob.model.connect('inlet:ram_recovery', pt+'.inlet.ram_recovery')
    #     prob.model.connect('duct4:dPqP', pt+'.duct4.dPqP')
    #     prob.model.connect('duct6:dPqP', pt+'.duct6.dPqP')
    #     prob.model.connect('burner:dPqP', pt+'.burner.dPqP')
    #     prob.model.connect('duct11:dPqP', pt+'.duct11.dPqP')
    #     prob.model.connect('duct13:dPqP', pt+'.duct13.dPqP')
    #     prob.model.connect('nozzle:Cv', pt+'.nozzle.Cv')
    #     prob.model.connect('duct15:dPqP', pt+'.duct15.dPqP')
    #     prob.model.connect('hp_shaft:HPX', pt+'.hp_shaft.HPX')

    #     prob.model.connect('hpc:cool1:frac_W', pt+'.hpc.cool1:frac_W')
    #     prob.model.connect('hpc:cool1:frac_P', pt+'.hpc.cool1:frac_P')
    #     prob.model.connect('hpc:cool1:frac_work', pt+'.hpc.cool1:frac_work')
    #     prob.model.connect('hpc:cool2:frac_W', pt+'.hpc.cool2:frac_W')
    #     prob.model.connect('hpc:cool2:frac_P', pt+'.hpc.cool2:frac_P')
    #     prob.model.connect('hpc:cool2:frac_work', pt+'.hpc.cool2:frac_work')
    #     prob.model.connect('bld3:cool3:frac_W', pt+'.bld3.cool3:frac_W')
    #     prob.model.connect('bld3:cool4:frac_W', pt+'.bld3.cool4:frac_W')
    #     prob.model.connect('hpc:cust:frac_W', pt+'.hpc.cust:frac_W')
    #     prob.model.connect('hpc:cust:frac_P', pt+'.hpc.cust:frac_P')
    #     prob.model.connect('hpc:cust:frac_work', pt+'.hpc.cust:frac_work')
    #     prob.model.connect('hpt:cool3:frac_P', pt+'.hpt.cool3:frac_P')
    #     prob.model.connect('hpt:cool4:frac_P', pt+'.hpt.cool4:frac_P')
    #     prob.model.connect('lpt:cool1:frac_P', pt+'.lpt.cool1:frac_P')
    #     prob.model.connect('lpt:cool2:frac_P', pt+'.lpt.cool2:frac_P')
    #     prob.model.connect('bypBld:frac_W', pt+'.byp_bld.bypBld:frac_W')

    #     prob.model.connect('DESIGN.fan.s_PRdes', pt+'.fan.s_PRdes')
    #     prob.model.connect('DESIGN.fan.s_WcDes', pt+'.fan.s_WcDes')
    #     prob.model.connect('DESIGN.fan.s_effDes', pt+'.fan.s_effDes')
    #     prob.model.connect('DESIGN.fan.s_NcDes', pt+'.fan.s_NcDes')
    #     prob.model.connect('DESIGN.lpc.s_PRdes', pt+'.lpc.s_PRdes')
    #     prob.model.connect('DESIGN.lpc.s_WcDes', pt+'.lpc.s_WcDes')
    #     prob.model.connect('DESIGN.lpc.s_effDes', pt+'.lpc.s_effDes')
    #     prob.model.connect('DESIGN.lpc.s_NcDes', pt+'.lpc.s_NcDes')
    #     prob.model.connect('DESIGN.hpc.s_PRdes', pt+'.hpc.s_PRdes')
    #     prob.model.connect('DESIGN.hpc.s_WcDes', pt+'.hpc.s_WcDes')
    #     prob.model.connect('DESIGN.hpc.s_effDes', pt+'.hpc.s_effDes')
    #     prob.model.connect('DESIGN.hpc.s_NcDes', pt+'.hpc.s_NcDes')
    #     prob.model.connect('DESIGN.hpt.s_PRdes', pt+'.hpt.s_PRdes')
    #     prob.model.connect('DESIGN.hpt.s_WpDes', pt+'.hpt.s_WpDes')
    #     prob.model.connect('DESIGN.hpt.s_effDes', pt+'.hpt.s_effDes')
    #     prob.model.connect('DESIGN.hpt.s_NpDes', pt+'.hpt.s_NpDes')
    #     prob.model.connect('DESIGN.lpt.s_PRdes', pt+'.lpt.s_PRdes')
    #     prob.model.connect('DESIGN.lpt.s_WpDes', pt+'.lpt.s_WpDes')
    #     prob.model.connect('DESIGN.lpt.s_effDes', pt+'.lpt.s_effDes')
    #     prob.model.connect('DESIGN.lpt.s_NpDes', pt+'.lpt.s_NpDes')

    #     prob.model.connect('DESIGN.nozzle.Throat:stat:area',pt+'.core_flow_balance.rhs')

    #     prob.model.connect('DESIGN.inlet.Fl_O:stat:area', pt+'.inlet.area')
    #     prob.model.connect('DESIGN.fan.Fl_O:stat:area', pt+'.fan.area')
    #     prob.model.connect('DESIGN.splitter.Fl_O1:stat:area', pt+'.splitter.area1')
    #     prob.model.connect('DESIGN.splitter.Fl_O2:stat:area', pt+'.splitter.area2')
    #     prob.model.connect('DESIGN.duct4.Fl_O:stat:area', pt+'.duct4.area')
    #     prob.model.connect('DESIGN.lpc.Fl_O:stat:area', pt+'.lpc.area')
    #     prob.model.connect('DESIGN.duct6.Fl_O:stat:area', pt+'.duct6.area')
    #     prob.model.connect('DESIGN.hpc.Fl_O:stat:area', pt+'.hpc.area')
    #     prob.model.connect('DESIGN.bld3.Fl_O:stat:area', pt+'.bld3.area')
    #     prob.model.connect('DESIGN.burner.Fl_O:stat:area', pt+'.burner.area')
    #     prob.model.connect('DESIGN.hpt.Fl_O:stat:area', pt+'.hpt.area')
    #     prob.model.connect('DESIGN.duct11.Fl_O:stat:area', pt+'.duct11.area')
    #     prob.model.connect('DESIGN.lpt.Fl_O:stat:area', pt+'.lpt.area')
    #     prob.model.connect('DESIGN.duct13.Fl_O:stat:area', pt+'.duct13.area')
    #     prob.model.connect('DESIGN.byp_bld.Fl_O:stat:area', pt+'.byp_bld.area')
    #     prob.model.connect('DESIGN.duct15.Fl_O:stat:area', pt+'.duct15.area')

    prob.setup(check=False)

    # initial guesses
    prob['DESIGN.balance.FAR'] = 0.02261
    prob['DESIGN.balance.W'] = 10.76
    prob['DESIGN.balance.pt_PR'] = 4.939
    prob['DESIGN.balance.lpt_PR'] = 1.979
    prob['DESIGN.balance.hpt_PR'] = 4.236
    prob['DESIGN.fc.balance.Pt'] = 5.666
    prob['DESIGN.fc.balance.Tt'] = 440.0

    # for pt in pts:
    #     prob[pt+'.thrust_balance.indep'] = 0.02467
    #     prob[pt+'.core_flow_balance.indep'] = 320.931
    #     prob[pt+'.byp_flow_balance.indep'] = 5.105
    #     prob[pt+'.lp_shaft_balance.indep'] = 4666.1
    #     prob[pt+'.hp_shaft_balance.indep'] = 14705.7
    #     prob[pt+'.fc.balance.Pt'] = 5.2
    #     prob[pt+'.fc.balance.Tt'] = 440.0
    #     prob[pt+'.hpt.PR'] = 3.6200
    #     prob[pt+'.lpt.PR'] = 4.3645
    #     prob[pt+'.lpc.map.RlineMap'] = 2.0
    #     prob[pt+'.hpc.map.RlineMap'] = 2.0

    st = time.time()


    prob.set_solver_print(level=-1)
    prob.set_solver_print(level=2, depth=1)
    prob.run_model()

    for pt in ['DESIGN']+pts:
        viewer(prob, pt)

    print()
    print("time", time.time() - st)

