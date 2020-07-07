import numpy as np
from collections import OrderedDict

from pycycle.cea.species_data import Thermo
from pycycle.cea.thermo_data import co2_co_o2
from pycycle.cea.thermo_data import janaf


# thermo = Thermo(thermo_data_module=co2_co_o2, init_reacts=co2_co_o2.init_prod_amounts)
thermo = Thermo(thermo_data_module=janaf, init_reacts=janaf.init_prod_amounts)


# print(co2_co_o2.products['CO']['ranges'])
# quit()

T_range = np.linspace(200,6000,117)
# print(len(thermo.products))
# print(len(T_range))

H0 = np.empty((len(T_range),len(thermo.products)))
S0 = np.empty((len(T_range),len(thermo.products)))
Cp0 = np.empty((len(T_range),len(thermo.products)))
# print(S0)

for i, temp in enumerate(T_range):
    # print(i,temp)
    H0[i,:] = thermo.H0([temp])
    S0[i,:] = thermo.S0([temp])
    Cp0[i,:] = thermo.Cp0([temp])

# print(H0)
# print(S0)
# print(Cp0)

products = OrderedDict()
for i, species in enumerate(thermo.products):
    products[species] = {}
    products[species]['T_range'] = T_range
    products[species]['H0_T'] = H0[:,i].T
    products[species]['S0_T'] = S0[:,i].T
    products[species]['Cp0_T'] = Cp0[:,i].T

# print(products)
print(thermo.aij)

# f = open( 'co2_co_o2_table.py', 'w' )
f = open( 'janaf_table.py', 'w' )
f.write('import numpy as np\n')
f.write('from collections import OrderedDict\n\n')
f.write('class Composition(object):\n')
f.write('\t# Note: Temperature range for each species limited to 200-6000K in this file,\n')
f.write('\t# as this range covers temperatueres likely to be experienced in a gas turbine.\n')
f.write('\t# Additional data could be generated for some species up to 20000K based on CEA,\n')
f.write('\t# but would be best in a separate file for that specific application.\n')
f.write('\telements = '+ repr(thermo.elements)+'\n\n')
f.write('\tproducts = ' + repr(products) + '\n\n')
f.write('\taij = ' + repr(thermo.aij) + '\n\n')
f.write('comp = Composition()\n')
f.close()

