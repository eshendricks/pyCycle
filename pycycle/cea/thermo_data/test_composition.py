"""CO2, CO, O2 reactants, used for testing and development, a subset of the janaf thermo fit set"""

import numpy as np
from collections import OrderedDict

class TestComposition(object):

    temp_list_small = np.linspace(200,6000,59) #values every 100 K
    temp_list_large = np.linspace(200,20000,199) #values every 100 K

    products = OrderedDict([
        ('CO',{
          'T_range': temp_list_large,
          'H0_T': np.array([-6.81901E+01,-4.42926E+01,-3.23408E+01,-2.51619E+01,-2.03648E+01,-1.69263E+01,-1.43364E+01,
                  -1.23126E+01,-1.06859E+01,-9.34869E+00,-8.22941E+00,-7.27836E+00,-6.45998E+00,-5.74815E+00,
                  -5.12319E+00,-4.57002E+00,-4.07688E+00,-3.63445E+00,-3.23525E+00,-2.87321E+00,-2.54335E+00,
                  -2.24154E+00,-1.96433E+00,-1.70881E+00,-1.47253E+00,-1.25337E+00,-1.04954E+00,-8.59461E-01,
                  -6.81785E-01,-5.15327E-01,-3.59048E-01,-2.12037E-01,-7.34853E-02,5.73229E-02,1.81025E-01,
                  2.98191E-01,4.09330E-01,5.14901E-01,6.15317E-01,7.10952E-01,8.02144E-01,8.89200E-01,
                  9.72401E-01,1.05200E+00,1.12823E+00,1.20131E+00,1.27143E+00,1.33878E+00,1.40352E+00,
                  1.46580E+00,1.52578E+00,1.58357E+00,1.63931E+00,1.69311E+00,1.74507E+00,1.79530E+00,
                  1.84389E+00,1.89093E+00,1.93649E+00,1.98071E+00,2.02368E+00,2.06545E+00,2.10605E+00,
                  2.14555E+00,2.18397E+00,2.22138E+00,2.25782E+00,2.29333E+00,2.32797E+00,2.36177E+00,
                  2.39479E+00,2.42707E+00,2.45866E+00,2.48960E+00,2.51993E+00,2.54970E+00,2.57894E+00,
                  2.60770E+00,2.63601E+00,2.66392E+00,2.69146E+00,2.71866E+00,2.74555E+00,2.77218E+00,
                  2.79856E+00,2.82474E+00,2.85072E+00,2.87656E+00,2.90225E+00,2.92784E+00,2.95334E+00,
                  2.97878E+00,3.00417E+00,3.02953E+00,3.05487E+00,3.08022E+00,3.10559E+00,3.13099E+00,
                  3.15643E+00,3.18192E+00,3.20748E+00,3.23310E+00,3.25880E+00,3.28459E+00,3.31046E+00,
                  3.33643E+00,3.36250E+00,3.38866E+00,3.41493E+00,3.44129E+00,3.46776E+00,3.49433E+00,
                  3.52099E+00,3.54775E+00,3.57461E+00,3.60155E+00,3.62858E+00,3.65569E+00,3.68287E+00,
                  3.71012E+00,3.73743E+00,3.76480E+00,3.79221E+00,3.81967E+00,3.84715E+00,3.87466E+00,
                  3.90219E+00,3.92972E+00,3.95724E+00,3.98475E+00,4.01224E+00,4.03970E+00,4.06711E+00,
                  4.09446E+00,4.12176E+00,4.14897E+00,4.17610E+00,4.20314E+00,4.23007E+00,4.25687E+00,
                  4.28356E+00,4.31010E+00,4.33649E+00,4.36272E+00,4.38878E+00,4.41466E+00,4.44034E+00,
                  4.46583E+00,4.49110E+00,4.51615E+00,4.54097E+00,4.56555E+00,4.58987E+00,4.61394E+00,
                  4.63775E+00,4.66127E+00,4.68451E+00,4.70746E+00,4.73011E+00,4.75245E+00,4.77448E+00,
                  4.79619E+00,4.81757E+00,4.83862E+00,4.85933E+00,4.87970E+00,4.89972E+00,4.91938E+00,
                  4.93870E+00,4.95765E+00,4.97624E+00,4.99447E+00,5.01233E+00,5.02982E+00,5.04694E+00,
                  5.06369E+00,5.08008E+00,5.09609E+00,5.11173E+00,5.12701E+00,5.14193E+00,5.15648E+00,
                  5.17067E+00,5.18451E+00,5.19799E+00,5.21113E+00,5.22393E+00,5.23639E+00,5.24853E+00,
                  5.26035E+00,5.27185E+00,5.28305E+00,5.29395E+00,5.30457E+00,5.31491E+00,5.32499E+00,
                  5.33481E+00,5.34440E+00,5.35376E+00]),
          'S0_T': np.array([2.23745E+01,2.37946E+01,2.48055E+01,2.55982E+01,2.62581E+01,2.68290E+01,2.73354E+01,
                  2.77920E+01,2.82086E+01,2.85920E+01,2.89472E+01,2.92781E+01,2.95878E+01,2.98788E+01,
                  3.01531E+01,3.04126E+01,3.06588E+01,3.08928E+01,3.11159E+01,3.13290E+01,3.15330E+01,
                  3.17285E+01,3.19162E+01,3.20968E+01,3.22707E+01,3.24385E+01,3.26004E+01,3.27570E+01,
                  3.29086E+01,3.30554E+01,3.31979E+01,3.33361E+01,3.34704E+01,3.36010E+01,3.37280E+01,
                  3.38518E+01,3.39724E+01,3.40899E+01,3.42047E+01,3.43167E+01,3.44261E+01,3.45331E+01,
                  3.46377E+01,3.47400E+01,3.48402E+01,3.49384E+01,3.50345E+01,3.51288E+01,3.52212E+01,
                  3.53119E+01,3.54009E+01,3.54883E+01,3.55742E+01,3.56586E+01,3.57415E+01,3.58231E+01,
                  3.59033E+01,3.59823E+01,3.60600E+01,3.61366E+01,3.62121E+01,3.62866E+01,3.63601E+01,
                  3.64325E+01,3.65040E+01,3.65745E+01,3.66441E+01,3.67129E+01,3.67807E+01,3.68478E+01,
                  3.69141E+01,3.69796E+01,3.70445E+01,3.71086E+01,3.71721E+01,3.72350E+01,3.72974E+01,
                  3.73591E+01,3.74204E+01,3.74813E+01,3.75417E+01,3.76016E+01,3.76613E+01,3.77205E+01,
                  3.77795E+01,3.78382E+01,3.78966E+01,3.79548E+01,3.80128E+01,3.80706E+01,3.81282E+01,
                  3.81857E+01,3.82431E+01,3.83004E+01,3.83576E+01,3.84147E+01,3.84718E+01,3.85289E+01,
                  3.85859E+01,3.86429E+01,3.87000E+01,3.87570E+01,3.88141E+01,3.88712E+01,3.89283E+01,
                  3.89855E+01,3.90427E+01,3.91000E+01,3.91573E+01,3.92147E+01,3.92721E+01,3.93296E+01,
                  3.93872E+01,3.94448E+01,3.95025E+01,3.95603E+01,3.96181E+01,3.96759E+01,3.97338E+01,
                  3.97917E+01,3.98497E+01,3.99077E+01,3.99657E+01,4.00237E+01,4.00817E+01,4.01398E+01,
                  4.01978E+01,4.02558E+01,4.03138E+01,4.03717E+01,4.04296E+01,4.04874E+01,4.05452E+01,
                  4.06029E+01,4.06605E+01,4.07180E+01,4.07754E+01,4.08327E+01,4.08899E+01,4.09469E+01,
                  4.10037E+01,4.10604E+01,4.11170E+01,4.11733E+01,4.12294E+01,4.12853E+01,4.13411E+01,
                  4.13965E+01,4.14518E+01,4.15067E+01,4.15614E+01,4.16159E+01,4.16700E+01,4.17239E+01,
                  4.17774E+01,4.18307E+01,4.18836E+01,4.19362E+01,4.19884E+01,4.20403E+01,4.20918E+01,
                  4.21429E+01,4.21937E+01,4.22441E+01,4.22941E+01,4.23438E+01,4.23930E+01,4.24418E+01,
                  4.24902E+01,4.25381E+01,4.25857E+01,4.26328E+01,4.26795E+01,4.27258E+01,4.27716E+01,
                  4.28170E+01,4.28620E+01,4.29065E+01,4.29506E+01,4.29942E+01,4.30374E+01,4.30802E+01,
                  4.31225E+01,4.31644E+01,4.32059E+01,4.32469E+01,4.32875E+01,4.33278E+01,4.33676E+01,
                  4.34070E+01,4.34460E+01,4.34846E+01,4.35228E+01,4.35607E+01,4.35982E+01,4.36353E+01,
                  4.36721E+01,4.37086E+01,4.37448E+01]),
          'coeffs': [
                [ 1.489045326e+04, -2.922285939e+02,  5.724527170e+00, -8.176235030e-03, # 200 - 1000
                  1.456903469e-05, -1.087746302e-08,  3.027941827e-12,  -1.303131878e+04, -7.859241350e+00],
                [ 4.619197250e+05, -1.944704863e+03,5.916714180e+00,-5.664282830e-04, # 1000 - 6000
                  1.398814540e-07, -1.787680361e-11,9.620935570e-16,-2.466261084e+03, -1.387413108e+01],
                [ 8.868662960e+08, -7.500377840e+05,  2.495474979e+02, -3.956351100e-02, # 6000 - 20000
                  3.297772080e-06, -1.318409933e-10,  1.998937948e-15,  5.701421130e+06, -2.060704786e+03],
          ],
          'ranges': [200,1000,6000,20000],
          'wt': 28.01,
          'elements': OrderedDict([('C',1), ('O',1)]),
        }),
        ('CO2',{
          'T_range': temp_list_large,
          'H0_T': np.array([-2.38693E+02,-1.57733E+02,-1.17116E+02,-9.26580E+01,-7.62926E+01,-6.45608E+01,-5.57311E+01,
                  -4.88407E+01,-4.33111E+01,-3.87736E+01,-3.49822E+01,-3.17663E+01,-2.90035E+01,-2.66042E+01,
                  -2.45007E+01,-2.26414E+01,-2.09859E+01,-1.95024E+01,-1.81654E+01,-1.69540E+01,-1.58514E+01,
                  -1.48434E+01,-1.39184E+01,-1.30665E+01,-1.22794E+01,-1.15498E+01,-1.08717E+01,-1.02398E+01,
                  -9.64956E+00,-9.09691E+00,-8.57837E+00,-8.09085E+00,-7.63164E+00,-7.19832E+00,-6.78874E+00,
                  -6.40096E+00,-6.03328E+00,-5.68415E+00,-5.35216E+00,-5.03606E+00,-4.73470E+00,-4.44705E+00,
                  -4.17215E+00,-3.90915E+00,-3.65725E+00,-3.41572E+00,-3.18391E+00,-2.96119E+00,-2.74700E+00,
                  -2.54082E+00,-2.34215E+00,-2.15056E+00,-1.96562E+00,-1.78695E+00,-1.61418E+00,-1.44697E+00,
                  -1.28500E+00,-1.12798E+00,-9.75641E-01,-8.27706E-01,-6.83951E-01,-5.44166E-01,-4.08147E-01,
                  -2.75704E-01,-1.46655E-01,-2.08271E-02,1.01940E-01,2.21801E-01,3.38897E-01,4.53363E-01,
                  5.65327E-01,6.74905E-01,7.82209E-01,8.87343E-01,9.90402E-01,1.09148E+00,1.19065E+00,
                  1.28801E+00,1.38362E+00,1.47755E+00,1.56986E+00,1.66062E+00,1.74988E+00,1.83769E+00,
                  1.92410E+00,2.00915E+00,2.09289E+00,2.17535E+00,2.25657E+00,2.33658E+00,2.41542E+00,
                  2.49312E+00,2.56969E+00,2.64518E+00,2.71959E+00,2.79296E+00,2.86531E+00,2.93665E+00,
                  3.00700E+00,3.07638E+00,3.14481E+00,3.21230E+00,3.27887E+00,3.34454E+00,3.40930E+00,
                  3.47318E+00,3.53620E+00,3.59835E+00,3.65965E+00,3.72012E+00,3.77976E+00,3.83858E+00,
                  3.89659E+00,3.95381E+00,4.01023E+00,4.06587E+00,4.12074E+00,4.17485E+00,4.22820E+00,
                  4.28081E+00,4.33267E+00,4.38381E+00,4.43422E+00,4.48391E+00,4.53290E+00,4.58119E+00,
                  4.62878E+00,4.67570E+00,4.72193E+00,4.76749E+00,4.81240E+00,4.85665E+00,4.90025E+00,
                  4.94322E+00,4.98555E+00,5.02726E+00,5.06836E+00,5.10885E+00,5.14874E+00,5.18804E+00,
                  5.22675E+00,5.26488E+00,5.30245E+00,5.33945E+00,5.37591E+00,5.41181E+00,5.44718E+00,
                  5.48202E+00,5.51633E+00,5.55014E+00,5.58343E+00,5.61623E+00,5.64853E+00,5.68035E+00,
                  5.71169E+00,5.74257E+00,5.77298E+00,5.80295E+00,5.83246E+00,5.86154E+00,5.89019E+00,
                  5.91841E+00,5.94622E+00,5.97362E+00,6.00062E+00,6.02723E+00,6.05344E+00,6.07928E+00,
                  6.10474E+00,6.12984E+00,6.15458E+00,6.17896E+00,6.20300E+00,6.22670E+00,6.25006E+00,
                  6.27310E+00,6.29581E+00,6.31821E+00,6.34030E+00,6.36209E+00,6.38357E+00,6.40477E+00,
                  6.42567E+00,6.44630E+00,6.46665E+00,6.48672E+00,6.50653E+00,6.52607E+00,6.54536E+00,
                  6.56439E+00,6.58317E+00,6.60171E+00,6.62000E+00,6.63805E+00,6.65587E+00,6.67345E+00,
                  6.69081E+00,6.70794E+00,6.72484E+00]),
          'S0_T': np.array([2.40506E+01,2.57402E+01,2.70983E+01,2.82515E+01,2.92597E+01,3.01579E+01,3.09690E+01,
                  3.17088E+01,3.23888E+01,3.30178E+01,3.36027E+01,3.41490E+01,3.46612E+01,3.51432E+01,
                  3.55983E+01,3.60292E+01,3.64382E+01,3.68275E+01,3.71988E+01,3.75537E+01,3.78934E+01,
                  3.82193E+01,3.85324E+01,3.88336E+01,3.91238E+01,3.94038E+01,3.96743E+01,3.99358E+01,
                  4.01890E+01,4.04344E+01,4.06724E+01,4.09034E+01,4.11280E+01,4.13464E+01,4.15590E+01,
                  4.17661E+01,4.19680E+01,4.21650E+01,4.23573E+01,4.25451E+01,4.27288E+01,4.29084E+01,
                  4.30843E+01,4.32565E+01,4.34252E+01,4.35907E+01,4.37531E+01,4.39124E+01,4.40690E+01,
                  4.42228E+01,4.43741E+01,4.45229E+01,4.46693E+01,4.48136E+01,4.49557E+01,4.50959E+01,
                  4.52341E+01,4.53705E+01,4.55051E+01,4.56382E+01,4.57696E+01,4.58996E+01,4.60281E+01,
                  4.61553E+01,4.62811E+01,4.64057E+01,4.65290E+01,4.66512E+01,4.67724E+01,4.68925E+01,
                  4.70116E+01,4.71297E+01,4.72469E+01,4.73632E+01,4.74787E+01,4.75934E+01,4.77073E+01,
                  4.78205E+01,4.79329E+01,4.80446E+01,4.81556E+01,4.82659E+01,4.83756E+01,4.84846E+01,
                  4.85931E+01,4.87008E+01,4.88080E+01,4.89146E+01,4.90206E+01,4.91260E+01,4.92308E+01,
                  4.93350E+01,4.94387E+01,4.95417E+01,4.96442E+01,4.97462E+01,4.98475E+01,4.99483E+01,
                  5.00485E+01,5.01482E+01,5.02473E+01,5.03458E+01,5.04437E+01,5.05411E+01,5.06378E+01,
                  5.07340E+01,5.08296E+01,5.09247E+01,5.10191E+01,5.11130E+01,5.12062E+01,5.12989E+01,
                  5.13910E+01,5.14825E+01,5.15734E+01,5.16637E+01,5.17534E+01,5.18425E+01,5.19310E+01,
                  5.20190E+01,5.21063E+01,5.21930E+01,5.22791E+01,5.23646E+01,5.24495E+01,5.25338E+01,
                  5.26175E+01,5.27007E+01,5.27832E+01,5.28651E+01,5.29464E+01,5.30272E+01,5.31073E+01,
                  5.31869E+01,5.32658E+01,5.33442E+01,5.34220E+01,5.34993E+01,5.35759E+01,5.36520E+01,
                  5.37275E+01,5.38025E+01,5.38769E+01,5.39507E+01,5.40240E+01,5.40967E+01,5.41689E+01,
                  5.42405E+01,5.43116E+01,5.43822E+01,5.44522E+01,5.45217E+01,5.45907E+01,5.46592E+01,
                  5.47272E+01,5.47947E+01,5.48616E+01,5.49281E+01,5.49941E+01,5.50596E+01,5.51246E+01,
                  5.51892E+01,5.52533E+01,5.53169E+01,5.53801E+01,5.54428E+01,5.55051E+01,5.55669E+01,
                  5.56283E+01,5.56893E+01,5.57499E+01,5.58100E+01,5.58697E+01,5.59290E+01,5.59879E+01,
                  5.60465E+01,5.61046E+01,5.61623E+01,5.62197E+01,5.62766E+01,5.63332E+01,5.63895E+01,
                  5.64453E+01,5.65008E+01,5.65560E+01,5.66108E+01,5.66653E+01,5.67194E+01,5.67731E+01,
                  5.68266E+01,5.68797E+01,5.69325E+01,5.69849E+01,5.70371E+01,5.70889E+01,5.71404E+01,
                  5.71916E+01,5.72424E+01,5.72930E+01]),
          'coeffs': [
                [ 4.943650540e+04, -6.264116010e+02,  5.301725240e+00,  2.503813816e-03, # 200 - 1000
                  -2.127308728e-07, -7.689988780e-10,  2.849677801e-13, -4.528198460e+04, -7.048279440e+00, ],
                [ 1.176962419e+05, -1.788791477e+03,8.291523190e+00,-9.223156780e-05, # 1000 - 6000
                  4.863676880e-09, -1.891053312e-12,6.330036590e-16,-3.908350590e+04, -2.652669281e+01],
                [ -1.544423287e+09, 1.016847056e+06, -2.561405230e+02,  3.369401080e-02, # 6000 - 20000
                  -2.181184337e-06, 6.991420840e-11, -8.842351500e-16, -8.043214510e+06,  2.254177493e+03]
          ],
          'ranges': [201,1000,6000,20000],
          'wt': 44.01,
          'elements': OrderedDict([('C',1), ('O',2)]),
        }),
        ('O2',{
          'T_range': temp_list_large,
          'H0_T': np.array([-1.72484E+00,2.17926E-02,9.09894E-01,1.46383E+00,1.85320E+00,2.14773E+00,2.38110E+00,
                  2.57174E+00,2.73103E+00,2.86673E+00,2.98383E+00,3.08601E+00,3.17613E+00,3.25642E+00,
                  3.32862E+00,3.39412E+00,3.45398E+00,3.50909E+00,3.56013E+00,3.60768E+00,3.65218E+00,
                  3.69403E+00,3.73354E+00,3.77097E+00,3.80654E+00,3.84043E+00,3.87282E+00,3.90383E+00,
                  3.93358E+00,3.96218E+00,3.98972E+00,4.01627E+00,4.04190E+00,4.06668E+00,4.09066E+00,
                  4.11389E+00,4.13641E+00,4.15826E+00,4.17948E+00,4.20010E+00,4.22016E+00,4.23968E+00,
                  4.25868E+00,4.27719E+00,4.29523E+00,4.31283E+00,4.32999E+00,4.34674E+00,4.36310E+00,
                  4.37908E+00,4.39469E+00,4.40995E+00,4.42486E+00,4.43945E+00,4.45373E+00,4.46770E+00,
                  4.48137E+00,4.49475E+00,4.50785E+00,4.52066E+00,4.53317E+00,4.54537E+00,4.55730E+00,
                  4.56894E+00,4.58031E+00,4.59140E+00,4.60222E+00,4.61277E+00,4.62305E+00,4.63305E+00,
                  4.64277E+00,4.65221E+00,4.66137E+00,4.67025E+00,4.67885E+00,4.68716E+00,4.69518E+00,
                  4.70292E+00,4.71037E+00,4.71753E+00,4.72440E+00,4.73098E+00,4.73728E+00,4.74329E+00,
                  4.74901E+00,4.75445E+00,4.75961E+00,4.76449E+00,4.76909E+00,4.77341E+00,4.77747E+00,
                  4.78125E+00,4.78477E+00,4.78803E+00,4.79102E+00,4.79377E+00,4.79626E+00,4.79850E+00,
                  4.80050E+00,4.80226E+00,4.80379E+00,4.80508E+00,4.80615E+00,4.80700E+00,4.80764E+00,
                  4.80805E+00,4.80827E+00,4.80827E+00,4.80808E+00,4.80770E+00,4.80712E+00,4.80637E+00,
                  4.80543E+00,4.80431E+00,4.80302E+00,4.80157E+00,4.79995E+00,4.79818E+00,4.79625E+00,
                  4.79418E+00,4.79196E+00,4.78959E+00,4.78710E+00,4.78446E+00,4.78171E+00,4.77882E+00,
                  4.77582E+00,4.77270E+00,4.76946E+00,4.76612E+00,4.76268E+00,4.75913E+00,4.75548E+00,
                  4.75174E+00,4.74790E+00,4.74398E+00,4.73998E+00,4.73589E+00,4.73173E+00,4.72749E+00,
                  4.72317E+00,4.71879E+00,4.71434E+00,4.70983E+00,4.70526E+00,4.70063E+00,4.69594E+00,
                  4.69120E+00,4.68641E+00,4.68158E+00,4.67669E+00,4.67176E+00,4.66679E+00,4.66178E+00,
                  4.65674E+00,4.65166E+00,4.64654E+00,4.64140E+00,4.63622E+00,4.63102E+00,4.62579E+00,
                  4.62054E+00,4.61526E+00,4.60996E+00,4.60465E+00,4.59931E+00,4.59396E+00,4.58860E+00,
                  4.58322E+00,4.57783E+00,4.57242E+00,4.56701E+00,4.56159E+00,4.55615E+00,4.55072E+00,
                  4.54527E+00,4.53983E+00,4.53437E+00,4.52892E+00,4.52346E+00,4.51801E+00,4.51255E+00,
                  4.50709E+00,4.50163E+00,4.49618E+00,4.49073E+00,4.48528E+00,4.47983E+00,4.47439E+00,
                  4.46896E+00,4.46353E+00,4.45810E+00,4.45268E+00,4.44727E+00,4.44187E+00,4.43647E+00,
                  4.43109E+00,4.42571E+00,4.42034E+00]),
          'S0_T': np.array([2.32707E+01,2.46955E+01,2.57232E+01,2.65437E+01,2.72362E+01,2.78395E+01,2.83755E+01,
                  2.88579E+01,2.92967E+01,2.96992E+01,3.00709E+01,3.04160E+01,3.07382E+01,3.10404E+01,
                  3.13251E+01,3.15944E+01,3.18500E+01,3.20933E+01,3.23257E+01,3.25481E+01,3.27615E+01,
                  3.29666E+01,3.31641E+01,3.33547E+01,3.35389E+01,3.37171E+01,3.38898E+01,3.40572E+01,
                  3.42198E+01,3.43779E+01,3.45316E+01,3.46814E+01,3.48273E+01,3.49696E+01,3.51085E+01,
                  3.52441E+01,3.53766E+01,3.55062E+01,3.56330E+01,3.57570E+01,3.58786E+01,3.59976E+01,
                  3.61143E+01,3.62287E+01,3.63410E+01,3.64511E+01,3.65593E+01,3.66655E+01,3.67698E+01,
                  3.68723E+01,3.69731E+01,3.70723E+01,3.71697E+01,3.72657E+01,3.73601E+01,3.74530E+01,
                  3.75445E+01,3.76346E+01,3.77233E+01,3.78108E+01,3.78969E+01,3.79817E+01,3.80653E+01,
                  3.81477E+01,3.82289E+01,3.83090E+01,3.83879E+01,3.84657E+01,3.85424E+01,3.86181E+01,
                  3.86927E+01,3.87662E+01,3.88387E+01,3.89102E+01,3.89808E+01,3.90503E+01,3.91188E+01,
                  3.91864E+01,3.92531E+01,3.93188E+01,3.93836E+01,3.94475E+01,3.95105E+01,3.95726E+01,
                  3.96338E+01,3.96942E+01,3.97537E+01,3.98124E+01,3.98703E+01,3.99273E+01,3.99836E+01,
                  4.00390E+01,4.00937E+01,4.01476E+01,4.02008E+01,4.02532E+01,4.03048E+01,4.03558E+01,
                  4.04060E+01,4.04556E+01,4.05044E+01,4.05526E+01,4.06001E+01,4.06469E+01,4.06931E+01,
                  4.07387E+01,4.07836E+01,4.08279E+01,4.08717E+01,4.09148E+01,4.09573E+01,4.09993E+01,
                  4.10407E+01,4.10815E+01,4.11219E+01,4.11616E+01,4.12009E+01,4.12396E+01,4.12778E+01,
                  4.13155E+01,4.13528E+01,4.13895E+01,4.14258E+01,4.14616E+01,4.14969E+01,4.15318E+01,
                  4.15663E+01,4.16003E+01,4.16340E+01,4.16671E+01,4.16999E+01,4.17323E+01,4.17643E+01,
                  4.17959E+01,4.18271E+01,4.18580E+01,4.18885E+01,4.19186E+01,4.19483E+01,4.19778E+01,
                  4.20068E+01,4.20356E+01,4.20640E+01,4.20921E+01,4.21199E+01,4.21474E+01,4.21745E+01,
                  4.22014E+01,4.22280E+01,4.22543E+01,4.22803E+01,4.23060E+01,4.23314E+01,4.23566E+01,
                  4.23815E+01,4.24062E+01,4.24306E+01,4.24547E+01,4.24787E+01,4.25023E+01,4.25258E+01,
                  4.25490E+01,4.25719E+01,4.25947E+01,4.26172E+01,4.26395E+01,4.26616E+01,4.26835E+01,
                  4.27051E+01,4.27266E+01,4.27479E+01,4.27690E+01,4.27898E+01,4.28105E+01,4.28311E+01,
                  4.28514E+01,4.28715E+01,4.28915E+01,4.29113E+01,4.29309E+01,4.29503E+01,4.29696E+01,
                  4.29888E+01,4.30077E+01,4.30265E+01,4.30451E+01,4.30636E+01,4.30820E+01,4.31002E+01,
                  4.31182E+01,4.31361E+01,4.31538E+01,4.31714E+01,4.31889E+01,4.32062E+01,4.32234E+01,
                  4.32405E+01,4.32574E+01,4.32742E+01]),
          'coeffs':[
                [ -3.425563420e+04, 4.847000970e+02, 1.119010961e+00, 4.293889240e-03, # 200 - 1000
                  -6.836300520e-07, -2.023372700e-09, 1.039040018e-12, -3.391454870e+03, 1.849699470e+01],
                [ -1.037939022e+06,2.344830282e+03,1.819732036e+00,1.267847582e-03, # 1000 - 6000
                  -2.188067988e-07,2.053719572e-11,-8.193467050e-16,-1.689010929e+04, 1.738716506e+01],
                [ 4.975294300e+08, -2.866106874e+05,  6.690352250e+01, -6.169959020e-03,  # 6000 - 20000
                  3.016396027e-07, -7.421416600e-12,  7.278175770e-17, 2.293554027e+06, -5.530621610e+02]
          ],
          'ranges': [200,999,6000,20000],
          'wt': 32,
          'elements': {'O':2},
        }),
    ])

    init_prod_amounts = OrderedDict([ # initial value used to set the atomic fractions in the mixture
        ('CO', 0),
        ('CO2', 1),
        ('O2', 0),
    ])

    elements = OrderedDict([
        ('C', {
            'wt': 12.01
            }),
        ('O', {
            'wt': 16.0
            })
    ])

    element_wts = {
      'C':12.01, 'O': 16.0,
    }

    aij = np.array([[1,1,0],[1,2,2]])

test = TestComposition()

# print(temp_list_small)
# print(temp_list_large)
# print(aij)

# print(list(products.keys()))