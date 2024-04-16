import numpy as np
from numpy import pi, exp, sqrt, log
import sys

mH = 1.6726e-24
kb = 1.38066e-16

oxa10 = 8.865e-5
oxa20 = 1.275e-10
oxa21 = 1.772e-5
oxe10 = 2.2771e2 * kb
oxe21 = 9.886e1  * kb
oxe20 = oxe10 + oxe21

cIIa10 = 2.291e-6
cIIe10 = 9.125e1 * kb

def three_level_pops(r01, r02, r12, r10, r20, r21):

#
# If excitation rates are negligibly small, then we assume that all
# of the atoms are in level 0:
#
    a1 = r01 + r02
    a2 = -r10
    a3 = -r20
    b1 = r01
    b2 = -(r10 + r12)
    b3 = r21

    n2 = -a1 * (a1 * b2 - b1 * a2) / ((a1 - a2) * \
         (a1 * b3 - b1 * a3) - (a1 - a3) * \
         (a1 * b2 - b1 * a2))

    n1 = (a1 / (a1 - a2)) - ((a1 - a3) / (a1 - a2)) * n2

    n0 = 1. - n1 - n2

    ii = ((r01 == 0.) & (r02 == 0.))
    n0[ii] = 1.
    n1[ii] = 0.
    n2[ii] = 0.

    return n0, n1, n2

def compute_CII_populations(temp, yn, abHI, abh2, abe, G0):
    cII_lambda = 1. / 63.395087
    cII_freq   = 1900.5369e9

    ynh  = abHI * yn
    ynh2 = abh2 * yn
    yne  = abe  * yn

    cl45, cl46, cl47, cl48 = compute_CII_excitation_rate_terms(temp)

    cIIc10 = cl45 * ynh + cl46 * ynh2 + cl47 * yne
    cIIc01 = cl48 * cIIc10
#
# Absorption and stimulated emission of ISRF; assume spectral shape 
# from Mathis et al (1983), normalized by our value of G0
#
    isrf_mathis = 1.24e-5    # Linear interpolation from Table B1

# Now rescale from erg/s/cm^2/micron to standard units
    isrf_mathis = isrf_mathis / (4.0 * pi)
    isrf_mathis = isrf_mathis * 1e4 * cII_lambda / cII_freq

# Finally, apply G0 scaling
    isrf_mathis = isrf_mathis * G0

    cIIb10 = (cII_lambda**2 / (2. * cIIe10)) * cIIa10 * isrf_mathis
    cIIb01 = 2. * cIIb10

    cIIvar0 = (cIIc10 + cIIa10 + cIIb10)
    cIIvar1 = cIIc01 + cIIb01

    cIIf0 = cIIvar0 / (cIIvar0 + cIIvar1)
    cIIf1 = cIIvar1 / (cIIvar0 + cIIvar1)

    return cIIf0, cIIf1, cIIb10

def compute_emissivities(temp, rho, abcII, aboI, abHI, abh2, abe, abhp, G0):

    cII_spec_emiss = np.zeros(np.size(temp))
    oI_spec_emiss_63 = np.zeros(np.size(temp))
    oI_spec_emiss_145 = np.zeros(np.size(temp))

    yn = rho / (1.4 * mH)
#
# CII 158 micron line
# 
    cIIf0, cIIf1, cIIb10 = compute_CII_populations(temp, yn, abHI, abh2, abe, G0)

    cII_emiss = cIIf1 * abcII * yn * (cIIa10 + cIIb10) * cIIe10   # erg s^-1 cm^-3
    cII_spec_emiss = cII_emiss / rho
#
# OI 63 and 145 micron lines
#
    oIf0, oIf1, oIf2, oxb10, oxb21 = compute_OI_populations(temp, yn, abHI, abh2, abe, abhp, G0)

    oI_emiss_63       = oIf1 * aboI * yn * (oxa10 + oxb10) * oxe10   # erg s^-1 cm^-3
    oI_spec_emiss_63  = oI_emiss_63 / rho

    oI_emiss_145      = oIf2 * aboI * yn * (oxa21 + oxb21) * oxe21   # erg s^-1 cm^-3
    oI_spec_emiss_145 = oI_emiss_145 / rho

    ind0 = ((rho == 0.) | (abcII == 0.))

    cII_spec_emiss[ind0] = 0. 
    oI_spec_emiss_63[ind0] = 0.
    oI_spec_emiss_145[ind0] = 0.

    return cII_spec_emiss, oI_spec_emiss_63, oI_spec_emiss_145

def compute_CII_excitation_rate_terms(temp):
#
# Adapted from coolinmo.F -- this explains the odd numbering
#
      temp2  = temp / 1e2
#
# Assume only J=0, J=1 levels populated, and that they have a thermal
# distribution
#
      fpara_inv = 9. * exp(-170.5/temp) + 1.
      fpara  = 1. / fpara_inv
      fortho = 1. - fpara
#
# (cl45) CII - HI (HM89 below 2000K; K86 above 2000K)
#
# (Note that the high T coefficient has been tweaked slightly to ensure that
#  the rate is continuous at 2000K -- the adjustment is well within the 
#  errors).
#
      i_tlow = temp<=2e3
      i_thigh = ~i_tlow

      cl45 = np.ones(temp.size)

      cl45[i_tlow] = 8e-10 * temp2[i_tlow]**0.07
      cl45[i_thigh] = 3.113619e-10 * temp2[i_thigh]**0.385
    
#
# (cl46) CII - H2 (From Wiesenfeld & Goldsmith, 2014, ApJ, 780, 183. 
#        Valid for 20 < T < 400 K, and fairly well-behaved outside of this 
#        range. Limit T to <= 1e4 K, as expect contribution above this to
#        be negligible)
#

    
      i_tlow = temp<1e4
      i_thigh = ~i_tlow

      tfix = np.ones(temp.size)

      tfix[i_tlow] = temp2[i_tlow]
      tfix[i_thigh] = 1e2

      f  = (4.43 + 0.33 * tfix) * 1e-10 * fpara
      hh = (5.33 + 0.11 * tfix) * 1e-10 * fortho

      cl46 = f + hh
#
# (cl47) CII - electron (WB02). 
#
# (Note that the high T coefficient has been tweaked slightly to ensure that
#  the rate is continuous at 2000K -- the adjustment is well within the 
#  errors).
#
      cl47 = 3.86e-7 / sqrt(temp2)
      i_thigh = temp>2e3

      cl47[i_thigh] = 2.426206e-7 / temp2[i_thigh]**0.345
#
# Proton rate negligible below 10^4K, ignorable below 10^5K.
#
# Finally, cl48 holds f(1,LTE)
#
      cl48 = 2. * exp(-91.25 / temp)

      return cl45, cl46, cl47, cl48


def compute_OI_populations(temp, yn, abHI, abh2, abe, abhp, G0):
      oI_63_freq  = oxe10 / 6.62618e-27
      oI_145_freq = oxe21 / 6.62618e-27

      oI_63_lambda  = 2.99792e10 / oI_63_freq
      oI_145_lambda = 2.99792e10 / oI_145_freq
#
# Absorption and stimulated emission of ISRF; assume spectral shape                                                                                                
# from Mathis et al (1983), normalized by our value of G0
#
      isrf_mathis_63   = 4.685e-5   # Linear interpolation from Table B1
      isrf_mathis_145  = 1.54e-5    # Linear interpolation from Table B1
#
# NB. We ignore the 2-0 transition, since the rate for this will be
# five orders of magnitude smaller than the other rates (since A20
# is so small)
#
# Now rescale from erg/s/cm^2/micron to standard units
#
      isrf_mathis_63  = isrf_mathis_63 / (4. * pi)
      isrf_mathis_63  = isrf_mathis_63 * 1e4 * oI_63_lambda / oI_63_freq

      isrf_mathis_145 = isrf_mathis_145 / (4. * pi)
      isrf_mathis_145 = isrf_mathis_145 * 1e4 * oI_145_lambda / oI_145_freq
#
# Finally, apply G0 scaling
#
      isrf_mathis_63  = isrf_mathis_63 * G0
      isrf_mathis_145 = isrf_mathis_145 * G0
#
# For debugging: check brightness temp for both lines
#
      TB_inv = (kb / oxe10) * log(1. + 2. * oxe10 / (oI_63_lambda**2 * isrf_mathis_63))
      TB_63 = 1. / TB_inv

      TB_inv = (kb / oxe21) * log(1. + 2. * oxe21 / (oI_145_lambda**2 * isrf_mathis_145))
      TB_145 = 1. / TB_inv
      
      oxb10 = (oI_63_lambda**2 / (2. * oxe10)) * oxa10 * isrf_mathis_63
      oxb01 = (3./5.) * oxb10

      oxb21 = (oI_145_lambda**2 / (2. * oxe21)) * oxa21 * isrf_mathis_145
      oxb12 = (1./3.) * oxb21

      ynh  = abHI * yn
      ynh2 = abh2 * yn
      yne  = abe  * yn
      ynhp = abhp * yn

      oxct01, oxct02, oxct12, oxct10, oxct20, oxct21 = compute_OI_CT_rates(temp, ynhp)

      cl18, cl19, cl20, cl21, cl22, cl23, cl24, cl25, cl26, cl27, cl28, cl29 = compute_OI_excitation_rate_terms(temp)

      oxc10  = cl18 * ynh + cl21 * ynh2 + cl24 * yne + cl27 * ynhp
      oxc20  = cl19 * ynh + cl22 * ynh2 + cl25 * yne + cl28 * ynhp
      oxc21  = cl20 * ynh + cl23 * ynh2 + cl26 * yne + cl29 * ynhp

      oxa =  exp(-oxe10 / (kb * temp))
      oxb =  exp(-oxe20 / (kb * temp))
      oxc =  exp(-oxe21 / (kb * temp))

      oxc01  = 0.6 * oxc10 * oxa
      oxc02  = 0.2 * oxc20 * oxb
      oxc12  = (1. / 3.) * oxc21 * oxc
#
# Total transition rates:
#
      oxR01  = oxc01 + oxb01 + oxct01
      oxR02  = oxc02 + oxct02
      oxR12  = oxc12 + oxb12 + oxct12
      oxR10  = oxc10 + oxa10 + oxb10 + oxct10
      oxR20  = oxc20 + oxa20 + oxct20
      oxR21  = oxc21 + oxa21 + oxb21 + oxct21
#
# Check for tiny excitation rates
#
      oIf0, oIf1, oIf2 = three_level_pops(oxR01, oxR02, oxR12, oxR10, oxR20, oxR21)

      i_oxR01_l = (oxR01 < 1e-40)
      oIf0[i_oxR01_l] = 1e0
      oIf1[i_oxR01_l] = 0.
      oIf2[i_oxR01_l] = 0.

      return oIf0, oIf1, oIf2, oxb10, oxb21

#
# Charge transfer with H+ produces O+. If this then transfers charge back to H,
# the newly formed O atom can end up in a different J level from the one that
# it started in. We here compute the effective rate at which this induces changes
# in J for atoms starting in each of the three different levels. We assume only
# that the H+/O+ ratio is in chemical equilibrium, so that each transition from
# O -> O+ is balanced immediately by a transition back to O. The charge transfer
# rates adopted here are from Stancil et al (1999) 
#
def compute_OI_CT_rates(tin, ynhp):

# Stancil et al fits only valid for T < 1e4 K
      temp = tin
      temp4 = temp / 1e4

      i_tinhigh = tin > 1e4
      temp[i_tinhigh]  = 1e4
      temp4[i_tinhigh] = 1.

#Given an O+ ion, what fraction of time does transition back to O put the
#resulting atom in a particular J level. NB Remember: level 0 == J=2, etc. 
#in our notation 
      total_rate = 0.

      rate_J0 = (4.47e-10 * temp4**0.257    * exp(temp / 5.49e4) \
              +  1.03e-11 * temp4**(-0.365) * exp(-temp / 85.7)) \
              * exp(-99. / temp)

      rate_J1 = 3.29e-10 * temp4**0.455 \
              + 1.97e-11 * temp4**(-0.209) * exp(-temp / 3.22e4)

      rate_J2 = 1.14e-9  * temp4**0.397 \
              + 1.38e-11 * temp4**(-0.298) * exp(-temp / 3.64e4)

      total_rate = rate_J0 + rate_J1 + rate_J2

#Fraction ending up in level 0 (i.e. J=2)
      frac_0 = rate_J2 / total_rate
      frac_1 = rate_J1 / total_rate
      frac_2 = rate_J0 / total_rate

#CT rate out of levels J = 0, 1, 2 respectively
      rate_out_J0 = 2.39e-9  * temp4**0.327    * exp(temp / 2.06e5) \
                  + 3.54e-11 * temp4**(-0.426) * exp(-temp / 5.27e3) 

      rate_out_J1 = 7.91e-10 * temp4**0.311   * exp(temp / 5.97e5) \
                  + 1.96e-11 * temp4**(-3.29) * exp(-temp / 2.21e2)

      rate_out_J2 = (1.57e-9 * temp4**0.298  * exp(temp / 7.51e4) \
                  +  1.62e-7 * temp4**(1.13) * exp(-temp / 19.4)) \
                  * exp(-227 / temp)

#Compute final rates, weight by H+ number density
      oxct01 = rate_out_J2 * frac_1 * ynhp
      oxct02 = rate_out_J2 * frac_2 * ynhp

      oxct10 = rate_out_J1 * frac_0 * ynhp
      oxct12 = rate_out_J1 * frac_2 * ynhp

      oxct20 = rate_out_J0 * frac_0 * ynhp
      oxct21 = rate_out_J0 * frac_1 * ynhp

# No transitions if no H+
      i_ynhp_0 = (ynhp==0.)
      oxct01[i_ynhp_0] = 0.
      oxct02[i_ynhp_0] = 0.
      oxct12[i_ynhp_0] = 0.
      oxct10[i_ynhp_0] = 0.
      oxct20[i_ynhp_0] = 0.
      oxct21[i_ynhp_0] = 0.

      return oxct01, oxct02, oxct12, oxct10, oxct20, oxct21

def compute_OI_excitation_rate_terms(temp):
      tinv  = 1. / temp
      tintq = tinv**0.75
      tisqt = 1. / sqrt(temp)

#Assume only J=0, J=1 levels populated, and that they have a thermal
#distribution

      fpara_inv = 9. * exp(-170.5/temp) + 1.
      fpara  = 1. / fpara_inv
      fortho = 1. - fpara

#(cl[18-29]): OI fine-structure lines 

#Collisional de-excitation rates: 
#                   
#HI: taken from AKD07 below 1000K, extended to higher temperatures
#    with a simple power-law extrapolation

#1 -> 0
      cl18 = 6.81e-11 * temp**0.376
        
      i_temp_lt5 = (temp < 5.)
      i_temp_lt1e3 = (temp < 1e3)

      cl18[i_temp_lt1e3] = (5e-11 / 3.) * exp(4.581\
                           - 156.118       * tintq[i_temp_lt1e3]\
                           + 2679.979      * tintq[i_temp_lt1e3]**2\
                           - 78996.962     * tintq[i_temp_lt1e3]**3\
                           + 1308323.468   * tintq[i_temp_lt1e3]**4\
                           - 13011761.861  * tintq[i_temp_lt1e3]**5\
                           + 71010784.971  * tintq[i_temp_lt1e3]**6\
                           - 162826621.855 * tintq[i_temp_lt1e3]**7)\
                           * exp(oxe10 / (kb * temp[i_temp_lt1e3]))

      tfix = 5.
      tfintq = 1. / tfix**0.75 
      cl18[i_temp_lt5] = (5e-11 / 3.) * exp(4.581\
                         - 156.118       * tfintq\
                         + 2679.979      * tfintq**2\
                         - 78996.962     * tfintq**3\
                         + 1308323.468   * tfintq**4\
                         - 13011761.861  * tfintq**5\
                         + 71010784.971  * tfintq**6\
                         - 162826621.855 * tfintq**7)\
                         * exp(oxe10 / (kb * tfix))

#2 -> 0
      cl19 = 6.34e-11 * temp**0.36

      cl19[i_temp_lt1e3] = 5e-11 * exp(3.297\
                           - 168.382       * tintq[i_temp_lt1e3]\
                           + 1844.099      * tintq[i_temp_lt1e3]**2\
                           - 68362.889     * tintq[i_temp_lt1e3]**3\
                           + 1376864.737   * tintq[i_temp_lt1e3]**4\
                           - 17964610.169  * tintq[i_temp_lt1e3]**5\
                           + 134374927.808 * tintq[i_temp_lt1e3]**6\
                           - 430107587.886 * tintq[i_temp_lt1e3]**7)\
                           * exp(oxe20 / (kb * temp[i_temp_lt1e3]))

      tfix = 5.
      tfintq = 1. / tfix**0.75
      cl19[i_temp_lt5] = 5e-11 * exp(3.297
                         - 168.382       * tfintq\
                         + 1844.099      * tfintq**2\
                         - 68362.889     * tfintq**3\
                         + 1376864.737   * tfintq**4\
                         - 17964610.169  * tfintq**5\
                         + 134374927.808 * tfintq**6\
                         - 430107587.886 * tfintq**7)\
                         * exp(oxe20 / (kb * tfix)) 

#2 -> 1:  Low T extrapolation here is necessary because the AKD07 
#         fitting function blows up

      cl20 = 3.61e-10 * temp**0.158

      cl20[i_temp_lt1e3] = 3e-11 * exp(3.437\
                           + 17.443    * tisqt[i_temp_lt1e3]\
                           - 618.761   * tisqt[i_temp_lt1e3]**2\
                           + 3757.156  * tisqt[i_temp_lt1e3]**3\
                           - 12736.468 * tisqt[i_temp_lt1e3]**4\
                           + 22785.266 * tisqt[i_temp_lt1e3]**5\
                           - 22759.228 * tisqt[i_temp_lt1e3]**6\
                           + 12668.261 * tisqt[i_temp_lt1e3]**7)\
                           * exp(oxe21 / (kb * temp[i_temp_lt1e3]))

      i_temp_lt50 = (temp < 50.)
      cl20[i_temp_lt50] = 2.62e-12 * temp[i_temp_lt50]**0.74
  
#H2 rates supplied by Flower (priv. comm.)
#(NB. If the H2 rate is based on the data from J92, then strictly 
#speaking it is only applicable for T < 1500K; however, the rate 
#doesn't misbehave too badly at higher T).

#H2 - ortho and para states must be accounted for separately

#1 -> 0
      f    = fortho * 2.70e-11 * (temp**0.362)
      hh   = fpara  * 3.46e-11 * (temp**0.316)
      cl21 = f + hh
#2 -> 0
      f    = fortho * 5.49e-11 * (temp**0.317)
      hh   = fpara  * 7.07e-11 * (temp**0.268)
      cl22 = f + hh
#2 -> 1
      f    = fortho * 2.74e-14 * (temp**1.060)
      hh   = fpara  * 3.33e-15 * (temp**1.360)
      cl23 = f + hh

#Electron rate -- from my fits to BBT98.

#1 -> 0
      cl24 = 5.12e-10  * (temp**(-0.075))
#2 -> 0
      cl25 = 4.863e-10 * (temp**(-0.026))
#2 -> 1
      cl26 = 1.082e-14 * (temp**(0.926))

#Proton rate -- from P90

#1 -> 0
      cl27 = 2.65e-10 * (temp**0.37)

      i_temp_l = (temp < 3686)
      cl27[i_temp_l] = 7.75e-12 * (temp[i_temp_l]**0.80)

      i_temp_ll = (temp < 194.)
      cl27[i_temp_ll] = 6.38e-11 * (temp[i_temp_ll]**0.40)
    
#2 -> 0
      cl28 = 4.49e-10 * (temp**0.30)

      i_temp_l = (temp < 7510)
      cl28[i_temp_l] = 2.12e-12 * (temp[i_temp_l]**0.90)
      
      i_temp_ll = (temp < 511)
      cl28[i_temp_ll] = 6.10e-13 * (temp[i_temp_ll]**1.10)

#2 -> 1
      cl29 = 3.434e-10 * (temp**0.19)

      i_temp_l = (temp < 2090)
      cl29[i_temp_l] = 2.029e-11 * (temp[i_temp_l]**0.56)

      return cl18, cl19, cl20, cl21, cl22, cl23, cl24, cl25, cl26, cl27, cl28, cl29
