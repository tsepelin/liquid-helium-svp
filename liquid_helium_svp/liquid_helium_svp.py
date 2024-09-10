# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 11:13:24 2015
Modified on Thursday 28/10/2016 by TsepelinV
library should take in arguments as float, list, numpy arrays or Pandas
@author: noblemt
"""
from scipy import interpolate
import numpy as np

########################################################################
###     Constants
########################################################################

atomic_mass_unit = 1.660539040e-27
Boltzmann_const = 1.38064852e-23
Plank_const = 6.626176e-34
Plankbar_const = Plank_const / (2.0 * np.pi)
Avogadro_const = 6.022140857e+23

molar_mass  = 4.002602e-3 #[kg]
kappa = Plank_const /(molar_mass / Avogadro_const)

T_Lambda = 2.1768 # K
density_Tlambda = 0.1461087 # g/cm**3
density_Tzero = 0.1451397 # g/cm**3
Heat_Latent_evaporation = 2.6e3 # J/l @4.2K

######################################################################
### decorator function
######################################################################

def validate_input_data_types(func):
    def wrapper(*args):
        def list2array(list_input):
            return np.array(list_input)

        def array2list(array_input):
            return array_input.tolist()

        def single2array(single_input):
            return np.array([single_input])

        def array2single(array_input):
            return array_input[0]

        treated_args = []
        for arg in args:
            if not isinstance(arg, (list, np.ndarray, float, int)):
                raise ValueError("Unsupported data type. Please use list, numpy array, float, or int.")
            if isinstance(arg, list):
                treated_args.append(array2list(func(list2array(arg))))
            elif isinstance(arg, (float, int)):
                treated_args.append(array2single(func(single2array(arg))))
            else:
                treated_args.append(func(arg))
        if len(args)==1:
            return treated_args[0]
        else:
            return treated_args
    return wrapper


##############################################################
### helium properties functions
##############################################################


@validate_input_data_types
def density(TemperatureK):
    """For a given temperature[K] function returns the helium-4 density [kg m^3].

    based on Russell J. Donnelly and Carlo F. Barenghi
    J. Phys. Chem. Ref. Data 27, 1217 (1998); http://dx.doi.org/10.1063/1.556028
   'The Observed Properties of Liquid Helium at the Saturated Vapor Pressure'
   """

    rho = np.tile(np.NaN, len(TemperatureK)) # array of NaN

    t = TemperatureK - np.tile(T_Lambda, len(TemperatureK))

    #For tempeartures below 1.334K
    #rho = SUM(from 1 = 1 to 4)m_i * T^(i+1) + rho_0
    m1 = np.array([ 8.75342, -16.7461, 7.12413, -1.26935, 0, 0]) * 1e-5
    rho_below1p334 = np.polyval(m1, TemperatureK) + density_Tzero

    #For temperatures between 1.334K and the Lambda point
    a_belTl = np.array([6.87483, -7.57537, 0]) * 1e-3
    b_belTl = np.array([0, 0, 0, 0, 4.88345, 1.86557, 3.79937, 0]) * 1e-3
    delta_rho_belTl = (np.polyval(a_belTl,t) * np.log(np.absolute(t)) + np.polyval(b_belTl,t))

    #For temperatures above the lambda point
    a_abTl = np.array([ 5.057051, -7.94605,0]) * 1e-3
    b_abTl = np.array([-0.308182, 1.53454, -2.45749, 0.240720, -3.00636, -10.2326, -30.3511,0]) * 1e-3
    delta_rho_abTl = (np.polyval(a_abTl,t) * np.log(np.absolute(t)) + np.polyval(b_abTl,t))

    rho_above1p334 = density_Tlambda * (1 + (t<0)*delta_rho_belTl + (t>0)* delta_rho_abTl)

    rho = (TemperatureK > np.tile(1.334,len(TemperatureK))) * rho_above1p334 + (TemperatureK < np.tile(1.334,len(TemperatureK))) * rho_below1p334

    return rho * 1000

@validate_input_data_types
def density_superfluid(TemperatureK):
    """Performs spline fit to interpolate the superfluid density.

    based on Russell J. Donnelly and Carlo F. Barenghi
    J. Phys. Chem. Ref. Data 27, 1217 (1998); http://dx.doi.org/10.1063/1.556028
    'The Observed Properties of Liquid Helium at the Saturated Vapor Pressure'
    """

    k = np.array([0.0, 0.0, 0.0, 0.0, 0.443, 0.9012, 1.5419, 1.7540, 1.918,
                  2.111, 2.156991, 2.173218, 2.175647, 2.176358, 2.176568,
                  2.176692, 2.176766, 2.176791, 2.176798, 2.176799,
                  2.17679999, 2.1768, 2.1768, 2.1768, 2.1768])

    c = np.array([1.451275432822459E-1, 1.451334563362309E-1,
                  1.449759191497576E-1, 1.455008000684433E-1, 1.4075E-1,
                  1.095E-1, 8.15E-2, 5.30E-2, 2.1E-2, 8.904576E-3, 3.053214E-3,
                  1.494043E-3, 8.342826E-4, 5.10686E-4, 2.8379E-4, 1.287426E-4,
                  5.202569E-5, 2.153580E-5, 8.564206E-6, 3.567958E-6,0])

    density_superfluid = np.tile(np.NaN,len(TemperatureK)) # array of NaN

    for tempindex in range(len(density_superfluid)):
        density_superfluid[tempindex] = interpolate.splev(TemperatureK[tempindex], (k, c, 3), ext = 1) * 1e3 #kg/m**3

    return density_superfluid

def density_from_pressure_low_temperature(pressurePa, temperatureK):
    '''
    Uses model to find the denisty of helium as function of pressure. From low temperature data in:
    Abraham B M, Eckstein Y, Ketterson J B, Kuchnir M and Roach P R 1970 Phys. Rev. A 1 (2) 50

    :param pressurePa: Pressure in Pa
    :param temperatureK: temperature in K
    :return: The density in kg/m^3
    '''
    A1 = 5.60e2 # atm cm^3 / g
    A2 = 1.0970e4 # atm cm^6 / g
    A3 = 7.33e4 # atm cm^9 / g

    rho_0 = float(density(temperatureK))

    densities = []
    if type(pressurePa) is float:
        pressurePa = np.array([pressurePa])

    for p in pressurePa.tolist():
        d = p * -9.86923e-6
        poly = np.array([A3, A2, A1, d])
        roots = np.roots(poly) * 1e3
        for root in roots.tolist():
            if np.imag(root) < 1e-5:
                real_root = np.real(root)
        densities.append(real_root + rho_0)

    return np.array(densities)


def density_normalfluid(TemperatureK):
    """Uses the suprefluid density to work out the normal fluid density.
   Returns the whole fluid density above the lambda point."""
    #Works out whole fluid density and subtracts the superfluid density.

    rho_he = density(TemperatureK)
    rho_sup_he = density_superfluid(TemperatureK)

    return np.subtract(rho_he, rho_sup_he)


@validate_input_data_types
def viscosity(TemperatureK):
    """For a given temperature[K] array returns the helium-4 viscosity[Pa s].
    not very accurate?

    based on Russell J. Donnelly and Carlo F. Barenghi
    J. Phys. Chem. Ref. Data 27, 1217 (1998); http://dx.doi.org/10.1063/1.556028
    'The Observed Properties of Liquid Helium at the Saturated Vapor Pressure'
    """
    k = np.array([7.913364e-1, 7.913364e-1, 7.913364e-1, 7.913364e-1,
                  9.705100e-1, 1.064730, 1.285930, 1.582100, 1.747010, 2.02568,
                  2.051740, 2.146961, 2.17680, 2.17680, 2.17680, 2.212906,
                  2.221800, 2.618000, 3.253700, 3.784200, 4.025400, 4.406982,
                  4.406982, 4.406982, 4.406982])

    c =  np.array([1.730865E-5, 6.577810E-6, 4.956473E-6, 1.862435E-6,
                   1.452672E-6, 1.308345E-6, 1.273173E-6, 1.338821E-6,
                   1.613257E-6, 1.956558E-6, 2.296259E-6, 2.514817E-6,
                   2.487748E-6, 2.715638E-6, 3.125798E-6, 3.487019E-6,
                   3.564378E-6, 3.486451E-6, 3.270547E-6, 3.226615E-6,
                   3.160000E-6])

    viscosity = np.tile(np.NaN,len(TemperatureK)) # array of NaN

    for tempindex in range(len(viscosity)):
        viscosity[tempindex] = interpolate.splev(TemperatureK[tempindex], (k, c, 3))

    return viscosity # Pa s


@validate_input_data_types
def friction_mutual_B(TemperatureK):
    """For a given temperature[K] array returns the helium-4 friction coefficient B.


    based on Russell J. Donnelly and Carlo F. Barenghi
    J. Phys. Chem. Ref. Data 27, 1217 (1998); http://dx.doi.org/10.1063/1.556028
    'The Observed Properties of Liquid Helium at the Saturated Vapor Pressure'
    """
    k = np.array([-5.0, -5.0, -5.0, -5.0, -2.5,
                  -2.0, -0.8, -0.387958059947, -0.387958059947, -0.387958059947, -0.387958059947])

    c =  np.array([1.31928144433, 1.12452707801, 0.639314792565, 0.313383532495,
                   -0.162687403543, 0.092047691284, 0.188452616588])

    Temperature_reduced = 1.0 - TemperatureK/T_Lambda
    friction_B = np.tile(np.NaN,len(TemperatureK)) # array of NaN

    for tempindex in range(len(friction_B)):
        if TemperatureK[tempindex] > 2.167:
            friction_B[tempindex] = 0.47 * Temperature_reduced[tempindex]**-0.33
        else:
            friction_B[tempindex] = 10.0**(interpolate.splev(np.log10(Temperature_reduced[tempindex]), (k, c, 3)))

    return friction_B # Pa s


@validate_input_data_types
def friction_mutual_B_prime(TemperatureK):
    """For a given temperature[K] array returns the helium-4 mutual friction coefficient B prime.


    based on Russell J. Donnelly and Carlo F. Barenghi
    J. Phys. Chem. Ref. Data 27, 1217 (1998); http://dx.doi.org/10.1063/1.556028
    'The Observed Properties of Liquid Helium at the Saturated Vapor Pressure'
    """
    k = np.array([-5.0, -5.0, -5.0, -5.0,
                  -3.55, -3.20, -2.5, -1.0,
                  -0.384067377871, -0.384067377871, -0.384067377871, -0.384067377871])

    c =  np.array([-8.47218032526e-2, 0.931621715174, 0.973263359433, 1.10543591819,
                   1.15904485127, 1.18311634566, 1.17480594214, 1.19458392766])


    Temperature_reduced = 1.0 - TemperatureK/T_Lambda
    friction_B_prime = np.tile(np.NaN,len(TemperatureK)) # array of NaN

    for tempindex in range(len(friction_B_prime)):
        if TemperatureK[tempindex] > 2.134:
            friction_B_prime[tempindex] = -0.34 * Temperature_reduced[tempindex]**-0.33 + 1.01
        else:
            friction_B_prime[tempindex] = 10.0**(interpolate.splev(np.log10(Temperature_reduced[tempindex]), (k, c, 3)))-15.0

    return friction_B_prime # Pa s


@validate_input_data_types
def friction_mutual_alpha(TemperatureK):
    """For a given temperature[K] array returns the helium-4 friction coefficient alpha = B *rho_n/2 rho.

    based on Russell J. Donnelly and Carlo F. Barenghi
    J. Phys. Chem. Ref. Data 27, 1217 (1998); http://dx.doi.org/10.1063/1.556028
    'The Observed Properties of Liquid Helium at the Saturated Vapor Pressure'
    """

    friction_B = friction_mutual_B(TemperatureK) # array of NaN

    return friction_B*density_normalfluid(TemperatureK)/(2.0*density(TemperatureK))


@validate_input_data_types
def friction_mutual_alpha(TemperatureK):
    """For a given temperature[K] array returns the helium-4 friction coefficient alpha = B *rho_n/2 rho.
    values not correct at high temperature???

    based on Russell J. Donnelly and Carlo F. Barenghi
    J. Phys. Chem. Ref. Data 27, 1217 (1998); http://dx.doi.org/10.1063/1.556028
    'The Observed Properties of Liquid Helium at the Saturated Vapor Pressure'
    """

    friction_B = friction_mutual_B(TemperatureK)
    density_normalfluid_arr = density_normalfluid(TemperatureK)
    density_arr = density(TemperatureK)

    return friction_B*density_normalfluid_arr/(2.0*density_arr)

@validate_input_data_types
def friction_mutual_alpha_prime(TemperatureK):
    """For a given temperature[K] array returns the helium-4 friction coefficient alpha = B_prime *rho_n/2 rho.
    values not correct at high temperature???

    based on Russell J. Donnelly and Carlo F. Barenghi
    J. Phys. Chem. Ref. Data 27, 1217 (1998); http://dx.doi.org/10.1063/1.556028
    'The Observed Properties of Liquid Helium at the Saturated Vapor Pressure'
    """
    # change float into numpy array for function len() to work

    friction_B_prime = friction_mutual_B_prime(TemperatureK)
    density_normalfluid_arr = density_normalfluid(TemperatureK)
    density_arr = density(TemperatureK)

    return friction_B_prime*density_normalfluid_arr/(2.0*density_arr)


@validate_input_data_types
def pressure_SVP(TemperatureK):
    """For a given temperature[K] array returns the helium-4 vapour pressure[Pa].

    based on Russell J. Donnelly and Carlo F. Barenghi
    J. Phys. Chem. Ref. Data 27, 1217 (1998); http://dx.doi.org/10.1063/1.556028
    'The Observed Properties of Liquid Helium at the Saturated Vapor Pressure'
    """

    k = np.array([0.5, 0.5, 0.5, 0.5, 0.52, 0.53, 0.56, 0.6, 0.65, 0.7, 0.75,
                  0.8, 0.85, 0.92, 1, 1.1, 1.25, 1.5, 1.7, 1.85, 2, 2.2, 2.5,
                  2.7, 3, 3.3, 3.7, 4.05, 4.5, 5.1, 5.1, 5.1, 5.1])

    c = np.array([0.00183797, 0.0025, 0.00376, 0.00624, 0.01236, 0.03306,
                   0.09339, 0.25856, 0.6269, 1.37025, 2.93305, 6.37318,
                   15.0042, 38.057, 116.050, 362.7, 990.9, 1886, 3194, 5569.5,
                   9279.1, 15370, 23480, 37355, 57050, 87170, 132825, 179650,
                   211567])

    PressurePa = np.tile(np.NaN,len(TemperatureK)) # array of NaN

    for tempindex in range(len(PressurePa)):
        PressurePa[tempindex] = interpolate.splev(TemperatureK[tempindex], (k, c, 3))

    return PressurePa # Pa


@validate_input_data_types
def temperature_from_pressure_SVP(PressurePa):
    """For a given (helium-4 vapour) pressure [Pa] returns corresponding temperature [K].
-    will return an array of temperature values for an 1D array of pressures
-
-    based on Russell J. Donnelly and Carlo F. Barenghi
-    J. Phys. Chem. Ref. Data 27, 1217 (1998); http://dx.doi.org/10.1063/1.556028
-    'The Observed Properties of Liquid Helium at the Saturated Vapor Pressure'
-    """

    Temperature = np.tile(np.NaN,len(PressurePa)) # array of NaN

    P_lambda = pressure_SVP(T_Lambda) # pressure at lambda point

    A_beltl = np.array([1.392408, 0.527153, 0.166756, 0.050988, 0.026514, 0.001975, -0.017976 ,0.005409, 0.013259, 0.0])
    B_beltl = 5.6
    C_beltl = 2.9

    A_abtl = np.array([3.146631,1.357655,0.413923,0.091159,0.016349,0.001826,
                       -0.004325,-0.004973,0.0,0.0])
    B_abtl = 10.3
    C_abtl = 1.9

    powers = np.arange(0, 10)

    for tempindex in range(len(PressurePa)):
        t_ab = (np.log(PressurePa[tempindex]) - B_abtl) / C_abtl
        t_bl = (np.log(PressurePa[tempindex]) - B_beltl) / C_beltl

        t_a = np.power(np.tile(t_ab,len(powers)), powers)
        t_b = np.power(np.tile(t_bl,len(powers)), powers)

        T_ab = np.tensordot(t_a, A_abtl, axes = 1)
        T_bl = np.tensordot(t_b, A_beltl, axes = 1)

        Temperature[tempindex] = np.where((PressurePa[tempindex] <= P_lambda), T_bl, T_ab)

    return Temperature # [K]


@validate_input_data_types
def sound_velocity_first(TemperatureK):
    """
    Calculates the first sound velocity of helium-4 for a given temperature

    :param TemperatureK: The temperature in Kelvin
    :return: The first sound velocity in m/s
    """
    k = np.array([0.0, 0.0, 0.0, 0.0, 0.5016077, 0.7021246, 1.002777, 1.804065, 2.004234, 2.154563, 2.169604, 2.1758,
                  2.1763, 2.17675, 2.176797, 2.176797, 2.176797, 2.17681, 2.17683, 2.17695, 2.1773, 2.178, 2.184625,
                  2.224629, 2.505156, 4.248035, 4.248035, 4.248035, 4.248035])
    c = np.array([2.3821, 2.382041, 2.382203, 2.382958, 2.375414, 2.36402, 2.270033, 2.221461, 2.187201, 2.18001,
                  2.176634, 2.175663, 2.174367, 2.173641, 2.175712, 2.176713, 2.178168, 2.179702, 2.181464, 2.187212,
                  2.193106, 2.216496, 2.238243, 2.056185, 1.795017]) * 1e2

    first_sound = np.zeros(len(TemperatureK))

    for temp_index in range(len(TemperatureK)):
        first_sound[temp_index] = interpolate.splev(TemperatureK[temp_index], (k, c, 3))

    return first_sound


def sound_velocity_first_from_pressure_low_temperature(pressurePa, temperatureK):
    '''
    Uses model to find the first sound velocity of helium-4 as function of pressure. From low temperature data in:
    Abraham B M, Eckstein Y, Ketterson J B, Kuchnir M and Roach P R 1970 Phys. Rev. A 1 (2) 50

    :param pressurePa: Pressure in Pa
    :param temperatureK: temperature in K
    :return: The first sound velocity in kg/m^3
    '''
    A1 = 5.60e2  # atm cm^3 / g
    A2 = 1.0970e4  # atm cm^6 / g
    A3 = 7.33e4  # atm cm^9 / g

    rho = density_from_pressure_low_temperature(pressurePa, temperatureK) * 1e-3
    rho_0 = float(density(temperatureK) * 1e-3)
    c = []
    if type(rho) is float:
        rho = np.array([rho])
    for r in rho.tolist():
        vel = A1 + 2 * A2 * (r - rho_0) + 3 * A3 * ((r - rho_0) ** 2)
        c.append(np.sqrt(1.01325e2 * vel))

    return np.array(c)

@validate_input_data_types
def sound_velocity_second(TemperatureK):
    """
    Calculates the second sound velocity of helium-4 for a given temperature

    :param TemperatureK: The temperature in Kelvin
    :return: The second sound velocity in m/s

	based on Donnelly and Barenghi
    """
    k = np.array([0.5517426, 0.5517426, 0.5517426, 0.5517426, 0.6419700, 0.8062300, 0.9297000, 1.0696400, 1.285500,
				  1.720200, 1.940430, 2.068850, 2.126220, 2.153079, 2.164207, 2.169182, 2.170551, 2.174832, 2.175738,
				  2.176319, 2.176600, 2.176735, 2.176780, 2.176787, 2.176797, 2.1768, 2.1768, 2.1768, 2.1768, 2.1768])
    c = np.array([1.050000e2, 9.287425e1, 4.973642e1, 2.712778e1, 1.931258e1, 1.795336e1, 1.923863e1, 2.115625e1,
				  1.921708e1, 1.575245e1, 1.192157e1, 8.953338, 6.771458, 5.506380, 4.456879, 3.840343, 2.542421,
				  1.957438, 1.427086, 1.001273, 6.743104e-2, 4.565972e-2, 3.519492e-2, 2.088277e-2, 1.059874e-2, 0.0])

    second_sound = np.zeros(len(TemperatureK))

    for temp_index in range(len(TemperatureK)):
        second_sound[temp_index] = interpolate.splev(TemperatureK[temp_index], (k, c, 3))

    return second_sound


@validate_input_data_types
def specific_heat_SVP(TemperatureK):
    """
    Calculates the specific heat of helium-4 for a given temperature at saturated vapour pressure

    :param TemperatureK: The temperature in Kelvin
    :return: The specific heat of helium-4 at SVP [J / kg K]

	based on Donnelly and Barenghi
    """

    # T < T_lambda
    k1 = np.array([1e-2, 1e-2, 1e-2, 1e-2, 2.49738990740e-2, 6.07641105990e-2, 6.36528653770e-2, 1.54157179076e-1,
                  2.49299135447e-1, 6.34497077644e-1, 9.04465380669e-1, 1.17474564385, 1.54255910867, 2.06499088720,
                  2.16070930746, 2.17157525125, 2.17590629200, 2.17655315900, 2.17672849900, 2.17678788100,
                  2.17679663200, 2.17679958200, 2.17679996468, 2.17679996468, 2.17679996468, 2.17679996468])
    c1 = np.array([-7.080029, -6.461204, -5.407887, -4.971098, -4.092648, -3.430290, -2.300493, -1.865551, -6.120157e-1,
                  1.800753e-1, 8.200057e-1, 1.231093, 1.494099, 1.633977, 1.694854, 1.760367, 1.813528, 1.849639,
                  1.900049, 1.935770, 1.974983, 2.017442])

    # T > T_lambda
    k2 = np.array([2.17680013200, 2.17680013200, 2.17680013200, 2.17680013200, 2.17680101200, 2.17680349500,
                   2.17681801800, 2.17692420600, 2.17750755600, 2.18086397100, 2.20374020794, 2.30368058865,
                   2.60849038128, 3.60335763894, 4.41358469422, 4.81404577042, 5.06143581698, 5.06143581698,
                   5.06143581698, 5.06143581698])
    c2 = np.array([1.900539, 1.862865, 1.822431, 1.759056, 1.679157, 1.584631, 1.464344, 1.274043, 1.095050,
                   9.546039e-1, 9.503531e-1, 1.100724, 1.260996, 1.421360, 1.609738, 1.732672])

    # Calc the base 10 log of the molar heat capacity at SVP
    log10_c = []
    for temp in np.nditer(TemperatureK):
        if temp < T_Lambda:
            log10_c.append(interpolate.splev(temp, (k1, c1, 3)))
        elif temp == T_Lambda:
            log10_c.append(np.NaN)
        else:
            log10_c.append(interpolate.splev(temp, (k2, c2, 3)))

    # Get the specific heat capacity in J / kg K
    c = np.power(10, np.array(log10_c)) / molar_mass

    return c

