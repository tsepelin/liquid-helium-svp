import numpy as np
import liquid_helium_svp as lhesvp

temperature_float = 4.2 #K
temperature_list = [4.2, 1.5]
temperature_nparray = np.array(temperature_list)

print(f'Liquid helium at {temperature_float} K has density {lhesvp.density(temperature_float):.2f} kg/m^3')
print(f'Liquid helium at temperatures {temperature_list} K has densities {lhesvp.density(temperature_list)} kg/m^3')
print(f'Liquid helium at temperatures {temperature_nparray} K has densities {lhesvp.density(temperature_nparray)} kg/m^3')

print(f'Liquid helium at {temperature_float} K has vapour pressure of {lhesvp.pressure_SVP(temperature_float):.2f} Pa')
print(f'Liquid helium vapour pressure {lhesvp.pressure_SVP(temperature_float):.2f} Pa correspond to temperature of {lhesvp.temperature_from_pressure_SVP(lhesvp.pressure_SVP(temperature_float)):.2f} K')
print(f'Liquid helium at {temperature_list} K has vapour pressure of {lhesvp.pressure_SVP(temperature_list)} Pa')