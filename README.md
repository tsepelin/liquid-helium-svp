# liquid-helium-svp

Python functions for properties of liquid helium-4 at saturated vapour pressure.

## Installation

```bash
python -m pip install liquid-helium-svp
```

## Usage

```python
import liquid_helium_svp as helium

density = helium.density(4.2)       # kg/m^3
pressure = helium.pressure_SVP(4.2) # Pa
```

Functions accept scalars, Python lists, and one-dimensional NumPy arrays where
applicable. Temperatures are in kelvin, pressures are in pascals, densities are
in kg/m^3, and velocities are in m/s unless stated otherwise.

## Available properties

The library describes helium-4 at saturated vapour pressure using fits and
interpolations based primarily on Donnelly and Barenghi (1998). The input
temperature is normally named `TemperatureK` and the input pressure
`PressurePa`.

| Function | Property | Result unit |
| --- | --- | --- |
| `density(T)` | Total liquid density | kg/m^3 |
| `density_superfluid(T)` | Superfluid component density | kg/m^3 |
| `density_normalfluid(T)` | Normal-fluid component density | kg/m^3 |
| `viscosity(T)` | Dynamic viscosity | Pa s |
| `pressure_SVP(T)` | Saturated vapour pressure | Pa |
| `temperature_from_pressure_SVP(P)` | Temperature corresponding to saturated vapour pressure | K |
| `surface_tension(T)` | Surface tension | N/m |
| `ion_mobilities(T)` | Ion mobility fit | See source data |
| `dispersion(T)` | Dispersion fit | See source data |
| `structure_factor(T)` | Structure factor fit | See source data |
| `enthalpy(T)` | Specific enthalpy fit | See source data |
| `specific_heat_SVP(T)` | Specific heat at saturated vapour pressure | J/(kg K) |
| `sound_velocity_first(T)` | First-sound velocity | m/s |
| `sound_velocity_second(T)` | Second-sound velocity | m/s |
| `sound_velocity_fourth(T)` | Fourth-sound velocity | m/s |
| `friction_mutual_B(T)` | Mutual-friction coefficient B | Dimensionless |
| `friction_mutual_B_prime(T)` | Mutual-friction coefficient B' | Dimensionless |
| `friction_mutual_alpha(T)` | Mutual-friction coefficient alpha | Dimensionless |
| `friction_mutual_alpha_prime(T)` | Mutual-friction coefficient alpha' | Dimensionless |

For low-temperature pressure corrections, the following functions accept both
pressure and temperature arrays. They are based on Abraham et al. (1970):

| Function | Property | Result unit |
| --- | --- | --- |
| `density_from_pressure_low_temperature(P, T)` | Density corrected for pressure | kg/m^3 |
| `sound_velocity_first_from_pressure_low_temperature(P, T)` | First-sound velocity corrected for pressure | m/s |

The fits are intended for the temperature and pressure ranges covered by the
source data. Values outside those ranges may be extrapolated by SciPy and
should be treated with caution.

The module is based on publication by:

Russell J. Donnelly and Carlo F. Barenghi
J. Phys. Chem. Ref. Data 27, 1217 (1998); http://dx.doi.org/10.1063/1.556028
"The Observed Properties of Liquid Helium at the Saturated Vapor Pressure"


In addition to the replication of the publication, the module has a function to infer temperature of liquid helium from saturated vapour pressure. 

```python
import liquid_helium_svp as helium

temperature = 4.2  # K

pressure = helium.pressure_SVP(temperature)
recovered_temperature = helium.temperature_from_pressure_SVP(pressure)

print(f'Liquid helium at {temperature} K has vapour pressure of {pressure:.2f} Pa')
print(f'{pressure:.2f} Pa corresponds to {recovered_temperature:.2f} K')
```


## License

MIT



