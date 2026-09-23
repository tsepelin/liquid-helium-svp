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
applicable. See the function docstrings for units and source-data limitations.

The module is based on publication by:

Russell J. Donnelly and Carlo F. Barenghi
J. Phys. Chem. Ref. Data 27, 1217 (1998); http://dx.doi.org/10.1063/1.556028
"The Observed Properties of Liquid Helium at the Saturated Vapor Pressure"


In addition to the replication of the publication, the module has a function to infer temperature of liquid helium from saturated vapour pressure. 

```python
import liquid_helium_svp as helium

temperature = 4.2 #K

print(f'Liquid helium at {temperature_float} K has vapour pressure of {lhesvp.pressure_SVP(temperature):.2f} Pa')
print(f'Liquid helium vapour pressure {lhesvp.pressure_SVP(temperature):.2f} Pa correspond to temperature of {lhesvp.temperature_from_pressure_SVP(lhesvp.pressure_SVP(temperature)):.2f} K')
```


## License

MIT



