import numpy as np

import liquid_helium_svp as helium


def test_root_api_and_reference_density():
    assert np.isclose(helium.density(4.2), 125.4075, atol=1e-4)
    assert np.isclose(helium.pressure_SVP(4.2), 99232.6573, atol=1e-3)


def test_array_inputs():
    temperatures = np.array([1.0, 1.5, 2.0])

    result = helium.density(temperatures)

    assert result.shape == temperatures.shape
    assert np.all(np.isfinite(result))


def test_pressure_dependent_array_inputs():
    temperatures = np.array([1.0, 1.5, 2.0])
    pressures = helium.pressure_SVP(temperatures)

    densities = helium.density_from_pressure_low_temperature(pressures, temperatures)
    velocities = helium.sound_velocity_first_from_pressure_low_temperature(
        pressures, temperatures
    )

    assert densities.shape == temperatures.shape
    assert velocities.shape == temperatures.shape
    assert np.all(np.isfinite(densities))
    assert np.all(np.isfinite(velocities))