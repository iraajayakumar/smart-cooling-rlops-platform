from backend.physics import (
    update_temperature,
    update_energy,
    WORKLOAD_HEAT_FACTOR,
    COOLING_FACTOR,
    ENERGY_PER_LEVEL,
)


def test_physics_constants_match_rl():
    assert WORKLOAD_HEAT_FACTOR == 0.3
    assert COOLING_FACTOR == 1.0
    assert ENERGY_PER_LEVEL == 2.0


def test_update_temperature_formula():
    temp = 25.0
    workload = 4.0
    cooling = 1
    assert update_temperature(temp, workload, cooling) == temp + workload * 0.3 - cooling * 1.0


def test_update_energy_formula():
    energy = 0.0
    cooling = 2
    assert update_energy(energy, cooling) == energy + cooling * 2.0