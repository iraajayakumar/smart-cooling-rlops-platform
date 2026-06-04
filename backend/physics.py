WORKLOAD_HEAT_FACTOR = 0.3
COOLING_FACTOR = 1.0
ENERGY_PER_LEVEL = 2.0


def update_temperature(temp: float, workload: float, cooling: int) -> float:
    temp += workload * WORKLOAD_HEAT_FACTOR
    temp -= cooling * COOLING_FACTOR
    return temp


def update_energy(cumulative_energy: float, cooling: int) -> float:
    return cumulative_energy + cooling * ENERGY_PER_LEVEL