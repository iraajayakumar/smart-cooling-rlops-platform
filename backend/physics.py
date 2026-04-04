def update_temperature(temp, workload, cooling):
    temp += workload * 0.4
    temp -= cooling * 0.6
    return temp


def update_energy(energy, cooling):
    energy += cooling * 2
    return energy
