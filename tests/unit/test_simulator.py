from backend.simulator import Simulator, TEMP_MIN, TEMP_MAX, WORKLOAD_MIN, WORKLOAD_MAX


def test_simulator_initial_state_ranges():
    sim = Simulator()
    state = sim.get_state()

    assert TEMP_MIN <= state["temperature"] <= TEMP_MAX
    assert WORKLOAD_MIN <= state["workload"] <= WORKLOAD_MAX
    assert state["cooling"] in {0, 1, 2}
    assert sim.energy >= 0.0


def test_simulator_step_updates_state_and_energy():
    sim = Simulator()
    initial_state = sim.get_state()
    initial_energy = sim.get_energy()

    action = 1  # medium cooling
    new_state = sim.step(action)

    # Cooling should be set to action
    assert new_state["cooling"] == action

    # Temperature stays within bounds
    assert TEMP_MIN <= new_state["temperature"] <= TEMP_MAX

    # Workload stays within bounds
    assert WORKLOAD_MIN <= new_state["workload"] <= WORKLOAD_MAX

    # Energy should increase or stay same (if cooling=0)
    assert sim.get_energy() >= initial_energy

    # Step count increments
    assert sim.step_count == 1