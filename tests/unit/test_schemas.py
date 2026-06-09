import pytest
from pydantic import ValidationError

from backend.schemas import State, Action


def test_state_valid():
    s = State(temperature=25.0, workload=4.0, cooling=1)
    assert s.temperature == 25.0
    assert s.workload == 4.0
    assert s.cooling == 1


def test_state_invalid_temperature_high():
    with pytest.raises(ValidationError):
        State(temperature=100.0, workload=4.0, cooling=1)


def test_state_invalid_temperature_low():
    with pytest.raises(ValidationError):
        State(temperature=0.0, workload=4.0, cooling=1)


def test_state_invalid_workload():
    with pytest.raises(ValidationError):
        State(temperature=25.0, workload=20.0, cooling=1)


def test_state_invalid_cooling():
    with pytest.raises(ValidationError):
        State(temperature=25.0, workload=4.0, cooling=5)


def test_action_valid():
    a = Action(action=2)
    assert a.action == 2


def test_action_invalid():
    with pytest.raises(ValidationError):
        Action(action=3)