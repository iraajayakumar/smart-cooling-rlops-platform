import pytest
from pydantic import ValidationError
from backend.schemas import State, Action


def test_state_valid():
    s = State(temperature=25.0, workload=4.0, cooling=1)
    assert s.temperature == 25.0


def test_state_invalid_temperature():
    with pytest.raises(ValidationError):
        State(temperature=100.0, workload=4.0, cooling=1)


def test_action_valid():
    a = Action(action=2)
    assert a.action == 2


def test_action_invalid():
    with pytest.raises(ValidationError):
        Action(action=5)