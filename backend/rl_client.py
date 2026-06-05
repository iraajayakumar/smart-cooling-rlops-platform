from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
RL_ENGINE_PATH = ROOT / "rl_engine"
if str(RL_ENGINE_PATH) not in sys.path:
    sys.path.insert(0, str(RL_ENGINE_PATH))

from inference import predict_action as rl_predict_action


def predict_action(state):
    return rl_predict_action(state)