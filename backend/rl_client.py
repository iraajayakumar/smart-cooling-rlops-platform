from pathlib import Path
import sys
from typing import Any, Dict

_ROOT = Path(__file__).resolve().parent.parent
_RL_ENGINE_PATH = _ROOT / "rl_engine"
if str(_RL_ENGINE_PATH) not in sys.path:
    sys.path.insert(0, str(_RL_ENGINE_PATH))

# lazy singleton
_rl_predict_action = None


def predict_action(state: Dict[str, Any]) -> int:
    """
    Lightweight wrapper for RL inference.

    Lazily imports rl_engine.inference.predict_action on first use
    so that FastAPI can start without loading the whole RL stack.
    """
    global _rl_predict_action
    if _rl_predict_action is None:
        from rl_engine.inference import predict_action as rl_predict_action  # type: ignore
        _rl_predict_action = rl_predict_action

    return int(_rl_predict_action(state))