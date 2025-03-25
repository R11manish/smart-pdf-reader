from typing import Dict, Any, List
import json
import os
from datetime import datetime

class AgentStateManager:
    def __init__(self, state_dir: str = "agent_states"):
        self.state_dir = state_dir
        os.makedirs(state_dir, exist_ok=True)

    def save_state(self, session_id: str, context: List[Dict[str, Any]]) -> None:
        """Save agent state for a session."""
        state_file = os.path.join(self.state_dir, f"{session_id}.json")
        state = {
            "context": context,
            "last_updated": datetime.now().isoformat()
        }
        with open(state_file, "w") as f:
            json.dump(state, f)

    def load_state(self, session_id: str) -> List[Dict[str, Any]]:
        """Load agent state for a session."""
        state_file = os.path.join(self.state_dir, f"{session_id}.json")
        if os.path.exists(state_file):
            with open(state_file, "r") as f:
                state = json.load(f)
                return state.get("context", [])
        return []

    def clear_state(self, session_id: str) -> None:
        """Clear agent state for a session."""
        state_file = os.path.join(self.state_dir, f"{session_id}.json")
        if os.path.exists(state_file):
            os.remove(state_file) 