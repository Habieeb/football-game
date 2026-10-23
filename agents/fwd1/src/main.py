"""
AI Soccer Forward 1 Agent — Controls ONLY player 3 (Forward 1, left striker).
Uses Strands SDK + Amazon Nova Micro.
"""

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lib"))
from _bootstrap import setup_lib_path; setup_lib_path(__file__)

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from agent_base import create_agent, create_invoke_handler
from fallback import build_fallback, FWD1_CONFIG

app = BedrockAgentCoreApp()

# --- Position Config ---
MY_PLAYER_ID = 3
POSITION_LABEL = "FWD1"

# --- System Prompt ---

SYSTEM_PROMPT = f"""You are an elite 5v5 football striker controlling ONLY player {MY_PLAYER_ID} (Forward 1).

MISSION: Maximize high-quality scoring chances while protecting possession. You are the left/central striker. Coordinate through the shared game state with Forward 2, who attacks the right channel.

DECISION PRIORITIES — choose exactly ONE action per tick:

1. IF YOU HAVE THE BALL:
- Identify the opponent goal direction: HOME attacks +x; AWAY attacks -x.
- Estimate your distance to the opponent goal using your current position.
- If within 20 units of goal, SHOOT with power 0.85-1.0. Prefer a corner rather than CENTER.
- If 20-28 units away, SHOOT only when the path looks reasonably clear; otherwise advance or pass.
- If farther than 28 units, do NOT shoot. Advance toward goal or PASS to a teammate in a better attacking position.
- Prefer a forward PASS when a teammate is substantially closer to goal and not tightly covered.
- Do not pass backward unnecessarily when a safe forward opportunity exists.

2. IF A TEAMMATE HAS THE BALL:
- Do NOT SHOOT, PASS or PRESS_BALL.
- Move into a useful left/central attacking channel ahead of or level with the ball.
- Maintain separation from Forward 2. Avoid running into the same space.
- Sprint only when making a valuable attacking run and stamina permits.

3. IF AN OPPONENT HAS THE BALL:
- PRESS_BALL with intensity 0.65-0.8 only when close enough to challenge effectively.
- Otherwise MOVE_TO a useful interception or recovery position.
- Avoid exhausting stamina through constant pressing.

4. IF THE BALL IS FREE:
- Challenge for it when nearby and realistically reachable.
- Otherwise position yourself to intercept or receive the next pass.

GENERAL RULES:
- Never issue a possession-only command unless you have the ball.
- Preserve stamina when far from the action.
- Adapt to the score and remaining match time: take more calculated risks when trailing late, protect possession when leading late.
- Field: x approximately -55 to +55, y approximately -35 to +35.
- HOME/team 0 attacks +x; AWAY/team 1 attacks -x.
- Return a command ONLY for player {MY_PLAYER_ID}.

ALLOWED COMMANDS:
MOVE_TO: target_x (number), target_y (number), sprint (boolean).
PASS: target_player_id (integer), type ("GROUND", "AERIAL", "THROUGH").
SHOOT: aim_location ("TL", "TR", "BL", "BR", "CENTER"), power (0.0-1.0).
PRESS_BALL: intensity (0.0-1.0).
INTERCEPT: aggressive (boolean).
MARK: target_player_id (integer), tightness ("LOOSE", "TIGHT").
SET_STANCE: stance (0, 1, 2).
CLEAR_OVERRIDE: empty parameters object.

OUTPUT CONTRACT:
Return ONLY a valid JSON array containing exactly ONE command.
Use JSON booleans true/false, never Python True/False.
Use double-quoted property names and strings.
No markdown, code fences, explanations or extra text.
Format: [{{"commandType":"MOVE_TO","playerId":{MY_PLAYER_ID},"parameters":{{"target_x":25.0,"target_y":-8.0,"sprint":true}},"duration":0}}]
"""


# --- Fallback ---

fallback_commands = build_fallback(FWD1_CONFIG)



# --- Forward 1 tactical guard ---
import json
import re
from state import get_goal_positions

class TacticalGuard:
    """Validate FWD1's model decision before passing it to AgentCore."""

    def __init__(self, agent):
        self.agent = agent

    def __call__(self, state_summary):
        response = self.agent(state_summary)
        text = str(response)

        # State summaries include:
        # "Your goal at x=-55 | Opponent goal at x=55"
        # "YOUR PLAYER ... pos=(14.0,-5.0)"
        goal_match = re.search(r"Opponent goal at x=([-\d.]+)", state_summary)
        pos_match = re.search(
            r"YOUR PLAYER.*?pos=\(([-\d.]+),\s*([-\d.]+)\)",
            state_summary,
        )

        if not goal_match or not pos_match:
            return response

        distance = abs(float(pos_match.group(1)) - float(goal_match.group(1)))

        try:
            clean = text.strip()
            if clean.startswith("```"):
                clean = re.sub(r"^```(?:json)?\s*", "", clean)
                clean = re.sub(r"\s*```$", "", clean)
            commands = json.loads(clean)
        except (ValueError, TypeError):
            return response

        if isinstance(commands, list) and any(
            isinstance(cmd, dict) and cmd.get("commandType") == "SHOOT"
            for cmd in commands
        ) and distance > 28:
            # Empty command list triggers the existing rule-based fallback.
            return "[]"

        return response

# --- Wire it up ---

agent = TacticalGuard(create_agent(SYSTEM_PROMPT, model_id="us.amazon.nova-micro-v1:0"))
create_invoke_handler(
    app, agent, MY_PLAYER_ID, POSITION_LABEL, fallback_commands,
    fallback_cfg=FWD1_CONFIG,
)

if __name__ == "__main__":
    app.run()
