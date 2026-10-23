"""Base agent factory for AI soccer position agents."""

import json
from typing import Callable
from strands import Agent
from strands.models import BedrockModel

from parsing import parse_commands
from state import summarize_state
from fallback import FallbackConfig, build_last_resort


def create_agent(system_prompt: str, model_id: str = "us.amazon.nova-micro-v1:0") -> Agent:
    """Create a Strands Agent with the given system prompt."""
    model = BedrockModel(model_id=model_id)
    return Agent(model=model, system_prompt=system_prompt)


def parse_prompt(payload) -> dict:
    """Parse the invocation payload's prompt, accepting either str or dict.

    The main path has always accepted both, but the error handler used to
    re-parse with a bare json.loads(), which raises TypeError on a dict. That
    exception was caught by the outer bare except, so the rule-based fallback
    was skipped entirely and control fell through to the last resort.
    """
    prompt = payload.get("prompt", "{}")
    if isinstance(prompt, dict):
        return prompt
    try:
        parsed = json.loads(prompt)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def team_id_from(prompt_data: dict, log, label: str) -> int:
    """teamId from the payload; HOME only as a loudly logged last resort.

    teamId is authoritative input, not a guessable default. Silently
    defaulting to 0 makes an away agent emit commands tagged HOME, which the
    server attributes to the opposing team — and leaves nothing in the logs
    to explain why the away side played as though it were the home side.
    """
    team_id = prompt_data.get("teamId")
    if team_id is None:
        log.error(
            f"{label}: payload contained no teamId; falling back to HOME (0). "
            "Commands this tick may be attributed to the wrong team."
        )
        return 0
    return team_id


def create_invoke_handler(
    app,
    agent: Agent,
    my_player_id: int,
    position_label: str,
    fallback_fn: Callable[[dict, int, int], list[dict]],
    fallback_cfg: FallbackConfig,
):
    """Create and register the @app.entrypoint invoke handler.

    Three layers of error handling, from best to worst:
      1. LLM response → parse into commands
      2. fallback_fn(game_state, team_id, my_player_id) → rule-based commands
      3. last-resort command from fallback_cfg → single safe command
    """
    log = app.logger
    last_resort = build_last_resort(fallback_cfg, my_player_id)

    @app.entrypoint
    async def invoke(payload, context):
        # Parsed once, outside the try, so the error handler below can reuse
        # it instead of re-parsing and throwing a second exception.
        prompt_data = parse_prompt(payload)
        team_id = team_id_from(prompt_data, log, position_label)
        try:
            game_state = prompt_data.get("gameState", {})

            # Honor myPlayers from payload if present, otherwise use configured player ID
            my_players = prompt_data.get("myPlayers", [my_player_id])
            effective_pid = my_players[0] if my_players else my_player_id


            # GALE_ARCS_FINAL_TEAM_TACTICS
            # Deterministic coordinated play: possession, spacing, and defense.
            if position_label != "GK":
                from state import (
                    holder_is_me, get_goal_positions, get_possession_info,
                    _player_idx, _is_my_team, dist,
                )
                players = game_state.get("players", [])
                ball = game_state.get("ball", {})
                bp = ball.get("position") or {}
                me = next(
                    (p for p in players if _is_my_team(p, team_id)
                     and _player_idx(p) == effective_pid), None
                )
                if me:
                    pos = me.get("position") or {}
                    x, y = float(pos.get("x", 0)), float(pos.get("y", 0))
                    bx, by = float(bp.get("x", 0)), float(bp.get("y", 0))
                    own_goal, enemy_goal = get_goal_positions(team_id)
                    direction = 1 if enemy_goal > own_goal else -1
                    progress = lambda px: direction * px
                    friends = [
                        p for p in players if _is_my_team(p, team_id)
                        and _player_idx(p) != effective_pid
                    ]
                    enemies = [p for p in players if not _is_my_team(p, team_id)]
                    has_ball = holder_is_me(ball, players, team_id, effective_pid)
                    _, status, ours = get_possession_info(ball, players, team_id)

                    def distance(a, b):
                        return dist(a, b)

                    def command(kind, params, duration=0):
                        return {
                            "commandType": kind, "playerId": effective_pid,
                            "teamId": team_id, "parameters": params,
                            "duration": duration,
                        }

                    def emit(kind, params, duration=0):
                        return json.dumps([command(kind, params, duration)])

                    def open_space(p):
                        return min(
                            (distance(p, q.get("position") or {}) for q in enemies),
                            default=100,
                        )

                    # Ball carrier: score from shooting range, otherwise
                    # prioritize progressive passes and advancing.
                    if has_ball:
                        goal_distance = abs(enemy_goal - x)
                        nearby = min(
                            (distance(pos, q.get("position") or {}) for q in enemies),
                            default=100,
                        )
                        if goal_distance <= 23 and abs(y) <= 15:
                            aim = "BR" if y >= 0 else "BL"
                            yield emit("SHOOT", {
                                "aim_location": aim, "power": 0.83,
                            })
                            return

                        options = []
                        for mate in friends:
                            mid = _player_idx(mate)
                            if mid == 0:
                                continue
                            mp = mate.get("position") or {}
                            gain = (float(mp.get("x", 0)) - x) * direction
                            length = distance(pos, mp)
                            space = open_space(mp)
                            if gain >= 2 and 3 <= length <= 42:
                                score = gain * 1.5 + min(space, 15) - length * 0.15
                                options.append((score, mid))
                        if options:
                            target = max(options)[1]
                            yield emit("PASS", {
                                "target_player_id": target, "type": "GROUND",
                            })
                            return

                        # Carry forward when no useful pass exists.
                        target_x = max(-48, min(48, x + direction * 13))
                        target_y = max(-23, min(23, y * 0.75))
                        yield emit("MOVE_TO", {
                            "target_x": target_x, "target_y": target_y,
                            "sprint": nearby > 5,
                        })
                        return

                    # No ball: assign one challenger, others keep their shape.
                    field_friends = [
                        p for p in players if _is_my_team(p, team_id)
                        and _player_idx(p) != 0
                    ]
                    closest = min(
                        field_friends,
                        key=lambda p: distance(p.get("position") or {}, bp),
                        default=None,
                    )
                    challenger = (
                        closest is not None
                        and _player_idx(closest) == effective_pid
                    )

                    if not ours and challenger and distance(pos, bp) < 19:
                        yield emit("PRESS_BALL", {"intensity": 0.85}, 3)
                        return

                    # Maintain lanes and push the formation forward with possession.
                    anchor = {
                        "DEF": (-23, 0),
                        "MID": (0, 0),
                        "FWD1": (23, -12),
                        "FWD2": (23, 12),
                    }
                    ax, ay = anchor.get(position_label, (0, 0))
                    if ours:
                        base_x = max(ax, min(40, progress(bx) + {
                            "DEF": -24, "MID": 4,
                            "FWD1": 14, "FWD2": 14,
                        }.get(position_label, 0)))
                    else:
                        base_x = max(ax - 8, min(ax + 9, progress(bx) - 12))

                    target_x = max(-48, min(48, base_x * direction))
                    target_y = ay if position_label != "MID" else by * 0.3

                    if not ours and position_label == "DEF":
                        threats = [
                            p for p in enemies
                            if abs(float(p.get("position", {}).get("x", 0))
                                   - own_goal) < 35
                        ]
                        if threats:
                            threat = min(
                                threats,
                                key=lambda p: abs(
                                    float(p.get("position", {}).get("x", 0))
                                    - own_goal
                                ),
                            )
                            yield emit("MARK", {
                                "target_player_id": _player_idx(threat),
                                "tightness": "TIGHT",
                            }, 3)
                            return

                    yield emit("MOVE_TO", {
                        "target_x": target_x,
                        "target_y": max(-24, min(24, target_y)),
                        "sprint": ours,
                    })
                    return

            # GALE_ARCS_GK_DEFENSIVE_OVERRIDE
            # Keep GK0 goal-side when an opponent advances.
            # Do not override possession or an incoming loose-ball shot.
            if position_label == "GK":
                from state import (
                    get_goal_positions,
                    get_possession_info,
                    holder_is_me,
                )

                ball = game_state.get("ball", {})
                players = game_state.get("players", [])
                ball_pos = ball.get("position") or {}
                velocity = ball.get("velocity") or {}
                my_goal_x, opp_goal_x = get_goal_positions(team_id)
                direction = 1 if opp_goal_x > my_goal_x else -1

                if holder_is_me(ball, players, team_id, effective_pid):
                    commands = fallback_fn(game_state, team_id, effective_pid)
                    log.info("GK possession: distribute ball")
                    yield json.dumps(commands)
                    return

                if not holder_is_me(ball, players, team_id, effective_pid):
                    _, status, we_have_ball = get_possession_info(
                        ball, players, team_id
                    )
                    bx = float(ball_pos.get("x", 0) or 0)
                    vx = float(velocity.get("x", 0) or 0)
                    in_defensive_half = (bx - my_goal_x) * direction < 55
                    incoming_loose_ball = (
                        status == "free" and vx * direction < -0.5
                    )

                    if (
                        status.startswith("OPP")
                        and not we_have_ball
                        and in_defensive_half
                        and not incoming_loose_ball
                    ):
                        commands = fallback_fn(
                            game_state, team_id, effective_pid
                        )
                        log.info("GK defensive positioning override")
                        yield json.dumps(commands)
                        return

            # GALE_ARCS_SMART_SHOOTING
            # Prioritize clear chances, one-on-ones, and open goals.
            if position_label in ("FWD1", "FWD2", "MID"):
                from state import holder_is_me, get_goal_positions, _player_idx, _is_my_team
                import math

                players = game_state.get("players", [])
                ball = game_state.get("ball", {})
                me = next(
                    (p for p in players
                     if _player_idx(p) == effective_pid
                     and _is_my_team(p, team_id)),
                    None,
                )

                if me and holder_is_me(ball, players, team_id, effective_pid):
                    _, goal_x = get_goal_positions(team_id)
                    pos = me.get("position", {})
                    x = float(pos.get("x", 0))
                    y = float(pos.get("y", 0))
                    goal_distance = abs(goal_x - x)
                    direction = 1 if goal_x > x else -1

                    opponents = [
                        p for p in players if not _is_my_team(p, team_id)
                    ]
                    keeper = next(
                        (p for p in opponents if _player_idx(p) == 0),
                        None,
                    )
                    defenders = [
                        p for p in opponents if _player_idx(p) != 0
                    ]

                    # Detect opponents directly between player and goal.
                    blockers = 0
                    for defender in defenders:
                        dp = defender.get("position", {})
                        dx = float(dp.get("x", 0)) - x
                        dy = float(dp.get("y", 0)) - y
                        if 0 < dx * direction < goal_distance and abs(dy) < 7:
                            blockers += 1

                    # A keeper far from the goal leaves it exposed.
                    keeper_out = False
                    keeper_y = 0.0
                    if keeper:
                        kp = keeper.get("position", {})
                        keeper_x = float(kp.get("x", goal_x))
                        keeper_y = float(kp.get("y", 0))
                        keeper_out = (
                            abs(goal_x - keeper_x) > 12
                            or abs(keeper_y) > 12
                        )

                    close_chance = goal_distance <= 15 and abs(y) <= 14
                    one_on_one = (
                        goal_distance <= 28
                        and abs(y) <= 12
                        and blockers == 0
                    )
                    open_goal = (
                        goal_distance <= 35
                        and abs(y) <= 14
                        and keeper_out
                        and blockers == 0
                    )

                    # GALE_ARCS_FORWARD_PASS_UPGRADE
                    # Pass forward under pressure, unless a clear shot exists.
                    pressure = min(
                        (
                            math.hypot(
                                float(p.get("position", {}).get("x", 0)) - x,
                                float(p.get("position", {}).get("y", 0)) - y,
                            )
                            for p in opponents
                        ),
                        default=100.0,
                    )

                    if pressure <= 7.0 and not (
                        close_chance or one_on_one or open_goal
                    ):
                        candidates = []

                        for teammate in players:
                            if not _is_my_team(teammate, team_id):
                                continue

                            teammate_id = _player_idx(teammate)
                            if teammate_id in (None, 0, effective_pid):
                                continue

                            tp = teammate.get("position", {})
                            tx = float(tp.get("x", 0))
                            ty = float(tp.get("y", 0))

                            forward_progress = (tx - x) * direction
                            pass_distance = math.hypot(tx - x, ty - y)

                            if forward_progress < 3 or not (4 <= pass_distance <= 30):
                                continue

                            receiver_space = min(
                                (
                                    math.hypot(
                                        float(o.get("position", {}).get("x", 0)) - tx,
                                        float(o.get("position", {}).get("y", 0)) - ty,
                                    )
                                    for o in opponents
                                ),
                                default=100.0,
                            )

                            if receiver_space < 8:
                                continue

                            candidates.append(
                                (forward_progress, receiver_space, -pass_distance, teammate_id)
                            )

                        if candidates:
                            _, _, _, target_id = max(candidates)
                            command = {
                                "commandType": "PASS",
                                "playerId": effective_pid,
                                "teamId": team_id,
                                "parameters": {
                                    "target_player_id": target_id,
                                    "type": "GROUND",
                                },
                                "duration": 0,
                            }
                            log.info(
                                f"{position_label} FORWARD PASS: "
                                f"target={target_id}, pressure={pressure:.1f}"
                            )
                            yield json.dumps([command])
                            return

                    should_shoot = (
                        close_chance or one_on_one or open_goal
                    )

                    if should_shoot:
                        # Favor a central target for an exposed goal.
                        if keeper_out:
                            aim = "CENTER"
                        elif keeper_y >= 0:
                            aim = "BR"
                        else:
                            aim = "BL"

                        power = 0.78 if goal_distance <= 15 else 0.88

                        command = {
                            "commandType": "SHOOT",
                            "playerId": effective_pid,
                            "teamId": team_id,
                            "parameters": {
                                "aim_location": aim,
                                "power": power,
                            },
                            "duration": 0,
                        }
                        log.info(
                            f"{position_label} SMART SHOOT: "
                            f"distance={goal_distance:.1f}, "
                            f"blockers={blockers}, "
                            f"keeper_out={keeper_out}, "
                            f"aim={aim}"
                        )
                        yield json.dumps([command])
                        return

            state_summary = summarize_state(
                game_state, team_id, effective_pid, position_label
            )
            log.info(f"{position_label} agent invoked for team {team_id}, controlling player {effective_pid}")

            response = agent(state_summary)
            response_text = str(response)

            def on_recovered(raw: str) -> None:
                # The model wrote Python-flavoured JSON (usually `True`/`False`/`None`).
                # We recovered it rather than dropping the command and falling back —
                # logged so you can see how often your model does this.
                log.warn(f"{position_label} recovered malformed JSON from the model: {raw[:200]}")

            commands = parse_commands(response_text, team_id, effective_pid, on_recovered)

            if commands:
                log.info(f"LLM returned {len(commands)} commands: "
                         f"{[c.get('commandType') for c in commands]}")
                yield json.dumps(commands)
            else:
                log.warn(f"LLM parse failed, using fallback. Response: {response_text[:200]}")
                commands = fallback_fn(game_state, team_id, effective_pid)
                log.info(f"Fallback returned {len(commands)} commands")
                yield json.dumps(commands)

        except Exception as e:
            log.error(f"{position_label} agent error: {e}")
            try:
                # Reuse the payload parsed at entry. Re-parsing here is what
                # broke this path: json.loads() on an already-decoded dict
                # raises TypeError, the bare except below swallowed it, and
                # the rule-based fallback never ran.
                my_players = prompt_data.get("myPlayers", [my_player_id])
                effective_pid = my_players[0] if my_players else my_player_id
                commands = fallback_fn(
                    prompt_data.get("gameState", {}),
                    team_id,
                    effective_pid,
                )
                yield json.dumps(commands)
            except Exception as fallback_error:
                log.error(f"{position_label} fallback also failed: {fallback_error}")
                cmd = dict(last_resort)
                # team_id came from the payload at entry, so the last resort
                # is no longer hardcoded to HOME.
                cmd["teamId"] = team_id
                yield json.dumps([cmd])

    return invoke
