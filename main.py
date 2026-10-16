"""
main.py - Runs the "WALL-E and EVE's Signal" simulation.

Usage:  python main.py [--config config.json] [--trace trace.json] [--no-grid]

Loop:  agent.decide_next_action(obs) -> env.step(action, target) -> log -> repeat
until EVE is found, the battery dies, or the step limit is hit. Every step is
written to trace.json (action, decision, reasoning, battery, observation,
belief summary) and a final mission report is printed.
"""

import argparse
import json

from agent import WallEAgent
from environment import Environment

STATUS_TEXT = {
    "eve_found": "SUCCESS - WALL-E reached EVE",
    "battery_depleted": "FAILURE - battery died",
    "timeout": "FAILURE - step limit reached",
    "running": "Interrupted",
}


def belief_snapshot(agent, include_grid):
    """Compact summary of the agent's belief for the trace."""
    p_map, mass3 = agent.get_confidence()
    snap = {
        "map_cell": list(agent.get_inferred_location()),
        "map_probability": round(p_map, 4),
        "mass_within_1_cell": round(mass3, 4),
        "entropy_bits": round(agent.get_entropy(), 4),
        "top3": agent.top_cells(3),
    }
    if include_grid:
        snap["grid"] = [[round(float(v), 4) for v in row] for row in agent.belief]
    return snap


def print_report(env, agent, status, steps):
    truth = env.reveal_ground_truth()
    inferred = agent.get_inferred_location()
    p_map, mass3 = agent.get_confidence()
    err = abs(inferred[0] - truth["eve_pos"][0]) + abs(inferred[1] - truth["eve_pos"][1])

    line = "=" * 64
    print(f"\n{line}\n  MISSION REPORT - WALL-E and EVE's Signal\n{line}")
    print(f"Outcome            : {STATUS_TEXT.get(status, status)}")
    print(f"Total steps        : {steps}")
    print(f"Battery remaining  : {env.battery}/{env.capacity}")
    print(f"Scans performed    : {len(agent.signal_log)}")

    print("\nEvidence found:")
    if agent.clues:
        for c in agent.clues.values():
            print(f"  - {c['id']} at {tuple(c['pos'])}: \"{c['text']}\"")
            print(f"      => inferred: {c['inference']}")
    else:
        print("  - (no text clues discovered)")
    if agent.signal_log:
        best = max(agent.signal_log, key=lambda s: s["strength"])
        print(f"  - Strongest signal: {best['strength']:.2f} at {tuple(best['pos'])} "
              f"(bearing {best['direction_deg']:.0f} deg, {agent._compass(best['direction_deg'])})")

    print(f"\nInferred location  : {inferred}")
    print(f"Confidence         : {p_map:.1%} exact cell  |  {mass3:.1%} within 1 cell")
    print(f"Ground truth (eval): EVE at {truth['eve_pos']}  -> "
          f"{'CORRECT' if err == 0 else f'off by {err} cells (Manhattan)'}")
    print(f"Recommended action : {agent.recommend_next_move()}")
    print(line)


def run(config_path, trace_path, include_grid):
    with open(config_path) as f:
        config = json.load(f)

    env = Environment(config)
    agent = WallEAgent(env.get_public_config())      # the agent never sees ground truth

    obs = env.reset()
    trace = []
    done, status = False, "running"

    try:
        while not done:
            battery_before = obs["battery"]
            action, target, reasoning = agent.decide_next_action(obs)
            obs_next, reward, done, info = env.step(action, target)
            status = info["status"]

            # Let the agent absorb the result now so the trace shows the updated belief.
            agent.ingest(obs_next)

            trace.append({
                "step": obs_next["step"],
                "decision": {"action": action, "target": list(target) if target else None},
                "reasoning": reasoning,
                "battery_before": battery_before,
                "battery_after": obs_next["battery"],
                "reward": reward,
                "observation_in": obs,               # what the decision was based on
                "observation_out": obs_next,         # what the action produced
                "belief": belief_snapshot(agent, include_grid),
            })
            print(f"[{obs_next['step']:>3}] {action:<8} -> {str(target):<8} "
                  f"battery {obs_next['battery']:>3}  | {reasoning.split('] ')[-1][:90]}")
            obs = obs_next
    finally:
        # Always write whatever we have, even if something crashed mid-run.
        with open(trace_path, "w") as f:
            json.dump({
                "mission": "WALL-E and EVE's Signal",
                "config": env.get_public_config(),
                "final_status": status,
                "steps": trace,
            }, f, indent=2)
        print(f"\nTrace written to {trace_path} ({len(trace)} steps)")

    print_report(env, agent, status, env.step_count)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--trace", default="trace.json")
    ap.add_argument("--no-grid", action="store_true", help="omit full belief grids from the trace")
    args = ap.parse_args()
    run(args.config, args.trace, not args.no_grid)
