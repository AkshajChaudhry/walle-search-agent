"""
environment.py - Simulated, partially observable world for "WALL-E and EVE's Signal".

The Environment owns the GROUND TRUTH (EVE's position, clue locations, noise
generator). The agent never sees it; it only receives noisy observations
returned by `step()`.

Coordinate convention: positions are (row, col). Row 0 is NORTH, col 0 is WEST.
Bearing convention:    0 deg = EAST, 90 deg = NORTH, 180 = WEST, 270 = SOUTH.
"""

import copy
import math
import random


class Environment:
    """Gym-style environment: reset() -> obs, step(action, target_pos) -> (obs, reward, done, info)."""

    ACTIONS = ("move", "scan", "recharge")

    def __init__(self, config):
        # ---- public / structural parameters -------------------------------
        self._config = copy.deepcopy(config)
        self.rows, self.cols = config["grid_size"]
        self.start_pos = tuple(config["start_pos"])
        self.max_steps = int(config["max_steps"])

        batt = config["battery"]
        self.capacity = batt["capacity"]
        self.costs = dict(batt["costs"])            # move / scan / recharge
        self.recharge_gain = batt["recharge_gain"]

        self.stations = {tuple(p) for p in config["recharge_stations"]}
        self.obstacles = {tuple(p) for p in config["obstacles"]}
        self.sensor = dict(config["sensor_model"])

        # ---- hidden ground truth (never exposed through public API) -------
        gt = config["ground_truth"]
        self._eve_pos = tuple(gt["eve_pos"])
        self._clues = [
            {"id": c["id"], "pos": tuple(c["pos"]), "text": c["text"]}
            for c in gt["clues"]
        ]
        self._rng = random.Random(config.get("seed"))

        self._validate()
        self.reset()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _validate(self):
        """Fail early on an inconsistent config."""
        if not self._in_bounds(self.start_pos) or self.start_pos in self.obstacles:
            raise ValueError("start_pos is out of bounds or on an obstacle")
        if not self._in_bounds(self._eve_pos) or self._eve_pos in self.obstacles:
            raise ValueError("EVE's position is out of bounds or on an obstacle")
        for s in self.stations:
            if not self._in_bounds(s) or s in self.obstacles:
                raise ValueError(f"recharge station {s} invalid")

    def _in_bounds(self, p):
        return 0 <= p[0] < self.rows and 0 <= p[1] < self.cols

    def get_public_config(self):
        """Everything the agent is allowed to know in advance (no ground truth)."""
        c = self._config
        return copy.deepcopy({
            "grid_size": c["grid_size"],
            "start_pos": c["start_pos"],
            "max_steps": c["max_steps"],
            "battery": c["battery"],
            "recharge_stations": c["recharge_stations"],
            "obstacles": c["obstacles"],
            "sensor_model": c["sensor_model"],
        })

    def reveal_ground_truth(self):
        """Post-mission evaluation only (used by main.py for the final report)."""
        return {"eve_pos": self._eve_pos, "clues": copy.deepcopy(self._clues)}

    # ------------------------------------------------------------------
    # Gym-style API
    # ------------------------------------------------------------------
    def reset(self):
        self.pos = self.start_pos
        self.battery = self.capacity
        self.step_count = 0
        self.done = False
        self.status = "running"
        self._discovered = set()
        return self._make_obs("init", True, "Mission start.", None, [])

    def step(self, action, target_pos=None):
        """
        Apply an action. Every action costs battery.
          move     target_pos = adjacent (4-neighbour) free cell
          scan     target_pos ignored; scans from the current cell
          recharge target_pos ignored; must be standing on a station
        Returns (obs, reward, done, info).
        """
        if self.done:
            raise RuntimeError("Episode finished; call reset().")

        self.step_count += 1
        valid, message = True, ""
        signal, new_clues = None, []

        # Pay the action cost up front (unknown actions cost a 'move').
        cost = self.costs.get(action, self.costs["move"])
        self.battery -= cost

        if action == "move":
            valid, message = self._try_move(target_pos)
            if valid:
                new_clues = self._collect_clues(radius=0)   # stepping on a clue reveals it
        elif action == "scan":
            signal = self._read_signal()
            new_clues = self._collect_clues(radius=self.sensor["clue_scan_radius"])
            message = "Scan complete."
        elif action == "recharge":
            if self.pos in self.stations:
                self.battery += self.recharge_gain
                message = "Recharged at station."
            else:
                valid, message = False, "Not on a recharge station; battery wasted."
        else:
            valid, message = False, f"Unknown action '{action}'."

        self.battery = max(0, min(self.capacity, self.battery))

        # ---- termination checks ------------------------------------------
        found = self.pos == self._eve_pos
        if found:
            self.done, self.status = True, "eve_found"
        elif self.battery <= 0:
            self.done, self.status = True, "battery_depleted"
        elif self.step_count >= self.max_steps:
            self.done, self.status = True, "timeout"

        reward = -float(cost) + (100.0 if found else 0.0)
        obs = self._make_obs(action, valid, message, signal, new_clues)
        info = {"status": self.status, "cost": cost}
        return obs, reward, self.done, info

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _try_move(self, target):
        if target is None:
            return False, "No target given for move."
        t = tuple(target)
        if abs(t[0] - self.pos[0]) + abs(t[1] - self.pos[1]) != 1:
            return False, f"Target {t} is not adjacent to {self.pos}."
        if not self._in_bounds(t):
            return False, f"Target {t} is outside the grid."
        if t in self.obstacles:
            return False, f"Target {t} is blocked by an obstacle."
        self.pos = t
        return True, f"Moved to {t}."

    def _collect_clues(self, radius):
        """Reveal not-yet-discovered clues within Chebyshev `radius` of the robot."""
        found = []
        for c in self._clues:
            if c["id"] in self._discovered:
                continue
            if max(abs(c["pos"][0] - self.pos[0]), abs(c["pos"][1] - self.pos[1])) <= radius:
                self._discovered.add(c["id"])
                found.append({"id": c["id"], "pos": list(c["pos"]), "text": c["text"]})
        return found

    def _read_signal(self):
        """
        Noisy beacon reading from the current cell:
          strength  = exp(-d / decay) + N(0, sigma_s)   (higher = closer)
          direction = bearing to EVE (deg) + N(0, sigma_dir)
          visual_contact = EVE is within a very short range (no noise)
        """
        s = self.sensor
        dx = self._eve_pos[1] - self.pos[1]          # east is positive
        dy = self.pos[0] - self._eve_pos[0]          # north is positive
        d = math.hypot(dx, dy)

        strength = math.exp(-d / s["signal_decay"]) + self._rng.gauss(0, s["strength_noise_std"])
        bearing = math.degrees(math.atan2(dy, dx)) + self._rng.gauss(0, s["direction_noise_std_deg"])
        return {
            "strength": round(strength, 4),
            "direction_deg": round(bearing % 360.0, 2),
            "visual_contact": d <= s["visual_contact_radius"],
        }

    def _make_obs(self, action, valid, message, signal, new_clues):
        return {
            "step": self.step_count,
            "action": action,
            "valid": valid,
            "message": message,
            "position": list(self.pos),
            "battery": self.battery,
            "signal": signal,                 # None unless the action was a scan
            "new_clues": new_clues,
            "found_eve": self.pos == self._eve_pos,
        }
