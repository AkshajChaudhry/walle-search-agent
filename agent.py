"""
agent.py - WALL-E: a Bayesian search-and-resource-planning agent.

The agent keeps a belief grid P(EVE is at cell) and updates it with
  * Bayes' rule for noisy signal strength (Gaussian likelihood on distance),
  * Bayes' rule for noisy bearing readings (von-Mises-style likelihood),
  * "visual contact" / negative evidence from scans and visited cells,
  * rule-based likelihoods for text clues (direction and proximity rules).

Battery safety rule: after any action the agent must still have enough charge
to walk to the nearest recharge station plus a safety margin. If it doesn't,
it routes to the station and recharges.

Coordinates: (row, col); row 0 = north. Bearings: 0 deg = east, 90 deg = north.
"""

import math
import re
from collections import deque

import numpy as np

INF = 10 ** 6


class WallEAgent:
    SAFETY_MARGIN = 5        # spare battery kept on top of the return trip
    RECHARGE_FRAC = 0.6      # at a station, top up when below this fraction
    COMMIT_MASS = 0.6        # commit to the best cell when 3x3 mass exceeds this
    LIK_FLOOR = 0.01         # outlier floor so one bad reading can't kill a cell

    def __init__(self, public_config):
        pc = public_config
        self.rows, self.cols = pc["grid_size"]
        self.pos = tuple(pc["start_pos"])
        batt = pc["battery"]
        self.capacity = batt["capacity"]
        self.costs = batt["costs"]
        self.recharge_gain = batt["recharge_gain"]
        self.battery = self.capacity
        self.stations = [tuple(p) for p in pc["recharge_stations"]]
        self.obstacles = {tuple(p) for p in pc["obstacles"]}
        self.sensor = pc["sensor_model"]

        # Belief grid: uniform over free cells.
        self.rr, self.cc = np.indices((self.rows, self.cols))
        self.free_mask = np.ones((self.rows, self.cols), dtype=bool)
        for (r, c) in self.obstacles:
            self.free_mask[r, c] = False
        self.belief = self.free_mask.astype(float)
        self._normalize()

        self.station_dist = self._compute_station_dist()

        # Memory
        self.scanned = set()
        self.clues = {}               # id -> clue dict
        self.signal_log = []          # every scan reading
        self.goal = None              # current exploration target
        self.found = False
        self.step_no = 0
        self._last_ingested = -1
        self._new_clue_flag = False

    # ==================================================================
    # Belief handling
    # ==================================================================
    def _normalize(self):
        self.belief = np.where(self.free_mask, self.belief, 0.0)
        total = self.belief.sum()
        if not np.isfinite(total) or total <= 0:          # degenerate: restart uniform
            self.belief = self.free_mask.astype(float)
            total = self.belief.sum()
        self.belief = self.belief / total

    def _apply_likelihood(self, lik):
        """Posterior is proportional to prior * likelihood."""
        self.belief = self.belief * lik
        self._normalize()

    def update_belief_signal(self, pos, signal):
        """Bayesian update from a scan: strength + bearing + visual contact."""
        pr, pc = pos
        s = self.sensor
        dx = (self.cc - pc).astype(float)                 # east positive
        dy = (pr - self.rr).astype(float)                 # north positive
        dist = np.hypot(dx, dy)

        # --- strength: z = exp(-d/decay) + N(0, sigma)  ->  Gaussian likelihood
        expected = np.exp(-dist / s["signal_decay"])
        z = (signal["strength"] - expected) / s["strength_noise_std"]
        lik_strength = np.exp(-0.5 * z ** 2)

        # --- bearing: wrapped Gaussian approximated by von Mises (kappa = 1/sigma^2)
        sigma = math.radians(s["direction_noise_std_deg"])
        kappa = 1.0 / sigma ** 2
        diff = np.arctan2(dy, dx) - math.radians(signal["direction_deg"])
        lik_dir = np.exp(kappa * (np.cos(diff) - 1.0))
        lik_dir = np.where(dist == 0, 1.0, lik_dir)       # bearing undefined at own cell

        eps = self.LIK_FLOOR
        lik = ((1 - eps) * lik_strength + eps) * ((1 - eps) * lik_dir + eps)

        # --- visual contact (exact) or its absence (negative evidence)
        near = dist <= s["visual_contact_radius"]
        lik = lik * np.where(near, 1.0 if signal["visual_contact"] else 1e-3,
                             1e-3 if signal["visual_contact"] else 1.0)
        self._apply_likelihood(lik)

    # ---- text clues (rule-based) -------------------------------------
    def update_belief_clue(self, clue):
        """
        Parse a clue's text with simple rules and apply a soft likelihood.
          "... within N cells ..."      -> Manhattan-distance proximity rule
          "... NOT within N cells ..."  -> exclusion zone
          "... north/south/east/west"   -> half-plane constraints relative to clue
        Returns a short human-readable description of what was inferred.
        """
        cr, cc_ = clue["pos"]
        text = clue["text"].lower()
        rel = self.sensor["clue_reliability"]
        manhattan = np.abs(self.rr - cr) + np.abs(self.cc - cc_)
        lik = np.ones((self.rows, self.cols))
        desc = "no usable rule found in the text"

        m = re.search(r"within (\d+) cells?", text)
        if m:
            n = int(m.group(1))
            inside = manhattan <= n
            negated = bool(re.search(r"\bnot\b[^.]*within", text))
            consistent = ~inside if negated else inside
            lik = np.where(consistent, rel, 1 - rel)
            desc = (f"EVE is NOT within {n} cells of {tuple(clue['pos'])} (excluding that zone)"
                    if negated else
                    f"EVE is within {n} cells of {tuple(clue['pos'])} (concentrating belief there)")
        else:
            dirs = sorted(set(re.findall(r"\b(north|south|east|west)\b", text)))
            if dirs:
                consistent = np.ones((self.rows, self.cols), dtype=bool)
                if "north" in dirs: consistent &= self.rr < cr
                if "south" in dirs: consistent &= self.rr > cr
                if "east" in dirs:  consistent &= self.cc > cc_
                if "west" in dirs:  consistent &= self.cc < cc_
                lik = np.where(consistent, rel, 1 - rel)
                desc = f"EVE lies {' and '.join(dirs)} of {tuple(clue['pos'])}"

        self._apply_likelihood(lik)
        return desc

    # ==================================================================
    # Observation intake
    # ==================================================================
    def ingest(self, obs):
        """Fold an observation into state and belief. Idempotent per step number."""
        if obs["step"] <= self._last_ingested:
            return
        self._last_ingested = obs["step"]
        self._new_clue_flag = False
        self.pos = tuple(obs["position"])
        self.battery = obs["battery"]
        self.step_no = obs["step"]

        if obs["found_eve"]:
            self.found = True
            self.belief = np.zeros((self.rows, self.cols))
            self.belief[self.pos] = 1.0
            return

        # Standing here and not meeting EVE is negative evidence for this cell.
        self.belief[self.pos] = 0.0
        self._normalize()

        if obs.get("signal"):
            sig = obs["signal"]
            self.scanned.add(self.pos)
            self.signal_log.append({"pos": list(self.pos), **sig})
            self.update_belief_signal(self.pos, sig)

        for clue in obs.get("new_clues", []):
            if clue["id"] not in self.clues:
                clue = dict(clue)
                clue["inference"] = self.update_belief_clue(clue)
                self.clues[clue["id"]] = clue
                self._new_clue_flag = True

    # ==================================================================
    # Belief summaries
    # ==================================================================
    def get_inferred_location(self):
        r, c = np.unravel_index(np.argmax(self.belief), self.belief.shape)
        return int(r), int(c)

    def _box_mass(self, center, radius):
        r, c = center
        return float(self.belief[max(0, r - radius):r + radius + 1,
                                 max(0, c - radius):c + radius + 1].sum())

    def get_confidence(self):
        """(P(exact MAP cell), P(EVE within 1 cell of MAP cell))."""
        m = self.get_inferred_location()
        return float(self.belief[m]), self._box_mass(m, 1)

    def get_entropy(self):
        p = self.belief[self.belief > 0]
        return float(-(p * np.log2(p)).sum())

    def top_cells(self, k=3):
        idx = np.argsort(self.belief, axis=None)[::-1][:k]
        return [{"cell": [int(i // self.cols), int(i % self.cols)],
                 "p": round(float(self.belief.flat[i]), 4)} for i in idx]

    def recommend_next_move(self):
        """What WALL-E should do next, given the current belief (for the final report)."""
        if self.found:
            return "Mission complete: EVE reached. Hold position and stay with EVE."
        m = self.get_inferred_location()
        if self.battery <= 0:
            return "Out of power. Wait for rescue; no move is possible."
        return (f"Recharge if needed, then travel to {m} (the most probable cell) "
                f"and scan its neighbours.")

    # ==================================================================
    # Path planning helpers
    # ==================================================================
    def _neighbors(self, cell):
        r, c = cell
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            n = (r + dr, c + dc)
            if 0 <= n[0] < self.rows and 0 <= n[1] < self.cols and n not in self.obstacles:
                yield n

    def _bfs(self, start):
        dist, parent, q = {start: 0}, {start: None}, deque([start])
        while q:
            cur = q.popleft()
            for n in self._neighbors(cur):
                if n not in dist:
                    dist[n], parent[n] = dist[cur] + 1, cur
                    q.append(n)
        return dist, parent

    @staticmethod
    def _path(parent, goal):
        path, cur = [], goal
        while parent.get(cur) is not None:
            path.append(cur)
            cur = parent[cur]
        return path[::-1]                                   # excludes the start cell

    def _compute_station_dist(self):
        """Multi-source BFS: walking distance from every cell to the nearest station."""
        d = np.full((self.rows, self.cols), INF, dtype=int)
        q = deque()
        for s in self.stations:
            d[s] = 0
            q.append(s)
        while q:
            cur = q.popleft()
            for n in self._neighbors(cur):
                if d[n] == INF:
                    d[n] = d[cur] + 1
                    q.append(n)
        return d

    def _can_afford(self, cost, new_pos):
        """After paying `cost` and standing on `new_pos`, can we still get home with margin?"""
        need = int(self.station_dist[new_pos]) * self.costs["move"] + self.SAFETY_MARGIN
        return self.battery - cost >= need

    @staticmethod
    def _compass(deg):
        names = ["east", "north-east", "north", "north-west",
                 "west", "south-west", "south", "south-east"]
        return names[int(((deg % 360) + 22.5) // 45) % 8]

    # ==================================================================
    # Decision making
    # ==================================================================
    def _head_to_station(self, why):
        """One step along the shortest path to the nearest recharge station."""
        here = int(self.station_dist[self.pos])
        for n in self._neighbors(self.pos):
            if self.station_dist[n] == here - 1:
                return ("move", n,
                        f"{why} Heading to the nearest recharge station "
                        f"({here} moves away); stepping to {n}.")
        return ("scan", self.pos, f"{why} No route to any station exists; scanning in place.")

    def _choose_goal(self):
        """
        Pick the next scan point. Score = belief mass within 2 cells (value of
        information), discounted by travel distance, plus a small exploration
        bonus for being far from already-scanned cells. Only goals the battery
        can reach, scan at, and still return from are considered.
        """
        dist, _ = self._bfs(self.pos)
        mv, sc = self.costs["move"], self.costs["scan"]

        def candidates(strict):
            out = []
            for cell, plen in dist.items():
                if strict and (cell in self.scanned or
                               any(abs(cell[0] - s[0]) + abs(cell[1] - s[1]) < 2 for s in self.scanned)):
                    continue
                if cell == self.pos:
                    continue
                if self.battery - plen * mv - sc < int(self.station_dist[cell]) * mv + self.SAFETY_MARGIN:
                    continue
                mass = self._box_mass(cell, 2)
                gap = min((abs(cell[0] - s[0]) + abs(cell[1] - s[1]) for s in self.scanned), default=6)
                out.append((mass / (1 + 0.08 * plen) + 0.01 * min(gap, 6), cell, mass, plen))
            return out

        cands = candidates(True) or candidates(False)
        if not cands:
            return None
        score, cell, mass, plen = max(cands)
        self._goal_note = (f"best scan point {cell}: {mass:.0%} of belief within 2 cells, "
                           f"{plen} moves away, score {score:.3f}")
        return cell

    def decide_next_action(self, obs):
        """Return (action, target_pos, reasoning_string)."""
        self.ingest(obs)
        pos, bat = self.pos, self.battery
        mv, sc, rc = self.costs["move"], self.costs["scan"], self.costs["recharge"]
        at_station = pos in self.stations
        ret_cost = int(self.station_dist[pos]) * mv
        map_cell = self.get_inferred_location()
        p_map, mass3 = self.get_confidence()

        status = (f"[Battery {bat}/{self.capacity}; return trip costs {ret_cost}; "
                  f"belief peak {map_cell} p={p_map:.0%}, 3x3 mass={mass3:.0%}, "
                  f"entropy={self.get_entropy():.2f} bits]")
        notes = []
        if self._new_clue_flag:
            notes.append("New clue(s) discovered: " +
                         "; ".join(c["inference"] for c in self.clues.values()
                                   if c["id"] in {x["id"] for x in obs.get("new_clues", [])}) + ".")
        if obs.get("signal"):
            sg = obs["signal"]
            notes.append(f"Last scan: strength {sg['strength']:.2f}, bearing "
                         f"{sg['direction_deg']:.0f} deg ({self._compass(sg['direction_deg'])}); "
                         f"belief updated.")
        prefix = " ".join(notes + [status])

        # ---- 1. Critical battery: must recharge ----------------------------
        critical = bat <= ret_cost + self.SAFETY_MARGIN
        if critical:
            if at_station:
                return ("recharge", pos, f"{prefix} Battery is critically low at a station: recharging.")
            return self._head_to_station(f"{prefix} Battery is critically low (<= return cost + margin).")

        # ---- 2. At a station and below top-up threshold --------------------
        if at_station and bat < self.RECHARGE_FRAC * self.capacity:
            return ("recharge", pos, f"{prefix} Standing on a station with battery below "
                                     f"{self.RECHARGE_FRAC:.0%}: topping up before searching on.")

        # ---- 3. Confident enough: commit to the most likely cell -----------
        if mass3 >= self.COMMIT_MASS and map_cell != pos:
            dist, parent = self._bfs(pos)
            if map_cell in dist:
                plen = dist[map_cell]
                safe = bat - plen * mv >= int(self.station_dist[map_cell]) * mv + self.SAFETY_MARGIN
                desperate = at_station and bat >= self.capacity and bat - plen * mv >= 1
                if safe or desperate:
                    nxt = self._path(parent, map_cell)[0]
                    return ("move", nxt,
                            f"{prefix} {mass3:.0%} of the belief lies within 1 cell of {map_cell}: "
                            f"committing and walking there ({plen} moves left); next cell {nxt}.")
                if at_station and bat < self.capacity:
                    return ("recharge", pos, f"{prefix} Want to commit to {map_cell} but the trip "
                                             f"({plen} moves) is unsafe; recharging first.")
                if not at_station:
                    return self._head_to_station(f"{prefix} Want to commit to {map_cell} but battery "
                                                 f"is too low for the trip.")

        # ---- 4. Scan on arrival at (or start of) a scan point --------------
        if pos not in self.scanned and (self.goal is None or self.goal == pos):
            if self._can_afford(sc, pos):
                self.goal = pos
                return ("scan", pos, f"{prefix} Unscanned cell and battery allows it: scanning to "
                                     f"sense EVE's signal strength and direction.")
            return self._head_to_station(f"{prefix} Cannot afford a scan safely here.")

        # ---- 5. Pick (or refresh) an exploration goal ----------------------
        if self.goal is None or self.goal == pos or self._new_clue_flag or self.goal in self.scanned:
            self.goal = self._choose_goal()
            if self.goal is not None:
                prefix += f" Chose {self._goal_note}."

        if self.goal is None:                               # nothing affordable
            if at_station and bat < self.capacity:
                return ("recharge", pos, f"{prefix} No reachable scan point is affordable; "
                                         f"recharging to extend range.")
            if at_station:
                self.scanned.discard(pos)
                return ("scan", pos, f"{prefix} Fully charged and nothing new to reach; "
                                     f"re-scanning here to average out noise.")
            return self._head_to_station(f"{prefix} No scan point is affordable.")

        # ---- 6. Travel toward the goal one cell at a time ------------------
        dist, parent = self._bfs(pos)
        if self.goal not in dist:
            self.goal = None
            return self._head_to_station(f"{prefix} Goal unreachable.")
        nxt = self._path(parent, self.goal)[0]
        if not self._can_afford(mv, nxt):
            return self._head_to_station(f"{prefix} Next step toward {self.goal} would leave too "
                                         f"little battery to get home.")
        return ("move", nxt, f"{prefix} Traveling to scan point {self.goal} "
                             f"({dist[self.goal]} moves left); next cell {nxt}.")
