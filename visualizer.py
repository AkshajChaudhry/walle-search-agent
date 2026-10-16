"""
visualizer.py - Maps WALL-E's search from config.json + trace.json using custom PNG icons.

Usage:   python visualizer.py [--config config.json] [--trace trace.json]
                              [--assets DIR] [--out mission_map.png] [--no-show]
Install: pip install matplotlib

Icon files (looked up in --assets, then next to this script, then the current folder):
    image_6.png   WALL-E robot   (drawn at the start cell and at the final cell)
    image_8.png   EVE            (drawn at EVE's true position)
    image_9.png   magnifier      (drawn on every clue cell)
    image_10.png  barricade      (drawn on every obstacle cell)
Any icon that is missing or unreadable falls back to a plain matplotlib shape
(circle / star / square / bar), so the script always produces a map.

Project coordinates are (row, col) with row 0 at the NORTH edge, so the plot uses
x = col, y = row and inverts the y-axis to keep north at the top.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Rectangle

# --- icon files ---------------------------------------------------------------
ICON_FILES = {
    "walle": "image_6.jpg",
    "eve": "image_8.jpg",
    "clue": "image_9.png",
    "obstacle": "image_10.png",
}

# --- colours (fallback symbols + path) -----------------------------------------
C_PATH = "#d9480f"        # red-orange path
C_WALLE = "#1f5fbf"       # blue
C_EVE = "#1e9e4a"         # green
C_CLUE = "#f2c200"        # yellow
C_OBST = "#6b7280"        # grey
C_STATION = "#7c3aed"     # purple
INK = "#1f2937"

CELL_FILL = 0.95          # target size of an icon, as a fraction of one grid cell
DPI = 150


# ==============================================================================
# Loading helpers
# ==============================================================================
def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_icons(search_dirs):
    """Return {key: image array or None}. A missing/unreadable file gives None."""
    icons = {}
    for key, fname in ICON_FILES.items():
        icons[key] = None
        for d in search_dirs:
            p = os.path.join(d, fname)
            if os.path.isfile(p):
                try:
                    icons[key] = plt.imread(p)
                    break
                except Exception as exc:          # corrupt / unsupported image
                    print(f"  ! could not read {p}: {exc}")
        print(f"  {fname:<13} -> {'loaded' if icons[key] is not None else 'not found, using fallback shape'}")
    return icons


def extract_path(config, trace):
    """Start cell followed by the robot's position after each logged step (moves only)."""
    path = [tuple(config["start_pos"])]
    for step in trace["steps"]:
        pos = tuple(step["observation_out"]["position"])
        if pos != path[-1]:
            path.append(pos)
    return path


# ==============================================================================
# Drawing helpers
# ==============================================================================
def put_icon(ax, img, xy, zoom, zorder, alpha=1.0):
    """Place an image centred on data coordinate xy = (x, y) using OffsetImage + AnnotationBbox."""
    box = OffsetImage(img, zoom=zoom, alpha=alpha)
    ab = AnnotationBbox(box, xy, frameon=False, box_alignment=(0.5, 0.5),
                        pad=0, zorder=zorder)
    ax.add_artist(ab)


class HandlerIcon(HandlerBase):
    """Legend handler that draws a small PNG as the legend key."""

    def __init__(self, img, size_px=34):
        super().__init__()
        self.img, self.zoom = img, size_px / max(img.shape[:2]) / (DPI / 72)

    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        centre = (xdescent + width / 2, ydescent + height / 2)
        box = OffsetImage(self.img, zoom=self.zoom)
        ab = AnnotationBbox(box, centre, xycoords=trans, frameon=False,
                            box_alignment=(0.5, 0.5), pad=0)
        return [ab]


def fit_zoom(img, cell_px, scale=1.0):
    """Zoom that makes the image's longest side fill CELL_FILL of a grid cell."""
    # OffsetImage already multiplies zoom by (figure dpi / 72), so divide that back out.
    return scale * CELL_FILL * cell_px / max(img.shape[:2]) / (DPI / 72)


# ==============================================================================
# Main drawing routine
# ==============================================================================
def draw(config, trace, icons, out_path, show):
    rows, cols = config["grid_size"]
    truth = config["ground_truth"]
    start = tuple(config["start_pos"])
    eve = tuple(truth["eve_pos"])
    path = extract_path(config, trace)
    final = path[-1]
    found_ids = {c["id"] for s in trace["steps"] for c in s["observation_out"]["new_clues"]}

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=DPI)

    # ---- clean 10x10 grid with labelled axes ----------------------------------
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)                 # inverted: row 0 on top
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xticks([x - 0.5 for x in range(cols + 1)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(rows + 1)], minor=True)
    ax.grid(which="minor", color="#d1d5db", linewidth=0.8)
    ax.tick_params(which="minor", length=0)
    ax.set_aspect("equal")
    ax.set_xlabel("Column (west → east)", color=INK)
    ax.set_ylabel("Row (north → south)", color=INK)
    fig.subplots_adjust(left=0.07, right=0.70, top=0.93, bottom=0.09)

    # Size of one grid cell in pixels, so icons scale to the cells.
    fig.canvas.draw()
    cell_px = ax.get_window_extent().width / cols

    # ---- recharge stations (marker, no icon file for these) -------------------
    for r, c in config["recharge_stations"]:
        ax.scatter(c, r, marker="P", s=150, color=C_STATION, edgecolor="white",
                   linewidth=1.2, zorder=2)

    # ---- obstacles: barricade icon or grey bar --------------------------------
    for r, c in config["obstacles"]:
        if icons["obstacle"] is not None:
            put_icon(ax, icons["obstacle"], (c, r), fit_zoom(icons["obstacle"], cell_px), 3)
        else:
            ax.add_patch(Rectangle((c - 0.45, r - 0.22), 0.9, 0.44, color=C_OBST, zorder=3))

    # ---- path (red-orange with arrows) ----------------------------------------
    ax.plot([c for _, c in path], [r for r, _ in path], color=C_PATH, linewidth=2.5,
            solid_capstyle="round", zorder=4)
    for (r0, c0), (r1, c1) in zip(path[:-1:3], path[1::3]):
        ax.annotate("", xy=(c1, r1), xytext=(c0, r0), zorder=5,
                    arrowprops=dict(arrowstyle="->", color=C_PATH, lw=2))

    # ---- clues: magnifier icon (dimmed if never discovered) -------------------
    for clue in truth["clues"]:
        r, c = clue["pos"]
        seen = clue["id"] in found_ids
        if icons["clue"] is not None:
            put_icon(ax, icons["clue"], (c, r), fit_zoom(icons["clue"], cell_px), 6,
                     alpha=1.0 if seen else 0.55)
        else:
            ax.scatter(c, r, marker="s", s=210, color=C_CLUE, zorder=6,
                       edgecolor=INK if seen else "#b59b00", linewidth=2 if seen else 1)
        ax.text(c, r - 0.58, clue["id"], ha="center", va="bottom", fontsize=9,
                color=INK, zorder=9)

    # ---- EVE and WALL-E (WALL-E is drawn at the START only) --------------------
    eve_xy = (eve[1], eve[0])

    if icons["eve"] is not None:
        put_icon(ax, icons["eve"], eve_xy, fit_zoom(icons["eve"], cell_px), 8)
    else:
        ax.scatter(*eve_xy, marker="*", s=650, color=C_EVE,
                   edgecolor="white", linewidth=1.5, zorder=8)

    def draw_walle(xy, sc, z):
        if icons["walle"] is not None:
            put_icon(ax, icons["walle"], xy, fit_zoom(icons["walle"], cell_px, sc), z)
        else:
            ax.scatter(*xy, marker="o", s=280 * sc, color=C_WALLE,
                       edgecolor="white", linewidth=2, zorder=z)

    draw_walle((start[1], start[0]), 1.0, 7)                    # starting position only
    ax.text(start[1] + 0.5, start[0], "start", ha="left", va="center", fontsize=8,
            color=INK, zorder=9)

    # ---- legend on the right (custom images or fallback markers) --------------
    handler_map, handles, labels = {}, [], []

    def add_key(key, label, fallback_handle):
        if icons[key] is not None:
            proxy = object()                       # unique key for this icon's handler
            handler_map[proxy] = HandlerIcon(icons[key])
            handles.append(proxy)
        else:
            handles.append(fallback_handle)
        labels.append(label)

    add_key("walle", "WALL-E (start)",
            Line2D([], [], marker="o", color="none", markerfacecolor=C_WALLE, markersize=13))
    add_key("eve", "EVE (true position)",
            Line2D([], [], marker="*", color="none", markerfacecolor=C_EVE, markersize=18))
    add_key("clue", "Clue (faded = never discovered)",
            Line2D([], [], marker="s", color="none", markerfacecolor=C_CLUE,
                   markeredgecolor="#b59b00", markersize=12))
    add_key("obstacle", "Obstacle (barricade)",
            Rectangle((0, 0), 1, 0.5, color=C_OBST))
    handles.append(Line2D([], [], color=C_PATH, linewidth=2.5)); labels.append("WALL-E's path")
    handles.append(Line2D([], [], marker="P", color="none", markerfacecolor=C_STATION,
                          markersize=12)); labels.append("Recharge station")

    ax.legend(handles, labels, handler_map=handler_map, loc="upper left",
              bbox_to_anchor=(1.03, 1.0), frameon=False, fontsize=10,
              labelspacing=1.3, handlelength=2.2, borderaxespad=0)

    status = trace.get("final_status", "unknown").replace("_", " ")
    ax.set_title(f"WALL-E and EVE's Signal  |  {len(path) - 1} moves, "
                 f"{len(trace['steps'])} steps  |  outcome: {status}",
                 fontsize=12, color=INK, pad=10)

    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    print(f"Saved map to {out_path}")
    if show:
        plt.show()


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Plot WALL-E's search using PNG icons")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--trace", default="trace.json")
    ap.add_argument("--assets", default=None, help="folder holding image_6/8/9/10.png")
    ap.add_argument("--out", default="mission_map.png")
    ap.add_argument("--no-show", action="store_true", help="save the PNG without opening a window")
    args = ap.parse_args()

    dirs = ([args.assets] if args.assets else []) + [here, os.getcwd()]
    print("Looking for icons in:", ", ".join(dirs))
    draw(load_json(args.config), load_json(args.trace), load_icons(dirs), args.out,
         not args.no_show)