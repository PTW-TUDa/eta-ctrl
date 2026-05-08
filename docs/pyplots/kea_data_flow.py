"""
Generate a documentation diagram for the KEA tank temperature example.

Shows the experiment flow: temperature evolution, method calls (reset/step),
actions, scenario data indices, and the observation-action stagger.

Usage:
    python kea_diagram.py          # saves to kea_experiment_flow.svg
    python kea_diagram.py --pdf    # saves to kea_experiment_flow.pdf
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np

# plt.rcParams["figure.dpi"] = 168

# ── Parameters ──────────────────────────────────────────────────────────
SAMPLING_TIME = 1
EPISODE_DURATION = 7
N_STEPS = int(EPISODE_DURATION / SAMPLING_TIME)

START_TANK_TEMP = 60.0
TEMP_MIN = 55.0
TEMP_MAX = 65.0
HEATING_CHANGE = 1
COOLING_CHANGE = -1

# Example action sequence for illustration
ACTIONS = [1, 1, 1, 0, 0, 0, 1, 1, 0, 0]
PRICES = [0.12, 0.15, 0.18, 0.22, 0.25, 0.20, 0.14, 0.10, 0.08, 0.11, 0.19]

# ── Simulate ────────────────────────────────────────────────────────────
temps = [START_TANK_TEMP]
for i in range(N_STEPS):
    delta = HEATING_CHANGE if ACTIONS[i] == 1 else COOLING_CHANGE
    t_next = np.clip(temps[-1] + delta, TEMP_MIN, TEMP_MAX)
    temps.append(round(t_next, 1))

t_points = np.arange(N_STEPS + 1)

# ── Colors ──────────────────────────────────────────────────────────────
C_TEMP_LINE = "#2563EB"
C_TEMP_DOT = "#2563EB"
C_HEAT_ON = "#FDE68A"
C_HEAT_OFF = "#F3F4F6"
C_BOUNDS = "#EF4444"
C_RESET = "#8B5CF6"
C_STEP = "#10B981"
C_OBS = "#3B82F6"
C_ACT_ON = "#F59E0B"
C_ACT_OFF = "#D1D5DB"
C_SCENARIO = "#EC4899"
C_GRID = "#E5E7EB"
C_TEXT = "#374151"
C_TEXT_LIGHT = "#6B7280"

# ── Figure setup ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6.5))
fig.subplots_adjust(left=0.06, right=0.97, top=0.82, bottom=0.14)

# ── Background: heating on/off bands ───────────────────────────────────
for i in range(N_STEPS):
    color = C_HEAT_ON if ACTIONS[i] == 1 else C_HEAT_OFF
    ax.axvspan(i, i + 1, alpha=0.4, color=color, linewidth=0)

# ── Temperature bounds ──────────────────────────────────────────────────
ax.axhline(TEMP_MIN, color=C_BOUNDS, linestyle="--", linewidth=1, alpha=0.5, zorder=1)
ax.axhline(TEMP_MAX, color=C_BOUNDS, linestyle="--", linewidth=1, alpha=0.5, zorder=1)
ax.text(
    N_STEPS + 0.15,
    TEMP_MIN,
    f"T_min = {TEMP_MIN}°",
    va="center",
    fontsize=8,
    color=C_BOUNDS,
    alpha=0.8,
)
ax.text(
    N_STEPS + 0.15,
    TEMP_MAX,
    f"T_max = {TEMP_MAX}°",
    va="center",
    fontsize=8,
    color=C_BOUNDS,
    alpha=0.8,
)

# ── Temperature line + dots ─────────────────────────────────────────────
ax.plot(t_points, temps, color=C_TEMP_LINE, linewidth=2, zorder=4, label="Temperature")
ax.scatter(t_points, temps, color="white", edgecolors=C_TEMP_DOT, s=50, linewidth=1.5, zorder=5)

# Temperature values above each dot
for x, t in zip(t_points, temps, strict=True):
    label = f"{t:.1f}°"
    ax.annotate(
        label,
        (x, t),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontsize=7.5,
        color=C_TEMP_DOT,
        fontweight="bold",
        zorder=6,
    )

# ── Method calls row (above plot, inside axes coords) ───────────────────
METHOD_Y = TEMP_MAX + 1.4 * 3

# reset() at t=0
ax.annotate(
    "env.reset()",
    (0, METHOD_Y),
    ha="center",
    va="center",
    fontsize=7.5,
    fontweight="bold",
    color="white",
    bbox=dict(boxstyle="round,pad=0.3", facecolor=C_RESET, edgecolor="none", alpha=0.9),
    zorder=10,
)

# step() calls at t=0.5, 1.5, ..., 9.5 (centered in each interval)
for i in range(N_STEPS):
    ax.annotate(
        "env.step()",
        (i + 1, METHOD_Y),
        ha="center",
        va="center",
        fontsize=7.5,
        fontweight="bold",
        color="white",
        bbox=dict(boxstyle="round,pad=0.25", facecolor=C_STEP, edgecolor="none", alpha=0.85),
        zorder=10,
    )

# ── Observation markers at each timepoint ───────────────────────────────
OBS_Y = TEMP_MAX + 1.4 * 2

for i in range(N_STEPS + 1):
    style = "bold" if i == 0 else "normal"
    suffix = "\n(terminal)" if i == N_STEPS else ""
    ax.annotate(
        f"obs[{i}]{suffix}",
        (i, OBS_Y),
        ha="center",
        va="center",
        fontsize=6.5,
        fontweight=style,
        color=C_OBS,
        zorder=10,
    )

# ── Scenario data row ──────────────────────────────────────────────────
SCN_Y = TEMP_MAX + 1.4
# SCN_Y = TEMP_MIN - 0.7

for i in range(N_STEPS + 1):
    ax.annotate(
        f"scn[{i}]\n€{PRICES[i]:.2f}",
        (i, SCN_Y),
        ha="center",
        va="center",
        fontsize=6,
        color=C_SCENARIO,
        fontweight="bold",
        zorder=10,
    )

# ── Action row (below x-axis) ──────────────────────────────────────────
ACT_Y = TEMP_MIN - 1.4

for i in range(N_STEPS):
    val = ACTIONS[i]
    color = C_ACT_ON if val == 1 else C_ACT_OFF
    text_color = "#92400E" if val == 1 else "#6B7280"
    label = f"act[{i}]={'ON' if val else 'OFF'}"
    ax.annotate(
        label,
        (i, ACT_Y),
        ha="center",
        va="center",
        fontsize=6.5,
        fontweight="bold",
        color=text_color,
        bbox=dict(boxstyle="round,pad=0.25", facecolor=color, edgecolor="none", alpha=0.9),
        zorder=10,
    )

# ── Axes formatting ────────────────────────────────────────────────────
ax.set_xlim(-0.5, N_STEPS + 0.5)
ax.set_ylim(ACT_Y - 1.5, METHOD_Y + 1.5)
ax.set_xticks(range(N_STEPS + 1))
ax.set_xticklabels([f"t={i}" for i in range(N_STEPS + 1)], fontsize=8, color=C_TEXT)
ax.set_xlabel("Timestep", fontsize=10, color=C_TEXT, labelpad=8)

# Only show y-ticks in the temperature range
y_ticks = np.arange(TEMP_MIN, TEMP_MAX + 1, 1)
ax.set_yticks(y_ticks)
ax.set_yticklabels([f"{v:.0f}°" for v in y_ticks], fontsize=8, color=C_TEXT)
ax.set_ylabel("Tank temperature (°C)", fontsize=10, color=C_TEXT, labelpad=8)

# Grid only in temperature range
ax.yaxis.grid(True, color=C_GRID, linewidth=0.5, alpha=0.8)
ax.xaxis.grid(True, color=C_GRID, linewidth=0.5, alpha=0.3)
ax.set_axisbelow(True)

# Hide spines
for spine in ax.spines.values():
    spine.set_visible(False)

# ── Row labels on left margin ──────────────────────────────────────────
label_x = -0.45
label_kw = dict(ha="right", va="center", fontsize=8, color=C_TEXT_LIGHT, fontstyle="italic")
ax.text(label_x, METHOD_Y, "method calls", **label_kw, transform=ax.transData)
ax.text(label_x, OBS_Y, "observations", **label_kw, transform=ax.transData)
ax.text(label_x, ACT_Y, "actions", **label_kw, transform=ax.transData)
ax.text(label_x, SCN_Y, "scenario data", **label_kw, transform=ax.transData)

# ── Title ───────────────────────────────────────────────────────────────
fig.suptitle(
    "KEA experiment flow: observation-action sequence",
    fontsize=13,
    fontweight="bold",
    color=C_TEXT,
    y=0.96,
)
subtitle = (
    f"episode_duration={EPISODE_DURATION}  |  sampling_time={SAMPLING_TIME}  |  "
    f"T₀={START_TANK_TEMP}°  |  heating=+{HEATING_CHANGE}/step  |  cooling={COOLING_CHANGE}/step"
)
fig.text(0.5, 0.92, subtitle, ha="center", fontsize=9, color=C_TEXT_LIGHT)

# ── Legend ──────────────────────────────────────────────────────────────
legend_y = 0.885
legend_items = [
    (C_HEAT_ON, 1, "■ heating ON"),
    (C_HEAT_OFF, 1, "■ heating OFF"),
    (C_TEMP_LINE, 1.0, "— temperature"),
    (C_BOUNDS, 0.5, "-- T bounds"),
]
for idx, (color, alpha, text) in enumerate(legend_items):
    fig.text(
        0.06 + idx * 0.14,
        legend_y,
        text,
        fontsize=8,
        color=color,
        alpha=alpha,
        va="center",
    )

# ── Array size annotation ──────────────────────────────────────────────
array_text = (
    f"observations: {N_STEPS + 1} entries (0..{N_STEPS})  |  "
    f"actions: {N_STEPS} entries (0..{N_STEPS - 1})  |  "
    f"rewards: {N_STEPS} entries (0..{N_STEPS - 1})  |  "
    f"scenario: {N_STEPS + 1} entries (0..{N_STEPS})"
)
fig.text(
    0.5,
    0.025,
    array_text,
    ha="center",
    fontsize=8,
    color=C_TEXT_LIGHT,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#F9FAFB", edgecolor=C_GRID),
)
# ── Fancy PowerPoint Annotations ────────────────────────────────────────
# Keep arrow endpoints separate from text position
text_position_scale = 0.3
arrow_offset = 0.5
# mid_y_predict = ACT_Y + (SCN_Y - ACT_Y) * text_position_scale
mid_y_predict = (1 - text_position_scale) * ACT_Y + text_position_scale * SCN_Y
# mid_y_step = ACT_Y + (SCN_Y - ACT_Y) * (1 - text_position_scale)
mid_y_step = text_position_scale * ACT_Y + (1 - text_position_scale) * SCN_Y
predict_color = "#e02c1488"
step_color = "#14e04087"

arrow_prows_step = dict(arrowstyle="->", color=step_color, lw=3)
arrow_prows_predict = dict(arrowstyle="->", color=predict_color, lw=3)

box_style_predict = dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor=predict_color)
box_style_step = dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor=step_color)
text_kwargs = {"ha": "center", "va": "center"}


# Text for Prediction
ax.annotate("", xy=(0, ACT_Y + arrow_offset), xytext=(0, SCN_Y - arrow_offset), arrowprops=arrow_prows_predict)
ax.text((text_position_scale), mid_y_step, "model.predict()", bbox=box_style_predict, **text_kwargs)

# Text for Step
ax.annotate("", xy=(1, SCN_Y - arrow_offset), xytext=(0, ACT_Y + arrow_offset), arrowprops=arrow_prows_step)
ax.text((1 - (1 - text_position_scale)), mid_y_predict, "env.step()", bbox=box_style_step, **text_kwargs)


# ── Save ────────────────────────────────────────────────────────────────
fmt = "pdf" if "--pdf" in sys.argv else "svg"
outfile = f"kea_experiment_flow.{fmt}"
# fig.show()
plt.show()
# fig.savefig(outfile, format=fmt, dpi=150, bbox_inches="tight")
# print(f"Saved: {outfile}")
# plt.close()
