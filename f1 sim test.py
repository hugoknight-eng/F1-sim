# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 17:19:57 2026

@author: Hugo Knight
"""

import numpy as np
import matplotlib.pyplot as plt

#Vehicle parameters
m0      = 798
frontal_a = 2 * 0.9
power_kw  = 710
Cd        = 0.8
Cl        = 3.0
compounds = {
    "soft":   {"Cf0": 1.75, "deg_rate": 2e-6},
    "medium": {"Cf0": 1.70, "deg_rate": 1e-6},
    "hard":   {"Cf0": 1.67, "deg_rate": 6e-7},
}
Cf_wet    = 0.9
rho       = 1.225
g         = 9.81
deg_rate  = 1e-6
eta = 0.5       
Q   = 44e6       
fuel_mass = 170
laps = 52
pit_stop_lap =25       
pit_stop_time = 17        
compound_stint1 = compounds["medium"]
compound_stint2 = compounds["hard"]
#Track
track1 = [
    {"type": "straight", "length": 500},
    {"type": "corner",   "radius": 100, "angle": 180},
    {"type": "straight", "length": 500},
    {"type": "corner",   "radius": 100, "angle": 180},
]
track = [
   {"type": "straight", "length": 500},
   {"type": "corner", "radius":110 ,"angle":120},
   {"type":"straight", "length":115},
   {"type":"corner", "radius":110,"angle":40},
   {"type":"straight", "length": 175},
   {"type":"corner","radius":17,"angle":70},
   {"type":"straight","length":100},
   {"type":"corner", "radius":19,"angle":130},
   {"type":"straight","length":150},
   {"type":"corner", "radius":30 ,"angle":40},
   {"type":"straight","length":640},
   {"type":"corner", "radius": 55.7,"angle":125},
   {"type": "straight", "length": 65},
   {"type": "corner", "radius":47.7 ,"angle":200},
   {"type":"straight", "length":140},
   {"type":"corner", "radius":193,"angle":44},
   {"type":"straight", "length": 385},
   {"type":"corner","radius":115,"angle":70},
   {"type":"straight","length":527},
   {"type":"corner", "radius":60.6,"angle":50},
   {"type": "straight", "length": 80},
   {"type": "corner", "radius":76.2 ,"angle":50},
   {"type":"straight", "length":50},
   {"type":"corner", "radius":58.2,"angle":90},
   {"type":"straight", "length": 830},
   {"type":"corner","radius":107,"angle":110},
   {"type":"straight","length":310},
   {"type":"corner", "radius":12.1,"angle":90},
   {"type":"straight","length":27},
   {"type":"corner", "radius":24.6,"angle":90},
   {"type":"straight","length":116},
   {"type":"corner", "radius":25.4,"angle":60},
]
# Compute arc lengths for corners
for seg in track:
    if seg["type"] == "corner":
        seg["length"] = seg["radius"] * np.radians(seg["angle"])

# Physics functions
t=0


TOP_SPEED = (power_kw * 1000 / (0.5 * rho * Cd * frontal_a)) ** (1/3)

def aero_drag(v):
    return 0.5 * rho * v**2 * Cd * frontal_a

def max_corner_speed(radius, Cf,mass):
    denom = mass / radius - 0.5 * rho * frontal_a * Cl * Cf
    if denom<= 0:
        return (power_kw*1000 / (0.5*rho*Cd*frontal_a))**(1/3)
    v = np.sqrt(Cf * mass * g / (denom))
    v = min(v, (power_kw*1000 / (0.5*rho*Cd*frontal_a))**(1/3))
    return v

def drive_acceleration(v,mass):
    v = max(v, 1.0)
    return (power_kw * 1000 / v - aero_drag(v)) / mass

def brake_deceleration(v, Cf,mass):
    a = min(Cf*(g+0.5*rho*frontal_a*Cl*v**2/mass)+aero_drag(v)/mass,6*g+aero_drag(v)/mass)
    return a
"""
# ── Speed profile along a straight ────────────────────────────────
def straight_speed_profile(length, v_start, v_end, Cf,mass):
    steps  = int(length)
    speeds = np.zeros(steps)

    # Forward pass: accelerate from v_start
    speeds[0] = v_start
    for i in range(1, steps):
        speeds[i] = np.sqrt(speeds[i-1]**2 + 2 * drive_acceleration(speeds[i-1],mass))

    # Backward pass: brake to v_end, overwrite where braking is limiting
    speeds[-1] = v_end
    for i in range(steps - 1, 0, -1):
        v_back = np.sqrt(speeds[i]**2 + 2 * brake_deceleration(speeds[i], Cf,mass))
        if v_back < speeds[i-1]:
            speeds[i-1] = v_back

    return speeds
"""
# lap simulation
def simulate_lap(track, Cf0, deg_rate, mass):
    v_max = []
    for seg in track:
        n = int(seg["length"])
        if seg["type"] == "corner":
            v_max.extend([max_corner_speed(seg["radius"], Cf0, mass)] * n)
        else:
            v_max.extend([TOP_SPEED] * n)
    v_max = np.array(v_max)
    N = len(v_max)

    k = np.argmin(v_max)
    v_max = np.roll(v_max, -k)
    v = v_max.copy()

    for i in range(1, N):
        v_reach = np.sqrt(v[i-1]**2 + 2 * drive_acceleration(v[i-1], mass))
        v[i] = min(v[i], v_reach)

    v_wrap = np.sqrt(v[0]**2 + 2 * brake_deceleration(v[0], Cf0, mass))
    v[-1] = min(v[-1], v_wrap)
    for i in range(N - 2, -1, -1):
        v_brake = np.sqrt(v[i+1]**2 + 2 * brake_deceleration(v[i+1], Cf0, mass))
        v[i] = min(v[i], v_brake)

    v = np.roll(v, k)
    return np.arange(N), v

# Run and plot
fuel_remaining = fuel_mass
lap_times = []
compound = compound_stint1 

tyre_wear=0
lap_length = sum(seg["length"] for seg in track)
for lap in range(laps):
    if lap+1 == pit_stop_lap:
        compound = compound_stint2
        Cf0 = compound_stint2["Cf0"]      # fresh hard tyres
        deg_rate = compound_stint2["deg_rate"]
        tyre_wear=0
        pit_penalty = pit_stop_time
    else:
           pit_penalty = 0
    mass = m0 + fuel_remaining
    current_Cf = compound["Cf0"] * np.exp(-compound["deg_rate"] * tyre_wear)
    lap_distance, lap_speed = simulate_lap(track, current_Cf, deg_rate, mass)
    
    # Lap time
    lap_time = np.sum(1 / lap_speed)+pit_penalty
    lap_times.append(lap_time)
    
    # Fuel burned 
    fuel_burned = np.sum((power_kw * 1000 / (eta * Q)) * (1 / lap_speed))
    fuel_remaining -= fuel_burned
    
    tyre_wear += lap_length
    
    if fuel_remaining <= 0:
        print(f"Out of fuel on lap {lap+1}")
        break
""
plt.figure(figsize=(12, 5))
plt.plot(range(1, len(lap_times)+1), lap_times)
plt.xlabel("Lap")
plt.ylabel("Lap time (s)")
plt.title("Lap Time Evolution - Fuel & Tyre Degradation")
plt.grid(True)
plt.show()
total_race_time = sum(lap_times)
print(f"Total race time: {total_race_time:.3f} seconds")
print(f"Total race time: {total_race_time/60:.2f} minutes")
""
plt.figure(figsize=(12, 5))
plt.plot(lap_distance, lap_speed * 3.6)
plt.xlabel("Distance (m)")
plt.ylabel("Speed (km/h)")
plt.title("Lap Speed Profile")
plt.grid(True)
plt.show()
from itertools import product

def simulate_race(strategy):
    """
    strategy: list of dicts, e.g.
      [{"compound": "soft", "start_lap": 1},
       {"compound": "hard", "start_lap": 31}]
    Returns total race time (seconds), or inf if fuel runs out.
    """
    fuel_remaining = fuel_mass
    lap_times = []
    tyre_wear = 0

    # Build lap
    stint_map = {}
    for k, stint in enumerate(strategy):
        end = strategy[k+1]["start_lap"] if k+1 < len(strategy) else laps + 1
        for lap in range(stint["start_lap"], end):
            stint_map[lap] = (stint["compound"], lap == stint["start_lap"] and k > 0)

    for lap in range(1, laps + 1):
        compound_key, is_pit_lap = stint_map.get(lap, (strategy[0]["compound"], False))
        compound = compounds[compound_key]

        #tyre wear
        if is_pit_lap:
            tyre_wear = 0

        pit_penalty = pit_stop_time if is_pit_lap else 0
        mass = m0 + fuel_remaining
        current_Cf = compound["Cf0"] * np.exp(-compound["deg_rate"] * tyre_wear)

        lap_distance, lap_speed = simulate_lap(track, current_Cf, deg_rate, mass)
        lap_time = np.sum(1 / lap_speed) + pit_penalty
        lap_times.append(lap_time)

        fuel_burned = np.sum((power_kw * 1000 / (eta * Q)) * (1 / lap_speed))
        fuel_remaining -= fuel_burned
        tyre_wear += lap_length

        if fuel_remaining <= 0:
            return float('inf')  

    return sum(lap_times)


def optimise_strategy_grid(max_stops=1):
    """Grid search over pit lap(s) and compound choices."""
    compound_keys = list(compounds.keys())
    best_time = float('inf')
    best_strategy = None

    # 1-stop: iterate over pit lap and both compound choices
    for pit_lap in range(2, laps +1):          # avoid pitting lap 1 or last lap
        for c1, c2 in product(compound_keys, repeat=2):
            if c1 == c2:
                continue 
            strategy = [
                {"compound": c1, "start_lap": 1},
                {"compound": c2, "start_lap": pit_lap},
            ]
            t = simulate_race(strategy)
            if t < best_time:
                best_time = t
                best_strategy = (pit_lap, c1, c2)

    return best_strategy, best_time


best_strat, best_time = optimise_strategy_grid()
pit_lap, c1, c2 = best_strat
print(f"Optimal 1-stop: {c1} → pit lap {pit_lap} → {c2}")
print(f"Best race time: {best_time:.2f}s ({best_time/60:.2f} min)")
# ── Re-run simulation for the optimal strategy to extract lap-by-lap data ──
optimal_strategy = [
    {"compound": c1, "start_lap": 1},
    {"compound": c2, "start_lap": pit_lap},
]


fuel_remaining = fuel_mass
tyre_wear = 0
stint_map = {}
for k, stint in enumerate(optimal_strategy):
    end = optimal_strategy[k+1]["start_lap"] if k+1 < len(optimal_strategy) else laps + 1
    for lap in range(stint["start_lap"], end):
        stint_map[lap] = (stint["compound"], lap == stint["start_lap"] and k > 0)

opt_laps = []
opt_lap_times = []
opt_colors = []

color_mapping = {
    "soft": "red",
    "medium": "yellow",
    "hard": "white"
}

for lap in range(1, laps + 1):
    compound_key, is_pit_lap = stint_map.get(lap, (optimal_strategy[0]["compound"], False))
    compound = compounds[compound_key]

    if is_pit_lap:
        tyre_wear = 0

    pit_penalty = pit_stop_time if is_pit_lap else 0
    mass = m0 + fuel_remaining
    current_Cf = compound["Cf0"] * np.exp(-compound["deg_rate"] * tyre_wear)

    lap_distance, lap_speed = simulate_lap(track, current_Cf, deg_rate, mass)
    lap_time = np.sum(1 / lap_speed) + pit_penalty
    
    opt_laps.append(lap)
    opt_lap_times.append(lap_time)
    opt_colors.append(color_mapping[compound_key])

    fuel_burned = np.sum((power_kw * 1000 / (eta * Q)) * (1 / lap_speed))
    fuel_remaining -= fuel_burned
    tyre_wear += lap_length

#Strategy 
plt.style.use('dark_background')  # Necessary so the white line is visible
plt.figure(figsize=(12, 6))

for i in range(len(opt_laps) - 1):
    # Use the color of the current lap
    plt.plot(opt_laps[i:i+2], opt_lap_times[i:i+2], color=opt_colors[i], lw=2)

from matplotlib.lines import Line2D
custom_lines = [
    plt.Line2D([0], [0], color='red', lw=2, label='Soft'),
    plt.Line2D([0], [0], color='yellow', lw=2, label='Medium'),
    plt.Line2D([0], [0], color='white', lw=2, label='Hard')
]
plt.legend(handles=custom_lines, loc='upper right')

plt.xlabel("Lap", fontsize=12)
plt.ylabel("Lap Time (seconds)", fontsize=12)
plt.title(f"Optimal Strategy Lap Time Evolution ({c1.upper()} → {c2.upper()})", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.3)

clean_times = [t for i, t in enumerate(opt_lap_times) if not stint_map[opt_laps[i]][1]]
plt.ylim(min(clean_times) - 1, max(clean_times) + 2)

plt.show()
""
