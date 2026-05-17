#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vykreslení membránového napětí a iontových proudů pro nejlepší evoluční vlnu.
Inspirováno klasickou analýzou v neurofyziologii.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from neuron import h

# ════════════════════════════════════════════════════════════════════════════
# 1. Znovupoužití funkcí pro generování vlny
# ════════════════════════════════════════════════════════════════════════════
N_CONTROL_POINTS_PER_PHASE = 8
PHASE_RESOLUTION = 100
DT_MS = 0.005
T_TOTAL_MS = 15.0
DELAY_MS = 2.0
ELEC_RADIAL_UM = 2000.0
RHO_E_OHM_CM = 300.0

def render_phase_curve(control_points, n_samples=PHASE_RESOLUTION):
    knots = np.linspace(0, 1, len(control_points))
    aug_knots = np.concatenate([[-0.05], knots, [1.05]])
    aug_vals  = np.concatenate([[0.0], control_points, [0.0]])
    spl = CubicSpline(aug_knots, aug_vals, bc_type='natural', extrapolate=False)
    return spl(np.linspace(0, 1, n_samples))

def render_waveform(waveform, pw_us, dt_ms=DT_MS, time_arr=None):
    if time_arr is None:
        time_arr = np.arange(0, T_TOTAL_MS, dt_ms)
    wave = np.zeros(len(time_arr))
    pw_ms = pw_us / 1000.0
    gap_us = np.clip(waveform[16], 0.0, 3000.0)
    gap_ms = gap_us / 1000.0
    
    t0 = DELAY_MS
    t1 = t0 + pw_ms
    t2 = t1 + gap_ms
    t3 = t2 + pw_ms
    
    cathodic_mask = (time_arr >= t0) & (time_arr < t1)
    anodic_mask   = (time_arr >= t2) & (time_arr < t3)
    
    n1 = int(np.sum(cathodic_mask))
    n2 = int(np.sum(anodic_mask))
    
    cathodic_curve = render_phase_curve(waveform[:8], n_samples=n1)
    anodic_curve   = render_phase_curve(waveform[8:16], n_samples=n2)
    
    if np.sum(cathodic_curve) > 0: cathodic_curve = -cathodic_curve
    if np.sum(anodic_curve) < 0: anodic_curve = -anodic_curve
        
    peak_cath = float(np.max(np.abs(cathodic_curve)))
    cathodic_curve = cathodic_curve / peak_cath
    
    peak_anod = float(np.max(np.abs(anodic_curve)))
    anodic_curve = anodic_curve / max(peak_anod, 1e-9)
    
    area_cath = float(np.sum(cathodic_curve)) * dt_ms
    area_anod_unit = float(np.sum(anodic_curve)) * dt_ms
    balance_factor = -area_cath / area_anod_unit
    
    wave[cathodic_mask] = cathodic_curve
    wave[anodic_mask]   = anodic_curve * balance_factor
    
    return wave

# ════════════════════════════════════════════════════════════════════════════
# 2. Načtení dat
# ════════════════════════════════════════════════════════════════════════════
print("Načítám data z /Users/stepanzelnicek/Desktop/CEITEC/bifurcation_am_project_simulation/3810/outputs_evolver/final_waveform_pool.json...")
with open("/Users/stepanzelnicek/Desktop/CEITEC/bifurcation_am_project_simulation/3810/outputs_evolver/final_waveform_pool.json", "r") as f:
    data = json.load(f)

# Bereme první (nejlepší) vlnu
best_wf_params = np.array(data["waveforms"][0])
threshold_mA = data["thresholds_mA"][0]
pw_us = 200 # Pevně dané v configu

time_arr = np.arange(0, T_TOTAL_MS, DT_MS)
stim_wave_unit = render_waveform(best_wf_params, pw_us, time_arr=time_arr)
stim_wave_mA = stim_wave_unit * threshold_mA

# ════════════════════════════════════════════════════════════════════════════
# 3. Inicializace NEURONu a MRG Axonu
# ════════════════════════════════════════════════════════════════════════════
print("Spouštím NEURON a MRG axon simulaci...")
h.load_file('stdrun.hoc')
h.load_file('MRGaxon.hoc')

h.celsius = 37.0
CENTER_NODE = int(h.axonnodes) // 2

# Vytvoření pole pro extarcelulární potenciál (jako ve vašem měřiči)
sections = []
x = 0.0
for i in range(int(h.axonnodes)):
    sections.append((h.node[i], x + 0.5 * h.nodelength))
    x += h.nodelength
    if i == int(h.axonnodes) - 1: break
    sections.append((h.MYSA[2*i], x + 0.5 * h.paralength1)); x += h.paralength1
    sections.append((h.FLUT[2*i], x + 0.5 * h.paralength2)); x += h.paralength2
    for k in range(6):
        sections.append((h.STIN[6*i + k], x + 0.5 * h.interlength)); x += h.interlength
    sections.append((h.FLUT[2*i + 1], x + 0.5 * h.paralength2)); x += h.paralength2
    sections.append((h.MYSA[2*i + 1], x + 0.5 * h.paralength1)); x += h.paralength1

elec_x_um = next(pos for sec, pos in sections if sec == h.node[CENTER_NODE])

# Záznamové vektory (To nejdůležitější pro náš graf!)
# Použijeme vektor času přímo z NEURONu, aby délka časové osy
# přesně odpovídala délce záznamových vektorů (v_mem, i_na, i_k).
# `times_vec` is the fixed time base used to play stimulus vectors into sections.
# `t_vec` is the NEURON-recorded time vector (recording `h._ref_t`). Keeping
# them separate avoids a circular dependency where we would try to play using
# a vector that is itself being recorded during the simulation.
times_vec = h.Vector()
times_vec.from_python(time_arr)

t_vec = h.Vector()
t_vec.record(h._ref_t)
v_mem = h.Vector().record(h.node[CENTER_NODE](0.5)._ref_v)
# Zaznamenáváme i proudy iontovými kanály z AXNODE modelu (mA/cm2)
# Některé verze/modely nemusí mít vložené mechanismy; ošetříme to
# a použijeme nulové vektory, pokud mechanismus selže.
try:
    i_na = h.Vector().record(h.node[CENTER_NODE](0.5)._ref_ina)
except Exception:
    print("Warning: 'ina' mechanism not present at selected node; using zeros.")
    i_na = h.Vector(int(len(time_arr)) + 1)

try:
    i_k = h.Vector().record(h.node[CENTER_NODE](0.5)._ref_ik)
except Exception:
    print("Warning: 'ik' mechanism not present at selected node; using zeros.")
    i_k = h.Vector(int(len(time_arr)) + 1)

# Aplikace stimulace do tkáně
amp_vecs = [h.Vector(len(time_arr)) for _ in range(len(sections))]
for n, (sec, pos) in enumerate(sections):
    dx = pos - elec_x_um
    r_cm = np.sqrt(dx*dx + ELEC_RADIAL_UM**2) / 1e4
    ve_per_mA = RHO_E_OHM_CM / (4.0 * np.pi * r_cm)
    
    v = amp_vecs[n]
    v.from_python(ve_per_mA * stim_wave_mA)
    v.play(sec(0.5)._ref_e_extracellular, times_vec, 1)

h.dt = DT_MS
h.finitialize(-80.0)
h.continuerun(T_TOTAL_MS)

# ════════════════════════════════════════════════════════════════════════════
# 4. Vykreslení grafu
# ════════════════════════════════════════════════════════════════════════════
print("Generuji graf...")
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# Použijeme skutečný časový vektor vytvořený NEURONem (t_vec),
# aby nedocházelo k nesouladu délek mezi záznamy.
# Použijeme časovou osu `times_vec` (ta odpovídá délkám vektoru
# stimulusů a záznamovým vektorům vytvořeným pomocí `v.play`).
times_vec_np = np.array(times_vec)
plot_start_ms = DELAY_MS - 0.5
plot_end_ms = DELAY_MS + 2.0
mask = (times_vec_np >= plot_start_ms) & (times_vec_np <= plot_end_ms)
t_plot = times_vec_np[mask]

# Stimulus již je na ose `times_vec`, nemusíme jej interpovat znovu
stim_resampled = np.array(stim_wave_mA)
stim_resampled = np.interp(times_vec_np, time_arr, stim_resampled)

# Změna jednotek pro vykreslení
def _align_to_timesvec(a):
    arr = np.array(a)
    n_target = times_vec_np.size
    if arr.size == n_target:
        return arr
    if arr.size == n_target + 1:
        return arr[:-1]
    if arr.size + 1 == n_target:
        return np.concatenate([arr, arr[-1:]])
    # fallback: interpolate over normalized index space
    old_idx = np.linspace(0.0, 1.0, arr.size)
    new_idx = np.linspace(0.0, 1.0, n_target)
    return np.interp(new_idx, old_idx, arr)

i_na_aligned = _align_to_timesvec(i_na)
i_k_aligned = _align_to_timesvec(i_k)
v_mem_aligned = _align_to_timesvec(v_mem)

i_na_plot = i_na_aligned[mask]
i_k_plot = i_k_aligned[mask]
v_mem_plot = v_mem_aligned[mask]
stim_plot = stim_resampled[mask]

# Panel 1: Stimulační proud
ax1.plot(t_plot, stim_plot, 'b-', linewidth=2)
ax1.axhline(0, color='k', linestyle=':', alpha=0.5)
ax1.set_ylabel('Stimulační proud\n[mA]', fontsize=12)
ax1.set_title('Detail fyziologické odezvy na vítěznou vlnu (evo_000)', fontsize=14)
ax1.fill_between(t_plot, stim_plot, 0, where=(stim_plot < 0), color='blue', alpha=0.2)
ax1.fill_between(t_plot, stim_plot, 0, where=(stim_plot > 0), color='red', alpha=0.2)

# Panel 2: Membránové napětí
ax2.plot(t_plot, v_mem_plot, 'g-', linewidth=2)
ax2.axhline(-80.0, color='k', linestyle=':', label='Klidové napětí')
ax2.axhline(-20.0, color='r', linestyle='--', alpha=0.5, label='Práh akčního potenciálu')
ax2.set_ylabel('Membránové napětí\n[mV]', fontsize=12)
ax2.legend(loc='upper right', fontsize=9)

# Panel 3: Iontové proudy (Na+ a K+)
# Poznámka: Záporný proud (Inward) znamená, že kladné ionty tečou DO buňky (depolarizace)
ax3.plot(t_plot, i_na_plot, 'r-', linewidth=2, label='Sodík ($I_{Na}$)')
ax3.plot(t_plot, i_k_plot, 'purple', linewidth=2, label='Draslík ($I_K$)')
ax3.axhline(0, color='k', linestyle=':')
ax3.set_ylabel('Iontový proud membránou\n[mA/cm²]', fontsize=12)
ax3.set_xlabel('Čas [ms]', fontsize=12)
ax3.legend(loc='upper right', fontsize=9)

plt.tight_layout()
# Ensure target directory exists when script is run from a different CWD
os.makedirs('outputs_evolver', exist_ok=True)
out_path = 'outputs_evolver/fig_ion_channels_analysis.png'
plt.savefig(out_path, dpi=200)
print(f"Hotovo! Graf je uložen jako: {out_path}")