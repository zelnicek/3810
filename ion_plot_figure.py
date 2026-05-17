#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analýza neurostimulačních vln podle standardů publikační literatury.

Standard: Wongsarnpigoon & Grill (2010), Foutz & McIntyre, Peña et al. (2024).

Hlavní rysy:
  - Threshold hledán binární bisekcí s tolerancí 0.1 µA na proud,
    který způsobí PROPAGOVANÝ akční potenciál (záznam ve vzdáleném uzlu).
  - Validace skrze druhý uzel v centru (oba musí spike).
  - Metriky: I_th, Q (náboj na fázi), E (energie), P_peak, conduction velocity.
  - Layout publikační kvality: 5 panelů (vlny, V_mem, I_iont, tabulka metrik).
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from scipy.interpolate import CubicSpline
from neuron import h

# ════════════════════════════════════════════════════════════════════════════
# 1. NASTAVENÍ A KONSTANTY
# ════════════════════════════════════════════════════════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs_evolver")
DATA_FILE = os.path.join(OUTPUT_DIR, "final_waveform_pool.json")
SAVE_FILE = os.path.join(OUTPUT_DIR, "fig_publication_comparison.png")

# Časové parametry simulace
DT_MS         = 0.005      # časový krok [ms]
T_TOTAL_MS    = 15.0       # celková doba simulace [ms]
DELAY_MS      = 2.0        # zpoždění začátku stimulace [ms]
PW_US         = 200        # délka jedné fáze pulzu [µs]

# Geometrie a tkáň
ELEC_RADIAL_UM = 2000.0    # vzdálenost elektrody od nervu [µm]
RHO_E_OHM_CM   = 300.0     # specifický odpor tkáně [Ω·cm]

# Detekce akčního potenciálu (validační kritéria)
V_REST            = -80.0  # klidový potenciál [mV]
SPIKE_PEAK_MV     = 0.0    # AP musí dosáhnout aspoň 0 mV
DVDT_THRESHOLD    = 100.0  # minimální dV/dt pro spike [mV/ms]
DETECTION_OFFSET_NODES = 5 # vzdálenost detekčního uzlu od centra (počet uzlů)

# Bisekce (standardně 0.1 µA tolerance dle Peña 2024)
TOL_MA            = 0.0001 # 0.1 µA
MAX_BISECT_ITER   = 60
MAX_SEARCH_MA     = 50.0   # absolutní horní mez

# ════════════════════════════════════════════════════════════════════════════
# 2. GENERÁTORY VLN
# ════════════════════════════════════════════════════════════════════════════
def render_phase_curve(control_points, n_samples=100):
    knots = np.linspace(0, 1, len(control_points))
    aug_knots = np.concatenate([[-0.05], knots, [1.05]])
    aug_vals  = np.concatenate([[0.0], control_points, [0.0]])
    spl = CubicSpline(aug_knots, aug_vals, bc_type='natural', extrapolate=False)
    return spl(np.linspace(0, 1, n_samples))

def render_evolved_waveform(waveform, pw_us, time_arr):
    wave = np.zeros(len(time_arr))
    pw_ms = pw_us / 1000.0
    gap_us = float(np.clip(waveform[16], 0.0, 3000.0))
    gap_ms = gap_us / 1000.0

    t0 = DELAY_MS
    t1 = t0 + pw_ms
    t2 = t1 + gap_ms
    t3 = t2 + pw_ms

    cathodic_mask = (time_arr >= t0) & (time_arr < t1)
    anodic_mask   = (time_arr >= t2) & (time_arr < t3)
    n1, n2 = int(np.sum(cathodic_mask)), int(np.sum(anodic_mask))

    if n1 > 0 and n2 > 0:
        cathodic_curve = render_phase_curve(waveform[:8], n_samples=n1)
        anodic_curve   = render_phase_curve(waveform[8:16], n_samples=n2)

        if np.sum(cathodic_curve) > 0: cathodic_curve = -cathodic_curve
        if np.sum(anodic_curve) < 0:   anodic_curve   = -anodic_curve

        peak_cath = float(np.max(np.abs(cathodic_curve)))
        if peak_cath > 0: cathodic_curve = cathodic_curve / peak_cath
        peak_anod = float(np.max(np.abs(anodic_curve)))
        anodic_curve = anodic_curve / max(peak_anod, 1e-9)

        area_cath = float(np.sum(cathodic_curve)) * DT_MS
        area_anod_unit = float(np.sum(anodic_curve)) * DT_MS
        if area_anod_unit != 0:
            balance_factor = -area_cath / area_anod_unit
            wave[cathodic_mask] = cathodic_curve
            wave[anodic_mask]   = anodic_curve * balance_factor
    return wave

def render_square_waveform(pw_us, time_arr):
    wave = np.zeros(len(time_arr))
    pw_ms = pw_us / 1000.0
    t0, t1, t2 = DELAY_MS, DELAY_MS + pw_ms, DELAY_MS + 2*pw_ms
    wave[(time_arr >= t0) & (time_arr < t1)] = -1.0
    wave[(time_arr >= t1) & (time_arr < t2)] =  1.0
    return wave

# ════════════════════════════════════════════════════════════════════════════
# 3. NEURON SETUP
# ════════════════════════════════════════════════════════════════════════════
if not os.path.exists(DATA_FILE):
    print(f"[CHYBA] Nebyl nalezen soubor: {DATA_FILE}")
    sys.exit(1)

with open(DATA_FILE, "r") as f:
    data = json.load(f)

best_wf_params = np.array(data["waveforms"][0])

time_drive = np.arange(0, T_TOTAL_MS, DT_MS)
t_drive_vec = h.Vector(time_drive)
evo_unit = render_evolved_waveform(best_wf_params, PW_US, time_drive)
sq_unit  = render_square_waveform(PW_US, time_drive)

print("Spouštím prostředí NEURON a načítám MRG Axon...")
os.chdir(SCRIPT_DIR)
h.load_file('stdrun.hoc')
h.load_file('MRGaxon.hoc')

h.celsius = 36.0
N_NODES = int(h.axonnodes)
CENTER_NODE = N_NODES // 2
DETECTION_NODE = min(CENTER_NODE + DETECTION_OFFSET_NODES, N_NODES - 1)

# Geometrie axonu (poziční mapa)
sections = []
x = 0.0
for i in range(N_NODES):
    sections.append((h.node[i], x + 0.5 * h.nodelength)); x += h.nodelength
    if i == N_NODES - 1: break
    sections.append((h.MYSA[2*i], x + 0.5 * h.paralength1)); x += h.paralength1
    sections.append((h.FLUT[2*i], x + 0.5 * h.paralength2)); x += h.paralength2
    for k in range(6):
        sections.append((h.STIN[6*i + k], x + 0.5 * h.interlength)); x += h.interlength
    sections.append((h.FLUT[2*i + 1], x + 0.5 * h.paralength2)); x += h.paralength2
    sections.append((h.MYSA[2*i + 1], x + 0.5 * h.paralength1)); x += h.paralength1

elec_x_um = next(pos for sec, pos in sections if sec == h.node[CENTER_NODE])
center_seg = h.node[CENTER_NODE](0.5)
detect_seg = h.node[DETECTION_NODE](0.5)

center_pos_um = next(pos for sec, pos in sections if sec == h.node[CENTER_NODE])
detect_pos_um = next(pos for sec, pos in sections if sec == h.node[DETECTION_NODE])
node_distance_mm = abs(detect_pos_um - center_pos_um) / 1000.0

print(f"  Centrální uzel: {CENTER_NODE}, detekční uzel: {DETECTION_NODE}")
print(f"  Vzdálenost mezi uzly: {node_distance_mm:.2f} mm")

# Záznamové vektory
t_rec     = h.Vector().record(h._ref_t)
v_center  = h.Vector().record(center_seg._ref_v)
v_detect  = h.Vector().record(detect_seg._ref_v)
i_na, i_k = None, None
if hasattr(center_seg, '_ref_ina_axnode'):
    i_na = h.Vector().record(center_seg._ref_ina_axnode)
    i_k  = h.Vector().record(center_seg._ref_ik_axnode)
elif hasattr(center_seg, '_ref_ina'):
    i_na = h.Vector().record(center_seg._ref_ina)
    i_k  = h.Vector().record(center_seg._ref_ik)

# Aplikace pole z elektrody
amp_vecs = [h.Vector(len(time_drive)) for _ in range(len(sections))]
ve_per_mA = np.zeros(len(sections))
for n, (sec, pos) in enumerate(sections):
    r_cm = np.sqrt((pos - elec_x_um)**2 + ELEC_RADIAL_UM**2) / 1e4
    ve_per_mA[n] = RHO_E_OHM_CM / (4.0 * np.pi * r_cm)

def run_sim(stim_wave_mA):
    for n, (sec, pos) in enumerate(sections):
        v = amp_vecs[n]
        v.play_remove()
        v.from_python(ve_per_mA[n] * stim_wave_mA)
        v.play(sec(0.5)._ref_e_extracellular, t_drive_vec, 1)

    h.dt = DT_MS
    h.finitialize(V_REST)
    h.continuerun(T_TOTAL_MS)

    return {
        't':       np.array(t_rec),
        'v_cen':   np.array(v_center),
        'v_det':   np.array(v_detect),
        'i_na':    np.array(i_na) if i_na else None,
        'i_k':     np.array(i_k)  if i_k  else None,
    }

# ════════════════════════════════════════════════════════════════════════════
# 4. DETEKCE PROPAGOVANÉHO AP A METRIKY
# ════════════════════════════════════════════════════════════════════════════
def detect_propagated_ap(result, stim_end_ms):
    """Validuje propagovaný AP na obou uzlech."""
    t = result['t']
    detect_start = stim_end_ms + 0.05
    mask_t = t >= detect_start
    if not np.any(mask_t):
        return False, {}

    info = {}
    for label, v_arr in [('center', result['v_cen']), ('detect', result['v_det'])]:
        v_w = v_arr[mask_t]
        t_w = t[mask_t]
        if len(v_w) < 3:
            return False, {}

        peak = float(np.max(v_w))
        peak_idx = int(np.argmax(v_w))
        peak_t = float(t_w[peak_idx])
        dvdt = np.gradient(v_w, t_w)
        max_dvdt = float(np.max(dvdt))

        info[f'{label}_peak_mv']  = peak
        info[f'{label}_peak_t']   = peak_t
        info[f'{label}_max_dvdt'] = max_dvdt

        if peak < SPIKE_PEAK_MV:    return False, info
        if max_dvdt < DVDT_THRESHOLD: return False, info

    if info['detect_peak_t'] <= info['center_peak_t']:
        return False, info

    return True, info

def find_threshold(wave_unit_array, label=""):
    """Standardní bisekce s validovanou propagací AP."""
    nonzero = np.where(np.abs(wave_unit_array) > 1e-9)[0]
    if len(nonzero) == 0:
        return float('nan'), {}
    stim_end_ms = (nonzero[-1] + 1) * DT_MS

    # Krok 1: najdi spolehlivé hi
    hi = 0.5
    hi_info = None
    while hi <= MAX_SEARCH_MA:
        res = run_sim(wave_unit_array * hi)
        ok, info = detect_propagated_ap(res, stim_end_ms)
        if ok:
            hi_info = info
            break
        hi *= 2.0

    if hi_info is None:
        print(f"  [!] {label}: práh nenalezen ani při {MAX_SEARCH_MA} mA.")
        return float('nan'), {}

    # Krok 2: najdi spolehlivé lo
    lo = hi / 2.0
    while lo > 1e-4:
        res = run_sim(wave_unit_array * lo)
        ok, _ = detect_propagated_ap(res, stim_end_ms)
        if not ok:
            break
        hi = lo
        lo /= 2.0

    print(f"  [{label}] Bracket: lo={lo*1000:.2f} µA, hi={hi*1000:.2f} µA")

    # Krok 3: jemná bisekce
    last_info = hi_info
    it = 0
    for it in range(MAX_BISECT_ITER):
        if (hi - lo) <= TOL_MA: break
        mid = 0.5 * (lo + hi)
        res = run_sim(wave_unit_array * mid)
        ok, info = detect_propagated_ap(res, stim_end_ms)
        if ok:
            hi = mid
            last_info = info
        else:
            lo = mid

    print(f"  [{label}] Konvergováno za {it+1} iterací → I_th = {hi*1000:.2f} µA")
    return hi, last_info

def compute_metrics(wave_unit, threshold_mA, ap_info):
    """Publikační metriky podle Wongsarnpigoon & Grill 2010."""
    R_LOAD_KOHM = 1.0  # referenční odpor pro srovnatelnost vln

    if np.isnan(threshold_mA):
        return {'I_th_uA': float('nan'), 'Q_nC': float('nan'),
                'E_nJ': float('nan'), 'P_peak_uW': float('nan'),
                'v_cv_ms': float('nan')}

    wave_real_mA = wave_unit * threshold_mA

    # Q: jen katodická fáze
    cath = wave_real_mA[wave_real_mA < 0]
    Q_nC = float(np.sum(np.abs(cath))) * DT_MS * 1000.0  # mA·ms × 1000 = nC

    # E: ∫I² dt × R
    nonzero_mask = np.abs(wave_real_mA) > 1e-9
    i_squared = wave_real_mA[nonzero_mask] ** 2
    E_nJ = float(np.sum(i_squared)) * DT_MS * R_LOAD_KOHM * 1e3

    # P_peak
    P_peak_uW = float(np.max(i_squared)) * R_LOAD_KOHM * 1e3

    # CV
    if ap_info and 'center_peak_t' in ap_info and 'detect_peak_t' in ap_info:
        dt_ms = ap_info['detect_peak_t'] - ap_info['center_peak_t']
        v_cv = node_distance_mm / dt_ms if dt_ms > 0 else float('nan')
    else:
        v_cv = float('nan')

    return {
        'I_th_uA':   threshold_mA * 1000.0,
        'Q_nC':      Q_nC,
        'E_nJ':      E_nJ,
        'P_peak_uW': P_peak_uW,
        'v_cv_ms':   v_cv,
    }

# ════════════════════════════════════════════════════════════════════════════
# 5. VÝPOČET PRAHŮ A METRIK
# ════════════════════════════════════════════════════════════════════════════
print("\n[ FÁZE 1: Hledání prahů s validací propagace ]")
print("\nČtvercová vlna:")
sq_th, sq_info = find_threshold(sq_unit, label="Square")

print("\nEvoluční vlna:")
ev_th, ev_info = find_threshold(evo_unit, label="Evolved")

print("\n[ FÁZE 2: Výpočet metrik ]")
sq_metrics = compute_metrics(sq_unit, sq_th, sq_info)
ev_metrics = compute_metrics(evo_unit, ev_th, ev_info)

def fmt(v, prec=2):
    return f"{v:.{prec}f}" if not np.isnan(v) else "—"

print(f"\n  Čtvercová: I_th={fmt(sq_metrics['I_th_uA'])} µA, Q={fmt(sq_metrics['Q_nC'])} nC, "
      f"E={fmt(sq_metrics['E_nJ'])} nJ, P_pk={fmt(sq_metrics['P_peak_uW'])} µW, "
      f"v_CV={fmt(sq_metrics['v_cv_ms'])} m/s")
print(f"  Evoluční:  I_th={fmt(ev_metrics['I_th_uA'])} µA, Q={fmt(ev_metrics['Q_nC'])} nC, "
      f"E={fmt(ev_metrics['E_nJ'])} nJ, P_pk={fmt(ev_metrics['P_peak_uW'])} µW, "
      f"v_CV={fmt(ev_metrics['v_cv_ms'])} m/s")

# ════════════════════════════════════════════════════════════════════════════
# 6. FINÁLNÍ SIMULACE PRO GRAF
# ════════════════════════════════════════════════════════════════════════════
print("\n[ FÁZE 3: Generování publikačního grafu ]")
res_sq = run_sim(sq_unit  * sq_th)
res_ev = run_sim(evo_unit * ev_th)

plot_start_ms = DELAY_MS - 0.3
plot_end_ms   = DELAY_MS + 4.0
mask = (res_sq['t'] >= plot_start_ms) & (res_sq['t'] <= plot_end_ms)
t_p = res_sq['t'][mask]

stim_sq_p = render_square_waveform(PW_US, t_p) * sq_th
stim_ev_p = render_evolved_waveform(best_wf_params, PW_US, t_p) * ev_th

# ════════════════════════════════════════════════════════════════════════════
# 7. VYKRESLENÍ
# ════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'DejaVu Sans',
    'axes.linewidth': 1.0,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'legend.frameon': False,
    'legend.fontsize': 9,
})

fig = plt.figure(figsize=(15, 10))
gs = GridSpec(3, 3, figure=fig,
              height_ratios=[1.0, 1.6, 1.6],
              width_ratios=[1, 1, 0.85],
              hspace=0.45, wspace=0.32,
              left=0.06, right=0.98, top=0.92, bottom=0.07)

fig.suptitle('Srovnání vln: standardní bifázický obdélník vs. AI-evoluční optimalizovaný pulz',
             fontsize=14, fontweight='bold', y=0.97)

v_max_stim = max(np.max(np.abs(stim_sq_p)), np.max(np.abs(stim_ev_p))) * 1.15
v_max_ion  = max(np.max(np.abs(res_sq['i_na'][mask])),
                 np.max(np.abs(res_ev['i_na'][mask]))) * 1.15

COL_STIM, COL_VCEN, COL_VDET = '#1f4e79', '#2ca02c', '#ff7f0e'
COL_NA, COL_K, COL_TOT       = '#d62728', '#9467bd', '#8c564b'

# Řádek 1: tvar vln
for col, (label, stim_p) in enumerate([
    ('A  •  Bifázický obdélník', stim_sq_p),
    ('B  •  Evoluční vlna',      stim_ev_p)
]):
    ax = fig.add_subplot(gs[0, col])
    ax.plot(t_p, stim_p, color=COL_STIM, linewidth=2.0)
    ax.fill_between(t_p, stim_p, 0, where=(stim_p < 0), color=COL_STIM, alpha=0.25)
    ax.fill_between(t_p, stim_p, 0, where=(stim_p > 0), color='#c44', alpha=0.20)
    ax.axhline(0, color='k', linewidth=0.5, alpha=0.5)
    ax.set_title(label, fontweight='bold', loc='left')
    ax.set_ylabel('I_stim  [mA]')
    ax.set_ylim(-v_max_stim, v_max_stim)
    ax.set_xlim(plot_start_ms, plot_end_ms)
    ax.grid(True, alpha=0.2, linestyle='--')

# Řádek 2: V_mem
for col, (title, res, info) in enumerate([
    ('C  •  Membránové napětí (čtverec)', res_sq, sq_info),
    ('D  •  Membránové napětí (evoluční)', res_ev, ev_info)
]):
    ax = fig.add_subplot(gs[1, col])
    ax.plot(t_p, res['v_cen'][mask], color=COL_VCEN, linewidth=2.0,
            label=f'Centrální uzel (#{CENTER_NODE})')
    ax.plot(t_p, res['v_det'][mask], color=COL_VDET, linewidth=2.0,
            label=f'Vzdálený uzel (#{DETECTION_NODE}, +{node_distance_mm:.2f} mm)')
    ax.axhline(V_REST, color='k', linestyle=':', linewidth=1.0, label='V_rest')
    ax.axhline(0, color='gray', linestyle=':', linewidth=0.7, alpha=0.6)
    ax.set_title(title, fontweight='bold', loc='left')
    ax.set_ylabel('V_mem  [mV]')
    ax.set_ylim(-95, 45)
    ax.set_xlim(plot_start_ms, plot_end_ms)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.legend(loc='upper right', fontsize=8)

    if info and 'center_peak_t' in info:
        for k, c in [('center_peak_t', COL_VCEN), ('detect_peak_t', COL_VDET)]:
            tp = info[k]
            if plot_start_ms <= tp <= plot_end_ms:
                ax.axvline(tp, color=c, linestyle='--', alpha=0.4, linewidth=0.8)

# Řádek 3: Iontové proudy
for col, (title, res) in enumerate([
    ('E  •  Iontové proudy (čtverec)',  res_sq),
    ('F  •  Iontové proudy (evoluční)', res_ev)
]):
    ax = fig.add_subplot(gs[2, col])
    ina = res['i_na'][mask]
    ik  = res['i_k'][mask]
    itot = ina + ik
    ax.plot(t_p, ina,  color=COL_NA,  linewidth=1.8, label='I_Na')
    ax.plot(t_p, ik,   color=COL_K,   linewidth=1.8, label='I_K')
    ax.plot(t_p, itot, color=COL_TOT, linewidth=1.5, label='I_total', alpha=0.85)
    ax.axhline(0, color='k', linewidth=0.5, alpha=0.5)
    ax.set_title(title, fontweight='bold', loc='left')
    ax.set_ylabel('J_mem  [mA/cm²]')
    ax.set_xlabel('Čas  [ms]')
    ax.set_ylim(-v_max_ion, v_max_ion)
    ax.set_xlim(plot_start_ms, plot_end_ms)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.legend(loc='lower right', ncol=3, fontsize=8)

    if col == 1:
        ax.annotate('', xy=(plot_end_ms - 0.05, v_max_ion * 0.85),
                    xytext=(plot_end_ms - 0.05, v_max_ion * 0.30),
                    arrowprops=dict(arrowstyle='->', color='k', lw=1.2))
        ax.text(plot_end_ms - 0.18, v_max_ion * 0.55, 'Outward',
                ha='right', va='center', fontsize=8, fontweight='bold')
        ax.annotate('', xy=(plot_end_ms - 0.05, -v_max_ion * 0.85),
                    xytext=(plot_end_ms - 0.05, -v_max_ion * 0.30),
                    arrowprops=dict(arrowstyle='->', color='k', lw=1.2))
        ax.text(plot_end_ms - 0.18, -v_max_ion * 0.55, 'Inward',
                ha='right', va='center', fontsize=8, fontweight='bold')

# Tabulka metrik
ax_tab = fig.add_subplot(gs[0:3, 2])
ax_tab.axis('off')
ax_tab.set_title('G  •  Souhrnné metriky', fontweight='bold', loc='left',
                 fontsize=11, pad=12)

def pct_change(new, old):
    if np.isnan(new) or np.isnan(old) or old == 0: return "—"
    delta = (new - old) / old * 100
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.1f}%"

table_rows = [
    ['Metrika',           'Čtverec',                          'Evoluční',                          'Δ'],
    ['I_th [µA]',         fmt(sq_metrics['I_th_uA'],   1),   fmt(ev_metrics['I_th_uA'],   1),   pct_change(ev_metrics['I_th_uA'], sq_metrics['I_th_uA'])],
    ['Q [nC]',            fmt(sq_metrics['Q_nC'],      2),   fmt(ev_metrics['Q_nC'],      2),   pct_change(ev_metrics['Q_nC'], sq_metrics['Q_nC'])],
    ['E [nJ] @ 1 kΩ',     fmt(sq_metrics['E_nJ'],      2),   fmt(ev_metrics['E_nJ'],      2),   pct_change(ev_metrics['E_nJ'], sq_metrics['E_nJ'])],
    ['P_peak [µW] @1 kΩ', fmt(sq_metrics['P_peak_uW'], 2),   fmt(ev_metrics['P_peak_uW'], 2),   pct_change(ev_metrics['P_peak_uW'], sq_metrics['P_peak_uW'])],
    ['v_CV [m/s]',        fmt(sq_metrics['v_cv_ms'],   2),   fmt(ev_metrics['v_cv_ms'],   2),   pct_change(ev_metrics['v_cv_ms'], sq_metrics['v_cv_ms'])],
]

tab = ax_tab.table(cellText=table_rows, loc='upper center',
                   cellLoc='center', colWidths=[0.34, 0.22, 0.22, 0.22])
tab.auto_set_font_size(False)
tab.set_fontsize(10)
tab.scale(1.0, 2.0)

for j in range(4):
    cell = tab[0, j]
    cell.set_facecolor('#2c3e50')
    cell.set_text_props(weight='bold', color='white')
    cell.set_height(0.10)

for i in range(1, len(table_rows)):
    for j in range(4):
        cell = tab[i, j]
        cell.set_height(0.085)
        if i % 2 == 0:
            cell.set_facecolor('#f5f5f5')
        if j == 3 and i > 0:
            txt = table_rows[i][3]
            if txt.startswith('-'):
                cell.set_text_props(color='#1a7d1a', weight='bold')
            elif txt.startswith('+'):
                cell.set_text_props(color='#c0392b', weight='bold')

note_text = (
    f"Detekce: AP propagovaný\nz uzlu #{CENTER_NODE} → #{DETECTION_NODE}\n\n"
    f"Kritéria propagace:\n"
    f"  • V_peak ≥ {SPIKE_PEAK_MV:.0f} mV\n"
    f"  • dV/dt ≥ {DVDT_THRESHOLD:.0f} mV/ms\n"
    f"  • t_detect > t_center\n\n"
    f"Bisekce: tol. {TOL_MA*1000:.1f} µA\n"
    f"PW = {PW_US} µs (na fázi)\n"
    f"R_load = 1 kΩ (referenční)"
)
ax_tab.text(0.02, 0.30, note_text, transform=ax_tab.transAxes,
            fontsize=8.5, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f9f9f9',
                      edgecolor='#ccc', linewidth=0.8))

os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.savefig(SAVE_FILE, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nHotovo! Publikační graf uložen jako:\n  {SAVE_FILE}")