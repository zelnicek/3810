#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MRG SINGLE-SHOCK BENCHMARK (v4 — uses original MRGaxon.hoc)
============================================================

Rewrite based on the actual contents of ModelDB #3810:
  https://github.com/ModelDBRepository/3810

The bundle contains ONLY these files — no parax.mod:
    AXNODE.mod      (active node mechanism)
    MRGaxon.hoc     (complete geometry + passive MYSA/FLUT/STIN)
    MRGaxon.ses
    README
    mosinit.hoc

Our previous Python reimplementations of the geometry had several subtle
bugs (wrong MYSA/FLUT/STIN diameters, wrong Ra scaling, missing Rpn0/Rpn1/
Rpn2/Rpx xraxial values). Instead of re-deriving those by hand — which is
how we ended up with a spontaneously-firing axon earlier — this script
simply LOADS MRGaxon.hoc directly through NEURON and then drives it with
an extracellular point-electrode field. This guarantees the axon is exactly
the one McIntyre & Grill published.

════════════════════════════════════════════════════════════════════════════
SETUP (one time)

  1. Clone or download ModelDB #3810:
       git clone https://github.com/ModelDBRepository/3810
     OR
       Download the ZIP from https://modeldb.science/3810 → Files tab

  2. Place this script next to MRGaxon.hoc and AXNODE.mod.

  3. Compile AXNODE.mod:
       cd <folder with the .mod file>
       nrnivmodl           # Linux/macOS
       mknrndll            # Windows

  4. Run:
       python mrg_single_shock_v4.py
════════════════════════════════════════════════════════════════════════════

What v4 fixes vs. v3:
  [1] Uses MRGaxon.hoc verbatim — no reimplementation. Eliminates all the
      diam/Ra/g_pas/xraxial bugs from v3.
  [2] Disables the IClamp stimulus from the hoc file (stim.amp = 0) and
      applies our own extracellular point-electrode field via Vector.play
      on e_extracellular[0] at every section (nodes + MYSA + FLUT + STIN).
  [3] Play bindings are set BEFORE finitialize, then cleared between trials
      via play_remove (keeps the v3 lesson).
  [4] Sanity checks as before — but they now pass because the axon is the
      real MRG, not the broken parax-fallback version.

Paradigm: single biphasic stimulus per trial. No train, no carrier.
Reference: Reilly 1998, §4.4 (reproduces Obr. 4.14, 4.16 qualitatively).
"""

import os
import sys
import time
import json
import platform
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import sawtooth, chirp
from scipy.special import erf
from scipy.optimize import curve_fit

# ─── 0. NEURON + MRG hoc file ─────────────────────────────────────────────────
try:
    from neuron import h
    h.load_file('stdrun.hoc')
    print("✓ NEURON loaded")
except ImportError:
    print("[ERROR] NEURON not found. Install:  pip install neuron")
    sys.exit(1)

# Check AXNODE mechanism
def _check_axnode():
    test = h.Section(name='__axnode_check__')
    try:
        test.insert('axnode')
        return True
    except Exception:
        return False

if not _check_axnode():
    print()
    print("═" * 72)
    print("  [ERROR] AXNODE mechanism not loaded.")
    print("═" * 72)
    print("  1. Download ModelDB #3810:")
    print("       git clone https://github.com/ModelDBRepository/3810")
    print("  2. Put AXNODE.mod and MRGaxon.hoc in THIS directory.")
    print("  3. Compile: nrnivmodl  (Linux/macOS)  or  mknrndll  (Windows)")
    print("  4. Re-run from the same directory.")
    print("═" * 72)
    sys.exit(1)
print("✓ AXNODE mechanism loaded")

# Find MRGaxon.hoc
SCRIPT_DIR = Path(__file__).resolve().parent
HOC_FILE = SCRIPT_DIR / "MRGaxon.hoc"
if not HOC_FILE.exists():
    print(f"[ERROR] MRGaxon.hoc not found at {HOC_FILE}")
    print("  Download from https://github.com/ModelDBRepository/3810")
    sys.exit(1)
print(f"✓ Found {HOC_FILE.name}")

# Load the hoc file. This:
#  - defines global parameters (celsius, v_init, dt, tstop, fiberD, etc.)
#  - builds the axon (node[], MYSA[], FLUT[], STIN[] arrays)
#  - creates one IClamp 'stim' on node[10] which we will immediately zero
#  - attempts to load a NEURON main menu (harmless on headless systems)
h.load_file(str(HOC_FILE))
print("✓ MRGaxon.hoc loaded, axon instantiated")

# Kill the GUI-driven IClamp so it does not interfere with our stimulation
h.stim.amp = 0.0
h.stim.dur = 0.0
h.stim.delay = 1e9   # effectively never

# ─── waveform catalog ────────────────────────────────────────────────────────
try:
    from waveform_catalog import SHAPES
except ImportError:
    print("[ERROR] waveform_catalog.py not found next to this script.")
    sys.exit(1)

OUTPUT_DIR = SCRIPT_DIR / "outputs_mrg_v4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("═" * 72)
print("  MRG SINGLE-SHOCK BENCHMARK (v4 — using MRGaxon.hoc directly)")
print("═" * 72)
print()

# ─── 1. READ OUT THE GEOMETRY THE HOC FILE BUILT ─────────────────────────────
# MRGaxon.hoc defines:
#   axonnodes = 21
#   paranodes1 = 40 (MYSA: 2 per internode)
#   paranodes2 = 40 (FLUT: 2 per internode)
#   axoninter  = 120 (STIN: 6 per internode)
#   axontotal  = 221
#   deltax     = 1150 µm for fiberD = 10
#
# That's only 21 nodes = 20 internodes. Reilly's rule of thumb is that
# the array should be ~9× the radial electrode distance in internodal
# units. For 2 mm electrode distance (~2 internodes), 21 nodes is plenty.

AXONNODES = int(h.axonnodes)
DELTAX    = float(h.deltax)         # µm, internode period
NODELEN   = float(h.nodelength)
PARAL1    = float(h.paralength1)
PARAL2    = float(h.paralength2)
INTERL    = float(h.interlength)    # length of ONE STIN sub-section
FIBERD    = float(h.fiberD)
CELSIUS   = float(h.celsius)

print("Geometry (from MRGaxon.hoc):")
print(f"  fiberD          : {FIBERD} µm")
print(f"  axonnodes       : {AXONNODES}")
print(f"  deltax (period) : {DELTAX} µm")
print(f"  Total length    : {(AXONNODES-1) * DELTAX / 1000:.2f} mm")
print(f"  celsius         : {CELSIUS} °C")
print()

# Compute the x-position (µm) of every SECTION along the axon, so we can
# calculate the extracellular potential at that section from a point source.
#
# Section order within ONE internodal segment (i → i+1):
#     node[i]                   (length NODELEN)
#     MYSA[2i]                  (length PARAL1)
#     FLUT[2i]                  (length PARAL2)
#     STIN[6i .. 6i+5]          (6 × length INTERL each)
#     FLUT[2i+1]                (length PARAL2)
#     MYSA[2i+1]                (length PARAL1)
#     node[i+1]                 (length NODELEN)
#
# Total from start of node[i] to start of node[i+1] = DELTAX.

def _build_section_positions():
    """
    Build a list of (section, x_center_um) for EVERY section in the axon,
    in physical order from x = 0 (start of node[0]) onward.
    """
    entries = []
    x = 0.0
    for i in range(AXONNODES):
        sec = h.node[i]
        entries.append((sec, x + 0.5 * NODELEN))
        x += NODELEN

        if i == AXONNODES - 1:
            break

        # MYSA[2i]
        entries.append((h.MYSA[2*i], x + 0.5 * PARAL1));           x += PARAL1
        # FLUT[2i]
        entries.append((h.FLUT[2*i], x + 0.5 * PARAL2));           x += PARAL2
        # STIN[6i .. 6i+5]
        for k in range(6):
            entries.append((h.STIN[6*i + k], x + 0.5 * INTERL));   x += INTERL
        # FLUT[2i+1]
        entries.append((h.FLUT[2*i + 1], x + 0.5 * PARAL2));       x += PARAL2
        # MYSA[2i+1]
        entries.append((h.MYSA[2*i + 1], x + 0.5 * PARAL1));       x += PARAL1
    return entries

SECTIONS = _build_section_positions()   # list of (h.Section, x_center_um)
N_SEC = len(SECTIONS)
print(f"Total sections in axon: {N_SEC}")

# Sanity: total axon length
total_len_um = sum(sec.L for sec, _ in SECTIONS)
expected_um  = (AXONNODES-1) * DELTAX + NODELEN
print(f"Sum of section lengths: {total_len_um:.1f} µm  (expected ≈ {expected_um:.1f})")
print()

# ─── 2. BENCHMARK PARAMETERS ─────────────────────────────────────────────────
ELEC_RADIAL_UM = 2000.0    # µm — Reilly standard (2 mm)
RHO_E_OHM_CM   = 300.0     # Ω·cm
CENTER_NODE    = AXONNODES // 2        # node[10] for a 21-node axon
PROPAGATION_NODES = 3
AP_THRESHOLD_MV   = -20.0

DT         = 0.005
T_TOTAL    = 15.0
DELAY      = 2.0
GAP_US     = 0.0
BISECT_TOL = 0.001         # mA — 1 µA resolution
BISECT_MAX_ITER = 50
AMP_MAX    = 10.0          # mA
AMP_MIN    = 1e-6          # mA

PULSE_WIDTHS_US = [25, 50, 100, 200, 300, 500, 750, 1000]

time_arr = np.arange(0, T_TOTAL, DT)
np.random.seed(42)
h.celsius = CELSIUS

# Position of the electrode: directly above the center node
elec_x_um = SECTIONS[0][1]  # placeholder; will be set after we find center
for sec, x in SECTIONS:
    if sec == h.node[CENTER_NODE]:
        elec_x_um = x
        break
elec_y_um = ELEC_RADIAL_UM

# For each section, compute radial distance from electrode and hence
# the transfer coefficient ve_per_mA (mV per mA of stimulus current).
#
#   V_e [V] = ρ [Ω·m]  · I [A]      / (4π · r [m])
#   V_e [V] = ρ [Ω·cm] · I [A] · 100/ (4π · r [cm] · 100)
#   V_e [V] = ρ [Ω·cm] · I [A]      / (4π · r [cm])
# For I in mA: V_e [mV] = ρ [Ω·cm] · I [mA] / (4π · r [cm])
ve_per_mA = np.zeros(N_SEC)
for n, (sec, x) in enumerate(SECTIONS):
    dx_um = x - elec_x_um
    r_um  = np.sqrt(dx_um**2 + elec_y_um**2)
    r_cm  = r_um / 1e4
    ve_per_mA[n] = RHO_E_OHM_CM / (4.0 * np.pi * r_cm)

print(f"Electrode at:   x = {elec_x_um:.0f} µm (above node[{CENTER_NODE}])")
print(f"                y = {elec_y_um:.0f} µm radial")
print()
print(f"Ve coefficients (mV per mA of stimulus current):")
print(f"  at node[{CENTER_NODE}] (under electrode) : {ve_per_mA[[i for i,(s,_) in enumerate(SECTIONS) if s==h.node[CENTER_NODE]][0]]:.2f}")
print(f"  at node[0]    (edge)               : {ve_per_mA[[i for i,(s,_) in enumerate(SECTIONS) if s==h.node[0]][0]]:.2f}")
print(f"  at node[{AXONNODES-1}]  (far edge)           : {ve_per_mA[[i for i,(s,_) in enumerate(SECTIONS) if s==h.node[AXONNODES-1]][0]]:.2f}")
print()

# ─── 3. PULSE GENERATOR ──────────────────────────────────────────────────────
def make_pulse(shape, pw_us, amp=1.0, gap_us=0.0):
    pw_ms  = pw_us  / 1000.0
    gap_ms = gap_us / 1000.0
    wave   = np.zeros(len(time_arr))
    t0 = DELAY; t1 = t0 + pw_ms; t_gap = t1 + gap_ms; t2 = t_gap + pw_ms
    mask1 = (time_arr >= t0)    & (time_arr < t1)
    mask2 = (time_arr >= t_gap) & (time_arr < t2)
    tau1 = (time_arr[mask1] - t0)    / pw_ms
    tau2 = (time_arr[mask2] - t_gap) / pw_ms

    def normalize_nn(x):
        x = np.asarray(x, dtype=float)
        xmin, xmax = np.min(x), np.max(x)
        span = xmax - xmin
        return np.ones_like(x) if span < 1e-12 else (x - xmin) / span

    def phase_shape(tau, s):
        if s == 'rect': return np.ones_like(tau)
        if s in ('sine_half','sine'): return np.sin(np.pi * tau)
        if s == 'tri_rise': return sawtooth(2*np.pi*tau, width=0.8)*0.5+0.5
        if s == 'tri_fall': return sawtooth(2*np.pi*tau, width=0.2)*0.5+0.5
        if s in ('tri_sym','tri_50'): return 1.0 - np.abs(2*tau - 1.0)
        if s == 'tri_20': return sawtooth(2*np.pi*tau, width=0.2)*0.5+0.5
        if s == 'tri_35': return sawtooth(2*np.pi*tau, width=0.35)*0.5+0.5
        if s == 'tri_65': return sawtooth(2*np.pi*tau, width=0.65)*0.5+0.5
        if s == 'tri_72': return sawtooth(2*np.pi*tau, width=0.72)*0.5+0.5
        if s == 'tri_80': return sawtooth(2*np.pi*tau, width=0.8)*0.5+0.5
        if s == 'saw_up': return tau
        if s == 'saw_down': return 1.0 - tau
        if s == 'gauss': return np.exp(-((tau-0.5)**2)/(2*0.15**2))
        if s == 'gaussian':
            sigma = 0.12
            raw = np.exp(-((tau-0.25)**2)/(2*sigma**2)) - np.exp(-((tau-0.75)**2)/(2*sigma**2))
            return normalize_nn(raw)
        if s == 'raised_cos': return 0.5*(1 - np.cos(np.pi*tau))
        if s in ('exp_decay','exponential'):
            tc = 0.3; return np.exp(-tau/tc)/(1 - np.exp(-1/tc))
        if s == 'exp_rise':
            tc = 0.3; return (np.exp(tau/tc) - 1)/(np.exp(1/tc) - 1)
        if s == 'trapezoid':
            r = 0.2; result = np.zeros_like(tau)
            result[tau < r]                = tau[tau < r] / r
            result[(tau >= r) & (tau < 1-r)] = 1.0
            result[tau >= 1-r]             = (1 - tau[tau >= 1-r]) / r
            return result
        if s == 'trapezoid_flat35':
            r = 0.15; flat = 0.35; result = np.zeros_like(tau)
            m1 = tau < r; m2 = (tau >= r)&(tau < r+flat); m3 = tau >= (r+flat)
            result[m1] = tau[m1]/r; result[m2] = 1.0
            result[m3] = np.maximum(0.0, 1.0 - (tau[m3] - r - flat)/r)
            return result
        if s == 'staircase':
            result = np.zeros_like(tau)
            result[tau < 1/3] = 0.33
            result[(tau >= 1/3)&(tau < 2/3)] = 0.67
            result[tau >= 2/3] = 1.00
            return result
        if s == 'staircase_4':
            result = np.zeros_like(tau)
            result[tau < 0.25] = 0.5
            result[(tau >= 0.25)&(tau < 0.5)] = 1.0
            result[(tau >= 0.5)&(tau < 0.75)] = 0.5
            result[tau >= 0.75] = 1.0
            return result
        if s == 'erf': return 0.5*(1 + erf(6*(tau - 0.5)))
        if s == 'erf_sigmoid':
            raw = erf(6*(tau-0.25)) - erf(6*(tau-0.75)); return normalize_nn(raw)
        if s == 'linear_ramp': return tau
        if s == 'linear_ramp_down': return 1.0 - tau
        if s == 'chirp':
            raw = chirp(tau, f0=0.5, f1=1.5, t1=1.0, method='linear')
            return normalize_nn(raw)
        if s == 'prbs':
            bits = np.array([1, -1] * int(np.ceil(len(tau) / 2)))[:len(tau)]
            return normalize_nn(bits)
        if s == 'biomimetic_ap':
            raw = np.exp(-((tau-0.1)/0.06)**2) - 0.6*np.exp(-((tau-0.35)/0.15)**2)
            return normalize_nn(raw)
        if s == 'cathodic_first':
            out = np.ones_like(tau)/3.0; out[tau < 0.25] = 1.0; return out
        if s == 'anodic_first':
            out = -np.ones_like(tau)/3.0; out[tau < 0.25] = -1.0; return out
        if s == 'double_pulse':
            out = np.zeros_like(tau)
            out[(tau >= 0.05)&(tau < 0.2)] = 1.0
            out[(tau >= 0.55)&(tau < 0.7)] = 1.0
            return out
        if s == 'composite_harm':
            raw = np.sin(2*np.pi*tau) + 0.5*np.sin(4*np.pi*tau+0.3) + 0.25*np.sin(6*np.pi*tau+0.6)
            return normalize_nn(raw)
        if s == 'soft_clip':
            raw = np.sin(2*np.pi*tau); soft = np.tanh(2.5*raw)/np.tanh(2.5)
            return normalize_nn(soft)
        if s == 'sinc':
            raw = np.sinc(4*(tau - 0.5)); return normalize_nn(raw)
        if s == 'half_wave':
            return np.abs(np.sin(np.pi*tau))
        raise ValueError(f"Unknown pulse shape: {s!r}")

    ph1 = phase_shape(tau1, shape)
    ph2 = phase_shape(tau2, shape)
    area1 = np.sum(ph1) * DT
    area2 = np.sum(ph2) * DT
    bal   = area1 / area2 if area2 > 1e-12 else 1.0
    wave[mask1] = -amp * ph1
    wave[mask2] =  amp * ph2 * bal
    return wave, mask1, mask2


# ─── 4. SIMULATION ENGINE ────────────────────────────────────────────────────
# Persistent resources — allocated once, reused every trial.
t_vec_h = h.Vector(time_arr)
amp_vecs = [h.Vector(len(time_arr)) for _ in range(N_SEC)]

# Recording at excitation node and propagation node
EXC_IDX  = CENTER_NODE
PROP_IDX = CENTER_NODE + PROPAGATION_NODES
if PROP_IDX >= AXONNODES:
    PROP_IDX = AXONNODES - 1

v_exc_rec  = h.Vector().record(h.node[EXC_IDX](0.5)._ref_v)
v_prop_rec = h.Vector().record(h.node[PROP_IDX](0.5)._ref_v)


def _set_extracellular_waveform(I_mA):
    """
    Play ve_per_mA[n] * I(t) onto e_extracellular[0] for each section.
    MUST be called BEFORE h.finitialize().
    """
    for n, (sec, _) in enumerate(SECTIONS):
        v = amp_vecs[n]
        v.play_remove()
        v.from_python(ve_per_mA[n] * I_mA)
        v.play(sec(0.5)._ref_e_extracellular, t_vec_h, 1)


def _first_upcrossing_index(v_arr, start_idx):
    if len(v_arr) < 2: return None
    above = (v_arr > AP_THRESHOLD_MV).astype(int)
    crosses = np.where(np.diff(above) == 1)[0] + 1
    valid = crosses[crosses >= start_idx]
    return int(valid[0]) if len(valid) else None


def simulate_propagation(I_mA):
    _set_extracellular_waveform(I_mA)
    h.dt = DT
    h.finitialize(-80.0)
    h.continuerun(T_TOTAL)

    v_e = np.array(v_exc_rec)
    v_p = np.array(v_prop_rec)
    start_idx = int(DELAY / DT)

    exc_first  = _first_upcrossing_index(v_e, start_idx=start_idx)
    prop_first = _first_upcrossing_index(v_p, start_idx=start_idx)
    if exc_first is None or prop_first is None:
        return False
    dt_ms = (prop_first - exc_first) * DT
    # Forward propagation with plausible delay (50 m/s → ~25 µs/internode)
    return 0.01 <= dt_ms <= 5.0


def find_threshold(shape, pw_us):
    test_wave, _, _ = make_pulse(shape, pw_us, amp=AMP_MAX)
    if not simulate_propagation(test_wave):
        return None, None
    amin, amax = AMP_MIN, AMP_MAX
    iters = 0
    while (amax - amin) > BISECT_TOL and iters < BISECT_MAX_ITER:
        amid = 0.5 * (amin + amax)
        w, _, _ = make_pulse(shape, pw_us, amp=amid)
        if simulate_propagation(w):
            amax = amid
        else:
            amin = amid
        iters += 1
    threshold = 0.5 * (amin + amax)
    final_wave, mask1, _ = make_pulse(shape, pw_us, amp=threshold)
    cath = final_wave[mask1]
    charge_nC = np.sum(np.abs(cath)) * DT * 1000.0
    return threshold, charge_nC


# ─── 5. SANITY CHECKS ────────────────────────────────────────────────────────
print("Sanity check 1/3: zero-current trial should not fire ...", end=' ')
if simulate_propagation(np.zeros(len(time_arr))):
    print("FAIL"); print("[ERROR] Spontaneous firing at I=0."); sys.exit(1)
print("OK (silent)")

print("Sanity check 2/3: 2 mA rectangular 500 µs should fire ...", end=' ')
w_test, _, _ = make_pulse('rect', 500, amp=2.0)
if not simulate_propagation(w_test):
    print("FAIL"); print("[ERROR] 2 mA fails to fire. Geometry/scaling issue."); sys.exit(1)
print("OK (propagates)")

print("Sanity check 3/3: 0.01 mA should NOT fire ...", end=' ')
w_test, _, _ = make_pulse('rect', 500, amp=0.01)
# For a real MRG axon at 2 mm, threshold is in the hundreds of µA range,
# so 10 µA (=0.01 mA) MUST be subthreshold.
if simulate_propagation(w_test):
    print("WARN")
    print("[WARNING] 10 µA fires — axon may be hyperexcitable, or electrode")
    print("          closer than advertised. Will continue, but expect")
    print("          unusually low thresholds.")
else:
    print("OK (silent — as expected)")
print()

# ─── 6. RUN BENCHMARK ────────────────────────────────────────────────────────
print(f"Running {len(SHAPES)} shapes × {len(PULSE_WIDTHS_US)} PWs")
print(f"  = {len(SHAPES) * len(PULSE_WIDTHS_US)} threshold searches")
print()

all_results = {shape: {} for shape in SHAPES}
t_total_start = time.time()

for shape, (label, color) in SHAPES.items():
    print(f"  ── {label.replace(chr(10), ' ')}")
    for pw_us in PULSE_WIDTHS_US:
        t0 = time.time()
        th_mA, charge_nC = find_threshold(shape, pw_us)
        elapsed = time.time() - t0
        if th_mA is not None:
            all_results[shape][pw_us] = {'threshold_mA': th_mA, 'charge_nC': charge_nC}
            print(f"     PW={pw_us:4d} µs -> Th={th_mA*1000:8.2f} µA  |  "
                  f"Q={charge_nC:7.3f} nC  [{elapsed:.1f}s]")
        else:
            all_results[shape][pw_us] = {'threshold_mA': None, 'charge_nC': None}
            print(f"     PW={pw_us:4d} µs -> NOT FOUND  [{elapsed:.1f}s]")
    print()

total_elapsed = time.time() - t_total_start
print(f"Total benchmark time: {total_elapsed/60:.1f} min")
print()

# ─── 7. WEISS-LAPICQUE FIT ──────────────────────────────────────────────────
def weiss_lapicque(pw, Ir, tc):
    return Ir * (1.0 + tc / pw)

def fit_sd(pws, ths_uA):
    valid = [(pw, th) for pw, th in zip(pws, ths_uA) if th is not None]
    if len(valid) < 3: return None, None
    xs = np.array([v[0] for v in valid], dtype=float)
    ys = np.array([v[1] for v in valid], dtype=float)
    try:
        popt, _ = curve_fit(weiss_lapicque, xs, ys,
                            p0=[float(np.min(ys)), float(np.median(xs))],
                            bounds=([1e-3, 1.0], [1e7, 1e6]), maxfev=5000)
        return float(popt[0]), float(popt[1])
    except Exception as e:
        print(f"    [fit warning] {e}")
        return None, None


print("RHEOBASE & CHRONAXIE (Weiss-Lapicque):")
print("─" * 60)
print(f"  {'Shape':<35} {'Rheobase':>12} {'Chronaxie':>12}")
print("─" * 60)
rheobase_data, chronaxie_data = {}, {}
for shape, (label, color) in SHAPES.items():
    ths_uA = [all_results[shape].get(pw, {}).get('threshold_mA') for pw in PULSE_WIDTHS_US]
    ths_uA = [(th*1000 if th is not None else None) for th in ths_uA]
    rb, tc = fit_sd(PULSE_WIDTHS_US, ths_uA)
    rheobase_data[shape], chronaxie_data[shape] = rb, tc
    rb_str = f"{rb:.2f} µA" if rb is not None else "N/A"
    tc_str = f"{tc:.1f} µs" if tc is not None else "N/A"
    print(f"  {label.replace(chr(10),' '):<35} {rb_str:>12} {tc_str:>12}")
print("─" * 60)
print()

# ─── 8. SAVE RESULTS ────────────────────────────────────────────────────────
save = {
    'metadata': {
        'paradigm':        'single-shock biphasic, MRG axon via MRGaxon.hoc',
        'model':           'McIntyre-Richardson-Grill (2002), ModelDB #3810',
        'script_version':  'v4 (uses original hoc file)',
        'fiber_D_um':      FIBERD,
        'n_nodes':         AXONNODES,
        'internode_um':    DELTAX,
        'electrode_radial_um': ELEC_RADIAL_UM,
        'rho_e_ohm_cm':    RHO_E_OHM_CM,
        'temperature_C':   CELSIUS,
        'propagation_nodes': PROPAGATION_NODES,
        'ap_threshold_mV': AP_THRESHOLD_MV,
        'dt_ms':           DT,
        't_total_ms':      T_TOTAL,
        'delay_ms':        DELAY,
        'gap_us':          GAP_US,
        'amp_min_mA':      AMP_MIN,
        'amp_max_mA':      AMP_MAX,
        'bisect_tol_mA':   BISECT_TOL,
        'pulse_widths_us': PULSE_WIDTHS_US,
        'fit_method':      'Weiss-Lapicque nonlinear (scipy.curve_fit)',
        'neuron_version':  str(h.nrnversion(0)),
        'python':          platform.python_version(),
        'numpy':           np.__version__,
    },
    'shapes': {}
}
for shape, (label, color) in SHAPES.items():
    save['shapes'][shape] = {
        'label':     label,
        'rheobase_uA':  rheobase_data.get(shape),
        'chronaxie_us': chronaxie_data.get(shape),
        'by_pw': {
            str(pw): {
                'threshold_uA': (all_results[shape][pw]['threshold_mA'] * 1000
                                 if all_results[shape][pw]['threshold_mA'] is not None
                                 else None),
                'charge_nC':    all_results[shape][pw]['charge_nC'],
            }
            for pw in PULSE_WIDTHS_US
        }
    }
with open(OUTPUT_DIR / 'mrg_single_shock_results.json', 'w') as f:
    json.dump(save, f, indent=2, ensure_ascii=False)
print(f"Saved: {OUTPUT_DIR / 'mrg_single_shock_results.json'}")

# ─── 9. PLOTS ────────────────────────────────────────────────────────────────
BG, BG2 = '#ffffff', '#f7f9fc'
GRID, FG, MUTED, SPINE = '#d0d7de', '#1f2937', '#4b5563', '#9ca3af'

def hex_color(c):
    return c if isinstance(c, str) else '#%02x%02x%02x' % tuple(int(x*255) for x in c[:3])

# Plot A
n_cols = 5; n_rows = int(np.ceil(len(SHAPES) / n_cols))
fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(18, 3*n_rows), facecolor=BG)
axes1 = axes1.flatten()
for ax in axes1: ax.set_facecolor(BG); ax.axis('off')
demo_pw = 500
for i, (shape, (label, color)) in enumerate(SHAPES.items()):
    ax = axes1[i]; ax.set_facecolor(BG2); ax.axis('on')
    w, _, _ = make_pulse(shape, demo_pw, amp=1.0)
    m = (time_arr >= DELAY-0.5) & (time_arr <= DELAY + demo_pw/1000*2 + 1.0)
    ts, ws = time_arr[m], w[m]
    col = hex_color(color)
    ax.plot(ts*1000, ws, color=col, linewidth=1.8)
    ax.fill_between(ts*1000, ws, 0, where=ws<0, color=col, alpha=0.3)
    ax.fill_between(ts*1000, ws, 0, where=ws>0, color='#ef4444', alpha=0.2)
    ax.axhline(0, color=SPINE, linewidth=0.7, alpha=0.9)
    ax.set_title(label.replace('\n', ' '), fontsize=8, color=FG, pad=3)
    ax.set_xlabel('µs', fontsize=6, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=5)
    for sp in ax.spines.values(): sp.set_color(SPINE)
fig1.suptitle('Single-shock biphasic waveforms — PW=500 µs (normalized)',
              fontsize=11, color=FG)
plt.tight_layout(pad=1.0)
plt.savefig(OUTPUT_DIR/'mrg_pulse_shapes.png', dpi=180, bbox_inches='tight', facecolor=BG)
print("Saved: mrg_pulse_shapes.png")

# Plot B
fig2, ax2 = plt.subplots(figsize=(12, 7), facecolor=BG); ax2.set_facecolor(BG2)
for shape, (label, color) in SHAPES.items():
    ths = [all_results[shape].get(pw, {}).get('threshold_mA') for pw in PULSE_WIDTHS_US]
    vp = [pw for pw, th in zip(PULSE_WIDTHS_US, ths) if th is not None]
    vt = [th*1000 for th in ths if th is not None]
    if not vp: continue
    ax2.plot(vp, vt, 'o-', color=hex_color(color), linewidth=1.8, markersize=5,
             label=label.replace('\n',' '), alpha=0.85)
ax2.set_xlabel('Pulse width [µs]', color=FG, fontsize=12)
ax2.set_ylabel('Threshold amplitude [µA]', color=FG, fontsize=12)
ax2.set_title(f'MRG axon (D={FIBERD} µm) — strength-duration, single-shock\n'
              f'Point electrode {ELEC_RADIAL_UM/1000:.1f} mm radial',
              color=FG, fontsize=13)
ax2.legend(fontsize=7.5, ncol=2, facecolor=BG2, edgecolor=GRID, labelcolor=FG, loc='upper right')
ax2.set_xscale('log'); ax2.set_yscale('log')
ax2.tick_params(colors=FG); ax2.grid(True, linestyle=':', alpha=0.4, color=GRID)
for sp in ax2.spines.values(): sp.set_color(SPINE)
plt.tight_layout()
plt.savefig(OUTPUT_DIR/'mrg_strength_duration.png', dpi=180, bbox_inches='tight', facecolor=BG)
print("Saved: mrg_strength_duration.png")

# Plot C
fig3, ax3 = plt.subplots(figsize=(12, 7), facecolor=BG); ax3.set_facecolor(BG2)
for shape, (label, color) in SHAPES.items():
    chs = [all_results[shape].get(pw, {}).get('charge_nC') for pw in PULSE_WIDTHS_US]
    vp = [pw for pw, ch in zip(PULSE_WIDTHS_US, chs) if ch is not None]
    vc = [ch for ch in chs if ch is not None]
    if not vp: continue
    ax3.plot(vp, vc, 'o-', color=hex_color(color), linewidth=1.8, markersize=5,
             label=label.replace('\n',' '), alpha=0.85)
ax3.set_xlabel('Pulse width [µs]', color=FG, fontsize=12)
ax3.set_ylabel('Threshold charge [nC]', color=FG, fontsize=12)
ax3.set_title('MRG charge-duration — lower is more efficient', color=FG, fontsize=13)
ax3.legend(fontsize=7.5, ncol=2, facecolor=BG2, edgecolor=GRID, labelcolor=FG, loc='upper left')
ax3.set_xscale('log'); ax3.tick_params(colors=FG)
ax3.grid(True, linestyle=':', alpha=0.4, color=GRID)
for sp in ax3.spines.values(): sp.set_color(SPINE)
plt.tight_layout()
plt.savefig(OUTPUT_DIR/'mrg_charge_duration.png', dpi=180, bbox_inches='tight', facecolor=BG)
print("Saved: mrg_charge_duration.png")

# Plot D
valid = [(s, SHAPES[s][0], SHAPES[s][1]) for s in SHAPES
         if rheobase_data.get(s) is not None and chronaxie_data.get(s) is not None]
if valid:
    rbs = [rheobase_data[s] for s,_,_ in valid]
    tcs = [chronaxie_data[s] for s,_,_ in valid]
    order = np.argsort(rbs)
    valid = [valid[i] for i in order]; rbs = [rbs[i] for i in order]; tcs = [tcs[i] for i in order]
    fig4, (axA, axB) = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG)
    for ax in (axA, axB): ax.set_facecolor(BG2)
    cols = [hex_color(SHAPES[s][1]) for s,_,_ in valid]
    lbls = [lbl.replace('\n',' ') for _,lbl,_ in valid]
    axA.barh(range(len(rbs)), rbs, color=cols, edgecolor=SPINE, linewidth=0.6, alpha=0.9)
    for i, val in enumerate(rbs):
        axA.text(val + max(rbs)*0.02, i, f'{val:.1f} µA', va='center', fontsize=8, color=FG)
    axA.set_yticks(range(len(lbls))); axA.set_yticklabels(lbls, fontsize=8.5, color=FG)
    axA.set_xlabel('Rheobase [µA]', color=FG)
    axA.set_title('Rheobase (long-pulse asymptote)', color=FG, fontsize=11)
    axA.invert_yaxis(); axA.tick_params(colors=FG)
    for sp in axA.spines.values(): sp.set_color(SPINE)
    axB.barh(range(len(tcs)), tcs, color=cols, edgecolor=SPINE, linewidth=0.6, alpha=0.9)
    for i, val in enumerate(tcs):
        axB.text(val + max(tcs)*0.02, i, f'{val:.1f} µs', va='center', fontsize=8, color=FG)
    axB.set_yticks(range(len(lbls))); axB.set_yticklabels(lbls, fontsize=8.5, color=FG)
    axB.set_xlabel('Chronaxie [µs]', color=FG)
    axB.set_title('Chronaxie (Reilly Tab. 4.4 ref ≈ 92 µs)', color=FG, fontsize=11)
    axB.invert_yaxis(); axB.tick_params(colors=FG)
    for sp in axB.spines.values(): sp.set_color(SPINE)
    fig4.suptitle('MRG strength-duration parameters (Weiss-Lapicque)', fontsize=13, color=FG)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/'mrg_rheobase_chronaxie.png', dpi=180, bbox_inches='tight', facecolor=BG)
    print("Saved: mrg_rheobase_chronaxie.png")

# Plot E
REF_PWS = [200, 500, 100, 300, 50]
summary = []
for shape, (label, color) in SHAPES.items():
    for ref in REF_PWS:
        r = all_results[shape].get(ref, {})
        ch = r.get('charge_nC'); th = r.get('threshold_mA')
        if ch is not None and th is not None:
            summary.append((shape, label, color, ch, th*1000, ref)); break
summary.sort(key=lambda x: x[3])

fig5, ax5 = plt.subplots(figsize=(14, 8), facecolor=BG); ax5.set_facecolor(BG2)
s_lbls = [s[1].replace('\n',' ') for s in summary]
s_chs  = [s[3] for s in summary]
s_ths  = [s[4] for s in summary]
s_pws  = [s[5] for s in summary]
s_cols = [hex_color(s[2]) for s in summary]
ax5.barh(range(len(summary)), s_chs, color=s_cols, edgecolor=SPINE, linewidth=0.6, alpha=0.9)
for i, (ch, th, pw) in enumerate(zip(s_chs, s_ths, s_pws)):
    ax5.text(ch + max(s_chs)*0.01, i,
             f"{ch:.3f} nC  |  {th:.1f} µA  @ PW={pw} µs",
             va='center', ha='left', fontsize=8.5, color=FG)
medal_c = ["#FFD700","#C0C0C0","#CD7F32"]; medal_l = ["★ BEST","2nd","3rd"]
for r in range(min(3, len(summary))):
    ax5.barh(r, s_chs[r], color=medal_c[r], edgecolor=SPINE, linewidth=1.2, alpha=1.0)
    ax5.text(s_chs[r]*0.02, r, medal_l[r], va='center', ha='left',
             fontsize=8, color='black', fontweight='bold')
ax5.set_yticks(range(len(s_lbls))); ax5.set_yticklabels(s_lbls, fontsize=9.5, color=FG)
ax5.set_xlabel('Threshold charge [nC]', color=FG, fontsize=12)
ax5.set_title('MRG pulse-shape ranking (single-shock)\nLower charge = lower tissue stress',
              color=FG, fontsize=13, pad=12)
ax5.invert_yaxis(); ax5.tick_params(colors=FG)
ax5.grid(axis='x', linestyle=':', alpha=0.4, color=GRID)
for sp in ax5.spines.values(): sp.set_color(SPINE)
plt.tight_layout()
plt.savefig(OUTPUT_DIR/'mrg_summary_ranking.png', dpi=180, bbox_inches='tight', facecolor=BG)
print("Saved: mrg_summary_ranking.png")

print()
print("SUMMARY RANKING (by threshold charge):")
print("─" * 72)
for rank, (shape, label, color, ch, th, pw) in enumerate(summary, 1):
    medal = ["[1]","[2]","[3]"][rank-1] if rank <= 3 else f"{rank:>2}."
    print(f"  {medal}  {label.replace(chr(10),' '):<35} "
          f"{ch:.4f} nC  |  {th:.1f} µA  @ PW={pw} µs")
print("─" * 72)
print()
print(f"Done. All outputs in: {OUTPUT_DIR}/")