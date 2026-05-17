#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════════════
  MRG MULTI-FIBER ADVANCED ANALYSIS  (v4 - finalni stabilni verze)
  ────────────────────────────────────────────────────────────────────────────
  Zlepseni v4:
    • Spravne cisteni IClamp pri prepinani prumeru (zadne 'objref stim')
    • Pri spusteni s '--skip-phase1' nacte checkpoint a delam jen faze 2+3
    • Pri opetovnem spusteni automaticky preskoci hotove prumery

  POUZITI:
    python3 mrg_multifiber_analysis.py                # bezne spusteni
    python3 mrg_multifiber_analysis.py --skip-phase1  # preskoci hotovou fazi 1
    python3 mrg_multifiber_analysis.py --reset        # zaciname od zacatku
═══════════════════════════════════════════════════════════════════════════════
"""

import os, sys, time, json, platform, copy, re, pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import sawtooth, chirp
from scipy.special import erf

# ─── 0. CLI ──────────────────────────────────────────────────────────────────
RESET = '--reset' in sys.argv
SKIP_PHASE1 = '--skip-phase1' in sys.argv

# ─── 1. NEURON ───────────────────────────────────────────────────────────────
try:
    from neuron import h
    h.load_file('stdrun.hoc')
    print("✓ NEURON loaded")
except ImportError:
    print("[ERROR] NEURON not installed: pip install neuron"); sys.exit(1)

SCRIPT_DIR = Path(__file__).resolve().parent
HOC_FILE   = SCRIPT_DIR / "MRGaxon.hoc"
if not HOC_FILE.exists():
    print(f"[ERROR] MRGaxon.hoc not found at {HOC_FILE}"); sys.exit(1)

def _check_axnode():
    test = h.Section(name='__check__')
    try: test.insert('axnode'); return True
    except: return False
if not _check_axnode():
    print("[ERROR] AXNODE.mod not compiled. Run: nrnivmodl"); sys.exit(1)
print("✓ AXNODE mechanism available")

try:
    from waveform_catalog import SHAPES
except ImportError:
    print("[ERROR] waveform_catalog.py not found"); sys.exit(1)
print(f"✓ waveform_catalog loaded ({len(SHAPES)} shapes available)")

OUTPUT_DIR = SCRIPT_DIR / "outputs_multifiber"
OUTPUT_DIR.mkdir(exist_ok=True)
CHECKPOINT_FILE = OUTPUT_DIR / "_checkpoint.pkl"

if RESET and CHECKPOINT_FILE.exists():
    CHECKPOINT_FILE.unlink()
    print("✓ Checkpoint smazan, zaciname od zacatku")

print("═" * 75)
print("  MRG MULTI-FIBER ADVANCED ANALYSIS v4")
print("═" * 75)
print()

# ─── 2. KONFIGURACE ──────────────────────────────────────────────────────────
FIBER_DIAMETERS = [5.7, 8.7, 10.0, 11.5, 14.0, 16.0]
REF_DIAMETER = 10.0
PULSE_WIDTHS_US = [50, 100, 200, 500, 1000]
ELECTRODE_AXIAL_SHIFTS_UM = [-575.0, -287.5, 0.0, 287.5, 575.0]
DEFAULT_DISTANCE_UM = 2000.0

RHO_E_OHM_CM   = 300.0
ACCESS_R_OHM   = 1000.0
ELECTRODE_AREA_MM2 = 0.05

DT         = 0.005
T_TOTAL    = 15.0
DELAY      = 2.0
GAP_US     = 0.0
BISECT_TOL = 0.001
BISECT_MAX_ITER = 50
AMP_MAX    = 20.0
AMP_SEARCH_MAX = 320.0
AMP_GROWTH_FACTOR = 2.0
# Nektere tvary (hlavne anodic_first) muzou mit vyrazne vyssi prah.
SHAPE_AMP_SEARCH_MAX = {
    'anodic_first': 2000.0,
}
_SHAPE_AMP_SEARCH_MAX_NORM = {k.strip().lower(): float(v)
                              for k, v in SHAPE_AMP_SEARCH_MAX.items()}
AMP_MIN    = 1e-6
PROPAGATION_NODES = 3
AP_THRESHOLD_MV   = -20.0

time_arr = np.arange(0, T_TOTAL, DT)
np.random.seed(42)

# Pouzij vsechny tvary dostupne v katalogu waveform_catalog.py
SELECTED_SHAPES = list(SHAPES.keys())
if not SELECTED_SHAPES:
    print("[ERROR] Zadny ze SELECTED_SHAPES neni v catalogu!"); sys.exit(1)

print("KONFIGURACE:")
print(f"  Prumery vlaken:    {FIBER_DIAMETERS} µm")
print(f"  Sirky pulzu:       {PULSE_WIDTHS_US} µs")
print(f"  Posuny elektrody:  {ELECTRODE_AXIAL_SHIFTS_UM} µm")
print(f"  Tvary ({len(SELECTED_SHAPES)}): {SELECTED_SHAPES}")
print()

# ─── 3. WAVEFORM GENERATOR ───────────────────────────────────────────────────
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

# ─── 4. NACITANI MRG MODELU (zjednodusena robustni verze) ────────────────────
_LOADED_DIAMETER = None
_CURRENT_AXON = None

def _safe_zero_stim():
    """Vynulovat IClamp 'stim' - jen pokud existuje a je platny."""
    try:
        h.stim.amp = 0.0
        h.stim.dur = 0.0
        h.stim.delay = 1e9
    except Exception:
        # stim neexistuje nebo je vazany na smazanou sekci - ignoruj
        pass

def _safe_clear_axon():
    """Smazat axon - nejdriv vynulovat stim, pak sekce."""
    _safe_zero_stim()

    # Najdeme sekce ktere chceme smazat
    sec_to_delete = []
    for sec in h.allsec():
        nm = sec.name()
        if any(prefix in nm for prefix in ['node', 'MYSA', 'FLUT', 'STIN']):
            sec_to_delete.append(sec)

    # Smazat jednu po druhe
    for sec in sec_to_delete:
        try:
            h.delete_section(sec=sec)
        except Exception:
            pass

def load_axon_for_diameter(fiberD):
    """Nacist MRG axon pro dany prumer vlakna."""
    global _LOADED_DIAMETER, _CURRENT_AXON

    if _LOADED_DIAMETER == fiberD and _CURRENT_AXON is not None:
        return _CURRENT_AXON

    # Vytvorit modifikovany hoc soubor
    tmp_hoc = SCRIPT_DIR / f".tmp_mrg_d{fiberD}.hoc"
    with open(HOC_FILE, 'r') as f:
        hoc_content = f.read()
    new_content = re.sub(
        r'fiberD\s*=\s*[\d.]+',
        f'fiberD={fiberD}',
        hoc_content,
        count=1
    )
    with open(tmp_hoc, 'w') as f:
        f.write(new_content)

    # Smazat predchozi axon
    _safe_clear_axon()

    # Nacist novy hoc - vytvori novy stim na nove sekci node[10]
    h.load_file(str(tmp_hoc))

    # Hned vypneme stim, dokud je platny
    _safe_zero_stim()

    AXONNODES = int(h.axonnodes)
    DELTAX    = float(h.deltax)
    NODELEN   = float(h.nodelength)
    PARAL1    = float(h.paralength1)
    PARAL2    = float(h.paralength2)
    INTERL    = float(h.interlength)
    CELSIUS   = float(h.celsius)
    h.celsius = CELSIUS

    sections = []
    x = 0.0
    for i in range(AXONNODES):
        sections.append((h.node[i], x + 0.5 * NODELEN))
        x += NODELEN
        if i == AXONNODES - 1: break
        sections.append((h.MYSA[2*i], x + 0.5 * PARAL1));         x += PARAL1
        sections.append((h.FLUT[2*i], x + 0.5 * PARAL2));         x += PARAL2
        for k in range(6):
            sections.append((h.STIN[6*i + k], x + 0.5 * INTERL)); x += INTERL
        sections.append((h.FLUT[2*i + 1], x + 0.5 * PARAL2));     x += PARAL2
        sections.append((h.MYSA[2*i + 1], x + 0.5 * PARAL1));     x += PARAL1

    cache = {
        'sections':  sections,
        'n_sec':     len(sections),
        'axonnodes': AXONNODES,
        'deltax':    DELTAX,
        'fiberD':    fiberD,
    }

    _LOADED_DIAMETER = fiberD
    _CURRENT_AXON = cache

    try: tmp_hoc.unlink()
    except: pass

    return cache

# ─── 5. EXTRACELULARNI POLE ──────────────────────────────────────────────────
def compute_ve_coefficients(sections, elec_x_um, elec_y_um):
    n = len(sections)
    ve = np.zeros(n)
    for i, (sec, x) in enumerate(sections):
        dx = x - elec_x_um
        r_um = np.sqrt(dx**2 + elec_y_um**2)
        r_cm = r_um / 1e4
        ve[i] = RHO_E_OHM_CM / (4.0 * np.pi * r_cm)
    return ve

_t_vec_h = h.Vector(time_arr)
_amp_vec_pool = []

def _reset_vec_pool(n_sec):
    global _amp_vec_pool
    _amp_vec_pool = [h.Vector(len(time_arr)) for _ in range(n_sec)]

def _set_extracellular_waveform(I_mA, sections, ve_per_mA):
    for n_idx, (sec, _) in enumerate(sections):
        v = _amp_vec_pool[n_idx]
        v.play_remove()
        v.from_python(ve_per_mA[n_idx] * I_mA)
        v.play(sec(0.5)._ref_e_extracellular, _t_vec_h, 1)

# ─── 6. SIMULACE ─────────────────────────────────────────────────────────────
def _first_upcrossing(v_arr, start_idx):
    if len(v_arr) < 2: return None
    above = (v_arr > AP_THRESHOLD_MV).astype(int)
    crosses = np.where(np.diff(above) == 1)[0] + 1
    valid = crosses[crosses >= start_idx]
    return int(valid[0]) if len(valid) else None

def simulate_propagation(I_mA, axon, ve_per_mA, exc_idx, prop_idx):
    sections = axon['sections']
    _set_extracellular_waveform(I_mA, sections, ve_per_mA)
    v_e = h.Vector().record(h.node[exc_idx](0.5)._ref_v)
    v_p = h.Vector().record(h.node[prop_idx](0.5)._ref_v)
    h.dt = DT
    h.finitialize(-80.0)
    h.continuerun(T_TOTAL)
    v_e_arr = np.array(v_e)
    v_p_arr = np.array(v_p)
    start_idx = int(DELAY / DT)
    e_first = _first_upcrossing(v_e_arr, start_idx)
    p_first = _first_upcrossing(v_p_arr, start_idx)
    if e_first is None or p_first is None:
        return False
    dt_ms = (p_first - e_first) * DT
    return 0.01 <= dt_ms <= 5.0

def _shape_amp_search_max(shape):
    key = str(shape).strip().lower()
    return _SHAPE_AMP_SEARCH_MAX_NORM.get(key, AMP_SEARCH_MAX)

def find_threshold(shape, pw_us, axon, ve_per_mA, exc_idx, prop_idx, return_meta=False):
    # Adaptive probe: if AP is not reached at AMP_MAX, keep increasing current
    # up to AMP_SEARCH_MAX to reduce false FAILs for weak shapes.
    amp_search_max = _shape_amp_search_max(shape)
    amax = AMP_MAX
    while True:
        test_wave, _, _ = make_pulse(shape, pw_us, amp=amax)
        if simulate_propagation(test_wave, axon, ve_per_mA, exc_idx, prop_idx):
            break
        if amax >= amp_search_max - 1e-12:
            meta = {
                'status': 'no_ap',
                'max_tested_mA': float(amax),
                'search_max_mA': float(amp_search_max),
            }
            return (None, meta) if return_meta else None
        next_amax = min(amax * AMP_GROWTH_FACTOR, amp_search_max)
        if next_amax <= amax + 1e-12:
            meta = {
                'status': 'no_ap',
                'max_tested_mA': float(amax),
                'search_max_mA': float(amp_search_max),
            }
            return (None, meta) if return_meta else None
        amax = next_amax

    amin = AMP_MIN
    for _ in range(BISECT_MAX_ITER):
        if (amax - amin) <= BISECT_TOL: break
        amid = 0.5 * (amin + amax)
        w, _, _ = make_pulse(shape, pw_us, amp=amid)
        if simulate_propagation(w, axon, ve_per_mA, exc_idx, prop_idx):
            amax = amid
        else:
            amin = amid
    th = 0.5 * (amin + amax)
    meta = {'status': 'ok', 'max_tested_mA': float(amax)}
    return (th, meta) if return_meta else th

# ─── 7. METRIKY ──────────────────────────────────────────────────────────────
def compute_metrics(shape, pw_us, threshold_mA):
    if threshold_mA is None:
        return {'charge_nC': None, 'energy_uJ': None,
                'charge_density': None, 'shannon_k': None}
    wave, mask1, _ = make_pulse(shape, pw_us, amp=threshold_mA)
    cath = wave[mask1]
    charge_nC = np.sum(np.abs(cath)) * DT * 1000
    energy_nJ = np.sum(wave**2) * ACCESS_R_OHM * DT
    energy_uJ = energy_nJ / 1000.0
    area_cm2 = ELECTRODE_AREA_MM2 / 100.0
    charge_uC = charge_nC / 1000.0
    charge_density = charge_uC / area_cm2
    shannon_k = (np.log10(charge_uC) + np.log10(charge_density)) \
        if (charge_uC > 0 and charge_density > 0) else None
    return {
        'charge_nC':      float(charge_nC),
        'energy_uJ':      float(energy_uJ),
        'charge_density': float(charge_density),
        'shannon_k':      float(shannon_k) if shannon_k is not None else None,
    }

# ─── 8. CHECKPOINT ───────────────────────────────────────────────────────────
def save_checkpoint(state):
    tmp = CHECKPOINT_FILE.with_suffix('.tmp')
    with open(tmp, 'wb') as f:
        pickle.dump(state, f)
    tmp.replace(CHECKPOINT_FILE)

def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[warn] Checkpoint poskozeny: {e}, ignoruji.")
            return None
    return None

def _diameter_is_complete(main_results, fiberD):
    """True pouze kdyz dany prumer obsahuje vsechny tvary i vsechny PW."""
    if fiberD not in main_results:
        return False
    dct = main_results[fiberD]
    for shape in SELECTED_SHAPES:
        if shape not in dct:
            return False
        for pw in PULSE_WIDTHS_US:
            if pw not in dct[shape]:
                return False
            rec = dct[shape][pw]
            if 'threshold_mA' not in rec:
                return False
            # Pokud je v checkpointu no_ap s mensim limitem, je potreba prepocet.
            if rec.get('threshold_mA') is None:
                meta = rec.get('threshold_meta', {})
                if meta.get('status') == 'no_ap':
                    max_prev = meta.get('max_tested_mA')
                    shape_max = _shape_amp_search_max(shape)
                    if max_prev is None or max_prev < (shape_max - 1e-12):
                        return False
    return True

# ─── 9. FAZE 1 ───────────────────────────────────────────────────────────────
print()
print("═" * 75)
print(" FAZE 1: Hlavni sken prahu pres prumery × tvary × sirky pulzu")
print("═" * 75)

state = load_checkpoint()
if state is not None:
    main_results = state.get('main_results', {})
    cp_completed = state.get('completed_diameters', set())
    if isinstance(cp_completed, list):
        cp_completed = set(cp_completed)

    # Zpetna kompatibilita: checkpoint muze byt z drivejska s mensim seznamem tvaru.
    completed_diameters = {d for d in FIBER_DIAMETERS if _diameter_is_complete(main_results, d)}
    print(f"  ✓ Nacten checkpoint: hotovo {len(completed_diameters)}/{len(FIBER_DIAMETERS)} prumeru")
    print(f"    Hotove prumery: {sorted(completed_diameters)}")
    dropped = sorted(cp_completed - completed_diameters)
    if dropped:
        print(f"  [info] Neuplna data v checkpointu, prepocitam prumery: {dropped}")
else:
    main_results = {}
    completed_diameters = set()

t_phase1 = time.time()
all_done = all(fd in completed_diameters for fd in FIBER_DIAMETERS)

if SKIP_PHASE1 or all_done:
    if all_done:
        print(f"  ✓ Vsechny prumery uz hotove, preskakujeme fazi 1")
    else:
        print(f"  ✓ --skip-phase1 zadano, preskakujeme fazi 1")
else:
    # Pro stabilni navaznost faze 2 nechame referencni prumer jako posledni nacteny.
    phase1_order = [d for d in FIBER_DIAMETERS if d != REF_DIAMETER] + [REF_DIAMETER]
    for d_idx, fiberD in enumerate(phase1_order):
        if fiberD in completed_diameters:
            print(f"\n  ─── D = {fiberD} µm ({d_idx+1}/{len(phase1_order)}) [hotovo, preskakujem] ───")
            continue

        print(f"\n  ─── D = {fiberD} µm ({d_idx+1}/{len(phase1_order)}) ───")

        try:
            axon = load_axon_for_diameter(fiberD)
        except Exception as e:
            print(f"    [ERROR] Selhal load axonu: {e}")
            continue

        n_sec = axon['n_sec']
        AXONNODES = axon['axonnodes']
        sections = axon['sections']
        _reset_vec_pool(n_sec)

        CENTER_NODE = AXONNODES // 2
        PROP_NODE   = min(CENTER_NODE + PROPAGATION_NODES, AXONNODES - 1)
        elec_x_um = next(x for sec, x in sections if sec == h.node[CENTER_NODE])
        ve = compute_ve_coefficients(sections, elec_x_um, DEFAULT_DISTANCE_UM)

        if fiberD not in main_results:
            main_results[fiberD] = {}
        for shape in SELECTED_SHAPES:
            if shape not in main_results[fiberD]:
                main_results[fiberD][shape] = {}
            for pw in PULSE_WIDTHS_US:
                # Pokud je kombinace uz spocitana, preskoc.
                # Vyjimka: drivejsi no_ap s nizsim limitem po zmene konfigurace prepocitej.
                if pw in main_results[fiberD][shape] and 'threshold_mA' in main_results[fiberD][shape][pw]:
                    rec = main_results[fiberD][shape][pw]
                    th_prev = rec.get('threshold_mA')
                    if th_prev is not None:
                        continue
                    meta_prev = rec.get('threshold_meta', {})
                    if meta_prev.get('status') == 'no_ap':
                        max_prev = meta_prev.get('max_tested_mA')
                        shape_max = _shape_amp_search_max(shape)
                        if max_prev is not None and max_prev >= (shape_max - 1e-12):
                            continue
                t0 = time.time()
                th_meta = {'status': 'error', 'max_tested_mA': None}
                try:
                    th, th_meta = find_threshold(
                        shape, pw, axon, ve, CENTER_NODE, PROP_NODE, return_meta=True
                    )
                except Exception as e:
                    print(f"    {shape:>16s} PW={pw:>4d}µs  ERROR: {e}")
                    th = None
                metrics = compute_metrics(shape, pw, th)
                main_results[fiberD][shape][pw] = {
                    'threshold_mA': th,
                    'threshold_meta': th_meta,
                    **metrics,
                }
                elapsed = time.time() - t0
                if th is not None:
                    print(f"    {shape:>16s} PW={pw:>4d}µs  "
                          f"Ith={th*1000:>7.1f}µA  E={metrics['energy_uJ']:>7.3f}µJ  "
                          f"Q={metrics['charge_nC']:>6.2f}nC  [{elapsed:.1f}s]")
                else:
                    max_tested = th_meta.get('max_tested_mA')
                    if th_meta.get('status') == 'no_ap' and max_tested is not None:
                        search_max = th_meta.get('search_max_mA')
                        print(f"    {shape:>16s} PW={pw:>4d}µs  "
                              f"FAIL (no AP <= {max_tested:.1f} mA; "
                              f"search_max={search_max:.1f} mA)  [{elapsed:.1f}s]")
                    else:
                        print(f"    {shape:>16s} PW={pw:>4d}µs  FAIL  [{elapsed:.1f}s]")

        completed_diameters.add(fiberD)
        save_checkpoint({
            'main_results': main_results,
            'completed_diameters': list(completed_diameters),
        })
        print(f"    ✓ Checkpoint ulozen ({len(completed_diameters)}/{len(FIBER_DIAMETERS)} hotovo)")

    print(f"\n  Faze 1 hotova: {(time.time()-t_phase1)/60:.1f} min")

# ─── 10. FAZE 2: ROBUSTNOST ──────────────────────────────────────────────────
# DULEZITE: Zde nezavedame novy axon pres load_axon_for_diameter, protoze pri
# druhem volani by se snazil smazat sekce, na kterych vise stim. Misto toho:
# pouzijeme aktualne nacteny axon (z konce faze 1 nebo z noveho startu).
print()
print("═" * 75)
print(" FAZE 2: Robustnost — citlivost prahu na posun elektrody")
print("═" * 75)

ref_diameter = REF_DIAMETER
shape_avg_energy = {}
if ref_diameter in main_results:
    for shape in SELECTED_SHAPES:
        if shape not in main_results[ref_diameter]:
            continue
        energies = [main_results[ref_diameter][shape][pw]['energy_uJ']
                    for pw in PULSE_WIDTHS_US
                    if main_results[ref_diameter][shape][pw]['energy_uJ'] is not None]
        if energies:
            shape_avg_energy[shape] = np.mean(energies)

robustness_results = {}
PW_FOR_ROBUST = 200

if shape_avg_energy:
    top_shapes = sorted(shape_avg_energy.keys(), key=lambda s: shape_avg_energy[s])[:3]
    print(f"  Top-3 tvary podle prumerne energie: {top_shapes}")

    # Poznamka: opakovane mazani sekci + reload muze v NEURONu shodit proces
    # ("section was deleted"). Proto po fazi 1 preferujeme pouzit uz nacteny axon.
    # Pokud aktualni axon neexistuje (napr. beh jen s --skip-phase1), nacteme ref_diameter.
    if _CURRENT_AXON is None:
        print(f"  Neni nacten zadny axon, nacitam D={ref_diameter}...")
        try:
            axon = load_axon_for_diameter(ref_diameter)
        except Exception as e:
            print(f"  [ERROR] Nacteni D={ref_diameter} selhalo: {e}")
            axon = None
    else:
        axon = _CURRENT_AXON
        if _LOADED_DIAMETER != ref_diameter:
            print(f"  [info] Preskakuju reload D={ref_diameter} kvuli stabilite NEURON.")
            print(f"         Robustnost pocitam na aktualnim axonu D={_LOADED_DIAMETER} µm.")

    if axon is not None:
        try:
            n_sec = axon['n_sec']; sections = axon['sections']
            AXONNODES = axon['axonnodes']
            _reset_vec_pool(n_sec)
            CENTER_NODE = AXONNODES // 2
            PROP_NODE   = min(CENTER_NODE + PROPAGATION_NODES, AXONNODES - 1)
            elec_x_center = next(x for sec, x in sections if sec == h.node[CENTER_NODE])

            for shape in top_shapes:
                robustness_results[shape] = {}
                for shift_um in ELECTRODE_AXIAL_SHIFTS_UM:
                    elec_x = elec_x_center + shift_um
                    ve = compute_ve_coefficients(sections, elec_x, DEFAULT_DISTANCE_UM)
                    try:
                        th = find_threshold(shape, PW_FOR_ROBUST, axon, ve,
                                            CENTER_NODE, PROP_NODE)
                    except Exception as e:
                        print(f"    [warn] {shape} shift={shift_um}: {e}")
                        th = None
                    robustness_results[shape][shift_um] = th
                    if th is not None:
                        print(f"    {shape:>16s}  shift={shift_um:>+7.1f}µm  "
                              f"Ith={th*1000:>7.1f}µA")
                    else:
                        print(f"    {shape:>16s}  shift={shift_um:>+7.1f}µm  FAIL")

                # Checkpoint po kazdem tvaru
                save_checkpoint({
                    'main_results': main_results,
                    'completed_diameters': list(completed_diameters),
                    'robustness_results': robustness_results,
                })
        except Exception as e:
            print(f"  [error] Faze 2 selhala: {e}")
            print(f"  Pokracujeme s tim, co mame v robustness_results.")
else:
    print("  [warn] Zadne validni energie - preskakujeme robustnost")

# ─── 11. FAZE 3: SELEKTIVITA ─────────────────────────────────────────────────
print()
print("═" * 75)
print(" FAZE 3: Selektivita — schopnost rozlisit tlusta od tenkych vlaken")
print("═" * 75)
selectivity_results = {}
thin_d  = min(FIBER_DIAMETERS)
thick_d = max(FIBER_DIAMETERS)

for shape in SELECTED_SHAPES:
    selectivity_results[shape] = {}
    for pw in PULSE_WIDTHS_US:
        try:
            th_thin  = main_results[thin_d][shape][pw]['threshold_mA']
            th_thick = main_results[thick_d][shape][pw]['threshold_mA']
            ratio = th_thin / th_thick if (th_thin is not None and th_thick is not None
                                            and th_thick > 0) else None
        except KeyError:
            ratio = None
        selectivity_results[shape][pw] = ratio
    valid = [r for r in selectivity_results[shape].values() if r is not None]
    if valid:
        print(f"    {shape:>16s}  prum. selektivita = {np.mean(valid):.2f}×")

# ─── 12. EXPORT JSON ─────────────────────────────────────────────────────────
print()
print("═" * 75)
print(" Export vysledku")
print("═" * 75)

export = {
    'metadata': {
        'paradigm':             'multi-fiber MRG analysis v4',
        'fiber_diameters':      FIBER_DIAMETERS,
        'completed_diameters':  sorted(completed_diameters),
        'pulse_widths_us':      PULSE_WIDTHS_US,
        'shapes':               SELECTED_SHAPES,
        'electrode_distance_um': DEFAULT_DISTANCE_UM,
        'axial_shifts_um':      ELECTRODE_AXIAL_SHIFTS_UM,
        'rho_e_ohm_cm':         RHO_E_OHM_CM,
        'access_R_ohm':         ACCESS_R_OHM,
        'electrode_area_mm2':   ELECTRODE_AREA_MM2,
        'shannon_threshold_k':  1.85,
        'python':               platform.python_version(),
        'numpy':                np.__version__,
    },
    'main_thresholds': {
        str(d): {sh: {str(pw): main_results[d][sh][pw] for pw in PULSE_WIDTHS_US}
                 for sh in SELECTED_SHAPES if sh in main_results.get(d, {})}
        for d in FIBER_DIAMETERS if d in main_results
    },
    'robustness': {
        sh: {str(s): robustness_results[sh][s] for s in ELECTRODE_AXIAL_SHIFTS_UM
             if s in robustness_results[sh]}
        for sh in robustness_results
    },
    'selectivity': {
        sh: {str(pw): selectivity_results[sh][pw] for pw in PULSE_WIDTHS_US}
        for sh in selectivity_results
    },
}

json_path = OUTPUT_DIR / "multifiber_results.json"
with open(json_path, 'w') as f:
    json.dump(export, f, indent=2, ensure_ascii=False)
print(f"  ✓ {json_path}")

# Ulozime finalni checkpoint
save_checkpoint({
    'main_results': main_results,
    'completed_diameters': list(completed_diameters),
    'robustness_results': robustness_results,
    'selectivity_results': selectivity_results,
})

# ─── 13. VIZUALIZACE ─────────────────────────────────────────────────────────
plt.style.use('default')
BG = '#ffffff'; BG2 = '#f7f9fc'
GRID = '#d0d7de'; FG = '#1f2937'; SPINE = '#9ca3af'

PW_PLOT = 200
cmap = plt.cm.viridis

# Plot A: Prah vs. prumer
fig, ax = plt.subplots(figsize=(11, 6.5), facecolor=BG); ax.set_facecolor(BG2)
for i, shape in enumerate(SELECTED_SHAPES):
    ths = []
    for d in FIBER_DIAMETERS:
        try: ths.append(main_results[d][shape][PW_PLOT]['threshold_mA'])
        except KeyError: ths.append(None)
    ths_uA = [t*1000 if t is not None else np.nan for t in ths]
    color = cmap(i / max(1, len(SELECTED_SHAPES)-1))
    label_disp = SHAPES[shape][0].replace('\n',' ') if shape in SHAPES else shape
    ax.plot(FIBER_DIAMETERS, ths_uA, 'o-', label=label_disp, color=color, lw=1.6, ms=5)
ax.set_xlabel('Prumer vlakna [µm]', color=FG, fontsize=12)
ax.set_ylabel('Prah [µA]', color=FG, fontsize=12)
ax.set_title(f'Prah stimulace vs. prumer vlakna (PW = {PW_PLOT} µs)',
             color=FG, fontsize=13)
ax.set_yscale('log'); ax.grid(True, ls=':', alpha=0.4, color=GRID)
ax.legend(fontsize=8, ncol=2, loc='upper right')
for sp in ax.spines.values(): sp.set_color(SPINE)
plt.tight_layout()
plt.savefig(OUTPUT_DIR/'01_threshold_vs_diameter.png', dpi=180, bbox_inches='tight')
print(f"  ✓ 01_threshold_vs_diameter.png")
plt.close()

# Plot B: Energy heatmap
fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
energy_matrix = np.zeros((len(SELECTED_SHAPES), len(FIBER_DIAMETERS)))
for i, sh in enumerate(SELECTED_SHAPES):
    for j, d in enumerate(FIBER_DIAMETERS):
        try: e = main_results[d][sh][PW_PLOT]['energy_uJ']
        except KeyError: e = None
        energy_matrix[i, j] = e if e is not None else np.nan
im = ax.imshow(energy_matrix, aspect='auto', cmap='YlOrRd', origin='lower')
ax.set_xticks(range(len(FIBER_DIAMETERS)))
ax.set_xticklabels([f'{d}' for d in FIBER_DIAMETERS])
ax.set_yticks(range(len(SELECTED_SHAPES))); ax.set_yticklabels(SELECTED_SHAPES)
ax.set_xlabel('Prumer vlakna [µm]', color=FG)
ax.set_ylabel('Tvar pulzu', color=FG)
ax.set_title(f'Spotreba energie [µJ] (PW = {PW_PLOT} µs)\nNizsi = uspornejsi',
             color=FG, fontsize=13)
plt.colorbar(im, ax=ax, label='Energie [µJ]')
for i in range(energy_matrix.shape[0]):
    for j in range(energy_matrix.shape[1]):
        v = energy_matrix[i, j]
        if not np.isnan(v):
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    color='black', fontsize=7)
plt.tight_layout()
plt.savefig(OUTPUT_DIR/'02_energy_heatmap.png', dpi=180, bbox_inches='tight')
print(f"  ✓ 02_energy_heatmap.png")
plt.close()

# Plot C: Selektivita
fig, ax = plt.subplots(figsize=(11, 6), facecolor=BG); ax.set_facecolor(BG2)
for i, sh in enumerate(SELECTED_SHAPES):
    ratios = [selectivity_results[sh][pw] for pw in PULSE_WIDTHS_US]
    valid = [(pw, r) for pw, r in zip(PULSE_WIDTHS_US, ratios) if r is not None]
    if valid:
        xs, ys = zip(*valid)
        label_disp = SHAPES[sh][0].replace('\n',' ') if sh in SHAPES else sh
        ax.plot(xs, ys, 'o-', label=label_disp,
                color=cmap(i/max(1,len(SELECTED_SHAPES)-1)), lw=1.6, ms=5)
ax.axhline(1.0, color='gray', ls='--', alpha=0.5, label='zadna selektivita')
ax.set_xlabel('Sirka pulzu [µs]', color=FG, fontsize=12)
ax.set_ylabel('Selektivita = Ith(thin) / Ith(thick)', color=FG, fontsize=12)
ax.set_title(f'Selektivita pro tlusta vlakna ({thick_d} µm vs. {thin_d} µm)',
             color=FG, fontsize=13)
ax.set_xscale('log'); ax.grid(True, ls=':', alpha=0.4, color=GRID)
ax.legend(fontsize=8, ncol=2, loc='best')
for sp in ax.spines.values(): sp.set_color(SPINE)
plt.tight_layout()
plt.savefig(OUTPUT_DIR/'03_selectivity.png', dpi=180, bbox_inches='tight')
print(f"  ✓ 03_selectivity.png")
plt.close()

# Plot D: Shannon limit
fig, ax = plt.subplots(figsize=(10, 7), facecolor=BG); ax.set_facecolor(BG2)
k_safe   = 1.85
k_severe = 2.0
D_range = np.logspace(-2, 1, 100)
ax.plot(D_range, 10**k_safe   / D_range, 'g--', lw=2, label='k=1.85 (safe)')
ax.plot(D_range, 10**k_severe / D_range, 'r--', lw=2, label='k=2.0 (severe)')
for i, sh in enumerate(SELECTED_SHAPES):
    for d in FIBER_DIAMETERS:
        for pw in PULSE_WIDTHS_US:
            try: r = main_results[d][sh][pw]
            except KeyError: continue
            if r['charge_nC'] is not None and r['charge_density'] is not None:
                ax.scatter(r['charge_nC']/1000, r['charge_density'],
                           color=cmap(i/max(1,len(SELECTED_SHAPES)-1)),
                           alpha=0.45, s=18)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Naboj na fazi Q [µC]', color=FG, fontsize=12)
ax.set_ylabel('Hustota naboje Q/A [µC/cm²]', color=FG, fontsize=12)
ax.set_title('Shannon-Wallace bezpecnostni diagram', color=FG, fontsize=13)
ax.grid(True, ls=':', alpha=0.4); ax.legend(fontsize=10, loc='upper right')
for sp in ax.spines.values(): sp.set_color(SPINE)
plt.tight_layout()
plt.savefig(OUTPUT_DIR/'04_shannon_safety.png', dpi=180, bbox_inches='tight')
print(f"  ✓ 04_shannon_safety.png")
plt.close()

# Plot E: Robustnost
if robustness_results:
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG); ax.set_facecolor(BG2)
    for sh, data in robustness_results.items():
        shifts = sorted(data.keys())
        ths = [data[s]*1000 if data[s] is not None else np.nan for s in shifts]
        label_disp = SHAPES[sh][0].replace('\n',' ') if sh in SHAPES else sh
        ax.plot(shifts, ths, 'o-', label=label_disp, lw=2, ms=7)
    ax.set_xlabel('Posun elektrody [µm]', color=FG, fontsize=12)
    ax.set_ylabel('Prah [µA]', color=FG, fontsize=12)
    ax.set_title(f'Robustnost prahu pri posunu elektrody (PW = {PW_FOR_ROBUST} µs)',
                 color=FG, fontsize=13)
    ax.grid(True, ls=':', alpha=0.4); ax.legend(fontsize=10)
    for sp in ax.spines.values(): sp.set_color(SPINE)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/'05_robustness.png', dpi=180, bbox_inches='tight')
    print(f"  ✓ 05_robustness.png")
    plt.close()

# Plot F: Scorecard
fig, ax = plt.subplots(figsize=(12, 7), facecolor=BG); ax.set_facecolor(BG2)
scorecard = {}
for sh in SELECTED_SHAPES:
    energies_list = []
    charges_list  = []
    if ref_diameter in main_results:
        for pw in PULSE_WIDTHS_US:
            try:
                e = main_results[ref_diameter][sh][pw]['energy_uJ']
                if e is not None: energies_list.append(e)
                q = main_results[ref_diameter][sh][pw]['charge_nC']
                if q is not None: charges_list.append(q)
            except KeyError: pass
    avg_energy = np.mean(energies_list) if energies_list else np.nan
    avg_charge = np.mean(charges_list) if charges_list else np.nan

    selects_list = [selectivity_results[sh][pw]
                    for pw in PULSE_WIDTHS_US
                    if selectivity_results[sh][pw] is not None]
    avg_select = np.mean(selects_list) if selects_list else 1.0

    if sh in robustness_results:
        ths = [v for v in robustness_results[sh].values() if v is not None]
        if len(ths) > 1:
            robust = 1.0 / (np.std(ths) / np.mean(ths) + 0.01)
        else:
            robust = np.nan
    else:
        robust = np.nan

    scorecard[sh] = {
        'energy': avg_energy, 'charge': avg_charge,
        'selectivity': avg_select, 'robustness': robust,
    }

energies = [s['energy']      for s in scorecard.values()]
charges  = [s['charge']      for s in scorecard.values()]
selects  = [s['selectivity'] for s in scorecard.values()]

def norm_lower_better(values):
    arr = np.array(values, dtype=float)
    if np.all(np.isnan(arr)): return [0.5]*len(arr)
    vmin = np.nanmin(arr); vmax = np.nanmax(arr)
    return [(vmax - v) / (vmax - vmin) if (vmax > vmin and not np.isnan(v)) else 0.5
            for v in arr]

def norm_higher_better(values):
    arr = np.array(values, dtype=float)
    if np.all(np.isnan(arr)): return [0.5]*len(arr)
    vmin = np.nanmin(arr); vmax = np.nanmax(arr)
    return [(v - vmin) / (vmax - vmin) if (vmax > vmin and not np.isnan(v)) else 0.5
            for v in arr]

scores_energy = norm_lower_better(energies)
scores_charge = norm_lower_better(charges)
scores_select = norm_higher_better(selects)
robust_norm = []
for sh in SELECTED_SHAPES:
    r = scorecard[sh]['robustness']
    robust_norm.append(0.0 if np.isnan(r) else r)
scores_robust = norm_higher_better(robust_norm) if any(robust_norm) \
                else [0.5]*len(SELECTED_SHAPES)

metrics_arr = np.array([scores_energy, scores_charge, scores_select, scores_robust])
metric_labels = ['Energie\n(nizsi=lepsi)', 'Naboj\n(nizsi=lepsi)',
                 'Selektivita\n(vyssi=lepsi)', 'Robustnost\n(vyssi=lepsi)']

im = ax.imshow(metrics_arr, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
ax.set_xticks(range(len(SELECTED_SHAPES)))
ax.set_xticklabels(SELECTED_SHAPES, rotation=45, ha='right')
ax.set_yticks(range(len(metric_labels))); ax.set_yticklabels(metric_labels)
ax.set_title('Scorecard: normalizovane skore v kazde metrice (1=nejlepsi)',
             color=FG, fontsize=13)
plt.colorbar(im, ax=ax, label='Skore [0–1]')
total = metrics_arr.mean(axis=0)
for i, t in enumerate(total):
    ax.text(i, len(metric_labels)-0.5, f'{t:.2f}', ha='center', va='top',
            fontsize=8, color=FG, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR/'06_scorecard.png', dpi=180, bbox_inches='tight')
print(f"  ✓ 06_scorecard.png")
plt.close()

print()
print("═" * 75)
print(" SOUHRN — celkove skore tvaru")
print("═" * 75)
total_with_names = sorted(zip(SELECTED_SHAPES, total), key=lambda x: -x[1])
for rank, (sh, sc) in enumerate(total_with_names, 1):
    medal = ['[1]', '[2]', '[3]'][rank-1] if rank <= 3 else f'{rank:>3d}.'
    print(f"  {medal}  {sh:<20s}  total score = {sc:.3f}")
print()
print(f"Vsechny vystupy v: {OUTPUT_DIR}/")
print()
print("HOTOVO!")