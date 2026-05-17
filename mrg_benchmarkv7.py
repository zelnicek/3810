#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MRG SINGLE-SHOCK BENCHMARK v7  (final)
========================================

Benchmarks evolved or hand-crafted stimulation waveforms on the
McIntyre-Richardson-Grill (MRG) mammalian axon model. Designed as a
drop-in consumer of `waveform_catalog_evolved.py` produced by
`waveform_evolver.py`, but also accepts the original hand-built
catalog for comparison.

WHAT THIS DOES
--------------
For each waveform in the supplied catalog:

  1. Renders it via the catalog's API (uses get_gap_us when present)
  2. Bisects extracellular amplitude until the axon fires propagating AP
  3. Records threshold amplitude, threshold charge, and propagation delay
     across 8 pulse widths from 25 µs to 1 ms
  4. Fits Weiss-Lapicque (with bootstrap CIs) to extract rheobase and
     chronaxie
  5. Classifies each waveform as well-behaved or ill-conditioned based on
     cathodic-phase lobe count and anodic/cathodic peak ratio
  6. Runs an intracellular control (single-node AP detection) for a
     literature-comparable reference chronaxie
  7. Applies a post-hoc MAD outlier filter to the propagation-delay
     distribution

INPUTS
------
  MRGaxon.hoc                      (from ModelDB #3810)
  AXNODE.mod  (compiled w/ nrnivmodl)
  waveform_catalog_evolved.py      (from waveform_evolver.py)
    OR
  waveform_catalog.py              (the original hand-built one)

CATALOG AUTODETECTION
---------------------
Tries 'evolved' first (looks for EVOLVED_WAVEFORMS, evo_phase_profile,
get_gap_us). Falls back to 'classic' (just SHAPES + phase_profile).
Override with --catalog or --catalog-file.

OUTPUTS  (in --out, default outputs_mrg_v7/)
--------------------------------------------
  mrg_benchmark_v7_results.json    full results + provenance
  fig_A0_intracellular.png         intracellular SD curve (control)
  fig_A_pulse_shapes.png           gallery of waveforms
  fig_B_strength_duration.png      threshold I vs PW (log-log)
  fig_C_charge_duration.png        threshold Q vs PW (semilog)
  fig_D_rheobase_chronaxie_CI.png  rheobase + chronaxie with bootstrap CIs
  fig_E_propagation_delays.png     propagation-delay histogram
  fig_F_summary_ranking.png        top-K ranking by threshold charge
  fig_G_gap_distribution.png       (evolved only) gap histogram + scatter

USAGE
-----
  python mrg_benchmark_v7.py                      # auto-detect catalog
  python mrg_benchmark_v7.py --catalog evolved
  python mrg_benchmark_v7.py --catalog classic
  python mrg_benchmark_v7.py --catalog-file my_cat.py
  python mrg_benchmark_v7.py --quick              # fewer PWs for fast test

BUG FIXES from v5/v6 (reviewer-proof)
-------------------------------------
  Q1  Soft propagation window [10, 500] µs + post-hoc MAD outlier filter
      (no more 23/27 NaNs at long pulse widths).
  Q2  Well-behavedness diagnosed via lobe-count + peak-ratio (β-guard
      removed; β was always ≈ 1 with normalised phase profiles).
  Q3  Intracellular control uses LOCAL AP detection, not propagation,
      matching how literature SD curves were measured.
  Q4  Negative propagation delays excluded at the source.
"""

import os
import sys
import time
import json
import hashlib
import platform
import socket
import subprocess
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# ════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════
ELEC_RADIAL_UM    = 2000.0     # µm — radial electrode distance
RHO_E_OHM_CM      = 300.0      # Ω·cm
PROPAGATION_NODES = 3
AP_THRESHOLD_MV   = -20.0

DT          = 0.005            # ms simulation step
T_TOTAL     = 25.0             # ms — long enough for gaps up to 3 ms + 2×PW
DELAY       = 2.0              # ms equilibration before pulse onset

# Bisection
BISECT_TOL_MA   = 0.001        # 1 µA tolerance
BISECT_MAX_ITER = 50
AMP_MAX_MA      = 10.0
AMP_MIN_MA      = 1e-6

# Bug-fix #1: wide soft window + post-hoc MAD filter
PROP_WINDOW_LO_MS = 0.010
PROP_WINDOW_HI_MS = 0.500
MAD_OUTLIER_K     = 3.0

# Bug-fix #2: peak-ratio + lobe-count instead of β-guard
PEAK_RATIO_TOL = 1.5
MAX_LOBES      = 1

# Pulse widths
PULSE_WIDTHS_FULL  = [25, 50, 100, 200, 300, 500, 750, 1000]
PULSE_WIDTHS_QUICK = [50, 200, 500]

# WL fitting
N_BOOT       = 500
MULTISTART_N = 10
REF_PW_US    = 200             # pulse width used for the headline ranking

# Plot maxima
GALLERY_MAX_SHOW = 30          # waveforms shown in fig_A
SD_MAX_SHOW      = 50          # waveforms shown in figs_B and figs_C
RANK_MAX_SHOW    = 40          # rows in fig_F


# ════════════════════════════════════════════════════════════════════════════
#  0. NEURON + MRG axon
# ════════════════════════════════════════════════════════════════════════════
try:
    from neuron import h
    h.load_file('stdrun.hoc')
    print("✓ NEURON loaded")
except ImportError:
    print("[ERROR] NEURON not found. pip install neuron")
    sys.exit(1)


def _check_axnode():
    test = h.Section(name='__chk__')
    try:
        test.insert('axnode')
        return True
    except Exception:
        return False


if not _check_axnode():
    print("[ERROR] AXNODE not loaded. Compile AXNODE.mod with nrnivmodl.")
    sys.exit(1)
print("✓ AXNODE mechanism loaded")


# ════════════════════════════════════════════════════════════════════════════
#  1. CLI + catalog loading
# ════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--catalog', choices=['auto', 'evolved', 'classic'],
                   default='auto',
                   help='catalog type (default: auto-detect)')
    p.add_argument('--catalog-file', default=None,
                   help='explicit path to catalog .py file')
    p.add_argument('--hoc', default='MRGaxon.hoc',
                   help='path to MRGaxon.hoc')
    p.add_argument('--out', default='outputs_mrg_v7',
                   help='output directory')
    p.add_argument('--quick', action='store_true',
                   help='use 3 pulse widths instead of 8 for quick testing')
    return p.parse_args()


def load_catalog(catalog_file, force_type):
    """Import a catalog .py and decide whether it's 'evolved' or 'classic'."""
    p = Path(catalog_file).resolve()
    if not p.exists():
        print(f"[ERROR] catalog file {p} not found")
        sys.exit(1)
    spec = importlib.util.spec_from_file_location('catalog_module', p)
    cat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cat)
    if not hasattr(cat, 'SHAPES'):
        print(f"[ERROR] catalog {p} has no SHAPES dict")
        sys.exit(1)
    has_evolved = (hasattr(cat, 'EVOLVED_WAVEFORMS')
                   and hasattr(cat, 'evo_phase_profile')
                   and hasattr(cat, 'get_gap_us'))
    has_classic = hasattr(cat, 'phase_profile')
    if force_type == 'evolved':
        if not has_evolved:
            print(f"[ERROR] --catalog evolved but no evolved API in {p}")
            sys.exit(1)
        return cat, 'evolved'
    if force_type == 'classic':
        if not has_classic:
            print(f"[ERROR] --catalog classic but no phase_profile in {p}")
            sys.exit(1)
        return cat, 'classic'
    if has_evolved:
        return cat, 'evolved'
    if has_classic:
        return cat, 'classic'
    print(f"[ERROR] catalog {p} exposes neither evolved nor classic API")
    sys.exit(1)


# ════════════════════════════════════════════════════════════════════════════
#  2. UNIFIED RENDERER  — wraps both catalog types behind one interface
# ════════════════════════════════════════════════════════════════════════════
class WaveformRenderer:
    def __init__(self, catalog, catalog_type, time_arr, dt_ms, delay_ms):
        self.catalog = catalog
        self.catalog_type = catalog_type
        self.time_arr = time_arr
        self.dt = dt_ms
        self.delay = delay_ms
        self.shapes = catalog.SHAPES

    def get_gap_us(self, shape_key):
        if self.catalog_type == 'evolved':
            try:
                return float(self.catalog.get_gap_us(shape_key))
            except Exception:
                return 0.0
        return 0.0

    def get_phase_profile(self, tau, shape_key, phase):
        if self.catalog_type == 'evolved':
            return self.catalog.evo_phase_profile(tau,
                                                   f"{shape_key}_{phase}")
        return self.catalog.phase_profile(tau, shape_key)

    def make_pulse(self, shape_key, pw_us, amp_mA=1.0, gap_us=None):
        """Return (waveform_mA, mask_cathodic, mask_anodic, info)."""
        if gap_us is None:
            gap_us = self.get_gap_us(shape_key)
        n_t = len(self.time_arr)
        wave = np.zeros(n_t)
        pw_ms = pw_us / 1000.0
        gap_ms = gap_us / 1000.0
        t0 = self.delay
        t1 = t0 + pw_ms
        t2 = t1 + gap_ms
        t3 = t2 + pw_ms
        m1 = (self.time_arr >= t0) & (self.time_arr < t1)
        m2 = (self.time_arr >= t2) & (self.time_arr < t3)
        if not np.any(m1) or not np.any(m2):
            return wave, m1, m2, {'invalid': True, 'reason': 'mask_empty',
                                  'gap_us': gap_us}
        n1 = int(np.sum(m1))
        n2 = int(np.sum(m2))
        try:
            p1 = self.get_phase_profile(np.linspace(0, 1, n1),
                                         shape_key, 'p1')
            p2 = self.get_phase_profile(np.linspace(0, 1, n2),
                                         shape_key, 'p2')
        except Exception as e:
            return wave, m1, m2, {'invalid': True,
                                  'reason': f'render_err:{e}',
                                  'gap_us': gap_us}
        a1 = float(np.sum(p1)) * self.dt
        a2 = float(np.sum(p2)) * self.dt
        if a2 < 1e-12:
            return wave, m1, m2, {'invalid': True, 'reason': 'a2_zero',
                                  'gap_us': gap_us}
        beta = a1 / a2
        wave[m1] = -amp_mA * p1
        wave[m2] = amp_mA * p2 * beta
        info = {
            'invalid': False,
            'gap_us': gap_us,
            'beta': beta,
            'peak_cathodic': float(amp_mA * np.max(np.abs(p1))),
            'peak_anodic':   float(amp_mA * np.max(np.abs(p2)) * beta),
        }
        return wave, m1, m2, info

    def describe_shape(self, shape_key, pw_us=REF_PW_US):
        """Bug-fix #2: lobe-count + peak-ratio diagnostics.

        Waveforms with substantial gap (≥ 100 µs) are CGA/AGC-style and
        typically have peak_anodic << peak_cathodic by design (Hofmann
        et al. 2011). For these, the peak_ratio threshold is relaxed:
        only an upper bound is enforced (anodic must not exceed the
        cathodic), and the lower bound is dropped because a small
        anodic peak is exactly what makes such waveforms charge-efficient.
        """
        wave, m1, m2, info = self.make_pulse(shape_key, pw_us, amp_mA=1.0)
        if info.get('invalid'):
            return {'well_behaved': False, 'reason': info.get('reason'),
                    'peak_ratio': None, 'n_lobes': None,
                    'gap_us': info.get('gap_us', 0.0)}
        cath = wave[m1]
        cath_abs = np.abs(cath)
        pk = float(cath_abs.max())
        if pk < 1e-9:
            return {'well_behaved': False, 'reason': 'flat_p1',
                    'peak_ratio': None, 'n_lobes': 0,
                    'gap_us': info['gap_us']}
        cutoff = 0.3 * pk
        above = cath_abs > cutoff
        transitions = np.diff(above.astype(int))
        n_lobes = int(np.sum(transitions == 1))
        if above[0]:
            n_lobes += 1
        peak_ratio = info['peak_anodic'] / max(info['peak_cathodic'], 1e-9)
        gap_us = info['gap_us']
        # Adaptive peak-ratio criterion based on gap presence
        if gap_us >= 100.0:
            # CGA/AGC-like waveforms: only enforce upper bound
            pr_ok = (peak_ratio <= PEAK_RATIO_TOL)
            pr_reason = 'extreme_peak_ratio_with_gap'
        else:
            # Symmetric biphasic pulses: enforce both bounds
            pr_ok = (1/PEAK_RATIO_TOL <= peak_ratio <= PEAK_RATIO_TOL)
            pr_reason = 'extreme_peak_ratio'
        well_behaved = (n_lobes <= MAX_LOBES) and pr_ok
        reason = (None if well_behaved else
                  ('multi_lobe' if n_lobes > MAX_LOBES else pr_reason))
        return {
            'well_behaved': bool(well_behaved),
            'reason': reason,
            'peak_ratio': float(peak_ratio),
            'n_lobes': int(n_lobes),
            'gap_us': float(gap_us),
        }


# ════════════════════════════════════════════════════════════════════════════
#  3. MRG axon loading
# ════════════════════════════════════════════════════════════════════════════
def load_mrg(hoc_path):
    p = Path(hoc_path).resolve()
    if not p.exists():
        print(f"[ERROR] {p} not found")
        sys.exit(1)
    with open(p, 'rb') as f:
        sha = hashlib.sha256(f.read()).hexdigest()[:16]
    h.load_file(str(p))
    h.stim.amp = 0.0
    h.stim.dur = 0.0
    h.stim.delay = 1e9
    return sha


def build_section_list(axonnodes, nodelen, paral1, paral2, interl):
    entries, x = [], 0.0
    for i in range(axonnodes):
        entries.append((h.node[i], x + 0.5 * nodelen))
        x += nodelen
        if i == axonnodes - 1:
            break
        entries.append((h.MYSA[2*i],   x + 0.5*paral1)); x += paral1
        entries.append((h.FLUT[2*i],   x + 0.5*paral2)); x += paral2
        for k in range(6):
            entries.append((h.STIN[6*i + k], x + 0.5*interl))
            x += interl
        entries.append((h.FLUT[2*i + 1], x + 0.5*paral2)); x += paral2
        entries.append((h.MYSA[2*i + 1], x + 0.5*paral1)); x += paral1
    return entries


# ════════════════════════════════════════════════════════════════════════════
#  4. MRG simulator + threshold search
# ════════════════════════════════════════════════════════════════════════════
class MRGSimulator:
    def __init__(self):
        self.AXONNODES = int(h.axonnodes)
        self.DELTAX    = float(h.deltax)
        self.NODELEN   = float(h.nodelength)
        self.PARAL1    = float(h.paralength1)
        self.PARAL2    = float(h.paralength2)
        self.INTERL    = float(h.interlength)
        self.FIBERD    = float(h.fiberD)
        self.CELSIUS   = float(h.celsius)
        h.celsius = self.CELSIUS
        self.CENTER_NODE = self.AXONNODES // 2
        self.PROP_IDX = min(self.CENTER_NODE + PROPAGATION_NODES,
                            self.AXONNODES - 1)
        self.sections = build_section_list(
            self.AXONNODES, self.NODELEN, self.PARAL1, self.PARAL2,
            self.INTERL)
        self.elec_x_um = next(x for sec, x in self.sections
                              if sec == h.node[self.CENTER_NODE])
        self.ve_per_mA = self._build_transfer()
        self.time_arr = np.arange(0, T_TOTAL, DT)
        self.t_vec = h.Vector(self.time_arr)
        self.amp_vecs = [h.Vector(len(self.time_arr))
                         for _ in range(len(self.sections))]
        self.v_exc  = h.Vector().record(h.node[self.CENTER_NODE](0.5)._ref_v)
        self.v_prop = h.Vector().record(h.node[self.PROP_IDX](0.5)._ref_v)

    def _build_transfer(self):
        ve = np.zeros(len(self.sections))
        for n, (sec, x) in enumerate(self.sections):
            dx = x - self.elec_x_um
            r_um = np.sqrt(dx*dx + ELEC_RADIAL_UM**2)
            r_cm = r_um / 1e4
            ve[n] = RHO_E_OHM_CM / (4.0 * np.pi * r_cm)
        return ve

    def set_drive(self, I_mA):
        for n, (sec, _) in enumerate(self.sections):
            v = self.amp_vecs[n]
            v.play_remove()
            v.from_python(self.ve_per_mA[n] * I_mA)
            v.play(sec(0.5)._ref_e_extracellular, self.t_vec, 1)

    def clear_drive(self):
        zero = np.zeros(len(self.time_arr))
        for n, (sec, _) in enumerate(self.sections):
            v = self.amp_vecs[n]
            v.play_remove()
            v.from_python(zero.copy())
            v.play(sec(0.5)._ref_e_extracellular, self.t_vec, 1)

    @staticmethod
    def first_upcross(v_arr, s_idx):
        if len(v_arr) < 2:
            return None
        above = (v_arr > AP_THRESHOLD_MV).astype(int)
        crosses = np.where(np.diff(above) == 1)[0] + 1
        valid = crosses[crosses >= s_idx]
        return int(valid[0]) if len(valid) else None

    def fires_propagation(self, I_mA, return_delay=False):
        """Bug-fix #1+#4: soft window, only positive delays."""
        self.set_drive(I_mA)
        h.dt = DT
        h.finitialize(-80.0)
        h.continuerun(T_TOTAL)
        v_e = np.array(self.v_exc)
        v_p = np.array(self.v_prop)
        s_idx = int(DELAY / DT)
        e = self.first_upcross(v_e, s_idx)
        p = self.first_upcross(v_p, s_idx)
        if e is None or p is None:
            return (False, None) if return_delay else False
        dt_ms = (p - e) * DT
        valid = (dt_ms > 0) and (PROP_WINDOW_LO_MS <= dt_ms <= PROP_WINDOW_HI_MS)
        return (valid, dt_ms if valid else None) if return_delay else valid

    def fires_local_AP(self, I_nA, pw_us):
        """Bug-fix #3: single-node detection for intracellular control."""
        self.clear_drive()
        h.stim.delay = DELAY
        h.stim.dur = pw_us / 1000.0
        h.stim.amp = I_nA
        h.dt = DT
        h.finitialize(-80.0)
        h.continuerun(T_TOTAL)
        v_e = np.array(self.v_exc)
        s_idx = int(DELAY / DT)
        return self.first_upcross(v_e, s_idx) is not None

    def find_threshold_extracellular(self, renderer, shape_key, pw_us):
        wave, m1, _, info = renderer.make_pulse(shape_key, pw_us, amp_mA=1.0)
        if info.get('invalid'):
            return None, None, None, info
        if not self.fires_propagation(AMP_MAX_MA * wave):
            return None, None, None, {**info, 'reason': 'unfireable'}
        lo, hi = AMP_MIN_MA, AMP_MAX_MA
        for _ in range(BISECT_MAX_ITER):
            if (hi - lo) <= BISECT_TOL_MA:
                break
            mid = 0.5 * (lo + hi)
            if self.fires_propagation(mid * wave):
                hi = mid
            else:
                lo = mid
        threshold = 0.5 * (lo + hi)
        cathodic = (threshold * wave)[m1]
        charge_nC = float(np.sum(np.abs(cathodic)) * DT * 1000.0)
        valid, delay_ms = self.fires_propagation(threshold * wave,
                                                  return_delay=True)
        if not valid:
            return None, None, None, info
        return float(threshold), charge_nC, float(delay_ms), info

    def find_threshold_iclamp(self, pw_us):
        if not self.fires_local_AP(50.0, pw_us):
            return None
        lo, hi = 1e-3, 50.0
        for _ in range(40):
            if (hi - lo) < 1e-3:
                break
            mid = 0.5 * (lo + hi)
            if self.fires_local_AP(mid, pw_us):
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)


# ════════════════════════════════════════════════════════════════════════════
#  5. WL fitting + bootstrap
# ════════════════════════════════════════════════════════════════════════════
def fit_weiss_lapicque(pws, ths, multistart=MULTISTART_N):
    valid = [(p, t) for p, t in zip(pws, ths) if t is not None]
    if len(valid) < 3:
        return None, None, None
    xs = np.array([v[0] for v in valid], dtype=float)
    ys = np.array([v[1] for v in valid], dtype=float)
    rng = np.random.default_rng(0)
    p0_list = [(np.min(ys), np.median(xs))]
    for _ in range(multistart - 1):
        p0_list.append((np.min(ys) * rng.uniform(0.1, 10.0),
                        np.median(xs) * rng.uniform(0.1, 10.0)))
    best, best_loss = (None, None, None), np.inf
    for p0 in p0_list:
        try:
            popt, _ = curve_fit(
                lambda pw, Ir, T: Ir * (1 + T/pw),
                xs, ys, p0=p0,
                bounds=([1e-5, 0.1], [1e8, 1e7]),
                maxfev=5000)
            yhat = popt[0] * (1 + popt[1]/xs)
            ss_res = float(np.sum((ys - yhat)**2))
            ss_tot = float(np.sum((ys - ys.mean())**2))
            r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0.0
            if ss_res < best_loss:
                best_loss = ss_res
                best = (float(popt[0]), float(popt[1]), float(r2))
        except Exception:
            continue
    return best


def bootstrap_wl(pws, ths, n_boot=N_BOOT, seed=0):
    rb, tc, r2 = fit_weiss_lapicque(pws, ths)
    if rb is None:
        return rb, tc, r2, None, None, None, None
    valid = [(p, t) for p, t in zip(pws, ths) if t is not None]
    xs = np.array([v[0] for v in valid], dtype=float)
    ys = np.array([v[1] for v in valid], dtype=float)
    if len(xs) < 4:
        return rb, tc, r2, None, None, None, None
    rng = np.random.default_rng(seed)
    rbs, tcs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(xs), size=len(xs))
        if len(set(idx.tolist())) < 3:
            continue
        try:
            popt, _ = curve_fit(
                lambda pw, Ir, T: Ir*(1 + T/pw),
                xs[idx], ys[idx], p0=[rb, tc],
                bounds=([1e-5, 0.1], [1e8, 1e7]),
                maxfev=2000)
            rbs.append(popt[0]); tcs.append(popt[1])
        except Exception:
            continue
    if len(rbs) < 10:
        return rb, tc, r2, None, None, None, None
    rb_lo, rb_hi = np.percentile(rbs, [2.5, 97.5])
    tc_lo, tc_hi = np.percentile(tcs, [2.5, 97.5])
    return rb, tc, r2, float(rb_lo), float(rb_hi), float(tc_lo), float(tc_hi)


# ════════════════════════════════════════════════════════════════════════════
#  6. Plotting
# ════════════════════════════════════════════════════════════════════════════
BG, BG2 = '#ffffff', '#f7f9fc'
FG, MUTED, GRID, SPINE = '#1f2937', '#4b5563', '#d0d7de', '#9ca3af'


def _hex(c):
    return c if isinstance(c, str) else \
        '#%02x%02x%02x' % tuple(int(x*255) for x in c[:3])


def plot_intracellular(out_dir, pws, ths_pA, rb_pA, tc_us, r2):
    if rb_pA is None or len(pws) < 3:
        return
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
    ax.set_facecolor(BG2)
    ax.plot(pws, ths_pA, 'o-', color='#0066cc', linewidth=2, markersize=7,
            label='Measured (single-node AP at center)')
    pw_smooth = np.geomspace(min(pws), max(pws), 200)
    ax.plot(pw_smooth, rb_pA*(1 + tc_us/pw_smooth),
            '--', color='#cc0000', linewidth=1.5,
            label=f'WL fit: I_rh={rb_pA:.0f} pA, '
                  f'T_ch={tc_us:.0f} µs (R²={r2:.3f})')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Pulse width [µs]', color=FG)
    ax.set_ylabel('Threshold current [pA]', color=FG)
    ax.set_title('Intracellular control (literature ≈ 70-100 µs)',
                 color=FG, fontsize=11)
    ax.legend(facecolor=BG2, edgecolor=GRID, labelcolor=FG)
    ax.grid(True, linestyle=':', alpha=0.4, color=GRID)
    for sp in ax.spines.values(): sp.set_color(SPINE)
    plt.tight_layout()
    plt.savefig(out_dir/'fig_A0_intracellular.png',
                dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close()


def plot_pulse_gallery(out_dir, renderer, fit_data, time_arr, max_show):
    keys = list(renderer.shapes.keys())
    if len(keys) > max_show:
        scored = [(k, fit_data[k].get('charge_nC_at_ref')
                       if fit_data[k].get('charge_nC_at_ref') is not None
                       else float('inf'))
                  for k in keys if k in fit_data]
        scored.sort(key=lambda x: x[1])
        keys = [k for k, _ in scored[:max_show]]
    n_cols = 5
    n_rows = int(np.ceil(len(keys) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3*n_rows),
                              facecolor=BG)
    axes = np.atleast_2d(axes).flatten()
    for i, key in enumerate(keys):
        ax = axes[i]
        ax.set_facecolor(BG2)
        label, color = renderer.shapes[key]
        gap_us = renderer.get_gap_us(key)
        wave, m1, m2, info = renderer.make_pulse(key, 500, amp_mA=1.0)
        end_ms = DELAY + 1.0 + gap_us/1000 + 0.5
        m = (time_arr >= DELAY-0.3) & (time_arr <= end_ms)
        col = _hex(color)
        wb = fit_data.get(key, {}).get('well_behaved', False)
        ax.plot(time_arr[m]*1000, wave[m], color=col,
                linewidth=2.0 if wb else 1.0,
                alpha=1.0 if wb else 0.45)
        ax.fill_between(time_arr[m]*1000, wave[m], 0, where=wave[m]<0,
                        color=col, alpha=0.3 if wb else 0.12)
        ax.fill_between(time_arr[m]*1000, wave[m], 0, where=wave[m]>0,
                        color='#ef4444', alpha=0.2 if wb else 0.08)
        ax.axhline(0, color=SPINE, linewidth=0.5)
        flag = "✓" if wb else "✗"
        title = f"{label.replace(chr(10),' ')[:25]}\ngap={gap_us:.0f}µs {flag}"
        ax.set_title(title, fontsize=7, color=FG if wb else MUTED, pad=3)
        ax.set_xlabel('ms', fontsize=6, color=MUTED)
        ax.tick_params(colors=MUTED, labelsize=5)
        for sp in ax.spines.values(): sp.set_color(SPINE)
    for unused in axes[len(keys):]:
        unused.axis('off')
    fig.suptitle(f'Waveform gallery — {len(keys)} of '
                 f'{len(renderer.shapes)} shown (well-behaved bold)',
                 fontsize=11, color=FG)
    plt.tight_layout(pad=1.0)
    plt.savefig(out_dir/'fig_A_pulse_shapes.png',
                dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close()


def plot_strength_duration(out_dir, renderer, results, fit_data, fiber_D,
                            pulse_widths, max_show):
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=BG)
    ax.set_facecolor(BG2)
    items = list(renderer.shapes.items())
    if len(items) > max_show:
        scored = sorted(
            [(k, fit_data[k].get('charge_nC_at_ref') or float('inf'))
             for k, _ in items], key=lambda x: x[1])
        keep = set(k for k, _ in scored[:max_show])
        items = [(k, v) for k, v in items if k in keep]
    n_show = 0
    for key, (label, color) in items:
        wb = fit_data.get(key, {}).get('well_behaved', False)
        ths = [results[key][pw]['threshold_mA'] for pw in pulse_widths]
        vp = [pw for pw, t in zip(pulse_widths, ths) if t is not None]
        vt = [t*1000 for t in ths if t is not None]
        if not vp:
            continue
        n_show += 1
        ax.plot(vp, vt, 'o-', color=_hex(color),
                linewidth=2.0 if wb else 0.8,
                alpha=1.0 if wb else 0.3, markersize=4)
    ax.set_xlabel('Pulse width [µs]', color=FG, fontsize=12)
    ax.set_ylabel('Threshold amplitude [µA]', color=FG, fontsize=12)
    ax.set_title(f'MRG axon (D={fiber_D} µm) — strength-duration\n'
                 f'{ELEC_RADIAL_UM/1000:.1f} mm radial — '
                 f'{n_show} waveforms shown',
                 color=FG, fontsize=12)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.tick_params(colors=FG)
    ax.grid(True, linestyle=':', alpha=0.4, color=GRID)
    for sp in ax.spines.values(): sp.set_color(SPINE)
    plt.tight_layout()
    plt.savefig(out_dir/'fig_B_strength_duration.png',
                dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close()


def plot_charge_duration(out_dir, renderer, results, fit_data,
                          pulse_widths, max_show):
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=BG)
    ax.set_facecolor(BG2)
    items = list(renderer.shapes.items())
    if len(items) > max_show:
        scored = sorted(
            [(k, fit_data[k].get('charge_nC_at_ref') or float('inf'))
             for k, _ in items], key=lambda x: x[1])
        keep = set(k for k, _ in scored[:max_show])
        items = [(k, v) for k, v in items if k in keep]
    for key, (label, color) in items:
        wb = fit_data.get(key, {}).get('well_behaved', False)
        chs = [results[key][pw]['charge_nC'] for pw in pulse_widths]
        vp = [pw for pw, c in zip(pulse_widths, chs) if c is not None]
        vc = [c for c in chs if c is not None]
        if not vp:
            continue
        ax.plot(vp, vc, 'o-', color=_hex(color),
                linewidth=2.0 if wb else 0.8,
                alpha=1.0 if wb else 0.3, markersize=4)
    ax.set_xlabel('Pulse width [µs]', color=FG, fontsize=12)
    ax.set_ylabel('Threshold charge [nC]', color=FG, fontsize=12)
    ax.set_title('Charge-duration', color=FG, fontsize=13)
    ax.set_xscale('log')
    ax.tick_params(colors=FG)
    ax.grid(True, linestyle=':', alpha=0.4, color=GRID)
    for sp in ax.spines.values(): sp.set_color(SPINE)
    plt.tight_layout()
    plt.savefig(out_dir/'fig_C_charge_duration.png',
                dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close()


def plot_rheobase_chronaxie(out_dir, renderer, fit_data, ic_tc_us,
                             max_show=20):
    wb = [(k, d) for k, d in fit_data.items()
          if d.get('well_behaved') and d.get('rheobase_uA') is not None]
    wb.sort(key=lambda x: x[1]['rheobase_uA'])
    wb = wb[:max_show]
    if not wb:
        return
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(16, max(6, len(wb)*0.4)),
                                    facecolor=BG)
    for ax in (axA, axB):
        ax.set_facecolor(BG2)
    lbls = [renderer.shapes[k][0].replace('\n',' ')[:30] for k, _ in wb]
    cols = [_hex(renderer.shapes[k][1]) for k, _ in wb]
    rbs = [d['rheobase_uA'] for _, d in wb]
    tcs = [d['chronaxie_us'] for _, d in wb]
    rb_err = np.array([
        [r - (d['rb_ci95'][0] if d.get('rb_ci95') else r)
         for r, (_, d) in zip(rbs, wb)],
        [(d['rb_ci95'][1] if d.get('rb_ci95') else r) - r
         for r, (_, d) in zip(rbs, wb)],
    ])
    tc_err = np.array([
        [t - (d['tc_ci95'][0] if d.get('tc_ci95') else t)
         for t, (_, d) in zip(tcs, wb)],
        [(d['tc_ci95'][1] if d.get('tc_ci95') else t) - t
         for t, (_, d) in zip(tcs, wb)],
    ])
    axA.barh(range(len(rbs)), rbs, color=cols, edgecolor=SPINE,
             xerr=rb_err, error_kw={'ecolor':'#444', 'capsize':3, 'lw':1},
             alpha=0.85)
    axA.set_yticks(range(len(lbls))); axA.set_yticklabels(lbls, fontsize=8)
    axA.set_xlabel('Rheobase [µA] (95% bootstrap CI)', color=FG)
    axA.set_title('Rheobase', color=FG, fontsize=11)
    axA.invert_yaxis(); axA.tick_params(colors=FG)
    for sp in axA.spines.values(): sp.set_color(SPINE)
    axB.barh(range(len(tcs)), tcs, color=cols, edgecolor=SPINE,
             xerr=tc_err, error_kw={'ecolor':'#444', 'capsize':3, 'lw':1},
             alpha=0.85)
    axB.set_yticks(range(len(lbls))); axB.set_yticklabels(lbls, fontsize=8)
    axB.set_xlabel('Chronaxie [µs] (95% bootstrap CI)', color=FG)
    if ic_tc_us is not None:
        axB.axvline(ic_tc_us, color='#cc0000', linestyle='--', linewidth=1.5,
                    label=f'Intracellular ref ({ic_tc_us:.0f} µs)')
        axB.legend(facecolor=BG2, edgecolor=GRID, labelcolor=FG, fontsize=8)
    axB.set_title('Chronaxie', color=FG, fontsize=11)
    axB.invert_yaxis(); axB.tick_params(colors=FG)
    for sp in axB.spines.values(): sp.set_color(SPINE)
    fig.suptitle(f'SD parameters — top {len(wb)} well-behaved by rheobase',
                 fontsize=13, color=FG)
    plt.tight_layout()
    plt.savefig(out_dir/'fig_D_rheobase_chronaxie_CI.png',
                dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close()


def plot_propagation_delays(out_dir, trial_log, mad_window):
    delays = [t['delay_us'] for t in trial_log if t['delay_us'] is not None]
    if not delays:
        return
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
    ax.set_facecolor(BG2)
    ax.hist(delays, bins=40, color='#3b82f6', edgecolor=FG, alpha=0.85)
    if mad_window is not None:
        ax.axvspan(mad_window[0], mad_window[1], color='#22c55e', alpha=0.15,
                   label=f'MAD acceptance [{mad_window[0]:.0f}, '
                         f'{mad_window[1]:.0f}] µs')
        ax.legend(facecolor=BG2, edgecolor=GRID, labelcolor=FG)
    ax.set_xlabel('Measured propagation delay [µs]', color=FG)
    ax.set_ylabel('Number of trials at threshold', color=FG)
    ax.set_title('Propagation delays — soft window + MAD outlier filter',
                 color=FG)
    ax.tick_params(colors=FG)
    for sp in ax.spines.values(): sp.set_color(SPINE)
    plt.tight_layout()
    plt.savefig(out_dir/'fig_E_propagation_delays.png',
                dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close()


def plot_summary_ranking(out_dir, ranking, max_show):
    if not ranking:
        return
    rows = ranking[:max_show]
    fig, ax = plt.subplots(figsize=(14, max(6, len(rows)*0.35)),
                            facecolor=BG)
    ax.set_facecolor(BG2)
    labels, charges, amps, gaps, cols = [], [], [], [], []
    for r in rows:
        flag = ""
        if not r['well_behaved']:    flag += " [✗]"
        if r.get('delay_outlier'):   flag += " [out]"
        labels.append(r['label'].replace('\n',' ')[:32] + flag)
        charges.append(r['charge_nC'])
        amps.append(r['threshold_uA'])
        gaps.append(r.get('gap_us', 0))
        wb_clean = r['well_behaved'] and not r.get('delay_outlier')
        cols.append(_hex(r['color']) if wb_clean else '#cccccc')
    ax.barh(range(len(rows)), charges, color=cols, edgecolor=SPINE,
            alpha=0.85)
    for i, (ch, th, g, r) in enumerate(zip(charges, amps, gaps, rows)):
        wb_clean = r['well_behaved'] and not r.get('delay_outlier')
        gap_str = f"  gap={g:.0f}µs" if g > 50 else ""
        ax.text(ch + max(charges)*0.01, i,
                f"{ch:.2f}nC | {th:.0f}µA{gap_str}",
                va='center', fontsize=8, color=FG if wb_clean else MUTED)
    medal_c = ["#FFD700","#C0C0C0","#CD7F32"]
    wb_only = [(i, r) for i, r in enumerate(rows)
               if r['well_behaved'] and not r.get('delay_outlier')]
    for k, (idx, _) in enumerate(wb_only[:3]):
        ax.barh(idx, charges[idx], color=medal_c[k],
                edgecolor=SPINE, linewidth=1.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(f'Threshold charge at PW={REF_PW_US} µs [nC]',
                  color=FG, fontsize=12)
    ax.set_title(f'Headline ranking — top {len(rows)} '
                 f'(medals: well-behaved + non-outlier only)',
                 color=FG, fontsize=13, pad=10)
    ax.invert_yaxis(); ax.tick_params(colors=FG)
    ax.grid(axis='x', linestyle=':', alpha=0.4, color=GRID)
    for sp in ax.spines.values(): sp.set_color(SPINE)
    plt.tight_layout()
    plt.savefig(out_dir/'fig_F_summary_ranking.png',
                dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close()


def plot_gap_distribution(out_dir, renderer, fit_data, results):
    if renderer.catalog_type != 'evolved':
        return
    gaps, charges_at_ref, wbs = [], [], []
    for key in renderer.shapes:
        gap = renderer.get_gap_us(key)
        ch = results[key].get(REF_PW_US, {}).get('charge_nC')
        if ch is not None:
            gaps.append(gap)
            charges_at_ref.append(ch)
            wbs.append(fit_data.get(key, {}).get('well_behaved', False))
    if not gaps:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(BG2)
    ax1.hist(gaps, bins=30, color='#9333ea', edgecolor=FG, alpha=0.8)
    ax1.set_xlabel('Interphase gap [µs]', color=FG)
    ax1.set_ylabel('Number of waveforms', color=FG)
    ax1.set_title('Distribution of evolved gaps', color=FG)
    ax1.tick_params(colors=FG)
    for sp in ax1.spines.values(): sp.set_color(SPINE)
    cols = ['#22c55e' if w else '#cbd5e1' for w in wbs]
    ax2.scatter(gaps, charges_at_ref, c=cols, edgecolor=FG, s=40, alpha=0.8)
    ax2.set_xlabel('Interphase gap [µs]', color=FG)
    ax2.set_ylabel(f'Threshold charge at PW={REF_PW_US} µs [nC]', color=FG)
    ax2.set_title('Charge vs. gap (green = well-behaved)', color=FG)
    ax2.tick_params(colors=FG)
    ax2.grid(True, linestyle=':', alpha=0.4, color=GRID)
    for sp in ax2.spines.values(): sp.set_color(SPINE)
    plt.tight_layout()
    plt.savefig(out_dir/'fig_G_gap_distribution.png',
                dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
#  7. Main
# ════════════════════════════════════════════════════════════════════════════
def _git_commit():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL).decode().strip()[:12]
    except Exception:
        return None


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    out_dir = script_dir / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    pulse_widths = (PULSE_WIDTHS_QUICK if args.quick else PULSE_WIDTHS_FULL)

    # Choose catalog file
    if args.catalog_file:
        catalog_file = args.catalog_file
    else:
        evolved = script_dir / 'waveform_catalog_evolved.py'
        classic = script_dir / 'waveform_catalog.py'
        if args.catalog == 'evolved' or \
           (args.catalog == 'auto' and evolved.exists()):
            catalog_file = evolved
        else:
            catalog_file = classic

    # Load axon
    hoc_sha = load_mrg(args.hoc)
    print(f"✓ MRGaxon.hoc loaded (SHA256[:16]={hoc_sha})")
    sim = MRGSimulator()
    print(f"✓ MRG axon: D={sim.FIBERD} µm, {sim.AXONNODES} nodes, "
          f"{sim.CELSIUS} °C")

    # Load catalog
    catalog, catalog_type = load_catalog(catalog_file, args.catalog)
    print(f"✓ Catalog '{Path(catalog_file).name}' ({catalog_type}, "
          f"{len(catalog.SHAPES)} shapes)")
    renderer = WaveformRenderer(catalog, catalog_type, sim.time_arr,
                                  DT, DELAY)

    print()
    print("═" * 72)
    print(f"  MRG SINGLE-SHOCK BENCHMARK v7  ({catalog_type})")
    print("═" * 72)

    # Sanity
    print()
    print("Sanity checks:")
    print("  1/3 zero current is silent ...", end=' ')
    if sim.fires_propagation(np.zeros(len(sim.time_arr))):
        print("FAIL"); sys.exit(1)
    print("OK")
    print("  2/3 2 mA × 500 µs rect fires ...", end=' ')
    rect_test = np.zeros(len(sim.time_arr))
    i0 = int(DELAY/DT); i1 = i0 + int(0.5/DT)
    rect_test[i0:i1] = -2.0
    if not sim.fires_propagation(rect_test):
        print("FAIL"); sys.exit(1)
    print("OK")
    print("  3/3 0.01 mA rect silent ...", end=' ')
    rect_silent = rect_test * 0.005
    print("OK" if not sim.fires_propagation(rect_silent) else "WARN")

    # Intracellular control
    print()
    print("Intracellular control (single-node AP, IClamp on center node):")
    ic_thresholds_nA = {}
    for pw in pulse_widths:
        th = sim.find_threshold_iclamp(pw)
        ic_thresholds_nA[pw] = th
        if th is not None:
            print(f"  PW={pw:4d} µs -> {th*1000:7.1f} pA")
    h.stim.amp = 0.0; h.stim.dur = 0.0; h.stim.delay = 1e9
    pws_with_ic = [p for p in pulse_widths if ic_thresholds_nA[p] is not None]
    ths_pA_list = [ic_thresholds_nA[p]*1000 for p in pws_with_ic]
    ic_rb_pA, ic_tc_us, ic_r2 = fit_weiss_lapicque(pws_with_ic, ths_pA_list)
    if ic_rb_pA is not None:
        print(f"  WL fit: I_rh={ic_rb_pA:.1f} pA, T_ch={ic_tc_us:.1f} µs, "
              f"R²={ic_r2:.4f}")

    # Main benchmark
    print()
    n_trials = len(renderer.shapes) * len(pulse_widths)
    print(f"Main benchmark: {len(renderer.shapes)} shapes × "
          f"{len(pulse_widths)} PWs = {n_trials} trials")
    print()
    all_results = {key: {} for key in renderer.shapes}
    trial_log = []
    descriptors = {}
    t_start = time.time()
    for shape_idx, (shape_key, (label, color)) in enumerate(renderer.shapes.items(), 1):
        flat_label = label.replace('\n',' ')[:35]
        gap = renderer.get_gap_us(shape_key)
        desc = renderer.describe_shape(shape_key)
        descriptors[shape_key] = desc
        wb_flag = "✓" if desc['well_behaved'] else "✗"
        gap_str = f" gap={gap:.0f}µs" if gap > 1 else ""
        elapsed = (time.time() - t_start) / 60
        eta = (elapsed / shape_idx * len(renderer.shapes)) - elapsed if shape_idx > 0 else 0
        print(f"  [{shape_idx}/{len(renderer.shapes)}] {wb_flag} "
              f"{flat_label}{gap_str}  (ETA: {eta:.0f} min)")
        for pw in pulse_widths:
            th, q, dly, info = sim.find_threshold_extracellular(
                renderer, shape_key, pw)
            all_results[shape_key][pw] = {
                'threshold_mA': th, 'charge_nC': q, 'delay_ms': dly}
            trial_log.append({
                'shape': shape_key, 'pw_us': pw,
                'threshold_uA': th*1000 if th is not None else None,
                'charge_nC': q,
                'delay_us': dly*1000 if dly is not None else None,
                'gap_us': gap,
            })
            if th is not None:
                print(f"      PW={pw:4d}µs -> Th={th*1000:8.1f}µA  "
                      f"Q={q:7.2f}nC  d={dly*1000:5.1f}µs")
            else:
                print(f"      PW={pw:4d}µs -> NOT FOUND")
    print()
    print(f"Benchmark complete in {(time.time()-t_start)/60:.1f} min")

    # Post-hoc MAD outlier filter
    delays = np.array([t['delay_us'] for t in trial_log
                       if t['delay_us'] is not None])
    mad_window = None
    outlier_lookup = {}
    if len(delays) > 5:
        med = float(np.median(delays))
        mad = float(np.median(np.abs(delays - med)))
        if mad < 1.0:
            mad = 1.0
        lo_cut = med - MAD_OUTLIER_K * mad
        hi_cut = med + MAD_OUTLIER_K * mad
        mad_window = (lo_cut, hi_cut)
        print()
        print(f"MAD outlier filter:")
        print(f"  median delay : {med:.1f} µs")
        print(f"  MAD          : {mad:.1f} µs")
        print(f"  window       : [{lo_cut:.1f}, {hi_cut:.1f}] µs")
        n_out = 0
        for t in trial_log:
            if t['delay_us'] is None:
                t['delay_outlier'] = False
            else:
                t['delay_outlier'] = bool(
                    t['delay_us'] < lo_cut or t['delay_us'] > hi_cut)
                if t['delay_outlier']:
                    n_out += 1
                outlier_lookup[(t['shape'], t['pw_us'])] = t['delay_outlier']
        print(f"  outliers     : {n_out} / {len(delays)}")
    else:
        for t in trial_log:
            t['delay_outlier'] = False

    # WL fits
    print()
    print("Weiss-Lapicque fits (with bootstrap CIs) ...")
    fit_data = {}
    for shape_key, (label, color) in renderer.shapes.items():
        ths_uA = [(all_results[shape_key][pw]['threshold_mA']*1000
                   if all_results[shape_key][pw]['threshold_mA'] is not None
                   else None) for pw in pulse_widths]
        rb, tc, r2, rb_lo, rb_hi, tc_lo, tc_hi = bootstrap_wl(
            pulse_widths, ths_uA)
        desc = descriptors[shape_key]
        well_behaved = (desc['well_behaved']
                        and r2 is not None and r2 >= 0.95)
        ref = all_results[shape_key].get(REF_PW_US, {})
        fit_data[shape_key] = {
            'rheobase_uA': rb, 'chronaxie_us': tc, 'r2': r2,
            'rb_ci95': [rb_lo, rb_hi] if rb_lo is not None else None,
            'tc_ci95': [tc_lo, tc_hi] if tc_lo is not None else None,
            'peak_ratio': desc.get('peak_ratio'),
            'n_lobes': desc.get('n_lobes'),
            'gap_us': desc.get('gap_us', 0),
            'well_behaved': well_behaved,
            'charge_nC_at_ref': ref.get('charge_nC'),
            'threshold_uA_at_ref': (ref['threshold_mA']*1000
                                     if ref.get('threshold_mA') is not None
                                     else None),
        }

    # Build ranking
    ranking = []
    for shape_key, (label, color) in renderer.shapes.items():
        r = all_results[shape_key].get(REF_PW_US, {})
        ch = r.get('charge_nC'); th = r.get('threshold_mA')
        if ch is None or th is None:
            continue
        ranking.append({
            'shape': shape_key,
            'label': label, 'color': color,
            'charge_nC': ch,
            'threshold_uA': th*1000,
            'gap_us': renderer.get_gap_us(shape_key),
            'well_behaved': fit_data[shape_key]['well_behaved'],
            'delay_outlier': outlier_lookup.get((shape_key, REF_PW_US), False),
        })
    ranking.sort(key=lambda r: r['charge_nC'])

    print()
    print(f"Top 10 (well-behaved, non-outlier) at PW={REF_PW_US} µs:")
    print("─"*78)
    wb_ranking = [r for r in ranking
                  if r['well_behaved'] and not r['delay_outlier']]
    for i, r in enumerate(wb_ranking[:10], 1):
        medal = ["🥇","🥈","🥉"][i-1] if i <= 3 else f"{i:>2}."
        gap_s = f" gap={r['gap_us']:.0f}µs" if r['gap_us'] > 1 else ""
        print(f"  {medal} {r['label'].replace(chr(10),' ')[:30]:<30} "
              f"Q={r['charge_nC']:6.2f}nC  "
              f"I={r['threshold_uA']:6.0f}µA{gap_s}")
    print("─"*78)

    # JSON
    metadata = {
        'paradigm': 'single-shock biphasic, MRG axon, point electrode',
        'script_version': 'v7 final',
        'catalog_file': str(Path(catalog_file).name),
        'catalog_type': catalog_type,
        'n_shapes': len(renderer.shapes),
        'fiber_D_um': sim.FIBERD,
        'n_nodes': sim.AXONNODES,
        'electrode_radial_um': ELEC_RADIAL_UM,
        'rho_e_ohm_cm': RHO_E_OHM_CM,
        'temperature_C': sim.CELSIUS,
        'propagation_nodes': PROPAGATION_NODES,
        'prop_window_soft_us': [PROP_WINDOW_LO_MS*1000, PROP_WINDOW_HI_MS*1000],
        'mad_filter_window_us': list(mad_window) if mad_window else None,
        'peak_ratio_tol': PEAK_RATIO_TOL,
        'max_lobes': MAX_LOBES,
        'ap_threshold_mV': AP_THRESHOLD_MV,
        'dt_ms': DT,
        't_total_ms': T_TOTAL,
        'pulse_widths_us': pulse_widths,
        'ref_pw_us': REF_PW_US,
        'fit_method': 'WL + multi-start + bootstrap',
        'n_boot': N_BOOT,
        'hoc_sha256_16': hoc_sha,
        'neuron_version': str(h.nrnversion(0)),
        'python_version': platform.python_version(),
        'numpy_version': np.__version__,
        'hostname': socket.gethostname(),
        'git_commit': _git_commit(),
        'total_runtime_min': float((time.time() - t_start) / 60),
    }
    payload = {
        'metadata': metadata,
        'intracellular_control': {
            'rheobase_pA': ic_rb_pA, 'chronaxie_us': ic_tc_us, 'r2': ic_r2,
            'thresholds_by_pw_nA': {str(p): ic_thresholds_nA[p]
                                     for p in pulse_widths},
        },
        'shapes': {
            key: {
                'label': renderer.shapes[key][0],
                **fit_data[key],
                'by_pw': {
                    str(pw): {
                        'threshold_uA': (all_results[key][pw]['threshold_mA']*1000
                                          if all_results[key][pw]['threshold_mA'] is not None
                                          else None),
                        'charge_nC': all_results[key][pw]['charge_nC'],
                        'delay_us': (all_results[key][pw]['delay_ms']*1000
                                      if all_results[key][pw]['delay_ms'] is not None
                                      else None),
                        'delay_outlier': outlier_lookup.get((key, pw), False),
                    }
                    for pw in pulse_widths
                },
            }
            for key in renderer.shapes
        },
        'ranking_at_ref_pw': ranking,
        'trial_log': trial_log,
    }
    json_path = out_dir / 'mrg_benchmark_v7_results.json'
    with open(json_path, 'w') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False,
                  default=lambda x: None)
    print()
    print(f"Saved: {json_path}")

    # Plots
    print("Generating plots ...")
    plot_intracellular(out_dir, pws_with_ic, ths_pA_list,
                       ic_rb_pA, ic_tc_us, ic_r2)
    print("  ✓ fig_A0_intracellular.png")
    plot_pulse_gallery(out_dir, renderer, fit_data, sim.time_arr,
                        max_show=GALLERY_MAX_SHOW)
    print("  ✓ fig_A_pulse_shapes.png")
    plot_strength_duration(out_dir, renderer, all_results, fit_data,
                            sim.FIBERD, pulse_widths, max_show=SD_MAX_SHOW)
    print("  ✓ fig_B_strength_duration.png")
    plot_charge_duration(out_dir, renderer, all_results, fit_data,
                          pulse_widths, max_show=SD_MAX_SHOW)
    print("  ✓ fig_C_charge_duration.png")
    plot_rheobase_chronaxie(out_dir, renderer, fit_data, ic_tc_us)
    print("  ✓ fig_D_rheobase_chronaxie_CI.png")
    plot_propagation_delays(out_dir, trial_log, mad_window)
    print("  ✓ fig_E_propagation_delays.png")
    plot_summary_ranking(out_dir, ranking, max_show=RANK_MAX_SHOW)
    print("  ✓ fig_F_summary_ranking.png")
    plot_gap_distribution(out_dir, renderer, fit_data, all_results)
    if renderer.catalog_type == 'evolved':
        print("  ✓ fig_G_gap_distribution.png")

    print()
    print("═" * 72)
    print(f"  All outputs in: {out_dir}/")
    print("═" * 72)


if __name__ == '__main__':
    main()