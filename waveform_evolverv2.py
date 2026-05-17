#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WAVEFORM EVOLVER v2  (multi-PW, multi-objective, asymmetric phases)
=====================================================================

Evolves charge-balanced biphasic stimulation waveforms for the MRG
mammalian axon, optimizing simultaneously for:

  - low threshold charge (across multiple pulse widths)
  - low peak amplitude (realistic stimulator hardware)
  - low slew rate (soft-penalized at > 100 mA/µs)

Output is a Pareto front of non-dominated waveforms covering different
operating regimes (high-amp/low-charge vs low-amp/high-charge).

WHAT IS NEW vs v1
-----------------
  1. ASYMMETRIC PHASES: cathodic and anodic durations evolved
     independently. cath_pw ∈ [10, 1500] µs, anod_pw ∈ [10, 3000] µs.
     Charge balance via amplitude scaling (auto, not evolved).

  2. MULTI-PW FITNESS: threshold charge measured at 4 PWs
     (50, 200, 500, 1000 µs). Final score = geometric mean across
     all PWs at which the waveform fires.

  3. SLEW-RATE SOFT PENALTY: max |dI/dt| computed for each waveform.
     If > 100 mA/µs, penalty = 0.01 * (excess) added to charge score.

  4. NSGA-II PARETO SORTING: standard non-dominated sorting + crowding
     distance for multi-objective selection (Deb et al. 2002).

  5. PARETO ARCHIVE: persistent archive of non-dominated waveforms
     across all generations, capped at MAX_ARCHIVE_SIZE.

WAVEFORM GENOME (22 floats per waveform)
-----------------------------------------
  [0:8]   cathodic phase shape (8 cubic-spline control points, [-1.5, 1.5])
  [8:16]  anodic phase shape   (8 cubic-spline control points, [-1.5, 1.5])
  [16]    gap_us               in [0, 3000]
  [17]    cath_pw_us           in [10, 1500]
  [18]    anod_pw_us           in [10, 3000]
  [19]    amp_ratio_bias       in [0.1, 10]  (anodic_peak / cathodic_peak)
                               (final amps re-scaled for charge balance)
  [20:22] spare slots          (reserved for future extensions, ignored)

USAGE
-----
  python waveform_evolver_v2.py --quick               # ~1h test (small)
  python waveform_evolver_v2.py                        # default 12-16h
  python waveform_evolver_v2.py --pool 80 --gens 60   # weekend ~30-40h

OUTPUTS  (in outputs_v2/)
-------------------------
  waveform_catalog_evolved_v2.py   drop-in for mrg_benchmark_v7_2.py
  pareto_archive.json              full archive with all scores + metadata
  evolution_history_v2.json        per-generation stats
  fig_evolution_charge.png         convergence of best charge per generation
  fig_evolution_peak.png           convergence of best peak amplitude
  fig_pareto_front.png             charge vs peak Pareto scatter
  fig_pareto_quadrants.png         representative waveforms across the front
  fig_top_waveforms_v2.png         top 20 by charge
"""

import os
import sys
import time
import json
import argparse
import platform
import hashlib
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


# ════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════
N_CONTROL_POINTS_PER_PHASE = 8
N_TOTAL_PARAMS = 22
PHASE_RESOLUTION = 100

# Indices into the genome
IDX_CATH_CTRL_START = 0
IDX_CATH_CTRL_END   = 8
IDX_ANOD_CTRL_START = 8
IDX_ANOD_CTRL_END   = 16
IDX_GAP             = 16
IDX_CATH_PW         = 17
IDX_ANOD_PW         = 18
IDX_AMP_RATIO       = 19
# 20, 21 are spare

# Multi-PW fitness
FITNESS_PWS_US = [50, 200, 500, 1000]
REF_PW_FOR_LABELS_US = 200

# Multi-objective fitness
SLEW_LIMIT_mA_per_us = 100.0
SLEW_PENALTY_COEFF   = 0.01
MIN_PWS_FIRED        = 2     # waveform must fire at ≥ 2 PWs to count

# GA hyperparameters
DEFAULT_POOL_SIZE   = 80
DEFAULT_GENERATIONS = 60
ELITISM_FRACTION    = 0.20
PERTURB_RATE        = 0.15
PERTURB_SIGMA_CTRL  = 0.25
PERTURB_SIGMA_GAP   = 200.0
PERTURB_SIGMA_PW    = 100.0
PERTURB_SIGMA_AMP_RATIO = 0.3
COMBINE_RATE        = 0.70
EARLY_STOP_PATIENCE = 8
MAX_ARCHIVE_SIZE    = 50

# Parameter bounds
GAP_MIN_US    = 0.0
GAP_MAX_US    = 3000.0
CATH_PW_MIN   = 10.0
CATH_PW_MAX   = 1500.0
ANOD_PW_MIN   = 10.0
ANOD_PW_MAX   = 3000.0
AMP_RATIO_MIN = 0.1
AMP_RATIO_MAX = 10.0

# MRG geometry (matches v7.2 benchmark)
ELEC_RADIAL_UM    = 2000.0
RHO_E_OHM_CM      = 300.0
AP_THRESHOLD_MV   = -20.0
DT_MS             = 0.005
T_TOTAL_MS        = 30.0    # increased for max PW * 2 + max gap = ~7.5ms
DELAY_MS          = 2.0
BISECT_TOL_MA     = 0.001
BISECT_MAX_ITER   = 50
AMP_MAX_MA        = 10.0
AMP_MIN_MA        = 1e-6
N_PROPAGATION_NODES = 4
INTERNODE_DELAY_MIN_MS = 0.005
INTERNODE_DELAY_MAX_MS = 0.150

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs_v2"


# ════════════════════════════════════════════════════════════════════════════
#  WAVEFORM CONSTRUCTION
# ════════════════════════════════════════════════════════════════════════════
def get_cath_ctrl(waveform):
    return waveform[IDX_CATH_CTRL_START:IDX_CATH_CTRL_END]

def get_anod_ctrl(waveform):
    return waveform[IDX_ANOD_CTRL_START:IDX_ANOD_CTRL_END]

def get_gap_us(waveform):
    return float(np.clip(waveform[IDX_GAP], GAP_MIN_US, GAP_MAX_US))

def get_cath_pw_us(waveform):
    return float(np.clip(waveform[IDX_CATH_PW], CATH_PW_MIN, CATH_PW_MAX))

def get_anod_pw_us(waveform):
    return float(np.clip(waveform[IDX_ANOD_PW], ANOD_PW_MIN, ANOD_PW_MAX))

def get_amp_ratio(waveform):
    return float(np.clip(waveform[IDX_AMP_RATIO], AMP_RATIO_MIN, AMP_RATIO_MAX))


def render_phase_curve(control_points, n_samples=PHASE_RESOLUTION):
    """Cubic spline through control points, zero at phase boundaries."""
    knots = np.linspace(0, 1, len(control_points))
    aug_knots = np.concatenate([[-0.05], knots, [1.05]])
    aug_vals  = np.concatenate([[0.0], control_points, [0.0]])
    spl = CubicSpline(aug_knots, aug_vals,
                      bc_type='natural', extrapolate=False)
    return spl(np.linspace(0, 1, n_samples))


def render_waveform(waveform, dt_ms=DT_MS, time_arr=None,
                    delay_ms=DELAY_MS, total_ms=T_TOTAL_MS):
    """Render a waveform with its own internal cath_pw, anod_pw, gap.

    Returns (wave_array_mA_unit, cath_mask, anod_mask, info)
    where wave_array is unit-normalized to peak |cathodic| = 1 mA.
    The actual amplitude when used as stimulus is the caller's job.

    NOTE: cath_pw, anod_pw, and gap come FROM THE WAVEFORM ITSELF, not
    from arguments. This is the key difference from v1 (where pw_us was
    an external argument).
    """
    if time_arr is None:
        time_arr = np.arange(0, total_ms, dt_ms)
    n_t = len(time_arr)
    wave = np.zeros(n_t)

    cath_pw_us = get_cath_pw_us(waveform)
    anod_pw_us = get_anod_pw_us(waveform)
    gap_us     = get_gap_us(waveform)
    amp_ratio  = get_amp_ratio(waveform)

    cath_pw_ms = cath_pw_us / 1000.0
    anod_pw_ms = anod_pw_us / 1000.0
    gap_ms     = gap_us / 1000.0

    t0 = delay_ms
    t1 = t0 + cath_pw_ms
    t2 = t1 + gap_ms
    t3 = t2 + anod_pw_ms

    # Sanity: pulse must fit in simulation window
    if t3 > total_ms - 0.5:
        return wave, np.zeros(n_t, dtype=bool), np.zeros(n_t, dtype=bool), \
               {'invalid': True, 'reason': 'pulse_too_long',
                'cath_pw_us': cath_pw_us, 'anod_pw_us': anod_pw_us,
                'gap_us': gap_us, 'pulse_end_ms': t3}

    cath_mask = (time_arr >= t0) & (time_arr < t1)
    anod_mask = (time_arr >= t2) & (time_arr < t3)

    if not np.any(cath_mask) or not np.any(anod_mask):
        return wave, cath_mask, anod_mask, {
            'invalid': True, 'reason': 'mask_empty',
            'cath_pw_us': cath_pw_us, 'anod_pw_us': anod_pw_us,
            'gap_us': gap_us, 'pulse_end_ms': t3}

    n1 = int(np.sum(cath_mask))
    n2 = int(np.sum(anod_mask))

    cath_curve = render_phase_curve(get_cath_ctrl(waveform), n_samples=n1)
    anod_curve = render_phase_curve(get_anod_ctrl(waveform), n_samples=n2)

    # Force conventions: cathodic negative, anodic positive
    if np.sum(cath_curve) > 0:
        cath_curve = -cath_curve
    if np.sum(anod_curve) < 0:
        anod_curve = -anod_curve

    pk_cath = float(np.max(np.abs(cath_curve)))
    if pk_cath < 1e-9:
        return wave, cath_mask, anod_mask, {
            'invalid': True, 'reason': 'flat_cathodic',
            'cath_pw_us': cath_pw_us, 'anod_pw_us': anod_pw_us,
            'gap_us': gap_us, 'pulse_end_ms': t3}
    pk_anod = float(np.max(np.abs(anod_curve)))
    if pk_anod < 1e-9:
        return wave, cath_mask, anod_mask, {
            'invalid': True, 'reason': 'flat_anodic',
            'cath_pw_us': cath_pw_us, 'anod_pw_us': anod_pw_us,
            'gap_us': gap_us, 'pulse_end_ms': t3}

    # Step 1: normalize cathodic curve so peak = 1
    cath_curve = cath_curve / pk_cath

    # Step 2: normalize anodic curve so peak = amp_ratio (bias)
    anod_curve = (anod_curve / pk_anod) * amp_ratio

    # Step 3: charge balance via secondary scaling of anodic curve
    # area_cath is negative (cathodic), area_anod is positive.
    # We want area_cath + scale * area_anod = 0
    area_cath_signed = float(np.sum(cath_curve)) * dt_ms     # negative
    area_anod_signed = float(np.sum(anod_curve)) * dt_ms     # positive
    if abs(area_anod_signed) < 1e-12:
        return wave, cath_mask, anod_mask, {
            'invalid': True, 'reason': 'zero_anodic_area',
            'cath_pw_us': cath_pw_us, 'anod_pw_us': anod_pw_us,
            'gap_us': gap_us, 'pulse_end_ms': t3}
    balance_scale = -area_cath_signed / area_anod_signed
    anod_balanced = anod_curve * balance_scale

    wave[cath_mask] = cath_curve
    wave[anod_mask] = anod_balanced

    # Slew rate: max |dI/dt| (in mA/µs)
    diffs = np.diff(wave) / (dt_ms * 1000.0)    # mA per µs
    max_slew = float(np.max(np.abs(diffs))) if len(diffs) else 0.0

    # Charge per phase (for diagnostics; in nC*1000 units, since amp=1 mA)
    area_cath_final = float(np.sum(wave[cath_mask])) * dt_ms
    area_anod_final = float(np.sum(wave[anod_mask])) * dt_ms

    info = {
        'invalid': False,
        'cath_pw_us': cath_pw_us,
        'anod_pw_us': anod_pw_us,
        'gap_us': gap_us,
        'amp_ratio': amp_ratio,
        'pulse_end_ms': t3,
        'peak_cathodic': float(np.max(np.abs(cath_curve))),
        'peak_anodic':   float(np.max(np.abs(anod_balanced))),
        'balance_scale': float(balance_scale),
        'max_slew_mA_per_us': max_slew,
        'charge_imbalance': float(abs(area_cath_final + area_anod_final)),
    }
    return wave, cath_mask, anod_mask, info


def describe_waveform(waveform):
    """Categorize a waveform (well-behaved / multi-lobe / extreme)."""
    wave, m1, m2, info = render_waveform(waveform)
    if info.get('invalid'):
        return {'well_behaved': False, 'reason': info.get('reason'),
                'peak_ratio': None, 'n_lobes_cathodic': None,
                'gap_us': get_gap_us(waveform),
                'cath_pw_us': get_cath_pw_us(waveform),
                'anod_pw_us': get_anod_pw_us(waveform)}
    cath_curve = wave[m1]
    cath_abs = np.abs(cath_curve)
    pk = float(cath_abs.max())
    if pk < 1e-9:
        return {'well_behaved': False, 'reason': 'flat',
                'peak_ratio': None, 'n_lobes_cathodic': 0,
                'gap_us': info['gap_us'],
                'cath_pw_us': info['cath_pw_us'],
                'anod_pw_us': info['anod_pw_us']}
    cutoff = 0.3 * pk
    above = cath_abs > cutoff
    transitions = np.diff(above.astype(int))
    n_lobes = int(np.sum(transitions == 1))
    if above[0]:
        n_lobes += 1
    peak_ratio = info['peak_anodic'] / max(info['peak_cathodic'], 1e-9)
    well_behaved = (n_lobes <= 1) and (peak_ratio <= 5.0)
    reason = (None if well_behaved else
              ('multi_lobe' if n_lobes > 1 else 'extreme_peak_ratio'))
    return {
        'well_behaved': bool(well_behaved),
        'reason': reason,
        'peak_ratio': float(peak_ratio),
        'n_lobes_cathodic': int(n_lobes),
        'gap_us': float(info['gap_us']),
        'cath_pw_us': float(info['cath_pw_us']),
        'anod_pw_us': float(info['anod_pw_us']),
        'max_slew_mA_per_us': float(info['max_slew_mA_per_us']),
    }


# ════════════════════════════════════════════════════════════════════════════
#  EVOLUTION OPERATORS
# ════════════════════════════════════════════════════════════════════════════
def random_waveform(rng, bias_unimodal=True):
    params = np.zeros(N_TOTAL_PARAMS)
    if bias_unimodal:
        bell = np.exp(-0.5 * ((np.arange(N_CONTROL_POINTS_PER_PHASE) -
                               (N_CONTROL_POINTS_PER_PHASE-1)/2) / 2.0)**2)
        bell = bell / bell.max()
        params[IDX_CATH_CTRL_START:IDX_CATH_CTRL_END] = (
            -bell + 0.3 * rng.normal(0, 1, N_CONTROL_POINTS_PER_PHASE))
        params[IDX_ANOD_CTRL_START:IDX_ANOD_CTRL_END] = (
            bell + 0.3 * rng.normal(0, 1, N_CONTROL_POINTS_PER_PHASE))
    else:
        params[:IDX_ANOD_CTRL_END] = rng.uniform(-1.0, 1.0,
                                                  2*N_CONTROL_POINTS_PER_PHASE)
    # Gap: small bias
    params[IDX_GAP] = float(rng.uniform(0.0, 500.0))
    # PWs: bias toward middle of fitness range
    params[IDX_CATH_PW] = float(rng.uniform(50, 500))
    params[IDX_ANOD_PW] = float(rng.uniform(50, 800))
    # Amplitude ratio: log-uniform from 0.3 to 3 (symmetric-ish bias)
    params[IDX_AMP_RATIO] = float(np.exp(rng.uniform(np.log(0.3),
                                                       np.log(3.0))))
    # Spare slots
    params[20] = 0.0
    params[21] = 0.0
    # Clamp control points
    params[:IDX_ANOD_CTRL_END] = np.clip(
        params[:IDX_ANOD_CTRL_END], -1.5, 1.5)
    params[IDX_GAP]       = np.clip(params[IDX_GAP], GAP_MIN_US, GAP_MAX_US)
    params[IDX_CATH_PW]   = np.clip(params[IDX_CATH_PW], CATH_PW_MIN, CATH_PW_MAX)
    params[IDX_ANOD_PW]   = np.clip(params[IDX_ANOD_PW], ANOD_PW_MIN, ANOD_PW_MAX)
    params[IDX_AMP_RATIO] = np.clip(params[IDX_AMP_RATIO],
                                     AMP_RATIO_MIN, AMP_RATIO_MAX)
    return params


def combine_waveforms(parent_a, parent_b, rng):
    """Uniform combination: each parameter from one parent."""
    mask = rng.random(N_TOTAL_PARAMS) < 0.5
    child = np.where(mask, parent_a, parent_b)
    return child


def perturb_waveform(waveform, rng,
                     rate=PERTURB_RATE,
                     sigma_ctrl=PERTURB_SIGMA_CTRL,
                     sigma_gap=PERTURB_SIGMA_GAP,
                     sigma_pw=PERTURB_SIGMA_PW,
                     sigma_amp_ratio=PERTURB_SIGMA_AMP_RATIO):
    out = waveform.copy()
    # Control points
    for i in range(2 * N_CONTROL_POINTS_PER_PHASE):
        if rng.random() < rate:
            out[i] += rng.normal(0, sigma_ctrl)
    out[:IDX_ANOD_CTRL_END] = np.clip(out[:IDX_ANOD_CTRL_END], -1.5, 1.5)
    # Gap
    if rng.random() < rate:
        out[IDX_GAP] += rng.normal(0, sigma_gap)
    out[IDX_GAP] = np.clip(out[IDX_GAP], GAP_MIN_US, GAP_MAX_US)
    # Cath PW
    if rng.random() < rate:
        out[IDX_CATH_PW] += rng.normal(0, sigma_pw)
    out[IDX_CATH_PW] = np.clip(out[IDX_CATH_PW], CATH_PW_MIN, CATH_PW_MAX)
    # Anod PW
    if rng.random() < rate:
        out[IDX_ANOD_PW] += rng.normal(0, sigma_pw)
    out[IDX_ANOD_PW] = np.clip(out[IDX_ANOD_PW], ANOD_PW_MIN, ANOD_PW_MAX)
    # Amp ratio (multiplicative perturbation in log space)
    if rng.random() < rate:
        out[IDX_AMP_RATIO] = float(out[IDX_AMP_RATIO] *
                                    np.exp(rng.normal(0, sigma_amp_ratio)))
    out[IDX_AMP_RATIO] = np.clip(out[IDX_AMP_RATIO],
                                  AMP_RATIO_MIN, AMP_RATIO_MAX)
    return out


# ════════════════════════════════════════════════════════════════════════════
#  NSGA-II PARETO SORTING
# ════════════════════════════════════════════════════════════════════════════
def dominates(score_a, score_b):
    """Returns True if score_a dominates score_b (both minimized)."""
    if any(np.isnan(score_a)) or any(np.isnan(score_b)):
        return False
    return all(a <= b for a, b in zip(score_a, score_b)) and \
           any(a < b for a, b in zip(score_a, score_b))


def fast_non_dominated_sort(scores):
    """Standard NSGA-II non-dominated sorting.

    Inputs:  list of score tuples (charge, peak)
    Returns: list of fronts, each front is a list of indices
    """
    n = len(scores)
    domination_count = [0] * n
    dominated_set = [[] for _ in range(n)]
    fronts = [[]]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(scores[i], scores[j]):
                dominated_set[i].append(j)
            elif dominates(scores[j], scores[i]):
                domination_count[i] += 1
        if domination_count[i] == 0:
            fronts[0].append(i)
    f = 0
    while fronts[f]:
        next_front = []
        for i in fronts[f]:
            for j in dominated_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        f += 1
        fronts.append(next_front)
    fronts.pop()    # drop the last empty
    return fronts


def crowding_distance(scores, indices):
    """NSGA-II crowding distance for tie-breaking within a front."""
    if len(indices) <= 2:
        return {idx: float('inf') for idx in indices}
    dist = {idx: 0.0 for idx in indices}
    n_obj = len(scores[indices[0]])
    for obj in range(n_obj):
        sorted_idx = sorted(indices, key=lambda i: scores[i][obj])
        dist[sorted_idx[0]] = float('inf')
        dist[sorted_idx[-1]] = float('inf')
        obj_range = scores[sorted_idx[-1]][obj] - scores[sorted_idx[0]][obj]
        if obj_range < 1e-12:
            continue
        for k in range(1, len(sorted_idx) - 1):
            prev_s = scores[sorted_idx[k-1]][obj]
            next_s = scores[sorted_idx[k+1]][obj]
            dist[sorted_idx[k]] += (next_s - prev_s) / obj_range
    return dist


def pareto_tournament_pick(pool, scores, fronts, rng):
    """NSGA-II tournament: pick lower-front index, or by crowding distance."""
    # Build a rank-and-distance map
    rank_map = {}
    for rank, front in enumerate(fronts):
        for idx in front:
            rank_map[idx] = rank
    dist_map = {}
    for front in fronts:
        dist_map.update(crowding_distance(scores, front))
    i, j = rng.integers(0, len(pool), size=2)
    if rank_map.get(i, 999) < rank_map.get(j, 999):
        return pool[i]
    if rank_map.get(j, 999) < rank_map.get(i, 999):
        return pool[j]
    # Same rank: prefer higher crowding distance
    if dist_map.get(i, 0) > dist_map.get(j, 0):
        return pool[i]
    return pool[j]


# ════════════════════════════════════════════════════════════════════════════
#  MRG SIMULATION (multi-node detection — same as v7.2)
# ════════════════════════════════════════════════════════════════════════════
class MRGSimulator:
    def __init__(self, hoc_path):
        from neuron import h
        self.h = h
        h.load_file('stdrun.hoc')
        test = h.Section(name='__chk__')
        try:
            test.insert('axnode')
        except Exception:
            raise RuntimeError("AXNODE not loaded. Run nrnivmodl AXNODE.mod")
        h.load_file(str(hoc_path))
        h.stim.amp = 0.0
        h.stim.dur = 0.0
        h.stim.delay = 1e9
        self.AXONNODES = int(h.axonnodes)
        self.NODELEN = float(h.nodelength)
        self.PARAL1 = float(h.paralength1)
        self.PARAL2 = float(h.paralength2)
        self.INTERL = float(h.interlength)
        self.FIBERD = float(h.fiberD)
        self.CELSIUS = float(h.celsius)
        h.celsius = self.CELSIUS
        self.EXC_IDX = self.AXONNODES // 2
        max_idx = self.AXONNODES - 1
        if self.EXC_IDX + N_PROPAGATION_NODES > max_idx:
            self.EXC_IDX = max_idx - N_PROPAGATION_NODES
        self.PROP_INDICES = [self.EXC_IDX + k + 1
                              for k in range(N_PROPAGATION_NODES)]
        self.sections = self._build_sections()
        self.elec_x_um = next(x for sec, x in self.sections
                              if sec == h.node[self.EXC_IDX])
        self.ve_per_mA = self._build_transfer()
        self.time_arr = np.arange(0, T_TOTAL_MS, DT_MS)
        self.t_vec = h.Vector(self.time_arr)
        self.amp_vecs = [h.Vector(len(self.time_arr))
                         for _ in range(len(self.sections))]
        self.v_exc = h.Vector().record(h.node[self.EXC_IDX](0.5)._ref_v)
        self.v_props = [h.Vector().record(h.node[idx](0.5)._ref_v)
                        for idx in self.PROP_INDICES]
        self._run_sanity_checks()

    def _build_sections(self):
        h = self.h
        out, x = [], 0.0
        for i in range(self.AXONNODES):
            out.append((h.node[i], x + 0.5 * self.NODELEN))
            x += self.NODELEN
            if i == self.AXONNODES - 1:
                break
            out.append((h.MYSA[2*i],   x + 0.5*self.PARAL1)); x += self.PARAL1
            out.append((h.FLUT[2*i],   x + 0.5*self.PARAL2)); x += self.PARAL2
            for k in range(6):
                out.append((h.STIN[6*i + k], x + 0.5*self.INTERL))
                x += self.INTERL
            out.append((h.FLUT[2*i + 1], x + 0.5*self.PARAL2)); x += self.PARAL2
            out.append((h.MYSA[2*i + 1], x + 0.5*self.PARAL1)); x += self.PARAL1
        return out

    def _build_transfer(self):
        ve = np.zeros(len(self.sections))
        for n, (sec, x) in enumerate(self.sections):
            dx = x - self.elec_x_um
            r_um = np.sqrt(dx*dx + ELEC_RADIAL_UM**2)
            r_cm = r_um / 1e4
            ve[n] = RHO_E_OHM_CM / (4.0 * np.pi * r_cm)
        return ve

    def _run_sanity_checks(self):
        if self._fires(np.zeros(len(self.time_arr))):
            raise RuntimeError("Spontaneous firing at I=0!")
        rect = np.zeros(len(self.time_arr))
        i0 = int(DELAY_MS / DT_MS)
        i1 = i0 + int(0.5 / DT_MS)
        rect[i0:i1] = -2.0
        if not self._fires(rect):
            raise RuntimeError("2 mA / 500 µs rect failed!")

    def _set_drive(self, I_mA):
        for n, (sec, _) in enumerate(self.sections):
            v = self.amp_vecs[n]
            v.play_remove()
            v.from_python(self.ve_per_mA[n] * I_mA)
            v.play(sec(0.5)._ref_e_extracellular, self.t_vec, 1)

    @staticmethod
    def _first_upcross(v_arr, gate_idx):
        if len(v_arr) < 2:
            return None
        above = (v_arr > AP_THRESHOLD_MV).astype(int)
        crosses = np.where(np.diff(above) == 1)[0] + 1
        valid = crosses[crosses >= gate_idx]
        return int(valid[0]) if len(valid) else None

    def _fires(self, I_mA):
        h = self.h
        self._set_drive(I_mA)
        h.dt = DT_MS
        h.finitialize(-80.0)
        h.continuerun(T_TOTAL_MS)
        gate = int(DELAY_MS / DT_MS)
        v_e = np.array(self.v_exc)
        e_idx = self._first_upcross(v_e, gate)
        if e_idx is None:
            return False
        prop_idxs = []
        for v_p in self.v_props:
            arr = np.array(v_p)
            p_idx = self._first_upcross(arr, gate)
            if p_idx is None:
                return False
            prop_idxs.append(p_idx)
        # Check monotonic forward order
        seq = [e_idx] + prop_idxs
        for k in range(len(seq) - 1):
            if seq[k+1] <= seq[k]:
                return False
        # Check inter-node delays in physiological range
        deltas_ms = np.diff(seq) * DT_MS
        for d in deltas_ms:
            if not (INTERNODE_DELAY_MIN_MS <= d <= INTERNODE_DELAY_MAX_MS):
                return False
        return True

    def measure_threshold_charge_at_pw(self, waveform, fitness_pw_us):
        """Re-render waveform with cath_pw OVERRIDDEN to fitness_pw_us.

        The waveform's anodic PW and shape are kept; only cathodic PW
        scales to fitness_pw_us. This lets us test the same shape at
        multiple PWs while keeping the asymmetry consistent.

        Strategy: Make a temporary copy of the waveform with cath_pw set
        to fitness_pw_us; the anod_pw is scaled proportionally to keep
        the cath:anod duration ratio.
        """
        original_cath_pw = get_cath_pw_us(waveform)
        original_anod_pw = get_anod_pw_us(waveform)
        # Scale anod_pw to maintain the ratio (otherwise the waveform
        # changes identity between PWs)
        ratio = original_anod_pw / max(original_cath_pw, 1e-6)
        wf_test = waveform.copy()
        wf_test[IDX_CATH_PW] = float(np.clip(fitness_pw_us,
                                              CATH_PW_MIN, CATH_PW_MAX))
        wf_test[IDX_ANOD_PW] = float(np.clip(fitness_pw_us * ratio,
                                              ANOD_PW_MIN, ANOD_PW_MAX))
        wave_unit, m1, _, info = render_waveform(
            wf_test, dt_ms=DT_MS, time_arr=self.time_arr,
            delay_ms=DELAY_MS, total_ms=T_TOTAL_MS)
        if info.get('invalid'):
            return None, None, info
        if not self._fires(AMP_MAX_MA * wave_unit):
            return None, None, {**info, 'reason': 'unfireable_at_max'}
        lo, hi = AMP_MIN_MA, AMP_MAX_MA
        for _ in range(BISECT_MAX_ITER):
            if (hi - lo) <= BISECT_TOL_MA:
                break
            mid = 0.5 * (lo + hi)
            if self._fires(mid * wave_unit):
                hi = mid
            else:
                lo = mid
        threshold = 0.5 * (lo + hi)
        cathodic = (threshold * wave_unit)[m1]
        charge_nC = float(np.sum(np.abs(cathodic)) * DT_MS * 1000.0)
        peak_anodic = float(np.max((threshold * wave_unit)[~m1]) if np.any(~m1) else 0)
        peak_at_threshold = max(
            float(np.max(np.abs((threshold * wave_unit)[m1]))),
            peak_anodic
        )
        return float(threshold), charge_nC, {
            **info,
            'peak_mA_at_threshold': peak_at_threshold,
        }


# ════════════════════════════════════════════════════════════════════════════
#  FITNESS EVALUATION (multi-PW, multi-objective)
# ════════════════════════════════════════════════════════════════════════════
def evaluate_waveform(waveform, simulator, fitness_pws=FITNESS_PWS_US,
                      min_pws_fired=MIN_PWS_FIRED):
    """Measure waveform across multiple PWs.

    Returns
    -------
    score : (charge_obj, peak_obj)
        Both to MINIMIZE. NaN scores are dominated by anything.
    record : dict
        Detailed per-PW thresholds + diagnostic.
    """
    desc = describe_waveform(waveform)
    per_pw = {}
    charges_nC = []
    peaks_mA = []
    n_fired = 0
    for pw_us in fitness_pws:
        th_mA, q_nC, info = simulator.measure_threshold_charge_at_pw(
            waveform, pw_us)
        per_pw[pw_us] = {
            'threshold_mA': th_mA,
            'charge_nC': q_nC,
            'peak_mA': info.get('peak_mA_at_threshold')
                       if info and not info.get('invalid') else None,
        }
        if q_nC is not None and th_mA is not None:
            charges_nC.append(q_nC)
            peaks_mA.append(info.get('peak_mA_at_threshold', th_mA))
            n_fired += 1
    record = {
        'descriptor': desc,
        'per_pw': per_pw,
        'n_pws_fired': n_fired,
        'genome': [float(x) for x in waveform],
    }
    if n_fired < min_pws_fired:
        # Doesn't fire enough — strongly dominated
        record['charge_obj'] = float('inf')
        record['peak_obj']   = float('inf')
        record['slew_penalty'] = 0.0
        record['failed'] = True
        return (float('inf'), float('inf')), record
    # Geometric mean of charges (robust to outliers)
    charge_obj_raw = float(np.exp(np.mean(np.log(charges_nC))))
    peak_obj       = float(np.max(peaks_mA))
    # Slew rate penalty (from descriptor)
    max_slew = float(desc.get('max_slew_mA_per_us', 0.0))
    slew_penalty = SLEW_PENALTY_COEFF * max(0.0,
                                              max_slew - SLEW_LIMIT_mA_per_us)
    charge_obj = charge_obj_raw + slew_penalty * charge_obj_raw
    record['charge_obj'] = charge_obj
    record['peak_obj']   = peak_obj
    record['charge_obj_raw'] = charge_obj_raw
    record['slew_penalty'] = slew_penalty
    record['max_slew_mA_per_us'] = max_slew
    record['failed'] = False
    return (charge_obj, peak_obj), record


# ════════════════════════════════════════════════════════════════════════════
#  EVOLUTION LOOP
# ════════════════════════════════════════════════════════════════════════════
def evolve_waveforms(simulator, pool_size, n_generations,
                     fitness_pws, seed=42, verbose=True):
    rng = np.random.default_rng(seed)
    pool = [random_waveform(rng) for _ in range(pool_size)]
    scores = [None] * pool_size
    records = [None] * pool_size
    archive = []           # list of (waveform, score, record)
    history = []
    no_improve_count = 0
    best_charge_seen = float('inf')

    for gen in range(n_generations):
        t_gen = time.time()
        # Evaluate any unscored
        for i, wf in enumerate(pool):
            if scores[i] is None:
                sc, rec = evaluate_waveform(wf, simulator, fitness_pws)
                scores[i] = sc
                records[i] = rec
        # Combine pool + archive scores for sorting
        all_wfs = pool + [a[0] for a in archive]
        all_scores = scores + [a[1] for a in archive]
        all_records = records + [a[2] for a in archive]
        fronts = fast_non_dominated_sort(all_scores)
        # Update archive: keep top MAX_ARCHIVE_SIZE non-dominated
        new_archive = []
        for front in fronts:
            for idx in front:
                if len(new_archive) >= MAX_ARCHIVE_SIZE:
                    break
                new_archive.append((all_wfs[idx].copy(),
                                     all_scores[idx],
                                     all_records[idx]))
            if len(new_archive) >= MAX_ARCHIVE_SIZE:
                break
        archive = new_archive
        # Statistics
        valid_charges = [s[0] for s in scores if s[0] < float('inf')]
        valid_peaks   = [s[1] for s in scores if s[1] < float('inf')]
        n_valid = len(valid_charges)
        n_wb = sum(1 for r in records if r and
                   r.get('descriptor', {}).get('well_behaved'))
        if valid_charges:
            best_charge = float(min(valid_charges))
            best_peak   = float(min(valid_peaks))
        else:
            best_charge, best_peak = float('nan'), float('nan')
        n_pareto = len(fronts[0]) if fronts else 0
        history.append({
            'gen': gen,
            'best_charge_obj_nC': best_charge,
            'best_peak_obj_mA':   best_peak,
            'pool_n_valid':       n_valid,
            'pool_n_well_behaved': n_wb,
            'pool_n_pareto_front': n_pareto,
            'archive_size':       len(archive),
            'gen_time_sec':       float(time.time() - t_gen),
        })
        if verbose:
            print(f"  Gen {gen:2d}: best_Q={best_charge:6.2f} nC | "
                  f"best_Ipeak={best_peak:6.2f} mA | "
                  f"pareto={n_pareto} | archive={len(archive)} | "
                  f"WB={n_wb}/{pool_size} | "
                  f"t={time.time()-t_gen:.0f}s")
        # Improvement tracking
        if best_charge < best_charge_seen - 0.01:
            best_charge_seen = best_charge
            no_improve_count = 0
        else:
            no_improve_count += 1
        if no_improve_count >= EARLY_STOP_PATIENCE:
            if verbose:
                print(f"  [early-stop] no charge improvement for "
                      f"{EARLY_STOP_PATIENCE} generations")
            break
        if gen == n_generations - 1:
            break
        # Selection + breeding using NSGA-II
        n_elite = max(1, int(ELITISM_FRACTION * pool_size))
        # Elite = top of front 0 by crowding distance
        front0 = fronts[0] if fronts else []
        dist_map = crowding_distance(all_scores, front0)
        sorted_front0 = sorted(front0,
                                key=lambda i: -dist_map.get(i, 0))
        elite = [all_wfs[i].copy() for i in sorted_front0[:n_elite]]
        new_pool = elite[:]
        new_scores  = [all_scores[i]  for i in sorted_front0[:n_elite]]
        new_records = [all_records[i] for i in sorted_front0[:n_elite]]
        while len(new_pool) < pool_size:
            pa = pareto_tournament_pick(pool, scores, fronts[:1], rng)
            pb = pareto_tournament_pick(pool, scores, fronts[:1], rng)
            if rng.random() < COMBINE_RATE:
                child = combine_waveforms(pa, pb, rng)
            else:
                child = pa.copy()
            child = perturb_waveform(child, rng)
            new_pool.append(child)
            new_scores.append(None)
            new_records.append(None)
        pool = new_pool
        scores = new_scores
        records = new_records
    # Final sweep
    for i, wf in enumerate(pool):
        if scores[i] is None:
            sc, rec = evaluate_waveform(wf, simulator, fitness_pws)
            scores[i] = sc
            records[i] = rec
    return pool, scores, records, archive, history


# ════════════════════════════════════════════════════════════════════════════
#  CATALOG EXPORT (drop-in for v7.2 benchmark)
# ════════════════════════════════════════════════════════════════════════════
EVOLVED_CATALOG_TEMPLATE = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WAVEFORM CATALOG — evolved (v2: multi-PW, multi-objective Pareto)
==================================================================
Auto-generated by waveform_evolver_v2.py. Do not hand-edit.

Provenance
----------
  pool_size       : {POOL_SIZE}
  n_generations   : {N_GENS}
  fitness_PWs_us  : {FITNESS_PWS}
  seed            : {SEED}
  hoc_sha256_16   : {HOC_SHA}
  archive_size    : {N_WF}
  produced        : {TIMESTAMP}

Each entry in EVOLVED_WAVEFORMS is a 22-float vector:
  [0:8]   cathodic ctrl pts
  [8:16]  anodic ctrl pts
  [16]    gap_us
  [17]    cath_pw_us
  [18]    anod_pw_us
  [19]    amp_ratio
  [20:22] reserved
"""

import numpy as np
from scipy.interpolate import CubicSpline

N_CONTROL_POINTS_PER_PHASE = {N_CP}
PHASE_RESOLUTION = {PHASE_RES}


def _render_phase_curve(control_points, n_samples=PHASE_RESOLUTION):
    knots = np.linspace(0, 1, len(control_points))
    aug_k = np.concatenate([[-0.05], knots, [1.05]])
    aug_v = np.concatenate([[0.0], control_points, [0.0]])
    spl = CubicSpline(aug_k, aug_v, bc_type="natural", extrapolate=False)
    return spl(np.linspace(0, 1, n_samples))


EVOLVED_WAVEFORMS = {WAVEFORMS_DICT}

EVOLVED_META = {META_DICT}

SHAPES = {SHAPES_DICT}


def get_gap_us(shape_key):
    base = shape_key.rsplit("_", 1)[0] if shape_key.endswith(("_p1", "_p2")) else shape_key
    if base not in EVOLVED_WAVEFORMS:
        return 0.0
    return float(np.clip(EVOLVED_WAVEFORMS[base][16], 0.0, 3000.0))


def get_cath_pw_us(shape_key):
    base = shape_key.rsplit("_", 1)[0] if shape_key.endswith(("_p1", "_p2")) else shape_key
    if base not in EVOLVED_WAVEFORMS:
        return 200.0
    return float(np.clip(EVOLVED_WAVEFORMS[base][17], 10.0, 1500.0))


def get_anod_pw_us(shape_key):
    base = shape_key.rsplit("_", 1)[0] if shape_key.endswith(("_p1", "_p2")) else shape_key
    if base not in EVOLVED_WAVEFORMS:
        return 200.0
    return float(np.clip(EVOLVED_WAVEFORMS[base][18], 10.0, 3000.0))


def evo_phase_profile(tau, shape_key):
    """Backward compat with v7.2 benchmark: returns positive profile on [0,1]."""
    parts = shape_key.rsplit("_", 1)
    if len(parts) != 2 or parts[1] not in ("p1", "p2"):
        raise ValueError(f"Bad key: {{shape_key!r}}")
    base, phase = parts
    if base not in EVOLVED_WAVEFORMS:
        raise ValueError(f"Unknown shape: {{base!r}}")
    params = np.asarray(EVOLVED_WAVEFORMS[base])
    cp = (params[:N_CONTROL_POINTS_PER_PHASE] if phase == "p1"
          else params[N_CONTROL_POINTS_PER_PHASE:2*N_CONTROL_POINTS_PER_PHASE])
    p = _render_phase_curve(cp, n_samples=len(tau))
    if phase == "p1" and np.sum(p) > 0:
        p = -p
    if phase == "p2" and np.sum(p) < 0:
        p = -p
    abs_p = np.abs(p)
    pk = np.max(abs_p)
    return abs_p / max(pk, 1e-9)
'''


def write_evolved_catalog(path, archive, pool_size, n_gens, fitness_pws,
                           seed, hoc_sha):
    """archive: list of (waveform, score, record). Sorted by charge."""
    archive_sorted = sorted(archive, key=lambda a: a[1][0])
    waveforms_dict = {}
    meta_dict = {}
    shapes_dict = {}
    cmap = plt.get_cmap('viridis')
    for rank, (wf, sc, rec) in enumerate(archive_sorted):
        key = f"evo_v2_{rank:03d}"
        waveforms_dict[key] = [float(x) for x in wf]
        meta_dict[key] = {
            'rank':           int(rank),
            'charge_obj_nC':  float(sc[0]) if sc[0] != float('inf') else None,
            'peak_obj_mA':    float(sc[1]) if sc[1] != float('inf') else None,
            'charge_obj_raw_nC': rec.get('charge_obj_raw'),
            'slew_penalty':   rec.get('slew_penalty'),
            'max_slew_mA_per_us': rec.get('max_slew_mA_per_us'),
            'n_pws_fired':    rec.get('n_pws_fired'),
            'well_behaved':   rec.get('descriptor', {}).get('well_behaved'),
            'gap_us':         float(get_gap_us(wf)),
            'cath_pw_us':     float(get_cath_pw_us(wf)),
            'anod_pw_us':     float(get_anod_pw_us(wf)),
            'amp_ratio':      float(get_amp_ratio(wf)),
            'per_pw_charges_nC': {str(pw): rec['per_pw'][pw]['charge_nC']
                                   for pw in fitness_pws
                                   if pw in rec.get('per_pw', {})},
            'per_pw_thresholds_mA': {str(pw): rec['per_pw'][pw]['threshold_mA']
                                      for pw in fitness_pws
                                      if pw in rec.get('per_pw', {})},
        }
        c = cmap(rank / max(1, len(archive_sorted) - 1))
        color_hex = '#%02x%02x%02x' % (int(c[0]*255), int(c[1]*255),
                                        int(c[2]*255))
        label = (f"Pareto #{rank:03d}\n"
                 f"Q={meta_dict[key]['charge_obj_nC']:.1f}nC "
                 f"I={meta_dict[key]['peak_obj_mA']:.1f}mA")
        shapes_dict[key] = (label, color_hex)
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    body = EVOLVED_CATALOG_TEMPLATE.format(
        POOL_SIZE=pool_size, N_GENS=n_gens,
        FITNESS_PWS=str(fitness_pws), SEED=seed,
        HOC_SHA=hoc_sha,
        N_CP=N_CONTROL_POINTS_PER_PHASE, PHASE_RES=PHASE_RESOLUTION,
        WAVEFORMS_DICT=repr(waveforms_dict),
        META_DICT=repr(meta_dict),
        SHAPES_DICT=repr(shapes_dict),
        N_WF=len(archive_sorted), TIMESTAMP=timestamp,
    )
    with open(path, 'w') as f:
        f.write(body)


# ════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ════════════════════════════════════════════════════════════════════════════
def plot_evolution_curves(history, out_dir):
    if not history:
        return
    gens = [h['gen'] for h in history]
    bq = [h['best_charge_obj_nC'] for h in history]
    bp = [h['best_peak_obj_mA'] for h in history]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                    facecolor='white', sharex=True)
    ax1.set_facecolor('#f7f9fc')
    ax2.set_facecolor('#f7f9fc')
    ax1.plot(gens, bq, 'g-', linewidth=2, marker='o')
    ax1.set_ylabel('Best charge_obj [nC]')
    ax1.set_title('Multi-objective evolution: charge & peak amp', fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.4)
    ax2.plot(gens, bp, 'b-', linewidth=2, marker='s')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Best peak_obj [mA]')
    ax2.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir/'fig_evolution_charge.png', dpi=180,
                bbox_inches='tight')
    plt.close()


def plot_pareto_front(archive, history, out_dir):
    if not archive:
        return
    charges = [a[1][0] for a in archive
               if a[1][0] != float('inf') and not np.isnan(a[1][0])]
    peaks   = [a[1][1] for a in archive
               if a[1][1] != float('inf') and not np.isnan(a[1][1])]
    if not charges:
        return
    fig, ax = plt.subplots(figsize=(10, 7), facecolor='white')
    ax.set_facecolor('#f7f9fc')
    sc = ax.scatter(charges, peaks, c=range(len(charges)), cmap='viridis',
                     s=80, edgecolor='black', alpha=0.85)
    plt.colorbar(sc, ax=ax, label='Rank (by charge)')
    ax.set_xlabel('Geometric-mean threshold charge [nC]', fontsize=12)
    ax.set_ylabel('Max peak amplitude across PWs [mA]', fontsize=12)
    ax.set_title(f'Pareto front: {len(charges)} non-dominated waveforms\n'
                 f'(lower-left is better)', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir/'fig_pareto_front.png', dpi=180, bbox_inches='tight')
    plt.close()


def plot_top_waveforms(archive, out_dir, top_k=20):
    if not archive:
        return
    archive_sorted = sorted(archive, key=lambda a: a[1][0])
    rows = archive_sorted[:top_k]
    n_cols = 5
    n_rows = int(np.ceil(top_k / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3*n_rows),
                              facecolor='white')
    axes = np.atleast_2d(axes).flatten()
    time_arr = np.arange(0, T_TOTAL_MS, DT_MS)
    for plot_i, (wf, sc, rec) in enumerate(rows):
        ax = axes[plot_i]
        ax.set_facecolor('#f7f9fc')
        wave, m1, m2, info = render_waveform(wf, dt_ms=DT_MS,
                                              time_arr=time_arr)
        if info.get('invalid'):
            ax.text(0.5, 0.5, f"invalid:\n{info.get('reason')}",
                    ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            continue
        end_ms = info['pulse_end_ms'] + 0.5
        m = (time_arr >= DELAY_MS - 0.3) & (time_arr <= end_ms)
        wb = rec.get('descriptor', {}).get('well_behaved', False)
        col = '#2563eb' if wb else '#94a3b8'
        ax.plot(time_arr[m]*1000, wave[m], color=col, linewidth=1.6)
        ax.fill_between(time_arr[m]*1000, wave[m], 0,
                        where=wave[m]<0, color=col, alpha=0.3)
        ax.fill_between(time_arr[m]*1000, wave[m], 0,
                        where=wave[m]>0, color='#ef4444', alpha=0.2)
        ax.axhline(0, color='#9ca3af', linewidth=0.5)
        wb_flag = "✓" if wb else "✗"
        title = (f"#{plot_i:02d} Q={sc[0]:.1f}nC I={sc[1]:.1f}mA\n"
                 f"cath={info['cath_pw_us']:.0f}µs "
                 f"anod={info['anod_pw_us']:.0f}µs "
                 f"gap={info['gap_us']:.0f}µs {wb_flag}")
        ax.set_title(title, fontsize=7)
        ax.tick_params(labelsize=6)
    for unused in axes[top_k:]:
        unused.axis('off')
    fig.suptitle(f'Top {len(rows)} Pareto-front waveforms', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir/'fig_top_waveforms_v2.png', dpi=180,
                bbox_inches='tight')
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pool', type=int, default=DEFAULT_POOL_SIZE)
    parser.add_argument('--gens', type=int, default=DEFAULT_GENERATIONS)
    parser.add_argument('--pws',  type=str, default=None,
                        help='comma-separated fitness PWs in µs '
                             '(default: 50,200,500,1000)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quick', action='store_true',
                        help='small run: pool=15, gens=10, 2 PWs')
    parser.add_argument('--hoc', default='MRGaxon.hoc')
    parser.add_argument('--out', default=str(OUTPUT_DIR))
    args = parser.parse_args()

    if args.quick:
        args.pool, args.gens = 15, 10
        fitness_pws = [200, 500]
    elif args.pws:
        fitness_pws = [int(x) for x in args.pws.split(',')]
    else:
        fitness_pws = FITNESS_PWS_US

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    hoc_path = Path(args.hoc).resolve()
    if not hoc_path.exists():
        print(f"[ERROR] {hoc_path} not found.")
        sys.exit(1)
    with open(hoc_path, 'rb') as f:
        hoc_sha = hashlib.sha256(f.read()).hexdigest()[:16]

    print("="*72)
    print("  WAVEFORM EVOLVER v2")
    print("="*72)
    print(f"  pool size      : {args.pool}")
    print(f"  generations    : {args.gens}")
    print(f"  fitness PWs    : {fitness_pws}")
    print(f"  seed           : {args.seed}")
    print(f"  hoc_sha256_16  : {hoc_sha}")
    n_evals = args.pool + (args.gens - 1) * int(args.pool * (1 - ELITISM_FRACTION))
    n_pw_evals = n_evals * len(fitness_pws)
    est_h = n_pw_evals * 30 / 3600
    print(f"  est PW-evals   : ~{n_pw_evals}")
    print(f"  est runtime    : ~{est_h:.1f} h  ({est_h*60:.0f} min)")
    print()

    print("Initializing NEURON + MRG axon ...")
    sim = MRGSimulator(hoc_path)
    print(f"✓ MRG: D={sim.FIBERD}µm, EXC=node[{sim.EXC_IDX}], "
          f"PROP={sim.PROP_INDICES}")
    print()

    print("Starting evolution ...")
    t0 = time.time()
    pool, scores, records, archive, history = evolve_waveforms(
        sim, args.pool, args.gens, fitness_pws, seed=args.seed)
    elapsed = time.time() - t0
    print()
    print(f"Evolution complete in {elapsed/60:.1f} min ({elapsed/3600:.2f} h)")
    print(f"Final archive: {len(archive)} non-dominated waveforms")

    # Save outputs
    print("Saving outputs ...")
    hist_path = out_dir / 'evolution_history_v2.json'
    with open(hist_path, 'w') as f:
        json.dump({
            'config': {
                'pool': args.pool, 'gens': args.gens,
                'fitness_pws_us': fitness_pws,
                'seed': args.seed, 'hoc_sha256_16': hoc_sha,
                'slew_limit_mA_per_us': SLEW_LIMIT_mA_per_us,
                'slew_penalty_coeff':   SLEW_PENALTY_COEFF,
            },
            'history': history,
            'total_runtime_sec': elapsed,
            'numpy_version': np.__version__,
            'python_version': platform.python_version(),
        }, f, indent=2, default=lambda x: None)
    print(f"  ✓ {hist_path.name}")

    archive_path = out_dir / 'pareto_archive.json'
    with open(archive_path, 'w') as f:
        json.dump({
            'archive': [
                {
                    'genome': [float(x) for x in wf],
                    'score_charge_obj_nC': float(sc[0])
                        if sc[0] != float('inf') else None,
                    'score_peak_obj_mA':   float(sc[1])
                        if sc[1] != float('inf') else None,
                    'record': rec,
                }
                for wf, sc, rec in archive
            ],
            'fitness_pws_us': fitness_pws,
        }, f, indent=2, default=lambda x: None)
    print(f"  ✓ {archive_path.name}")

    catalog_path = out_dir / 'waveform_catalog_evolved_v2.py'
    write_evolved_catalog(catalog_path, archive, args.pool, args.gens,
                           fitness_pws, args.seed, hoc_sha)
    print(f"  ✓ {catalog_path.name}")

    # Plots
    plot_evolution_curves(history, out_dir)
    print("  ✓ fig_evolution_charge.png")
    plot_pareto_front(archive, history, out_dir)
    print("  ✓ fig_pareto_front.png")
    plot_top_waveforms(archive, out_dir)
    print("  ✓ fig_top_waveforms_v2.png")

    print()
    print("="*72)
    print(f"  All outputs in: {out_dir}/")
    print("="*72)


if __name__ == '__main__':
    main()