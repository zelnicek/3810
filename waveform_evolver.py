#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WAVEFORM EVOLVER FOR NEUROSTIMULATION
======================================

Evolves charge-balanced biphasic stimulation waveforms against the MRG
single-shock benchmark using a genetic-algorithm-style optimizer.

DESIGN
------
Each candidate waveform is described by 17 numbers:

    control_points[0:8]   = 8 control points for the CATHODIC phase
                            (the depolarising/activating phase)
    control_points[8:16]  = 8 control points for the ANODIC phase
                            (the charge-balancing recovery phase)
    control_points[16]    = INTERPHASE GAP duration (µs, in [0, 3000])

These control points are passed through a cubic spline to produce the
final smooth waveform. Charge balance is enforced automatically by
scaling the anodic phase amplitude.

OPTIMIZATION CYCLE
------------------
1. Generate a pool of N random waveforms (default N=30)
2. For each waveform, measure its threshold charge on an MRG axon
3. Sort by charge (lower = more efficient = better)
4. Keep the top 20% verbatim (elitism)
5. Fill the rest by:
     - picking 2 good waveforms via tournament selection
     - combining them (crossover-like)
     - perturbing the result (small random mutation)
6. Repeat for ~20 generations or until improvement plateaus

PARAMETER NAMING
----------------
We use neutral terminology — "waveform_pool" not "population", "control
points" not "genes". The optimization mechanism is genetic-algorithm-like
but the underlying object is a stimulation waveform, not a biological
chromosome.

USAGE
-----
1. Place this file next to MRGaxon.hoc and AXNODE.mod (compiled).
2. Compile mechanism: nrnivmodl
3. Run:
     python waveform_evolver.py --quick           # ~1-2 hours, pool=12, gens=8
     python waveform_evolver.py                    # default ~6h, pool=30, gens=20
     python waveform_evolver.py --pool 50 --gens 40 --pw 200   # overnight

OUTPUT
------
outputs_evolver/
  waveform_catalog_evolved.py      ← drop-in replacement for v6 benchmark
  evolution_history.json            ← per-generation log
  final_waveform_pool.json          ← all waveforms + their charges + gaps
  fig_evolution_curve.png           ← fitness over generations
  fig_top_waveforms.png             ← top 20 evolved shapes
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
N_TOTAL_PARAMS = 2 * N_CONTROL_POINTS_PER_PHASE + 1  # +1 for gap
PHASE_RESOLUTION = 100

DEFAULT_PW_US      = 200      # cathodic phase width — phase 2 follows automatically
DEFAULT_POOL_SIZE  = 30
DEFAULT_GENERATIONS = 20

ELITISM_FRACTION   = 0.20
TOURNAMENT_SIZE    = 3
PERTURB_RATE       = 0.15     # per-parameter probability of being perturbed
PERTURB_SIGMA      = 0.25     # Gaussian sigma for control-point perturbation
GAP_PERTURB_SIGMA_US = 200    # Gaussian sigma for gap perturbation (µs)
COMBINE_RATE       = 0.70
EARLY_STOP_PATIENCE = 5
PEAK_RATIO_PENALTY = 0.05

# Gap bounds (µs) — Hofmann et al. (2011) showed effects up to ~5 ms
GAP_MIN_US = 0.0
GAP_MAX_US = 3000.0

# Stimulation geometry (matches v6 benchmark exactly)
ELEC_RADIAL_UM    = 2000.0
RHO_E_OHM_CM      = 300.0
PROPAGATION_NODES = 3
AP_THRESHOLD_MV   = -20.0
DT_MS             = 0.005
T_TOTAL_MS        = 25.0      # increased from 15 to accommodate large gaps
DELAY_MS          = 2.0
PROP_WIN_LO_MS    = 0.010
PROP_WIN_HI_MS    = 0.500
BISECT_TOL_MA     = 0.001
AMP_MAX_MA        = 10.0
AMP_MIN_MA        = 1e-6
BISECT_MAX_ITER   = 50

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs_evolver"


# ════════════════════════════════════════════════════════════════════════════
#  WAVEFORM REPRESENTATION
#
#  A "waveform" is a numpy array of N_TOTAL_PARAMS = 17 floats:
#    waveform[0:8]   — cathodic-phase control points (free, [-1.5, 1.5])
#    waveform[8:16]  — anodic-phase control points   (free, [-1.5, 1.5])
#    waveform[16]    — interphase gap in µs (clipped to [GAP_MIN, GAP_MAX])
#
#  Rendering:
#    1. Each phase's 8 control points are interpolated with a cubic spline
#       to PHASE_RESOLUTION samples (default 100).
#    2. Phase 1 is forced negative (cathodic convention).
#    3. Phase 2 is forced positive and scaled to balance the charge.
#    4. The two phases are placed on the time axis with the requested gap
#       between them.
# ════════════════════════════════════════════════════════════════════════════
def render_phase_curve(control_points, n_samples=PHASE_RESOLUTION):
    """Cubic spline through control points, clamped to zero at phase ends."""
    knots = np.linspace(0, 1, len(control_points))
    aug_knots = np.concatenate([[-0.05], knots, [1.05]])
    aug_vals  = np.concatenate([[0.0], control_points, [0.0]])
    spl = CubicSpline(aug_knots, aug_vals,
                      bc_type='natural', extrapolate=False)
    return spl(np.linspace(0, 1, n_samples))


def get_gap_us(waveform):
    """Extract the gap parameter and clip to allowed range."""
    return float(np.clip(waveform[16], GAP_MIN_US, GAP_MAX_US))


def render_waveform(waveform, pw_us, dt_ms=DT_MS, time_arr=None,
                    delay_ms=DELAY_MS, total_ms=T_TOTAL_MS):
    """
    Build the time-domain waveform on the simulation time grid.

    Returns
    -------
    wave_array : numpy array (length N_T)
        The unit-amplitude waveform (peak of cathodic phase = -1).
        Caller multiplies by desired amplitude in mA.
    cathodic_mask : bool array
        True at samples belonging to phase 1.
    anodic_mask : bool array
        True at samples belonging to phase 2.
    info : dict
        Diagnostic data (peak amplitudes, areas, balance factor, gap).
    """
    if time_arr is None:
        time_arr = np.arange(0, total_ms, dt_ms)
    n_t = len(time_arr)
    wave = np.zeros(n_t)
    pw_ms = pw_us / 1000.0
    gap_us = get_gap_us(waveform)
    gap_ms = gap_us / 1000.0
    t0 = delay_ms
    t1 = t0 + pw_ms
    t2 = t1 + gap_ms
    t3 = t2 + pw_ms
    cathodic_mask = (time_arr >= t0) & (time_arr < t1)
    anodic_mask   = (time_arr >= t2) & (time_arr < t3)
    if not np.any(cathodic_mask) or not np.any(anodic_mask):
        return wave, cathodic_mask, anodic_mask, {'invalid': True}
    n1 = int(np.sum(cathodic_mask))
    n2 = int(np.sum(anodic_mask))
    cathodic_curve = render_phase_curve(waveform[:N_CONTROL_POINTS_PER_PHASE],
                                         n_samples=n1)
    anodic_curve   = render_phase_curve(
        waveform[N_CONTROL_POINTS_PER_PHASE:2*N_CONTROL_POINTS_PER_PHASE],
        n_samples=n2)
    # Force conventions: cathodic negative, anodic positive
    if np.sum(cathodic_curve) > 0:
        cathodic_curve = -cathodic_curve
    if np.sum(anodic_curve) < 0:
        anodic_curve = -anodic_curve
    # Normalize cathodic peak to -1 (so the unit-amplitude waveform peaks
    # at exactly 1 mA in absolute terms when the caller asks for amp=1)
    peak_cath = float(np.max(np.abs(cathodic_curve)))
    if peak_cath < 1e-9:
        return wave, cathodic_mask, anodic_mask, {'invalid': True,
                                                   'reason': 'flat_cathodic'}
    cathodic_curve = cathodic_curve / peak_cath
    peak_anod = float(np.max(np.abs(anodic_curve)))
    if peak_anod < 1e-9:
        return wave, cathodic_mask, anodic_mask, {'invalid': True,
                                                   'reason': 'flat_anodic'}
    anodic_curve = anodic_curve / peak_anod
    # Charge balance: scale anodic curve so its area equals cathodic area
    area_cath = float(np.sum(cathodic_curve)) * dt_ms      # negative
    area_anod_unit = float(np.sum(anodic_curve)) * dt_ms   # positive
    if abs(area_anod_unit) < 1e-9:
        return wave, cathodic_mask, anodic_mask, {'invalid': True,
                                                   'reason': 'zero_anodic_area'}
    balance_factor = -area_cath / area_anod_unit  # positive scalar
    anodic_balanced = anodic_curve * balance_factor
    wave[cathodic_mask] = cathodic_curve
    wave[anodic_mask]   = anodic_balanced
    info = {
        'invalid': False,
        'gap_us': gap_us,
        'peak_cathodic': float(np.max(np.abs(cathodic_curve))),
        'peak_anodic':   float(np.max(np.abs(anodic_balanced))),
        'area_cathodic': area_cath,
        'area_anodic':   float(np.sum(anodic_balanced)) * dt_ms,
        'balance_factor': float(balance_factor),
        'charge_imbalance': float(abs(area_cath +
                                      np.sum(anodic_balanced) * dt_ms)),
    }
    return wave, cathodic_mask, anodic_mask, info


def describe_waveform(waveform, pw_us=DEFAULT_PW_US):
    """Categorize a waveform after the fact: count lobes in cathodic phase,
    measure peak ratio, decide if it counts as 'well-behaved'."""
    wave, m1, m2, info = render_waveform(waveform, pw_us)
    if info.get('invalid'):
        return {'well_behaved': False, 'reason': info.get('reason', 'invalid'),
                'peak_ratio': None, 'n_lobes_cathodic': None,
                'gap_us': get_gap_us(waveform)}
    cath_curve = wave[m1]
    cath_abs = np.abs(cath_curve)
    pk = float(cath_abs.max())
    if pk < 1e-9:
        return {'well_behaved': False, 'reason': 'flat',
                'peak_ratio': None, 'n_lobes_cathodic': 0,
                'gap_us': info['gap_us']}
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
    }


# ════════════════════════════════════════════════════════════════════════════
#  EVOLUTION OPERATORS
# ════════════════════════════════════════════════════════════════════════════
def random_waveform(rng, bias_unimodal=True):
    """Create a random waveform. If bias_unimodal is True, initialize with
    a bell-shaped cathodic phase (more likely to be useful) plus small noise."""
    params = np.zeros(N_TOTAL_PARAMS)
    if bias_unimodal:
        bell = np.exp(-0.5 * ((np.arange(N_CONTROL_POINTS_PER_PHASE) -
                               (N_CONTROL_POINTS_PER_PHASE-1)/2) / 2.0)**2)
        bell = bell / bell.max()
        params[:N_CONTROL_POINTS_PER_PHASE] = (
            -bell + 0.3 * rng.normal(0, 1, N_CONTROL_POINTS_PER_PHASE))
        params[N_CONTROL_POINTS_PER_PHASE:2*N_CONTROL_POINTS_PER_PHASE] = (
            bell + 0.3 * rng.normal(0, 1, N_CONTROL_POINTS_PER_PHASE))
    else:
        params[:2*N_CONTROL_POINTS_PER_PHASE] = rng.uniform(
            -1.0, 1.0, 2*N_CONTROL_POINTS_PER_PHASE)
    # Initial gap: slight bias toward small values, but allow exploration
    params[16] = float(rng.uniform(0.0, 500.0))
    # Clamp control points
    params[:2*N_CONTROL_POINTS_PER_PHASE] = np.clip(
        params[:2*N_CONTROL_POINTS_PER_PHASE], -1.5, 1.5)
    params[16] = float(np.clip(params[16], GAP_MIN_US, GAP_MAX_US))
    return params


def combine_waveforms(parent_a, parent_b, rng):
    """Uniform combination: each parameter is taken from parent_a or
    parent_b with equal probability. The gap parameter follows the same rule."""
    mask = rng.random(N_TOTAL_PARAMS) < 0.5
    child = np.where(mask, parent_a, parent_b)
    return child


def perturb_waveform(waveform, rng,
                     rate=PERTURB_RATE,
                     sigma_ctrl=PERTURB_SIGMA,
                     sigma_gap=GAP_PERTURB_SIGMA_US):
    """Random perturbation. Control points and gap use different sigmas
    because they have different natural scales (unitless vs. µs)."""
    out = waveform.copy()
    # Control points
    for i in range(2 * N_CONTROL_POINTS_PER_PHASE):
        if rng.random() < rate:
            out[i] += rng.normal(0, sigma_ctrl)
    out[:2*N_CONTROL_POINTS_PER_PHASE] = np.clip(
        out[:2*N_CONTROL_POINTS_PER_PHASE], -1.5, 1.5)
    # Gap parameter
    if rng.random() < rate:
        out[16] += rng.normal(0, sigma_gap)
    out[16] = float(np.clip(out[16], GAP_MIN_US, GAP_MAX_US))
    return out


def tournament_pick(waveform_pool, scores, rng, k=TOURNAMENT_SIZE):
    """Pick the best of k random waveforms from the pool."""
    idx = rng.integers(0, len(waveform_pool), size=k)
    best = idx[np.argmin([scores[i] for i in idx])]
    return waveform_pool[best]


# ════════════════════════════════════════════════════════════════════════════
#  MRG SIMULATION (for measuring threshold charge)
# ════════════════════════════════════════════════════════════════════════════
class MRGChargeMeasurer:
    """Wraps the NEURON+MRG simulation so the evolver can call
    measure_threshold_charge() without touching NEURON directly."""

    def __init__(self, hoc_path, verbose=False):
        from neuron import h
        self.h = h
        h.load_file('stdrun.hoc')
        # Verify mechanism is loaded
        test = h.Section(name='__verify__')
        try:
            test.insert('axnode')
        except Exception:
            raise RuntimeError("AXNODE not loaded. Run nrnivmodl AXNODE.mod.")
        h.load_file(str(hoc_path))
        h.stim.amp = 0.0
        h.stim.dur = 0.0
        h.stim.delay = 1e9
        # Geometry
        self.AXONNODES = int(h.axonnodes)
        self.DELTAX = float(h.deltax)
        self.NODELEN = float(h.nodelength)
        self.PARAL1 = float(h.paralength1)
        self.PARAL2 = float(h.paralength2)
        self.INTERL = float(h.interlength)
        self.FIBERD = float(h.fiberD)
        self.CELSIUS = float(h.celsius)
        h.celsius = self.CELSIUS
        self.CENTER_NODE = self.AXONNODES // 2
        self.PROP_IDX = min(self.CENTER_NODE + PROPAGATION_NODES,
                            self.AXONNODES - 1)
        self.sections = self._build_sections()
        self.elec_x_um = next(x for sec, x in self.sections
                              if sec == h.node[self.CENTER_NODE])
        self.ve_per_mA = self._build_transfer()
        self.time_arr = np.arange(0, T_TOTAL_MS, DT_MS)
        self.t_vec = h.Vector(self.time_arr)
        self.amp_vecs = [h.Vector(len(self.time_arr))
                         for _ in range(len(self.sections))]
        self.v_exc = h.Vector().record(h.node[self.CENTER_NODE](0.5)._ref_v)
        self.v_prop = h.Vector().record(h.node[self.PROP_IDX](0.5)._ref_v)
        self.verbose = verbose
        self._run_sanity_checks()

    def _build_sections(self):
        h = self.h
        out, x = [], 0.0
        for i in range(self.AXONNODES):
            out.append((h.node[i], x + 0.5 * self.NODELEN))
            x += self.NODELEN
            if i == self.AXONNODES - 1:
                break
            out.append((h.MYSA[2*i],   x + 0.5 * self.PARAL1)); x += self.PARAL1
            out.append((h.FLUT[2*i],   x + 0.5 * self.PARAL2)); x += self.PARAL2
            for k in range(6):
                out.append((h.STIN[6*i + k], x + 0.5 * self.INTERL))
                x += self.INTERL
            out.append((h.FLUT[2*i + 1], x + 0.5 * self.PARAL2)); x += self.PARAL2
            out.append((h.MYSA[2*i + 1], x + 0.5 * self.PARAL1)); x += self.PARAL1
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
        rect_test = np.zeros(len(self.time_arr))
        i0 = int(DELAY_MS / DT_MS)
        i1 = i0 + int(0.500 / DT_MS)
        rect_test[i0:i1] = -2.0
        if not self._fires(rect_test):
            raise RuntimeError("2 mA / 500 µs rect failed!")

    def _set_drive(self, I_mA):
        for n, (sec, _) in enumerate(self.sections):
            v = self.amp_vecs[n]
            v.play_remove()
            v.from_python(self.ve_per_mA[n] * I_mA)
            v.play(sec(0.5)._ref_e_extracellular, self.t_vec, 1)

    def _fires(self, I_mA):
        h = self.h
        self._set_drive(I_mA)
        h.dt = DT_MS
        h.finitialize(-80.0)
        h.continuerun(T_TOTAL_MS)
        v_e = np.array(self.v_exc)
        v_p = np.array(self.v_prop)
        s_idx = int(DELAY_MS / DT_MS)
        e = self._first_upcross(v_e, s_idx)
        p = self._first_upcross(v_p, s_idx)
        if e is None or p is None:
            return False
        dt_ms = (p - e) * DT_MS
        return (dt_ms > 0) and (PROP_WIN_LO_MS <= dt_ms <= PROP_WIN_HI_MS)

    @staticmethod
    def _first_upcross(v_arr, s_idx):
        if len(v_arr) < 2:
            return None
        above = (v_arr > AP_THRESHOLD_MV).astype(int)
        crosses = np.where(np.diff(above) == 1)[0] + 1
        valid = crosses[crosses >= s_idx]
        return int(valid[0]) if len(valid) else None

    def measure_threshold_charge(self, waveform, pw_us):
        """Bisection to find the smallest amplitude that fires.
        Returns (threshold_mA, charge_nC, info)."""
        wave_unit, mask1, _, info = render_waveform(
            waveform, pw_us, dt_ms=DT_MS, time_arr=self.time_arr,
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
        cathodic = (threshold * wave_unit)[mask1]
        charge_nC = float(np.sum(np.abs(cathodic)) * DT_MS * 1000.0)
        return float(threshold), charge_nC, info


def charge_score(threshold_mA, charge_nC, descriptor):
    """Lower is better. Returns the threshold charge plus a small penalty
    for ill-conditioned waveforms."""
    if threshold_mA is None or charge_nC is None:
        return 1e9
    score = charge_nC
    pr = descriptor.get('peak_ratio')
    if pr is not None and pr > 5.0:
        score += PEAK_RATIO_PENALTY * (pr - 5.0) * charge_nC
    return score


# ════════════════════════════════════════════════════════════════════════════
#  EVOLUTION LOOP
# ════════════════════════════════════════════════════════════════════════════
def evolve_waveforms(measurer, pool_size, n_generations, pw_us,
                     seed=42, verbose=True):
    rng = np.random.default_rng(seed)
    pool = [random_waveform(rng) for _ in range(pool_size)]
    scores       = [None] * pool_size
    descriptors  = [None] * pool_size
    thresholds   = [None] * pool_size
    charges      = [None] * pool_size
    history = []
    best_ever = (1e9, None, None)
    no_improve_count = 0

    for gen in range(n_generations):
        t_gen = time.time()
        # Evaluate any not-yet-scored waveforms
        for i, wf in enumerate(pool):
            if scores[i] is None:
                desc = describe_waveform(wf, pw_us=pw_us)
                th, q, _ = measurer.measure_threshold_charge(wf, pw_us)
                scores[i] = charge_score(th, q, desc)
                descriptors[i] = desc
                thresholds[i] = th
                charges[i] = q
        # Stats
        valid_scores = [s for s in scores if s < 1e8]
        best = min(scores)
        best_idx = int(np.argmin(scores))
        mean_s = float(np.mean(valid_scores)) if valid_scores else float('nan')
        worst = max(valid_scores) if valid_scores else float('nan')
        n_valid = len(valid_scores)
        n_wb = sum(1 for d in descriptors if d and d['well_behaved'])
        gap_at_best = (descriptors[best_idx]['gap_us']
                       if descriptors[best_idx] else None)
        history.append({
            'gen': gen,
            'best_score_nC': float(best),
            'mean_score_nC': mean_s,
            'worst_score_nC': float(worst),
            'n_valid_in_pool': n_valid,
            'n_well_behaved_in_pool': n_wb,
            'best_gap_us': gap_at_best,
            'gen_time_sec': float(time.time() - t_gen),
        })
        if verbose:
            gap_str = f"{gap_at_best:.0f}" if gap_at_best is not None else "?"
            print(f"  Gen {gen:2d}: best={best:6.2f} nC | mean={mean_s:6.2f} | "
                  f"valid={n_valid}/{pool_size} | WB={n_wb} | "
                  f"best_gap={gap_str} µs | t={time.time()-t_gen:.0f}s")
        if best < best_ever[0]:
            best_ever = (best, pool[best_idx].copy(), gen)
            no_improve_count = 0
        else:
            no_improve_count += 1
        if no_improve_count >= EARLY_STOP_PATIENCE:
            if verbose:
                print(f"  [early-stop] no improvement for "
                      f"{EARLY_STOP_PATIENCE} generations")
            break
        if gen == n_generations - 1:
            break
        # Selection + breeding
        n_elite = max(1, int(ELITISM_FRACTION * pool_size))
        order = np.argsort(scores)
        new_pool = [pool[i].copy() for i in order[:n_elite]]
        new_scores = [scores[i] for i in order[:n_elite]]
        new_desc = [descriptors[i] for i in order[:n_elite]]
        new_th = [thresholds[i] for i in order[:n_elite]]
        new_q  = [charges[i] for i in order[:n_elite]]
        while len(new_pool) < pool_size:
            pa = tournament_pick(pool, scores, rng)
            pb = tournament_pick(pool, scores, rng)
            if rng.random() < COMBINE_RATE:
                child = combine_waveforms(pa, pb, rng)
            else:
                child = pa.copy()
            child = perturb_waveform(child, rng)
            new_pool.append(child)
            new_scores.append(None)
            new_desc.append(None)
            new_th.append(None)
            new_q.append(None)
        pool = new_pool
        scores = new_scores
        descriptors = new_desc
        thresholds = new_th
        charges = new_q

    # Final sweep: ensure everything is evaluated
    for i, wf in enumerate(pool):
        if scores[i] is None:
            desc = describe_waveform(wf, pw_us=pw_us)
            th, q, _ = measurer.measure_threshold_charge(wf, pw_us)
            scores[i] = charge_score(th, q, desc)
            descriptors[i] = desc
            thresholds[i] = th
            charges[i] = q
    return pool, scores, descriptors, thresholds, charges, history, best_ever


# ════════════════════════════════════════════════════════════════════════════
#  CATALOG EXPORT  (drop-in for v6 benchmark)
# ════════════════════════════════════════════════════════════════════════════
EVOLVED_CATALOG_TEMPLATE = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WAVEFORM CATALOG — evolved
============================

Auto-generated by waveform_evolver.py. Do not hand-edit.

This catalog is a drop-in replacement for the original waveform_catalog.py.
Each shape key 'evo_NNN' refers to a waveform stored in EVOLVED_WAVEFORMS.

Provenance
----------
  pool_size       : {POOL_SIZE}
  n_generations   : {N_GENS}
  pw_us_fitness   : {PW_US}
  seed            : {SEED}
  hoc_sha256_16   : {HOC_SHA}
  best_score_nC   : {BEST_SCORE:.4f}
  total_waveforms : {N_WF}
  produced        : {TIMESTAMP}
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


# Each entry: 17 floats (8 cathodic + 8 anodic control points + gap_us)
EVOLVED_WAVEFORMS = {WAVEFORMS_DICT}


# Per-waveform metadata produced by the evolution
EVOLVED_META = {META_DICT}


# Compatible with the v6 benchmark format:  SHAPES[key] = (label, color)
SHAPES = {SHAPES_DICT}


def evo_phase_profile(tau, shape_key):
    """Render a phase of an evolved waveform.
    shape_key: 'evo_NNN_p1' (cathodic) or 'evo_NNN_p2' (anodic).
    Returns a positive-valued array on tau ∈ [0,1] — caller handles signs."""
    parts = shape_key.rsplit("_", 1)
    if len(parts) != 2 or parts[1] not in ("p1", "p2"):
        raise ValueError(f"Bad evolved key: {{shape_key!r}}")
    base, phase = parts
    if base not in EVOLVED_WAVEFORMS:
        raise ValueError(f"Unknown waveform: {{base!r}}")
    params = np.asarray(EVOLVED_WAVEFORMS[base])
    if phase == "p1":
        cp = params[:N_CONTROL_POINTS_PER_PHASE]
    else:
        cp = params[N_CONTROL_POINTS_PER_PHASE:2*N_CONTROL_POINTS_PER_PHASE]
    p = _render_phase_curve(cp, n_samples=len(tau))
    if phase == "p1" and np.sum(p) > 0:
        p = -p
    if phase == "p2" and np.sum(p) < 0:
        p = -p
    abs_p = np.abs(p)
    pk = np.max(abs_p)
    return abs_p / max(pk, 1e-9)


def get_gap_us(shape_key):
    """Return the interphase gap (µs) for an evolved waveform."""
    base = shape_key.rsplit("_", 1)[0] if shape_key.endswith(("_p1", "_p2")) \\
           else shape_key
    if base not in EVOLVED_WAVEFORMS:
        return 0.0
    return float(np.clip(EVOLVED_WAVEFORMS[base][16], 0.0, 3000.0))
'''


def write_evolved_catalog(path, pool, descriptors, thresholds, charges,
                           history, best_ever, pool_size, n_gens, pw_us,
                           seed, hoc_sha):
    final_scores = [charge_score(th, q, d) for th, q, d in
                    zip(thresholds, charges, descriptors)]
    order = np.argsort(final_scores)
    waveforms_dict = {}
    meta_dict = {}
    shapes_dict = {}
    cmap = plt.get_cmap('viridis')
    for rank, i in enumerate(order):
        key = f"evo_{rank:03d}"
        waveforms_dict[key] = [float(x) for x in pool[i]]
        meta_dict[key] = {
            'rank': int(rank),
            'score_nC': float(final_scores[i]),
            'threshold_uA': (float(thresholds[i] * 1000)
                             if thresholds[i] is not None else None),
            'charge_nC': (float(charges[i])
                          if charges[i] is not None else None),
            'gap_us': float(np.clip(pool[i][16], GAP_MIN_US, GAP_MAX_US)),
            'well_behaved': bool(descriptors[i]['well_behaved'])
                             if descriptors[i] else False,
            'reason': (descriptors[i]['reason']
                       if descriptors[i] else 'invalid'),
            'peak_ratio': (float(descriptors[i]['peak_ratio'])
                           if descriptors[i] and
                           descriptors[i]['peak_ratio'] is not None
                           else None),
            'n_lobes_cathodic': (int(descriptors[i]['n_lobes_cathodic'])
                                  if descriptors[i] and
                                  descriptors[i]['n_lobes_cathodic'] is not None
                                  else None),
        }
        c = cmap(rank / max(1, len(pool) - 1))
        color_hex = '#%02x%02x%02x' % (int(c[0]*255), int(c[1]*255), int(c[2]*255))
        gap_label = f", gap={meta_dict[key]['gap_us']:.0f}µs" \
                    if meta_dict[key]['gap_us'] > 50 else ""
        label = (f"Evolved #{rank:03d}\n"
                 f"(Q={meta_dict[key]['charge_nC']:.1f} nC{gap_label})"
                 if meta_dict[key]['charge_nC'] is not None
                 else f"Evolved #{rank:03d}\n(invalid)")
        shapes_dict[key] = (label, color_hex)
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    body = EVOLVED_CATALOG_TEMPLATE.format(
        POOL_SIZE=pool_size, N_GENS=n_gens, PW_US=pw_us, SEED=seed,
        HOC_SHA=hoc_sha, N_CP=N_CONTROL_POINTS_PER_PHASE,
        PHASE_RES=PHASE_RESOLUTION,
        WAVEFORMS_DICT=json.dumps(waveforms_dict, indent=2),
        META_DICT=json.dumps(meta_dict, indent=2, default=lambda x: None),
        SHAPES_DICT=repr(shapes_dict),
        BEST_SCORE=(best_ever[0] if best_ever[0] < 1e8 else float('nan')),
        N_WF=len(waveforms_dict), TIMESTAMP=timestamp,
    )
    with open(path, 'w') as f:
        f.write(body)


# ════════════════════════════════════════════════════════════════════════════
#  PLOTS
# ════════════════════════════════════════════════════════════════════════════
def plot_evolution_curve(history, out_path):
    if not history:
        return
    gens = [h['gen'] for h in history]
    bests = [h['best_score_nC'] for h in history]
    means = [h['mean_score_nC'] for h in history]
    worsts = [h['worst_score_nC'] for h in history]
    gaps = [h['best_gap_us'] for h in history]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), facecolor='white',
                                    sharex=True)
    ax1.set_facecolor('#f7f9fc')
    ax2.set_facecolor('#f7f9fc')
    ax1.plot(gens, bests, 'g-', linewidth=2, label='Best',  marker='o')
    ax1.plot(gens, means, 'b--', linewidth=1.5, label='Mean', alpha=0.7)
    ax1.plot(gens, worsts, 'r:', linewidth=1, label='Worst (valid only)',
             alpha=0.5)
    ax1.set_ylabel('Threshold charge [nC]')
    ax1.set_title('Waveform evolution: charge over generations')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.4)
    ax2.plot(gens, gaps, 'm-', linewidth=2, marker='s')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Best waveform: gap [µs]')
    ax2.set_title('Interphase gap of best waveform per generation')
    ax2.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()


def plot_top_waveforms(pool, descriptors, thresholds, charges, pw_us,
                        out_path, top_k=20):
    final_scores = [charge_score(th, q, d) for th, q, d in
                     zip(thresholds, charges, descriptors)]
    order = np.argsort(final_scores)[:top_k]
    n_cols = 5
    n_rows = int(np.ceil(top_k / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3*n_rows),
                              facecolor='white')
    axes = np.atleast_2d(axes).flatten()
    time_arr = np.arange(0, T_TOTAL_MS, DT_MS)
    for plot_i, i in enumerate(order):
        ax = axes[plot_i]
        ax.set_facecolor('#f7f9fc')
        gap = get_gap_us(pool[i])
        wave, m1, m2, info = render_waveform(
            pool[i], pw_us, dt_ms=DT_MS, time_arr=time_arr)
        # Show window: from delay-0.3 to end of phase 2 + 0.5 ms
        end_ms = DELAY_MS + 2 * pw_us / 1000 + gap / 1000 + 0.5
        m = (time_arr >= DELAY_MS - 0.3) & (time_arr <= end_ms)
        col = '#2563eb' if descriptors[i]['well_behaved'] else '#cbd5e1'
        ax.plot(time_arr[m]*1000, wave[m], color=col, linewidth=1.8)
        ax.fill_between(time_arr[m]*1000, wave[m], 0,
                        where=wave[m] < 0, color=col, alpha=0.3)
        ax.fill_between(time_arr[m]*1000, wave[m], 0,
                        where=wave[m] > 0, color='#ef4444', alpha=0.2)
        ax.axhline(0, color='#9ca3af', linewidth=0.5)
        wb_flag = "✓" if descriptors[i]['well_behaved'] else "✗"
        title = (f"#{plot_i:02d}  Q={charges[i]:.1f}nC  "
                 f"I={thresholds[i]*1000:.0f}µA  "
                 f"gap={gap:.0f}µs  {wb_flag}")
        ax.set_title(title, fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_xlabel('time [µs]', fontsize=6)
    for unused in axes[top_k:]:
        unused.axis('off')
    fig.suptitle(f'Top {top_k} evolved waveforms (lowest threshold charge)',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pool', type=int, default=DEFAULT_POOL_SIZE,
                        help=f'pool size (default {DEFAULT_POOL_SIZE})')
    parser.add_argument('--gens', type=int, default=DEFAULT_GENERATIONS,
                        help=f'generations (default {DEFAULT_GENERATIONS})')
    parser.add_argument('--pw',   type=int, default=DEFAULT_PW_US,
                        help=f'cathodic pulse width µs (default {DEFAULT_PW_US})')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quick', action='store_true',
                        help='quick mode: pool=12, gens=8')
    parser.add_argument('--hoc', default='MRGaxon.hoc',
                        help='path to MRGaxon.hoc')
    args = parser.parse_args()
    if args.quick:
        args.pool, args.gens = 12, 8

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    hoc_path = Path(args.hoc).resolve()
    if not hoc_path.exists():
        print(f"[ERROR] {hoc_path} not found.")
        sys.exit(1)
    with open(hoc_path, 'rb') as f:
        hoc_sha = hashlib.sha256(f.read()).hexdigest()[:16]

    print("="*72)
    print("  WAVEFORM EVOLVER FOR NEUROSTIMULATION")
    print("="*72)
    print(f"  pool size      : {args.pool}")
    print(f"  generations    : {args.gens}")
    print(f"  cathodic PW    : {args.pw} µs (anodic phase auto-balanced)")
    print(f"  gap range      : {GAP_MIN_US:.0f} – {GAP_MAX_US:.0f} µs (evolved)")
    print(f"  seed           : {args.seed}")
    print(f"  hoc_sha256_16  : {hoc_sha}")
    expected_evals = args.pool * args.gens * (1 - ELITISM_FRACTION)
    eta_min = expected_evals * 40 / 60
    print(f"  expected evals : ~{expected_evals:.0f}")
    print(f"  estimated time : ~{eta_min:.0f} min ({eta_min/60:.1f} h)")
    print()

    print("Initializing NEURON + MRG axon ...")
    measurer = MRGChargeMeasurer(hoc_path, verbose=True)
    print("✓ Sanity checks passed")
    print()

    print("Starting waveform evolution ...")
    t0 = time.time()
    pool, scores, descs, ths, qs, history, best_ever = evolve_waveforms(
        measurer, args.pool, args.gens, args.pw, seed=args.seed)
    elapsed = time.time() - t0
    print(f"\nEvolution complete in {elapsed/60:.1f} min")
    print(f"Best score:       {best_ever[0]:.4f} nC")
    print(f"Best at generation: {best_ever[2]}")
    if best_ever[1] is not None:
        gap_best = float(np.clip(best_ever[1][16], GAP_MIN_US, GAP_MAX_US))
        print(f"Best waveform gap:  {gap_best:.0f} µs")

    # Save outputs
    history_path = OUTPUT_DIR / "evolution_history.json"
    with open(history_path, 'w') as f:
        json.dump({
            'config': vars(args),
            'history': history,
            'best_ever_score_nC': float(best_ever[0]) if best_ever[0] < 1e8 else None,
            'best_ever_gen': best_ever[2],
            'total_runtime_sec': elapsed,
            'hoc_sha256_16': hoc_sha,
            'numpy_version': np.__version__,
            'python_version': platform.python_version(),
        }, f, indent=2, default=lambda x: None)
    print(f"Saved: {history_path}")

    pool_path = OUTPUT_DIR / "final_waveform_pool.json"
    with open(pool_path, 'w') as f:
        json.dump({
            'waveforms': [list(map(float, w)) for w in pool],
            'scores': [float(s) if s is not None else None for s in scores],
            'thresholds_mA': [float(x) if x is not None else None for x in ths],
            'charges_nC':    [float(x) if x is not None else None for x in qs],
            'gaps_us':       [float(np.clip(w[16], GAP_MIN_US, GAP_MAX_US))
                              for w in pool],
            'descriptors':   descs,
        }, f, indent=2, default=lambda x: None)
    print(f"Saved: {pool_path}")

    catalog_path = OUTPUT_DIR / "waveform_catalog_evolved.py"
    write_evolved_catalog(catalog_path, pool, descs, ths, qs, history,
                           best_ever, args.pool, args.gens, args.pw,
                           args.seed, hoc_sha)
    print(f"Saved: {catalog_path}")

    plot_evolution_curve(history, OUTPUT_DIR / "fig_evolution_curve.png")
    print("Saved: fig_evolution_curve.png")
    plot_top_waveforms(pool, descs, ths, qs, args.pw,
                        OUTPUT_DIR / "fig_top_waveforms.png", top_k=20)
    print("Saved: fig_top_waveforms.png")

    print()
    print("="*72)
    print(f"  All outputs in: {OUTPUT_DIR}/")
    print("="*72)


if __name__ == '__main__':
    main()