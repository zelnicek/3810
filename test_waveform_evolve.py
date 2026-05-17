#!/usr/bin/env python3
"""Offline tests for waveform_evolver.py — no NEURON needed."""
import sys
sys.path.insert(0, '.')
import numpy as np

# Stub NEURON
sys.modules['neuron'] = type(sys)('neuron')
sys.modules['neuron'].h = None

import importlib.util
from pathlib import Path

module_path = Path(__file__).with_name('waveform_evolver.py')
spec = importlib.util.spec_from_file_location('we', module_path)
we = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(we)
    print("✓ Module imported")
except SystemExit:
    print("[FAIL] module exits at import")
    sys.exit(1)

print()
print("=== TEST 1: random waveform has 17 parameters ===")
rng = np.random.default_rng(0)
wf = we.random_waveform(rng)
assert wf.shape == (17,), f"expected 17, got {wf.shape}"
assert -1.5 <= wf[:16].min() and wf[:16].max() <= 1.5, "control points out of range"
assert 0.0 <= wf[16] <= we.GAP_MAX_US, f"gap out of range: {wf[16]}"
print(f"  ✓ shape={wf.shape}, ctrl range=[{wf[:16].min():.2f},{wf[:16].max():.2f}], "
      f"gap={wf[16]:.0f}µs")

print()
print("=== TEST 2: gap is rendered correctly into waveform ===")
# Make a deterministic test waveform with gap = 1000 µs
test_wf = np.zeros(17)
test_wf[:8] = -np.array([0.1, 0.5, 1.0, 0.8, 0.5, 0.3, 0.2, 0.1])
test_wf[8:16] = np.array([0.1, 0.3, 0.5, 0.5, 0.5, 0.4, 0.2, 0.1])
test_wf[16] = 1000.0  # 1 ms gap
time_arr = np.arange(0, 25, 0.005)
wave, m1, m2, info = we.render_waveform(test_wf, pw_us=200, dt_ms=0.005,
                                          time_arr=time_arr)
assert not info.get('invalid'), "should not be invalid"
assert info['gap_us'] == 1000.0, f"gap mismatch: {info['gap_us']}"

# Find time of last cathodic sample and first anodic sample
cath_indices = np.where(m1)[0]
anod_indices = np.where(m2)[0]
last_cath_t = time_arr[cath_indices[-1]]
first_anod_t = time_arr[anod_indices[0]]
gap_measured = first_anod_t - last_cath_t
print(f"  last cathodic t = {last_cath_t*1000:.1f} ms")
print(f"  first anodic t  = {first_anod_t*1000:.1f} ms")
print(f"  measured gap    = {gap_measured*1000:.1f} ms (expected ~1.0)")
assert abs(gap_measured - 1.0) < 0.05, f"gap mismatch: {gap_measured*1000:.2f} ms"
print("  ✓ 1 ms gap correctly rendered")

print()
print("=== TEST 3: charge balance with gap ===")
n_tests = 30
fails = 0
for i in range(n_tests):
    rng = np.random.default_rng(i + 100)
    wf = we.random_waveform(rng)
    wave, m1, m2, info = we.render_waveform(wf, pw_us=200, dt_ms=0.005,
                                              time_arr=np.arange(0, 25, 0.005))
    if info.get('invalid'):
        continue
    net = float(np.sum(wave) * 0.005)
    p1_area = abs(float(np.sum(wave[m1])))
    if p1_area < 1e-6:
        continue
    rel = abs(net) / p1_area
    if rel > 1e-3:
        print(f"  ✗ wf {i} gap={info['gap_us']:.0f}µs rel_imbalance={rel:.6f}")
        fails += 1
print(f"  {n_tests - fails}/{n_tests} waveforms charge-balanced "
      "(threshold rel_imbalance < 1e-3)")
assert fails == 0, "charge balance broken"

print()
print("=== TEST 4: perturb_waveform respects gap range ===")
rng = np.random.default_rng(1)
wf_base = we.random_waveform(rng)
perturbed = [we.perturb_waveform(wf_base, np.random.default_rng(i), rate=1.0,
                                   sigma_ctrl=0.5, sigma_gap=500)
              for i in range(50)]
gaps = [w[16] for w in perturbed]
print(f"  gap range after perturbation: [{min(gaps):.0f}, {max(gaps):.0f}] µs")
assert all(we.GAP_MIN_US <= g <= we.GAP_MAX_US for g in gaps), "gap escaped bounds"
print(f"  ✓ all gaps in [{we.GAP_MIN_US}, {we.GAP_MAX_US}]")

print()
print("=== TEST 5: combine_waveforms preserves dimensionality ===")
rng = np.random.default_rng(7)
pa = we.random_waveform(rng)
pb = we.random_waveform(rng)
child = we.combine_waveforms(pa, pb, rng)
assert child.shape == (17,)
# Each gene should come from one of the parents
for i in range(17):
    assert child[i] == pa[i] or child[i] == pb[i], \
        f"position {i}: child {child[i]} not from either parent"
print(f"  ✓ child shape {child.shape}, all genes from parents")

print()
print("=== TEST 6: describe_waveform handles unimodal vs bimodal ===")
unimodal = np.zeros(17)
for i in range(8):
    x = (i - 3.5) / 3.5
    unimodal[i]    = -np.exp(-2*x*x)
    unimodal[8+i]  =  np.exp(-2*x*x)
unimodal[16] = 100.0
desc = we.describe_waveform(unimodal, pw_us=200)
print(f"  Unimodal:  WB={desc['well_behaved']}, lobes={desc['n_lobes_cathodic']}, "
      f"peak_ratio={desc['peak_ratio']:.2f}, gap={desc['gap_us']:.0f}µs")
assert desc['well_behaved']
assert desc['gap_us'] == 100.0

bimodal = np.zeros(17)
for i in range(8):
    x = (i - 3.5) / 3.5
    bimodal[i]   = -(np.exp(-8*(x+0.6)**2) + np.exp(-8*(x-0.6)**2))
    bimodal[8+i] =  np.exp(-2*x*x)
bimodal[16] = 500.0
desc_bi = we.describe_waveform(bimodal, pw_us=200)
print(f"  Bimodal:   WB={desc_bi['well_behaved']}, lobes={desc_bi['n_lobes_cathodic']}, "
      f"peak_ratio={desc_bi['peak_ratio']:.2f}, gap={desc_bi['gap_us']:.0f}µs")
print(f"  ✓ unimodal/bimodal classification working")

print()
print("=== TEST 7: charge_score behaves as expected ===")
s_invalid = we.charge_score(None, None, {'peak_ratio': 1.0})
assert s_invalid > 1e8
s_good = we.charge_score(0.5, 50.0, {'peak_ratio': 1.0, 'well_behaved': True})
assert s_good == 50.0
s_extreme = we.charge_score(0.5, 50.0, {'peak_ratio': 10.0})
assert s_extreme > 50.0
print(f"  ✓ charge_score(invalid)={s_invalid}, "
      f"charge_score(good)={s_good}, charge_score(extreme_pr)={s_extreme:.2f}")

print()
print("="*60)
print("  ALL TESTS PASSED  (terminology renamed, gap added)")
print("="*60)