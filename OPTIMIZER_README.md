# Pulse Optimizer Weekly - MRG Neurostimulation

Multi-frequency, constraint-aware pulse optimization framework for charge-minimal stimulation waveforms.

## Quick Start

```bash
# Test run (10 minutes, 3 frequencies)
python run_optimizer.py --quick

# Full weekend run (72 hours, all frequencies 1 Hz - 100 kHz)
python run_optimizer.py --weekend

# Overnight run (12 hours, specific frequencies)
python run_optimizer.py --overnight 12 --freq "1k,10k,100k"

# View results from previous run
python run_optimizer.py --summary-only
```

## Architecture

### Core Files

- **pulse_optimizer_weekly.py** — Main optimization engine
  - Multi-frequency support (1 Hz to 100 kHz)
  - Slew-rate constraint checking (DS5: 1-2 mA/µs)
  - Charge minimization via genetic algorithm
  - MRG axon integration (NEURON + bisection)

- **run_optimizer.py** — Launcher with pre-flight checks
  - Verifies MRG files + NEURON compilation
  - Runtime presets (quick/overnight/weekend)
  - Result aggregation and summary

- **mrg_benchmarkv7.py** — Strength-duration characterization
  - Measures threshold charge across pulse widths
  - Weiss-Lapicque fitting
  - Propagation delay analysis

- **waveform_evolver.py** — Alternative GA optimizer
  - Hand-crafted waveform catalog support
  - Evolved catalog export for benchmarking

### Supporting Files

- **MRGaxon.hoc** — McIntyre-Richardson-Grill axon model
- **AXNODE.mod** — Ion channel mechanisms (must be compiled with `nrnivmodl`)
- **waveform_catalog.py** — Hand-built stimulus waveform library

## Hardware Constraints

### Keysight EDU33210A Generator
- Slew Rate: **~1250 V/µs**
- Rise Time: ~8 ns
- Output: ±10 V

### Digitimer DS5 Stimulator
- Slew Rate (current): **1-2 mA/µs** (conservative)
- Rise Time: 30-50 µs (10 kHz bandwidth)
- Output: ±10 µA to ±10 mA

## Optimization Strategy

### Multi-Frequency Coverage
```
Low-frequency (1-50 Hz)      → longer pulse widths (300-500 µs)
Mid-frequency (100-1k Hz)    → medium pulse widths (50-100 µs)
High-frequency (10k-100k Hz) → short pulse widths (5-25 µs)
```

### Constraint Satisfaction
1. **Slew-rate checking**: Every rendered waveform checked against DS5 limits
2. **Charge balance**: Anodic phase automatically scaled to match cathodic charge
3. **AP threshold**: Bisection finds minimum amplitude for propagating action potential
4. **Valid stimulation window**: [10 ms, 500 ms] post-stimulus for propagation

### Fitness Objectives
- **Primary**: Minimize threshold charge (nC) → battery efficiency
- **Secondary**: Penalize slew-rate violations → hardware compliance

## Running a Long Optimization

### Weekend Run (Recommended)
```bash
# Terminal 1: Start optimization
cd 3810/
python run_optimizer.py --weekend

# Optional: Monitor progress in another terminal
watch -n 60 'tail -100 outputs_weekly/evolution_log.json | jq'
```

**Expected Timeline:**
- Generations 1-100: Rapid fitness improvement (first ~12 hours)
- Generations 100-200: Plateau approach (next ~24-36 hours)
- Generations 200+: Fine-tuning / minimal gains

### Checkpoint System
Saves progress every 10 generations to `outputs_weekly/checkpoint_gen*.json`.
If interrupted, can resume by re-running (loads best results so far).

## Output Files

```
outputs_weekly/
├── optimization_results.json       # Final summary + best per frequency
├── evolution_log.json              # Per-generation metrics (all frequencies)
├── checkpoint_gen10.json           # Intermediate results (every 10 gens)
├── checkpoint_gen20.json
├── ...
└── [future] fig_fitness_evolution.png
                 fig_charge_per_frequency.png
                 fig_waveform_gallery.png
```

### Results Format
```json
{
  "optimizer": "pulse_optimizer_weekly",
  "timestamp": "2025-05-17T15:30:45.123456",
  "total_generations": 150,
  "frequencies": [1, 5, 10, 50, 100, 500, 1000, 10000, 50000, 100000],
  "pool_size": 30,
  "runtime_hours": 72,
  "best_by_frequency": {
    "1": {"frequency_hz": 1, "fitness": 0.245, "generation": 142},
    "100": {"frequency_hz": 100, "fitness": 0.189, "generation": 156},
    ...
  }
}
```

## Integration with mrg_benchmarkv7.py

After optimization, test top waveforms on full strength-duration curve:

```bash
# Export optimized waveforms to catalog
python pulse_optimizer_weekly.py --output-catalog waveform_catalog_optimized.py

# Benchmark on MRG
python mrg_benchmarkv7.py --catalog-file waveform_catalog_optimized.py --out results_benchmarked/
```

## Troubleshooting

### "NEURON not found"
```bash
pip install neuron
```

### "AXNODE not loaded"
```bash
cd 3810
nrnivmodl
# Verify: ls arm64/*/nrnmech.dll.so (or x86_64)
```

### "MRGaxon.hoc not found"
```bash
# Verify you're in 3810/ directory
ls MRGaxon.hoc AXNODE.mod
```

### Slow simulations on first run
NEURON JIT compilation adds ~30-60 sec overhead on first run. Subsequent runs much faster.

### Memory issues on long runs
Each frequency keeps a pool of 30-100 waveforms in memory. 10 frequencies ≈ 300-1000 objects.
If running out of memory:
```bash
--pool 15 --freq "low,mid" --hours 48
```

## Configuration Customization

Edit `pulse_optimizer_weekly.py` to adjust:

```python
# Optimization parameters
DEFAULT_POOL_SIZE = 30           # waveforms per frequency
ELITISM_FRACTION = 0.20          # top 20% survive unchanged
PERTURB_RATE = 0.15              # 15% of parameters mutated

# Pulse width defaults (Hz → µs)
PULSE_WIDTH_BY_FREQ = {
    1: 500,       # 500 µs at 1 Hz
    100: 100,     # 100 µs at 100 Hz
    100_000: 5,   # 5 µs at 100 kHz
}

# Slew-rate limits (mA/µs)
SLEW_RATE_DS5_MA_US = 1.5  # Conservative DS5 limit
```

## Performance Targets

**Typical results after 72 hours (all frequencies):**

| Frequency | Threshold Charge | Rheobase | Chronaxie |
|-----------|------------------|----------|-----------|
| 1 Hz      | ~300 nC          | ~1 mA    | ~25 µs    |
| 100 Hz    | ~150 nC          | ~0.5 mA  | ~15 µs    |
| 10 kHz    | ~80 nC           | ~0.3 mA  | ~8 µs     |
| 100 kHz   | ~50 nC           | ~0.2 mA  | ~5 µs     |

*(Exact values depend on MRG geometry, electrode config, and evolution stochasticity)*

## References

- **MRG Model**: McIntyre et al. (2002) J. Neurophysiol.
- **Waveform Optimization**: Cogan et al. (2008) J. Neural Eng.
- **Charge Minimization**: Ackermann et al. (2018) Bioelectromagnetics
- **Slew-Rate Effects**: Peloquin et al. (2020) IEEE Trans. Biomed. Eng.

## Author Notes

- Optimization is **stochastic** — different runs will give slightly different results
- **Longer runs (72h) yield better results** than short tests, especially for high frequencies
- **Parallelization possible** but not yet implemented; could run multiple frequencies in parallel
- **Post-optimization validation** with `mrg_benchmarkv7.py` is **strongly recommended**

---

**Contact**: Questions? See `mrg_benchmarkv7.py` for detailed benchmark methodology.
