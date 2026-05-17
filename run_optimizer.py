#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PULSE OPTIMIZER RUNNER
======================

Helper script to launch, monitor, and manage pulse_optimizer_weekly.py
with proper environment setup and result aggregation.

USAGE
-----
  python run_optimizer.py --quick            # 10 min test
  python run_optimizer.py --weekend          # 72-hour weekend run
  python run_optimizer.py --overnight 12     # 12-hour run
  python run_optimizer.py --freq 1k,10k,100k --hours 24
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path
from datetime import datetime


def ensure_neuron_compiled():
    """Check that AXNODE.mod is compiled."""
    this_dir = Path(__file__).parent
    
    # Check for compiled mechanisms (arm64 or x86_64)
    import platform
    arch = platform.machine()
    mech_dir = this_dir / arch / f"nrnmech.dll.so"
    if mech_dir.exists():
        print(f"✓ Mechanisms compiled for {arch}")
        return True
    
    # Try to compile
    print("⚠  Mechanisms not found. Attempting compilation...")
    try:
        result = subprocess.run(
            ["nrnivmodl"],
            cwd=str(this_dir),
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            print("✓ nrnivmodl completed successfully")
            return True
        else:
            print(f"[WARNING] nrnivmodl failed:\n{result.stderr}")
            print("Continuing anyway (may fail at runtime)...")
            return False
    except FileNotFoundError:
        print("[WARNING] nrnivmodl not found. Install NEURON.")
        return False
    except subprocess.TimeoutExpired:
        print("[WARNING] nrnivmodl timeout. Try manual compilation.")
        return False


def check_mrgaxon_files():
    """Verify MRGaxon.hoc and AXNODE.mod exist."""
    this_dir = Path(__file__).parent
    files = ['MRGaxon.hoc', 'AXNODE.mod']
    
    all_exist = True
    for fname in files:
        if (this_dir / fname).exists():
            print(f"✓ {fname} found")
        else:
            print(f"✗ {fname} NOT FOUND")
            all_exist = False
    
    return all_exist


def run_optimizer(args):
    """Execute pulse_optimizer_weekly.py with given arguments."""
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "pulse_optimizer_weekly.py"),
    ]
    
    if args.quick:
        cmd.append("--quick")
    else:
        if args.freq:
            cmd.extend(["--freq", args.freq])
        if args.hours:
            cmd.extend(["--hours", str(args.hours)])
        if args.pool:
            cmd.extend(["--pool", str(args.pool)])
    
    if args.out:
        cmd.extend(["--out", args.out])
    
    print(f"\n{'='*70}")
    print(f"Launching optimizer:")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Time:    {datetime.now().isoformat()}")
    print(f"{'='*70}\n")
    
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\n[Interrupted by user]")
        return 1


def aggregate_results(output_dir):
    """Summarize results from output directory."""
    output_dir = Path(output_dir)
    
    results_file = output_dir / "optimization_results.json"
    if not results_file.exists():
        print(f"[WARNING] No results file at {results_file}")
        return
    
    with open(results_file) as f:
        results = json.load(f)
    
    print(f"\n{'='*70}")
    print(f"OPTIMIZATION SUMMARY")
    print(f"{'='*70}")
    print(f"Timestamp:      {results.get('timestamp', 'N/A')}")
    print(f"Total Gens:     {results.get('total_generations', 'N/A')}")
    print(f"Runtime (h):    {results.get('runtime_hours', 'N/A')}")
    print(f"Frequencies:    {', '.join(map(str, results.get('frequencies', [])))}")
    print()
    
    best_by_freq = results.get('best_by_frequency', {})
    if best_by_freq:
        print("Best Fitness per Frequency:")
        print("-" * 70)
        for freq_str in sorted(best_by_freq.keys(), key=lambda x: int(x)):
            data = best_by_freq[freq_str]
            print(f"  {freq_str:>8} Hz  →  Fitness: {data['fitness']:.6f}  "
                  f"(Gen {data['generation']})")
    print(f"{'='*70}\n")
    
    print(f"Full results: {results_file}")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    p.add_argument('--quick', action='store_true',
                   help='quick test (10 min, 3 frequencies)')
    p.add_argument('--weekend', action='store_true',
                   help='full weekend run (72 hours, all frequencies)')
    p.add_argument('--overnight', type=float, metavar='HOURS',
                   help='overnight run (default: 8 hours)')
    p.add_argument('--freq', default=None,
                   help='frequency preset or comma-separated list')
    p.add_argument('--hours', type=float, default=None,
                   help='runtime in hours')
    p.add_argument('--pool', type=int, default=30,
                   help='waveform pool size')
    p.add_argument('--out', default='outputs_weekly',
                   help='output directory')
    p.add_argument('--no-compile', action='store_true',
                   help='skip NEURON compilation check')
    p.add_argument('--summary-only', action='store_true',
                   help='just print summary, do not run')
    
    return p.parse_args()


def main():
    args = parse_args()
    
    # Infer runtime from preset
    if args.quick:
        pass  # --quick handled by optimizer
    elif args.weekend:
        args.hours = 72
        args.freq = 'all'
    elif args.overnight:
        args.hours = args.overnight
        if not args.freq:
            args.freq = 'all'
    
    # Summary-only mode
    if args.summary_only:
        aggregate_results(args.out)
        return 0
    
    # Pre-flight checks
    print("Pre-flight checks...")
    if not check_mrgaxon_files():
        print("\n[ERROR] Critical MRG files missing. Aborting.")
        return 1
    
    if not args.no_compile:
        ensure_neuron_compiled()
    
    # Run optimizer
    return_code = run_optimizer(args)
    
    # Post-run summary
    if return_code == 0:
        aggregate_results(args.out)
    
    return return_code


if __name__ == '__main__':
    sys.exit(main())
