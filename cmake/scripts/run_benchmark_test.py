#!/usr/bin/env python3
"""
run_benchmark_test.py - CTest wrapper for HeCBench benchmarks

This script runs a benchmark and validates its output against expected patterns.
It's designed to be called by CTest.

Usage:
    run_benchmark_test.py --regex <pattern> [--timeout <secs>] <binary> [args...]

Exit codes:
    0 - Success (output matched regex)
    1 - Failure (output didn't match regex or benchmark crashed)
    2 - Timeout
"""

import argparse
import re
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description='Run HeCBench benchmark test')
    parser.add_argument('--regex', required=True, help='Regex pattern to match in output')
    parser.add_argument('--timeout', type=int, default=300, help='Timeout in seconds')
    parser.add_argument('--working-dir', help='Working directory for the benchmark')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('cmd', nargs=argparse.REMAINDER, help='Benchmark binary followed by its arguments')

    args = parser.parse_args()

    cmd = args.cmd

    if args.verbose:
        print(f"Running: {' '.join(cmd)}")
        print(f"Regex: {args.regex}")
        print(f"Timeout: {args.timeout}s")

    try:
        start = time.time()
        result = subprocess.run(
            cmd,
            cwd=args.working_dir,
            capture_output=True,
            text=True,
            timeout=args.timeout,
        )
        elapsed = time.time() - start

        # Combine stdout and stderr
        output = result.stdout + result.stderr

        if args.verbose:
            print(f"\nOutput:\n{output[:2000]}")
            if len(output) > 2000:
                print(f"... (truncated, total {len(output)} chars)")
            print(f"\nElapsed: {elapsed:.2f}s")
            print(f"Return code: {result.returncode}")

        # Check return code
        if result.returncode != 0:
            print(f"FAIL: Benchmark exited with code {result.returncode}")
            return 1

        # Check regex match
        matches = re.findall(args.regex, output)
        if matches:
            # Extract numeric value if present
            try:
                value = float(matches[0])
                print(f"PASS: Matched value = {value}")
            except (ValueError, TypeError):
                print(f"PASS: Matched pattern")
            return 0
        else:
            print(f"FAIL: No match for regex pattern")
            print(f"Pattern: {args.regex}")
            print(f"Output (first 500 chars): {output[:500]}")
            return 1

    except subprocess.TimeoutExpired:
        print(f"FAIL: Timeout after {args.timeout}s")
        return 2

    except Exception as e:
        print(f"FAIL: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
