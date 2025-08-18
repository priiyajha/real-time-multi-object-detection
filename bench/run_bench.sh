#!/bin/bash

# This script runs a benchmark for a specified duration and mode.
# Usage: ./run_bench.sh --duration 30 --mode wasm

DURATION=30
MODE="wasm"

# Parse command line arguments
for i in "$@"; do
  case $i in
    --duration=*)
      DURATION="${i#*=}"
      shift
      ;;
    --mode=*)
      MODE="${i#*=}"
      shift
      ;;
    *)
      ;;
  esac
done

echo "Running benchmark for $DURATION seconds in $MODE mode..."

# Set up the environment for the backend to run in the correct mode
export MODE=$MODE

# Run the Python bench script which simulates a client connection
# This script will connect to the frontend server, simulate a stream,
# and collect metrics.
python3 bench/run_bench.py --duration $DURATION --mode $MODE

echo "Benchmark complete. Results saved to metrics.json."
