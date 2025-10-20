# Cirq Quantum Random Number Generator (QRNG) with Benchmarking

This project implements a **Quantum Random Number Generator (QRNG)** using Google's **Cirq** framework.
It compares performance and statistical quality with a classical **NumPy PRNG**.

## Features
- Quantum random bit generation using Cirq (Hadamard + Measurement)
- Benchmark vs. NumPy PRNG throughput (log-log plot)
- Basic quality analysis: frequency, min-entropy, autocorrelation, runs, chi-square
- Simple visualization with Matplotlib

## How to Run
```bash
pip install -r requirements.txt
python qrng_cirq_benchmark.py
```

Or in Google Colab:
```python
!pip install cirq numpy scipy matplotlib
!python qrng_cirq_benchmark.py
```

## Output
- Benchmark printout (Cirq vs NumPy)
- Log-log throughput plot
- Randomness quality summary

