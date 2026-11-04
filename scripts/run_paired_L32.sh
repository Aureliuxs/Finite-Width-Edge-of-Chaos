#!/bin/sh
# Extra paired networks for the Crinit / mean-field comparison at N = 64, L = 32 (seeds 100-999, same learning rate
# as the main run, multiplier 1.0).  Results: data/mnist/big_paired.pkl, keys (rule + "#s<seed0>", 32, 64).
cd "$(dirname "$0")"
for s in 100 200 300 400 500 600 700 800 900; do
  for r in tanh_crinit:1 tanh_eoc:1; do
    python3 run_mnist_big.py --N 64 --grid 32 --threads 2 --only $r --lr_mult 1.0 --seed0 $s --seeds 100 --tag "#s$s" --out big_paired.pkl
  done
done
