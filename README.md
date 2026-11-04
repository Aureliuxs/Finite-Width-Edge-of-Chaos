# Finite-width shift of the edge of chaos in deep neural networks: code

Code, data and figure scripts for the paper and its Supplemental Material.

## Layout

```
eocfw/            the package
  gauss.py        Gaussian averages of activations (Gauss-Hermite), Hermite coefficients
  meanfield.py    variance map, edge of chaos, correlation map, compression, shaping rules
  finite_width.py sigma_c^2(N) of the signal size; drift a_1 and diffusion b_1 of the distance walk;
                  finite-width edge of chaos (Gaussian and orthogonal weights)
  simulate.py     exact two-input chain, explicit-matrix walks, avalanches
  info.py         information about input similarity transmitted through a finite network
  mnist.py        MNIST data, initialisation rules (the Crinit rules call the crinit package), training loop
  mnist_batched.py many independent networks trained at once (the hundred-network comparison)
  tuned.py        Fashion-MNIST data; batches whose members differ in initialisation and learning rate (SM S9)
  style.py        figure style
scripts/
  run_fig2_data.py        Fig. 2: a_1, b_1, the shift and the avalanches at N = 16, 64, 256 (and 1024)
                                                                                  -> data/distance_walk_v2.json, eoc_shift_v2.json, survival_v2.npz
  run_fig3_data.py        Fig. 3: mean vs typical decorrelation and typical signal at N = 16, 64
                                                                                  -> data/two_input_chain_v2.npz, signal_walk_v2.npz
  run_distance_walk.py    a_1, b_1 for tanh and erf at N = 200, 800 and orthogonal tanh (Table S1)  -> data/distance_walk.json
  run_eoc_shift.py        the earlier shift measurement at N = 100-800            -> data/eoc_shift.json, survival_N400.npz
  run_signal_walk.py      typical signal of one input, tanh/ReLU/orthogonal (Fig. S4) -> data/signal_walk.npz
  run_two_input_chain.py  the earlier two-input runs at N = 100, 400              -> data/two_input_chain.npz
  run_info.py             information measurement, tanh and erf (hours)           -> data/info/*.npy
  run_mnist_big.py        the hundred-network comparison of Table I and Fig. 5     -> data/mnist/big.pkl
                          (--N 32 for the narrower networks of SM Table S7         -> data/mnist/big_n32.pkl)
  run_pairs_one_network.py many pairs of inputs through one network (SM S3.7)      -> data/pairs_one_network.json
  backward_crossentropy.py gradient walk with the cross-entropy loss on MNIST (SM S5.5) -> data/backward/crossentropy.json
  run_paired_L32.sh       900 more paired networks each for Crinit and the mean-field edge of chaos at N = 64, L = 32
                          (seeds 100-999; Table I and SM S8.2)                    -> data/mnist/big_paired.pkl
  sync_figures.sh         copies the figures used by the paper from figures/ into ../latex/figures/ (run after make_fig*.py)
  run_mnist.py            the smaller MNIST runs: jobs main, gamma, finiteN (SM)   -> data/mnist/*.pkl
  run_crinit_orders.py    first-order vs all-orders Crinit point (Table S2)       -> data/crinit_orders.json
  run_fmnist_tuned.py     Fashion-MNIST: learning rate, gain (standard rules) and compression (Crinit) tuned on a
                          validation set, with a confirmation step (SM S9) -> data/fmnist/tuned_L*.pkl
  run_fmnist_stable.py    the two leaky-ReLU 10-epoch runs at L = 96 repeated at half the rate (earlier version of S9)
  run_fmnist_firstlayer.py  first-layer signal swapped between Crinit and the tuned gain (input scaling) -> data/fmnist/firstlayer_L*.pkl
  run_fmnist_first_small.py the first-layer test of SM Table S9: weight vs input scaling, first-layer learning rate -> data/fmnist/first_small_L*.pkl
  backward_walk.py        forward and backward walks through the same explicit matrices (SM S5) -> data/backward/walk_*.pkl
  summarise_backward.py   compact summaries of those runs (shipped)                 -> data/backward/summary_*.npz
  make_fig1.py ... make_fig5.py, make_figS.py, make_figS_backward.py, make_tables.py   figures and tables -> figures/, ../latex/
legacy/           the earlier package (qcdl) and the script that drew an earlier figure of the activation classes
notebooks/        mnist_criticality.ipynb: a self-contained, configurable version of the MNIST experiment
data/             all simulation outputs used in the paper (MNIST raw files are downloaded on first use)
```

## Reproducing

```
pip install -r requirements.txt
pip install ../crinit                      # the Crinit package (sibling folder of this one; also found on the path automatically)
python scripts/run_fig2_data.py            # ~45 min on one core
python scripts/run_fig3_data.py            # ~5 min
python scripts/run_distance_walk.py        # ~40 min (Table S1: erf and orthogonal rows)
python scripts/run_eoc_shift.py            # ~30 min (earlier shift measurement)
python scripts/run_signal_walk.py          # ~5 min
python scripts/run_two_input_chain.py      # ~3 min
python scripts/run_mnist_big.py            # ~6 h on two cores (12 rules x 2 depths x (36 pilot + 100) networks)
python scripts/run_mnist_big.py --N 32 --grid 32,96 --only tanh_standard,tanh_finiteN,tanh_eoc:1,tanh_crinit:1,leaky_he,leaky_crinit:1 --out big_n32.pkl   # ~1 h (SM Table S7)
python scripts/run_pairs_one_network.py 1000   # ~2 min (SM S3.7)
python scripts/backward_crossentropy.py 2000   # ~5 min (SM S5.5)
sh scripts/run_paired_L32.sh               # ~70 min on two cores (Table I, L = 32 rows; SM S8.2)
python scripts/run_mnist.py main           # ~2 h on one core (the smaller runs of the SM)
python scripts/run_mnist.py gamma          # ~2 h
python scripts/run_mnist.py finiteN        # ~1 h
python scripts/run_info.py                 # a few hours; the results ship in data/info
python scripts/run_crinit_orders.py        # ~3 min (Table S2)
python scripts/run_fmnist_tuned.py         # ~20 h on two cores (SM S9; resumes where it stopped)
python scripts/run_fmnist_first_small.py 32; python scripts/run_fmnist_first_small.py 96   # ~2 h (SM Table S9)
python scripts/backward_walk.py 16 400 4000 1.7609546 13   # SM S5 (arguments: N L networks sigma_w^2 seed); ~1 h in all
python scripts/backward_walk.py 32 400 4000 1.7609546 14
python scripts/backward_walk.py 64 400 4000 1.7609546 1
python scripts/backward_walk.py 128 400 2000 1.7609546 15
python scripts/backward_walk.py 256 400 1000 1.7609546 3
python scripts/backward_walk.py 64 400 4000 1.8839670 2
python scripts/summarise_backward.py
for f in scripts/make_fig*.py scripts/make_tables.py scripts/make_fmnist.py; do python $f; done
sh scripts/sync_figures.sh                 # copy the figures into ../latex/figures
```

All scripts take the data they need from `data/` and write figures to `figures/` and table fragments to `../latex/`.
Every random number generator is seeded, so the numbers of the paper are reproduced exactly up to BLAS reduction
order. The MNIST runs use PyTorch on the CPU; `--threads` sets the number of threads.

`run_mnist_big.py` trains all networks of one rule at once with batched matrix products (`eocfw/mnist_batched.py`);
the members are exactly independent (own seeds for the weights and the data order, own momentum buffers).
Use `--threads 1` when other jobs share the machine: PyTorch's thread pool slows down badly when oversubscribed.

## Conventions

* `sw2`, `sb2`: the weight and bias variances sigma_w^2 (per fan-in, i.e. W_ij ~ N(0, sigma_w^2/N)) and sigma_b^2.
* `q`: pre-activation variance (signal size); `q*`: its steady value on the edge of chaos.
* `q1`: the signal of the first hidden layer. Rules with `q1 != sw2` (Crinit, the mean-field edge of chaos) draw the
  first-layer weights with variance `(q1 - sb2)/784` (`first_layer="weight"`, the default of both trainers, as the
  crinit package does). The runs stored under names ending in `@in` were made with the earlier implementation, which
  rescaled the input instead (`first_layer="input"`): same forward pass, smaller first-layer gradient, lower accuracy.
* Activations are named "tanh", "erf", "gelu", "relu", "linear", "leaky:alpha".
* `a1_b1(name, q, sw2=None, orthogonal=False)` returns the drift and diffusion coefficients and their parts.
