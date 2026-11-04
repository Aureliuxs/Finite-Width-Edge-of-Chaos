#!/bin/sh
# Copy the figures used by the paper and the SM from code/figures into latex/figures (run after any make_fig*.py).
cd "$(dirname "$0")/.."
for f in fig1 fig2 fig3 fig4 fig5 figS_backward figS_curves figS_erf figS_finiteN figS_fmnist figS_gamma figS_mnist figS_signal figS_spread; do
  cp figures/$f.pdf ../latex/figures/$f.pdf
done
echo "figures synced"
