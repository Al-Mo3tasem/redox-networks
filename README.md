# Redox Networks

A neural network whose neurons exchange a **conserved quantity** and **settle to an
equilibrium**, instead of computing an output in one forward pass. The idea came from
thinking about reduction–oxidation (redox) chemistry — atoms donating and accepting
electrons until a reaction reaches a stable state — and asking whether a network could work
the same way.

This repository holds the model, the experiments, the figures, and the paper draft.

## The short version

I set out to invent a new model architecture from a physics idea and test it on **tabular
data** — the one common setting where neural networks still lose to gradient-boosted trees
(XGBoost). The result is honest and, I think, interesting:

- **What works:** with a hard, learned gate on each input feature, the Redox Network becomes
  **robust to useless (uninformative) features** — the headline strength of decision trees.
  When I add 100 columns of pure noise, my model loses **less than half a point** of
  accuracy, the **smallest drop of every model I tested** (including XGBoost, LightGBM, MLP,
  ResNet, FT-Transformer, and TabNet). At that noise level it is the **most accurate neural
  network** of the group.
- **The honest limit:** it does **not** beat trees on clean accuracy (it trails by ~1–2
  points), and I found out exactly why. Settling to an equilibrium is, mathematically, a
  **smoothing** operation, and tabular targets are jagged — so the core mechanism works
  against the grain of the data. I show this with both theory and a direct experiment.

So the contribution is a novel architecture with a real, distinctive property (junk-feature
robustness, uncommon among neural nets), plus a clear explanation of where this kind of
mechanism helps and where it cannot.

## How it works (one paragraph)

Each neuron holds an amount of "charge." A fixed total is poured in and never changes — only
moves between neurons (a conservation law). Neurons have a learned pull, and charge flows
along learned connections from high to low potential until it balances. A gentle "washboard"
makes charge prefer discrete levels (sharp, switch-like decisions), and a hard on/off gate
lets the network ignore a feature completely. The prediction is read off the settled state.
Full details and the energy function are in the paper.

## Repository layout

```
paper/        the paper draft (method, experiments, results, limitations)
requirements.txt
experiments/v0/
  model.py            base Redox Network (conserved settling + washboard)
  model_v1..v6.py     variants (dynamic matching, feature gating, conditional gating, forest front-end)
  baselines_nn.py     tabular ResNet and FT-Transformer baselines
  datasets.py         dataset loading (from OpenML)
  eval.py             training + evaluation harness
  robustness.py       the junk-feature experiment
  compare.py          main comparison: Redox vs trees and other neural nets
  make_figures*.py    figure generation
  figures/            figures used in the paper
```

## Reproduce

```bash
pip install -r requirements.txt
cd experiments/v0
python compare.py        # main table: Redox vs trees and neural nets, clean + junk
python robustness.py     # robustness curves
python make_figures.py   # figures
```
Datasets are pulled from OpenML on first run and cached locally.

## Paper

See [`paper/redox-networks-draft.md`](paper/redox-networks-draft.md). A LaTeX/arXiv version
is in preparation.

## Author

ALMoatsim — Independent Researcher.
ORCID: [0009-0002-6350-1534](https://orcid.org/0009-0002-6350-1534).

If you use this work, please cite the paper (citation will be added on arXiv release).
