# Datasheet for the BBO Capstone Query History

## Document control

- **Dataset name:** BBO Capstone Query History
- **Version:** 1.4
- **Status date:** 29 October 2026
- **Maintainer:** Jane Kao
- **Scope:** Initial observations and iterative queries for eight synthetic black-box functions

## Motivation

This dataset was created to support a constrained black-box optimisation challenge. The analytical forms, gradients and true optima of eight functions are unknown. The dataset records the evidence available to the optimiser: input vectors proposed for evaluation and the scalar outputs returned by the competition portal.

The primary task is sequential decision-making: use a limited evaluation budget to select the next input for each function while balancing exploration of uncertain regions against exploitation of promising regions. The dataset also supports analysis of surrogate modelling, acquisition functions, local search, uncertainty, reproducibility and the effects of dimensionality.

## Composition

Each record represents one evaluation attempt for one function and contains:

- a function identifier from 1 to 8;
- a round or source label;
- an input vector of continuous coordinates;
- a scalar output when evaluation is complete;
- optionally, notes describing the strategy or whether the result improved the incumbent.

All functions are treated as maximisation problems. Functions 3 and 6 return negative transformed objectives, so a value closer to zero is better.

### Function-level composition

| Function | Dimensions | Initial evaluated records | Week 1–13 evaluated records | Final evaluated total | Status |
|---|---:|---:|---:|---:|---|
| 1 | 2 | 10 | 13 | 23 | Complete |
| 2 | 2 | 10 | 13 | 23 | Complete |
| 3 | 3 | 15 | 13 | 28 | Complete |
| 4 | 4 | 30 | 13 | 43 | Complete |
| 5 | 4 | 20 | 13 | 33 | Complete |
| 6 | 5 | 20 | 13 | 33 | Complete |
| 7 | 6 | 30 | 13 | 43 | Complete |
| 8 | 8 | 40 | 13 | 53 | Complete |
| **Total** | — | **175** | **104** | **279** | **Complete** |

Initial inputs and outputs are stored as NumPy `.npy` arrays using `float64`. The input shapes are `(n, d)`, where `d` is the function dimension; output shapes are `(n,)`. Weekly history is documented in Markdown. Portal submissions use six-decimal, hyphen-separated coordinates bounded by `0.000000` and `0.999999`.

### Known gaps

- The functions’ analytical definitions, gradients and true global optima are unavailable.
- Observation-noise distributions are not supplied.
- The optimisation phase is complete; no query outputs are pending.
- Later samples are concentrated around strong local regions rather than distributed uniformly.
- The dataset contains no demographic, personal or sensitive attributes.

## Collection process

The initial arrays were supplied by the course. Subsequent inputs were generated over twelve weekly rounds. One query per function was submitted in each round. Returned portal values were recorded without manual alteration.

The collection strategy evolved as evidence accumulated:

1. Gaussian Process surrogates and Upper Confidence Bound initially balanced exploration and exploitation.
2. Expected Improvement, Sobol candidate generation and dimension-specific Matérn kernels provided sensitivity checks.
3. Linear and logistic regression were considered as interpretable local approximations and threshold classifiers.
4. RBF SVMs classified relatively high- and low-performance regions.
5. Small neural-network ensembles and input gradients were used as flexible sensitivity checks, especially in higher dimensions.
6. Later rounds emphasised local response trends, bracketing, interpolation, controlled coordinate changes and transparent decision rules.
7. Round 11 used output-aware local clusters, medoids, interval tightening and ridge continuation to distinguish coherent trends from noisy neighbours.
8. Round 12 used repeated displacement vectors as PCA-style empirical directions, retaining movements associated with improvement and reducing redundant or unstable movement.
9. The final policy used an RL analogy: empirical improvement acted as reward, repeated query directions acted as actions and the finite remaining budget shifted the balance toward exploitation.

Candidate inputs were checked for dimensionality, bounds, six-decimal formatting and duplication. The chronological decision record is maintained in [`outputs/bbo-capstone-living-documentation.md`](../outputs/bbo-capstone-living-documentation.md).

## Preprocessing

Raw returned outputs are retained as the authoritative observations. Modelling-specific transformations included:

- response standardisation within each function;
- a logarithmic response representation for Function 1 because positive values span many orders of magnitude;
- `log(1 + y)` for Function 5 during selected surrogate analyses because of strong right skew;
- binary high/low labels derived from within-function thresholds for SVM or logistic-regression sensitivity analysis.

Transformed values are modelling aids and must not replace raw observations in the historical record. Inputs were not treated as categorical, even when an application description mentioned encoded hyperparameters.

## Intended and inappropriate uses

### Intended uses

- educational study of sequential black-box optimisation;
- comparison of surrogate and acquisition strategies under small evaluation budgets;
- reproduction and critique of the documented query decisions;
- analysis of local improvement, uncertainty and dimensionality.

### Inappropriate uses

- claims that a global optimum has been proved;
- safety-critical, medical, financial or operational decisions;
- causal claims about individual coordinates from correlated query movements;
- benchmarking unrelated optimisers without accounting for the adaptive sampling history;
- interpreting the synthetic application descriptions as real clinical or business data.

## Distribution and maintenance

The working copy is maintained in this project repository. The initial `.npy` files were supplied through the course and should be redistributed only if the course terms permit it. This repository adds derived query history, documentation and analysis; no broader licence is asserted for course-provided materials. Users should cite the course context and this datasheet when reusing the derived records.

Jane Kao maintains the dataset. New returned outputs should be appended rather than overwriting earlier records. Every update should include the date, submitted vector, returned output, improvement status and any modelling or preprocessing changes. Pending queries must remain visibly marked until an actual portal result is received.

