# Model Card: Adaptive BBO Surrogate Ensemble

## Overview

- **Approach name:** Adaptive BBO Surrogate Ensemble
- **Type:** Sequential, derivative-free black-box optimisation workflow
- **Version:** 1.3
- **Status date:** 29 October 2026
- **Owner:** Jane Kao
- **Functions supported:** Eight continuous synthetic objectives with 2–8 dimensions

This is an optimisation approach rather than one fixed predictive model. It combines probabilistic surrogate modelling, acquisition functions, classification-based sensitivity checks and transparent local heuristics. A separate model and decision policy are maintained for each function.

## Intended use

The approach is intended for educational, low-budget black-box maximisation where function values are expensive, gradients are unavailable and inputs lie in a bounded continuous space. It is especially useful for demonstrating how a search policy should evolve as observations accumulate.

It should not be used unchanged for safety-critical optimisation, high-noise production systems, categorical or conditional search spaces, or settings requiring guarantees of global optimality. It is also unsuitable for causal interpretation of feature effects.

## Approach details and evolution

### Rounds 1–2: probabilistic global search

Separate Gaussian Processes were fitted to each function. A Matérn kernel, response standardisation and an explicit noise term represented smoothness and uncertainty. Upper Confidence Bound selected points with either high predicted value or high uncertainty. Expected Improvement and scrambled Sobol candidates were added as checks. Exploration levels became function-specific after the first returned results.

### Round 3: SVM perspective

Continuous outputs were thresholded into relative high- and low-performance classes. Soft-margin RBF SVMs were used as sensitivity tools for nonlinear decision regions. Because labels depended on an arbitrary threshold and the positive class was small, SVMs informed but did not determine the final queries.

### Rounds 4–6: flexible surrogates and local sensitivity

Small neural-network ensembles were tested as nonlinear surrogates. Input gradients suggested locally influential directions, but the limited sample size made neural predictions unstable. Neural estimates were therefore combined with GP uncertainty, empirical local trends and conservative step sizes. These rounds increasingly favoured exploitation where repeated improvement was observed.

### Round 7: hyperparameter tuning

GP length scales, noise and exploration weights were tuned per function. SVM regularisation and kernel scale, plus neural hidden-layer size, weight decay and ensemble size, were treated as sensitivity parameters. Marginal likelihood, bounded search, repeated fits and performance history guided the settings.

### Rounds 8–9: structured evidence and robust local search

The workflow used compact structured histories, deterministic validation and explicit separation between predictions and observations. Functions with steady improvements received small continuation steps. Unstable functions received mirrored or one-variable probes to distinguish competing local explanations.

### Round 10: transparent rules

Every query was tied to a reproducible rule: interpolation, a fixed displacement, a one-dimensional continuation or a three-point local quadratic estimate. The goal was to improve auditability and avoid presenting opaque surrogate outputs as certainty.

### Round 11: cluster-guided refinement

High-output observations were treated as local, output-aware groups. Medoid tightening and bracket bisection were used for noisy or compact peaks, while repeated displacement vectors represented ridge-like clusters. Input distance was combined with output rank so that a nearby poor observation did not automatically pull a centroid away from a strong incumbent.

### Round 12: PCA-inspired directional refinement

Repeated query displacements were treated as empirical principal directions when they explained improvement across several rounds. Stable coordinated directions were retained, boundary-fixed or redundant coordinates were simplified, and unsuccessful directions were reversed or reduced. This was a structural analogy rather than a claim that sparse adaptive samples support a definitive PCA decomposition.

### Final round: RL-informed policy

Each query direction was treated as an action and its returned improvement as a reward. Repeatedly rewarded directions received controlled continuation, unstable functions returned to incumbent neighbourhoods and nearly flat rewards triggered smaller steps. This is a bandit and Q-update analogy rather than a formally trained reinforcement-learning agent.

## Performance

The primary metric is the **best observed output**, because all tasks are maximisation problems. Supporting indicators are improvement over the best initial observation, frequency of incumbent improvement and information gained from unsuccessful probes. The table reports the completed performance through Week 13.

| Function | Best initial output | Final best output | Best input | First achieved |
|---|---:|---:|---|---|
| 1 | `7.710875114502849e-16` | `0.0001362433711915106` | `[0.579980, 0.579580]` | Week 13 |
| 2 | `0.6112052157614438` | `0.6859447566834628` | `[0.681450, 0.999999]` | Week 9 |
| 3 | `-0.034835313350078584` | `-0.0028315397301727427` | `[0.517730, 0.605367, 0.423675]` | Week 6 |
| 4 | `-4.025542281908162` | `0.6789254464625576` | `[0.414080, 0.415890, 0.353110, 0.425590]` | Week 13 |
| 5 | `1088.8596181962705` | `1636.5047451170747` | `[0.000000, 0.300000, 0.999999, 0.999999]` | Week 13 |
| 6 | `-0.7142649478202404` | `-0.3056501399168364` | `[0.511157, 0.303311, 0.672180, 0.874815, 0.000000]` | Week 3 |
| 7 | `1.3649683044991994` | `1.8599092342834425` | `[0.043834, 0.441792, 0.263071, 0.270638, 0.313898, 0.765170]` | Week 13 |
| 8 | `9.598482002566342` | `9.9429688307286` | `[0.000000, 0.092573, 0.100796, 0.139476, 0.999999, 0.509537, 0.243978, 0.333447]` | Week 12 |

These results show improvement over the initial incumbent for every function. They do not establish proximity to the unknown global optima. Cross-function output magnitudes are not comparable, so outputs should never be averaged without normalisation.

## Assumptions

- Nearby inputs tend to have related outputs, at least locally.
- The portal values use a consistent maximisation convention.
- Recent improvement directions may remain useful for another small step.
- GP uncertainty is informative enough to rank candidates despite sparse coverage.
- Each coordinate is continuous within `[0, 1)`, and six-decimal rounding does not materially change the response.
- The evolving data-generating process is sufficiently stable for earlier observations to remain relevant.

## Limitations and failure modes

- The evaluation budget is small relative to dimensionality, especially for Functions 7 and 8.
- Adaptive sampling creates strong exploitation bias and leaves large regions unexplored.
- GP uncertainty depends on kernel and noise assumptions; neural surrogates may overfit.
- Threshold classifiers are sensitive to label definition and class imbalance.
- Correlated coordinate changes obscure individual feature attribution.
- Narrow peaks, discontinuities, heteroscedastic noise or interactions may violate local-smoothness assumptions.
- Boundary-heavy incumbents can cause the search to remain on a face of the hypercube.
- Function 2 demonstrates that nearby evaluations can vary sharply, limiting deterministic local interpretation.
- No known optimum is available, so regret and optimality gaps cannot be calculated.

## Ethical and reproducibility considerations

Transparency reduces the risk of overstating uncertain results. The project records raw observations separately from predictions, documents transformations, preserves unsuccessful evaluations and explains why each later query was chosen. This makes the workflow open to criticism and reduces hindsight bias.

Reproduction requires the initial arrays, chronological query history, software environment, random seeds, preprocessing rules, model hyperparameters, candidate-generation procedure and portal constraints. Any real-world adaptation would also require domain-specific safety limits, representative data, stakeholder review and monitoring for distribution shift. Synthetic success should not be treated as evidence that the method is safe or fair in a human-impacting application.

Additional detail improves this model card when it changes interpretation or reproducibility—for example, exact seeds, library versions and fitted kernels. Repeating every intermediate candidate would add volume without equivalent value; those details belong in machine-readable experiment logs rather than the main model card.

