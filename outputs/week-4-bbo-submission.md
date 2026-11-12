# Week 4 BBO Submission and Reflection

## 1. Week 3 results

All functions remain maximisation problems: larger outputs are better.

| Function | Week 3 output | Best before Week 3 | Result |
|---|---:|---:|---|
| 1 | `0.0001360393` | `6.5695e-9` | **New best; major improvement** |
| 2 | `0.4990046490` | `0.6112052158` | No improvement |
| 3 | `-0.0703464385` | `-0.0348353134` | No improvement |
| 4 | `0.4621722931` | `0.6767810480` | Strong, but below the Week 1 best |
| 5 | `1251.2196104545` | `1616.6675878037` | No improvement; lowering input 3 reduced yield |
| 6 | `-0.3056501399` | `-0.5191033545` | **New best** |
| 7 | `1.1777053321` | `1.3649683045` | No improvement |
| 8 | `9.9030156124` | `9.9125524546` | Close, but no improvement |

Week 3 improved Functions 1 and 6. It also clarified useful directions: Function 5 should return to the upper boundary for its third input, while Functions 7 and 8 should move back toward their earlier incumbents.

## 2. Week 4 technical strategy

The initial observations and all three submitted rounds were appended to separate datasets for the eight functions. Updated sample sizes range from 13 observations for Functions 1 and 2 to 43 for Function 8.

The primary model remained a Gaussian Process with a Matérn 5/2 kernel because it supplies both a continuous prediction and uncertainty. The following secondary models were fitted:

- a class-weighted, soft-margin RBF SVM for top-quartile versus lower performance;
- an ensemble of five small neural-network regressors with two hidden layers of 12 and 8 tanh units;
- finite-difference gradients of the neural-network ensemble at the current best input.

Candidate points combined global Sobol coverage with local perturbations around the five strongest observations. The final ranking considered GP Upper Confidence Bound, SVM decision score, mean neural-network prediction and disagreement within the neural-network ensemble. Neural-network dispersion was treated as instability rather than calibrated uncertainty.

Because the datasets are still small, gradients were used as sensitivity checks rather than direct instructions. Empirical evidence took priority where a gradient conflicted with recent observed results.

## 3. Final Week 4 queries

### Copy-ready portal entries

```text
Function 1: 0.590000-0.590000
Function 2: 0.678794-0.999999
Function 3: 0.494275-0.608066-0.369439
Function 4: 0.425000-0.430000-0.345000-0.424000
Function 5: 0.000000-0.025000-0.999999-0.999999
Function 6: 0.518634-0.292905-0.651733-0.857434-0.000000
Function 7: 0.011608-0.489865-0.261473-0.210305-0.412571-0.748090
Function 8: 0.035662-0.052610-0.145894-0.005116-0.999999-0.491591-0.233059-0.260954
```

### Rationale

| Function | Week 4 emphasis | Reason for selection |
|---|---|---|
| 1 | Careful exploitation | Continues a short distance along the diagonal from the rapidly improving Week 2–3 observations; avoids the GP's uncertainty-heavy jump |
| 2 | Balanced | GP/UCB selected the upper second-coordinate boundary near the best first-coordinate region; predicted mean about `0.596` |
| 3 | Conservative local refinement | Remains close to the initial incumbent; the neural model identified inputs 3 and 1 as influential, but evidence was insufficient for a large gradient step |
| 4 | Empirical plus gradient refinement | Returns toward the Week 1 best and follows the neural gradient by slightly increasing inputs 1–2 and reducing input 3 |
| 5 | Strong exploitation | Restores input 3 to `0.999999` after lowering it to `0.950000` reduced output by approximately 365; tests the midpoint of the two successful input-2 settings |
| 6 | Local refinement | Continues the Week 2–3 empirical direction with positive SVM support and low neural-ensemble disagreement |
| 7 | Return to incumbent region | GP, SVM and neural models agree near the initial best; the proposal has GP mean about `1.374` |
| 8 | Local refinement | Returns close to the Week 2 best; GP mean about `9.920`, with positive SVM support and low neural disagreement |

Predictions are surrogate estimates, not hidden-function outputs.

## 4. Discussion-board reflection

### Which observations acted like support vectors?

Several recent pairs behaved like points near a rapid transition. For Function 1, moving from `[0.553410, 0.547574]` to `[0.579688, 0.579836]` increased the output from `6.57e-9` to `1.36e-4`, suggesting a narrow high-performance region. For Function 5, reducing input 3 from `0.999999` to `0.950000` lowered the output from `1616.67` to `1251.22`. The Week 1–3 points for Function 4 and the Week 2–3 points for Function 7 also sit near apparent transitions between excellent and merely strong performance. Treating these as support-vector-like observations encouraged smaller queries across the implied boundary instead of distant exploration.

### Neural-network gradients and influential inputs

I trained a small ensemble of neural-network regressors and estimated local gradients by finite differences, which plays the same interpretive role as backpropagating output sensitivity to the inputs. Since every competition objective is maximised, I considered gradient-ascent directions; reversing the gradient would reduce the output. The largest local sensitivities were input 1 for Functions 1–2, inputs 3 and 1 for Function 3, inputs 2 and 1 for Function 4, inputs 4 and 3 for Function 5, inputs 5 and 4 for Function 6, inputs 1, 4 and 5 for Function 7, and inputs 3, 1 and 7 for Function 8. I used these rankings to prioritise small perturbations, but not as proof of causal importance.

### Classification framing and decision boundaries

I defined “good” as the top quartile of observed outputs. Logistic regression would provide interpretable linear coefficients but would miss curved or disconnected regions. A soft-margin RBF SVM can represent a nonlinear boundary and tolerate overlap, while a neural network can learn even more flexible boundaries. The trade-off is that false negatives may prevent exploration of a valuable region, whereas false positives waste expensive queries. Changing the threshold also changes the boundary. My regression neural network could be thresholded into a classifier, but the small positive class meant its apparent boundary was not reliable enough to use alone.

### Interpretability versus flexibility

At the current sample sizes, the SVM felt more appropriate than a neural network as a classification guide because regularisation and the soft margin offer useful control with limited data. The neural network was more expressive for Functions 7 and 8, but its gradients varied across random initialisations and its predictions were not calibrated uncertainties. Linear models were easiest to interpret but too restrictive for these nonlinear surfaces. I therefore retained the Gaussian Process as the main optimiser, used the SVM as a boundary check and used the neural network for secondary pattern and gradient evidence.

### Was neural-network flexibility worth it?

The neural ensemble captured nonlinear interactions that linear or logistic regression could not, especially in the higher-dimensional functions. Backpropagation-style sensitivity analysis also made local feature influence easier to inspect. However, with only 13–43 observations, the added flexibility creates overfitting risk and makes architecture, regularisation and random seed choices influential. It was worthwhile as a comparison and diagnostic, but not yet as a replacement for the uncertainty-aware GP. Its greatest value was highlighting directions to test conservatively and exposing disagreements that warranted caution.

## 5. Validation checklist

- [x] Initial data and Weeks 1–3 observations included
- [x] All eight objectives treated as maximisation problems
- [x] GP, SVM and small neural-network ensemble compared
- [x] Neural gradients interpreted as local sensitivity, not causal effects
- [x] Exactly one query selected per function
- [x] Correct dimensionality and six decimal places
- [x] Every coordinate between `0.000000` and `0.999999`
- [x] No exact duplicate of an earlier observation
- [x] Predictions clearly distinguished from returned outputs
