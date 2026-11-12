# BBO Capstone: Living Documentation

This document is the chronological record of the black-box optimisation challenge. It is updated whenever a new round’s instructions, submitted inputs, returned outputs, strategy, or reflection becomes available.

## Challenge overview

The project optimises eight unknown synthetic black-box functions. Their analytical forms and gradients are unavailable, and only a limited number of evaluations may be requested. Every task is a **maximisation problem**, including Functions 3 and 6, whose real-world descriptions involve undesirable quantities but whose competition outputs have already been transformed so that higher is better.

| Function | Dimensions | Illustrative application | Main modelling issue |
|---|---:|---|---|
| 1 | 2 | Detecting localised contamination sources | Extremely localised response spanning many orders of magnitude |
| 2 | 2 | Noisy, multimodal log-likelihood | Noise and narrow or multiple peaks |
| 3 | 3 | Reducing adverse reactions | Negative transformed objective; nonlinear response |
| 4 | 4 | Warehouse-allocation model tuning | Dynamic, multimodal surface |
| 5 | 4 | Chemical-process yield | Strongly skewed, typically unimodal response |
| 6 | 5 | Multi-criteria recipe optimisation | Negative composite score; sparse five-dimensional space |
| 7 | 6 | ML hyperparameter tuning | Sparse, high-dimensional search |
| 8 | 8 | Complex system optimisation | Very sparse eight-dimensional search |

### Submission format

Each portal query is a hyphen-separated vector with exactly six decimal places:

```text
0.xxxxxx-0.xxxxxx-...-0.xxxxxx
```

Coordinates must be between `0.000000` and `0.999999`, have the correct dimensionality, and not duplicate an earlier query.

---

# Week 1 — Initial Bayesian optimisation strategy

## Instructions

Use the supplied `initial_inputs.npy` and `initial_outputs.npy` observations to propose one query for each function. Treat all functions as maximisation tasks. Reflect on:

1. the principal heuristic used;
2. the most challenging functions and why;
3. how the strategy will be adjusted in later rounds.

## Submitted inputs and returned outputs

| Function | Input | Output |
|---|---|---:|
| 1 | `[0.370400, 0.000002]` | `1.0682651502197714e-126` |
| 2 | `[0.959936, 0.989779]` | `0.11375873572673137` |
| 3 | `[0.000000, 0.000000, 0.000000]` | `-0.17239781799687137` |
| 4 | `[0.412422, 0.414426, 0.354724, 0.426738]` | `0.6767810479720926` |
| 5 | `[0.000000, 0.000000, 0.999999, 0.999999]` | `1616.625747348229` |
| 6 | `[0.343814, 0.000000, 0.637712, 0.999999, 0.000000]` | `-0.7382660688946026` |
| 7 | `[0.000000, 0.000000, 0.912079, 0.193707, 0.361713, 0.783275]` | `1.1717833358639496` |
| 8 | `[0.000000, 0.000000, 0.000000, 0.000000, 0.999999, 0.403654, 0.000000, 0.000000]` | `9.7390176482835` |

## Strategy

A separate Gaussian Process surrogate was fitted to each function. Candidate points were ranked with Upper Confidence Bound:

```text
UCB(x) = predicted mean(x) + kappa × predicted standard deviation(x)
```

This selected points expected to perform well or located in uncertain regions. Exact and near duplicates were rejected.

## Reflection

The main principle was to balance exploration and exploitation through UCB. Functions 7 and 8 were most difficult because six- and eight-dimensional spaces were represented by very few observations. Function 2 was also challenging because noise and multiple local optima could make an observed high value unreliable. In future rounds, each returned observation would be appended before refitting the GP. Exploration would remain relatively high initially, then decrease when repeated evidence identified a credible strong region. Expected Improvement, kernel assumptions, observation noise, and candidate diversity would be used as sensitivity checks.

---

# Week 2 — Linear/logistic-regression perspective

## Instructions

Submit a second query for every function using the growing dataset. Reflect on:

1. the main strategy change and what prompted it;
2. the exploration–exploitation balance;
3. influences from recent outputs or class discussion;
4. assumptions a linear or logistic model would violate;
5. locally linear regions or possible decision boundaries;
6. whether interpretable individual feature effects were useful.

The discussion response was required to remain under 700 words.

## Submitted inputs and returned outputs

| Function | Input | Output | Improved incumbent? |
|---|---|---:|---|
| 1 | `[0.553410, 0.547574]` | `6.5695241954466015e-9` | Yes |
| 2 | `[0.731609, 0.000000]` | `0.4557039094263051` | No |
| 3 | `[0.968840, 0.980394, 0.401588]` | `-0.04402251816300856` | No |
| 4 | `[0.414402, 0.418661, 0.316846, 0.444711]` | `-0.35092512217900973` | No |
| 5 | `[0.000000, 0.050000, 0.999999, 0.999999]` | `1616.667587803653` | Yes |
| 6 | `[0.496284, 0.336975, 0.752584, 0.991492, 0.000000]` | `-0.5191033545054177` | Yes |
| 7 | `[0.000000, 0.522398, 0.243442, 0.228577, 0.389672, 0.664494]` | `1.3565136449945694` | No |
| 8 | `[0.044885, 0.036181, 0.177572, 0.000000, 0.999999, 0.458999, 0.243251, 0.229784]` | `9.9125524545679` | Yes |

## Strategy

The Week 1 observations were appended and the exploration setting became function-specific. Scrambled Sobol candidates provided global coverage, while local perturbations refined strong observations. Expected Improvement was compared with UCB. A Matérn 5/2 kernel, dimension-specific length scales, response standardisation and an explicit noise term were used. Function 1 was modelled on a logarithmic response scale because positive values spanned over 100 orders of magnitude. Function 5 used `log(1 + y)` because of extreme right skew.

## Reflection

The main change was adjusting exploration separately for each function rather than applying one UCB policy to all eight. Functions that improved received more exploitation; unsuccessful functions retained more exploration. Recent outputs were more influential than any participant coordinates.

Simple linear regression would violate linearity and constant-error assumptions on nonlinear, multimodal and sharply localised surfaces. Logistic regression would require an arbitrary continuous-to-binary transformation and assume a linear log-odds boundary. A local linear approximation might be useful within a small smooth neighbourhood, while a classifier could distinguish acceptable and unacceptable performance. However, disconnected peaks, class imbalance and few positive examples would limit it. Individual feature effects were useful as diagnostics, but interactions and sparse coverage prevented causal interpretation.

---

# Week 3 — SVM-guided refinement

## Instructions

Submit the third query and reflect on:

1. how the strategy changed from earlier rounds;
2. the current exploration–exploitation balance;
3. whether soft-margin or kernel SVMs could identify high-performance regions;
4. model limitations, overfitting and apparently irrelevant features;
5. how optimisation under incomplete knowledge resembles professional data science.

## Submitted inputs and returned outputs

| Function | Input | Output | Improved incumbent? |
|---|---|---:|---|
| 1 | `[0.579688, 0.579836]` | `0.00013603925381526093` | Yes |
| 2 | `[0.690971, 0.000000]` | `0.49900464899824254` | No |
| 3 | `[0.824017, 0.881484, 0.093403]` | `-0.07034643846855934` | No |
| 4 | `[0.399511, 0.414063, 0.386765, 0.427920]` | `0.462172293086621` | No |
| 5 | `[0.000000, 0.050000, 0.950000, 0.999999]` | `1251.2196104544594` | No |
| 6 | `[0.511157, 0.303311, 0.672180, 0.874815, 0.000000]` | `-0.3056501399168364` | Yes |
| 7 | `[0.000000, 0.540025, 0.206500, 0.272343, 0.413441, 0.709193]` | `1.1777053320929245` | No |
| 8 | `[0.180634, 0.126966, 0.118162, 0.000000, 0.999999, 0.305703, 0.203623, 0.434098]` | `9.9030156123721` | No |

## Strategy

The GP remained the primary optimiser because it provided a continuous response prediction and uncertainty. Two SVM-based checks were added:

- a class-weighted, soft-margin RBF SVM classified top-quartile observations as high performance;
- an RBF Support Vector Regressor provided a second nonlinear ranking.

The classifier did not select inputs alone because thresholding discarded response magnitude and did not provide GP-style uncertainty. Agreement among GP/UCB, SVM and SVR increased confidence; disagreement encouraged cautious local refinement.

## Reflection

The strategy became more dependent on model predictions where repeated observations identified strong regions, while unsuccessful functions retained exploration. An RBF SVM was more suitable than a linear boundary for curved or disconnected high-performance regions, and a soft margin tolerated overlapping or noisy labels.

Limitations became clearer as GP length scales approached their bounds and clustered points risked overconfidence. In high dimensions, apparently irrelevant features could reflect insufficient coverage. The exercise mirrored professional practice by requiring explicit assumptions, uncertainty management, competing-model validation and revision when evidence contradicted the current explanation.

---

# Week 4 — Neural networks, gradients and backpropagation

## Instructions

Submit the fourth query and reflect on:

1. observations that behaved like support vectors or lay near rapid transitions;
2. whether a neural surrogate and its gradients could guide movement;
3. good-versus-bad classification with logistic regression, SVMs or neural networks;
4. interpretability versus flexibility across linear, SVM and neural models;
5. which inputs showed the steepest neural gradients;
6. how effectively the neural model represented a decision boundary;
7. whether extra nonlinear flexibility justified its complexity.

## Submitted inputs and returned outputs

| Function | Input | Output | Improved incumbent? |
|---|---|---:|---|
| 1 | `[0.590000, 0.590000]` | `0.000014708654478022732` | No |
| 2 | `[0.678794, 0.999999]` | `0.6214931697535168` | Yes |
| 3 | `[0.494275, 0.608066, 0.369439]` | `-0.029713423749666873` | Yes |
| 4 | `[0.425000, 0.430000, 0.345000, 0.424000]` | `0.4860876387556705` | No |
| 5 | `[0.000000, 0.025000, 0.999999, 0.999999]` | `1616.6325923177533` | No |
| 6 | `[0.518634, 0.292905, 0.651733, 0.857434, 0.000000]` | `-0.3952553489269476` | No |
| 7 | `[0.011608, 0.489865, 0.261473, 0.210305, 0.412571, 0.748090]` | `1.3053706703390104` | No |
| 8 | `[0.035662, 0.052610, 0.145894, 0.005116, 0.999999, 0.491591, 0.233059, 0.260954]` | `9.9267356749929` | Yes |

## Strategy

An ensemble of five small neural-network regressors—with hidden layers of 12 and 8 tanh units—was added to the GP and SVM checks. Finite-difference gradients at the incumbent approximated backpropagated input sensitivity. Candidate ranking considered GP-UCB, SVM decision score, mean neural prediction and neural-ensemble disagreement. The ensemble dispersion was treated as instability, not calibrated uncertainty.

## Reflection

Function 1’s rapid change near its Week 2–3 points and Function 5’s decline after reducing input 3 behaved like support-vector-style boundary observations. Neural sensitivities identified influential coordinates, but they described the fitted surrogate rather than the hidden function and were therefore used for small perturbations only.

A top-quartile threshold defined “good” observations. Logistic regression offered interpretability but could not represent disconnected nonlinear regions. The soft-margin RBF SVM provided controlled nonlinear classification, while the neural network was more flexible but more vulnerable to overfitting and random initialisation. The GP remained the principal optimiser because its uncertainty was more useful with small data; the neural ensemble was valuable as a nonlinear diagnostic rather than a replacement.

---

# Week 5 — Advanced neural-network concepts

## Instructions

Submit the fifth query and reflect on:

1. hierarchical feature learning and strategy structure;
2. parallels between AlexNet/ImageNet breakthroughs and incremental optimisation gains;
3. complexity and training-efficiency trade-offs versus exploration and exploitation;
4. which neural-network building blocks changed the interpretation of learning;
5. whether the workflow resembled rapid prototyping or production design;
6. how real-world sports deep learning should influence success benchmarks.

## Submitted inputs and returned outputs

| Function | Input | Output | Improved incumbent? |
|---|---|---:|---|
| 1 | `[0.577500, 0.582500]` | `0.00012645740153453865` | No |
| 2 | `[0.686815, 0.999999]` | `0.5093634769880196` | No |
| 3 | `[0.503060, 0.605065, 0.397856]` | `-0.01661363736688786` | Yes |
| 4 | `[0.410000, 0.410000, 0.360000, 0.430000]` | `0.6352737153549346` | No |
| 5 | `[0.000000, 0.075000, 0.999999, 0.999999]` | `1616.7695595919934` | Yes |
| 6 | `[0.515000, 0.295000, 0.665000, 0.890000, 0.000000]` | `-0.4174564203618419` | No |
| 7 | `[0.055082, 0.481696, 0.250551, 0.228622, 0.399122, 0.737810]` | `1.5054761901872191` | Yes |
| 8 | `[0.000000, 0.064187, 0.126972, 0.083784, 0.999999, 0.502209, 0.237553, 0.303648]` | `9.9366164779331` | Yes |

## Strategy

The analysis was structured in layers: data validation and transformation, individual sensitivity inspection, interaction and neighbourhood modelling, then agreement among GP, SVM and neural surrogates. Stronger exploitation was used for recently improved functions. Functions with regressions returned toward earlier incumbents. Neural gradients were overridden when they conflicted with clearer empirical local trends.

## Reflection

Hierarchical feature learning suggested progressing from individual coordinates to interactions and multi-model consensus. AlexNet’s performance leap provided an analogy for occasional large improvements, such as Function 1, while also showing that a breakthrough observation does not validate the entire model.

The workflow remained closer to rapid prototyping than production: it compared models and revised heuristics quickly, but a production system would require versioned data, formal validation, testing, monitoring and reproducible artefacts. Real-world sports applications suggested benchmarking actual prospective improvement, robustness, query cost and repeatability rather than relying on an impressive prediction or training fit.

---

# Week 6 — CNN parallels

## Instructions

Submit the sixth query and reflect on:

1. progressive feature extraction from simple to complex representations;
2. parallels between LeNet/CNN advances and incremental BBO improvements;
3. depth, computation and overfitting versus exploration and exploitation;
4. how convolutions, pooling, activations or loss functions informed the strategy;
5. how edge-AI deployment considerations should influence success metrics.

## Submitted inputs and returned outputs

| Function | Input | Output | Improved incumbent? |
|---|---|---:|---|
| 1 | `[0.578600, 0.580800]` | `0.00013271218960912376` | No |
| 2 | `[0.675000, 0.999999]` | `0.5032987190688517` | No |
| 3 | `[0.517730, 0.605367, 0.423675]` | `-0.0028315397301727427` | Yes |
| 4 | `[0.413633, 0.416639, 0.352086, 0.425107]` | `0.6745051071718717` | No; close to incumbent |
| 5 | `[0.000000, 0.100000, 0.999999, 0.999999]` | `1616.996193052935` | Yes |
| 6 | `[0.511157, 0.295000, 0.672180, 0.874815, 0.000000]` | `-0.33392906816229223` | No |
| 7 | `[0.053676, 0.476708, 0.252116, 0.233874, 0.388469, 0.741230]` | `1.5705578188972789` | Yes |
| 8 | `[0.000000, 0.075000, 0.117000, 0.105000, 0.999999, 0.505000, 0.240000, 0.315000]` | `9.9404956999995` | Yes |

## Strategy

The model progressed from individual coordinate effects to interactions, local neighbourhoods and multi-model agreement. Strong local exploitation was used for Functions 3, 5, 7 and 8. Function 2 used a quadratic response curve around three upper-bound observations; Function 4 used local interpolation; Functions 1 and 6 returned toward earlier incumbents.

## Reflection

Progressive feature extraction inspired a hierarchy from individual inputs to interaction patterns and model consensus. Pooling was analogous to summarising evidence from a neighbourhood instead of reacting to one point. Nonlinear activations represented interactions, while the loss maintained alignment with maximising all eight outputs. Because BBO coordinates do not have image-like spatial order, this was a conceptual analogy rather than justification for literal convolution.

As with CNN depth, model flexibility had to be balanced against computation and overfitting. Edge-AI deployment highlighted that success should include query efficiency, reproducibility, robustness and sensitivity to modelling assumptions—not just the largest predicted value.

---

# Week 7 — Hyperparameter tuning

## Instructions

Submit the seventh query and reflect on:

1. which hyperparameters were tuned and why;
2. how tuning changed the query strategy;
3. which tuning methods were used and their trade-offs;
4. limitations revealed as the dataset grew;
5. how tuning would scale to larger datasets and models;
6. how black-box tuning develops professional ML practice under incomplete information.

## Submitted inputs and returned outputs

| Function | Input | Output | Improved incumbent? |
|---|---|---:|---|
| 1 | `[0.580232, 0.579354]` | `0.0001361304711345124` | Yes |
| 2 | `[0.680975, 0.999999]` | `0.6269295746914457` | Yes |
| 3 | `[0.526515, 0.602366, 0.452092]` | `-0.006191748510547173` | No |
| 4 | `[0.412848, 0.415204, 0.353796, 0.426164]` | `0.6776502541466454` | Yes |
| 5 | `[0.000000, 0.125000, 0.999999, 0.999999]` | `1617.4240637407345` | Yes |
| 6 | `[0.511157, 0.311622, 0.672180, 0.874815, 0.000000]` | `-0.37339881036452305` | No |
| 7 | `[0.052270, 0.471720, 0.253681, 0.239126, 0.377816, 0.744650]` | `1.6311090404290538` | Yes |
| 8 | `[0.000000, 0.080407, 0.112014, 0.115608, 0.999999, 0.506396, 0.241224, 0.320676]` | `9.9417617812329` | Yes |

### Copy-ready portal entries

```text
Function 1: 0.580232-0.579354
Function 2: 0.680975-0.999999
Function 3: 0.526515-0.602366-0.452092
Function 4: 0.412848-0.415204-0.353796-0.426164
Function 5: 0.000000-0.125000-0.999999-0.999999
Function 6: 0.511157-0.311622-0.672180-0.874815-0.000000
Function 7: 0.052270-0.471720-0.253681-0.239126-0.377816-0.744650
Function 8: 0.000000-0.080407-0.112014-0.115608-0.999999-0.506396-0.241224-0.320676
```

## Strategy

The main tuned GP hyperparameters were dimension-specific Matérn length scales, observation noise and the UCB exploration parameter. SVM regularisation, RBF scale and class weighting were sensitivity settings. Neural-surrogate tuning focused on hidden-layer size, activation, weight decay and ensemble size. Exploration weights decreased for functions with repeated improvements.

Kernel and noise parameters were estimated by GP marginal likelihood with multiple optimiser restarts. Sobol sampling provided quasi-random candidate search, while `kappa` and local-search radii were adjusted manually from improvement history. Response curves were used where several points bracketed a narrow optimum. Exhaustive grids were avoided in high dimensions, and Hyperband was not used because this was not a large multi-configuration neural-training problem.

## Reflection

Hyperparameter tuning changed the process from one broadly applied rule into eight function-specific policies. It also exposed limitations: some length scales reached their bounds, neural predictions varied with initialisation, SVM boundaries depended on the high-performance threshold, and apparent feature irrelevance remained difficult to distinguish from sparse coverage. Diminishing returns became visible for Functions 5 and 8.

With larger datasets, model selection should use cross-validation or a time-ordered holdout. Random or Bayesian search could tune kernels, regularisation and architectures more systematically, while Hyperband could allocate training resources efficiently for larger neural models. Experiment tracking should record search spaces, seeds, metrics and computational cost.

Professionally, this work reinforces the need to define objectives and search ranges, separate predictions from observations, compare complexity with robustness, document unsuccessful experiments and revise decisions when incomplete evidence changes.

---

# Week 8 — Prompt and decoding strategy

## Instructions

The activity is introduced as the eighth query round with 17 observations, although part of the supplied text refers to the “seventh” query and iteration. It is treated here as **Week 8**. Submit one new query per function and reflect on:

1. zero-shot, few-shot and structured prompt patterns;
2. temperature, top-k, top-p and maximum-token settings;
3. token boundaries, unusual numeric strings and truncation;
4. prompt overfitting, irrelevant attention and diminishing returns from long inputs;
5. strategies for reducing hallucinations;
6. scaling prompting and decoding to larger datasets or models;
7. balancing exploration, risk and computational constraints.

## Submitted inputs and returned outputs

| Function | Proposed input | Output |
|---|---|---|
| 1 | `[0.581048, 0.578631]` | `0.0001342342968192575` |
| 2 | `[0.680500, 0.999999]` | `0.4193619538587751` |
| 3 | `[0.520602, 0.604386, 0.432966]` | `-0.018040988288758178` |
| 4 | `[0.413200, 0.415400, 0.353600, 0.426000]` | `0.6782476545500953` |
| 5 | `[0.000000, 0.150000, 0.999999, 0.999999]` | `1618.1486943457433` |
| 6 | `[0.511157, 0.303311, 0.672180, 0.879815, 0.000000]` | `-0.32835631994722525` |
| 7 | `[0.050864, 0.466732, 0.255246, 0.244378, 0.367163, 0.748070]` | `1.6863528511188561` |
| 8 | `[0.000000, 0.085814, 0.107028, 0.126216, 0.999999, 0.507792, 0.242448, 0.326352]` | `9.9425788369331` |

### Copy-ready portal entries

```text
Function 1: 0.581048-0.578631
Function 2: 0.680500-0.999999
Function 3: 0.520602-0.604386-0.432966
Function 4: 0.413200-0.415400-0.353600-0.426000
Function 5: 0.000000-0.150000-0.999999-0.999999
Function 6: 0.511157-0.303311-0.672180-0.879815-0.000000
Function 7: 0.050864-0.466732-0.255246-0.244378-0.367163-0.748070
Function 8: 0.000000-0.085814-0.107028-0.126216-0.999999-0.507792-0.242448-0.326352
```

## Strategy

Week 7 established new incumbents for Functions 1, 2, 4, 5, 7 and 8. Function 3 returned toward the Week 6 region after Week 7 regressed, while Function 6 isolated a small fourth-coordinate change around its Week 3 incumbent. Exploration weights were reduced because strong local evidence now dominated distant uncertainty.

The final selections used:

- local interpolation or extrapolation from successive observed results;
- low-exploration GP estimates;
- SVM and neural-model agreement as sensitivity checks;
- rejection of distant UCB maxima with weak high-performance classification;
- six-decimal format, bound, dimensionality and duplicate checks.

## Reflection

### Prompt patterns

I used a structured zero-shot prompt for the main analysis. It specified the maximisation objective, supplied observations, function dimensions, modelling rules and an exact output schema. A small few-shot element demonstrated the required six-decimal, hyphen-separated portal format. Simplifying the prompt made the response faster but more likely to omit validation or confuse predictions with observations. The structured version produced more reproducible reasoning, although excessive instructions could distract attention from the most relevant recent observations.

### Decoding settings

For a repeatable decision task, I favoured low temperature (approximately `0.2`), top-p around `0.9`, a moderate top-k such as `20`, and a maximum output budget near `1,200` tokens. Low temperature reduced unnecessary variation. Top-p retained alternatives when the distribution was uncertain, while top-k removed a very long tail of implausible tokens. The final numerical queries did not rely on sampled creativity: Python calculations and deterministic validation remained authoritative. Higher temperature was useful only for brainstorming alternative modelling ideas.

### Token boundaries and limits

Decimal vectors and hyphenated portal strings may be split into many subword tokens. A model can therefore omit a digit, alter precision or interpret a minus-like separator incorrectly. I avoided this by generating vectors in a fixed schema and validating coordinate count, bounds and six-decimal formatting programmatically. No truncation was observed because the prompt used compact function-specific histories instead of repeatedly including the full discussion transcript. Output completeness was checked by requiring exactly eight labelled functions.

### Limitations with growing context

Longer prompts created diminishing returns. Early observations remained useful for global structure, but repeating every narrative reflection could pull attention away from the newest local evidence. Prompt overfitting was also possible: strongly worded instructions to exploit could bias the explanation even when exploration was warranted. Function-specific retrieval and concise numerical tables reduced irrelevant context. The LLM was used to organise evidence and explain decisions, not as the numerical surrogate itself.

### Hallucination controls

Hallucination risk was reduced through explicit instructions not to invent evaluations, retrieval of the actual initial and weekly observations, clear separation of predictions from returned outputs, fixed output templates and independent Python validation. Competing GP, SVM and neural signals were compared, and proposals were rejected when their apparent strength came mainly from unsupported uncertainty. Pending outputs are labelled `Pending` rather than filled with model predictions.

### Future scaling and professional relevance

For larger datasets, I would store observations in CSV or a database, retrieve only the relevant function history, and pass structured JSON rather than long prose. Prompt templates would be versioned, token counts monitored, and generation separated from a deterministic verifier. Larger models could use a low-temperature primary run plus a small number of diverse critique runs. These choices mirror professional practice: control stochasticity when precision matters, permit diversity during hypothesis generation, validate outputs outside the language model and allocate computation according to risk and expected value.

---

# Week 9 — Scaling, emergence and robust optimisation

## Instructions

With 18 observations available for each function, submit the ninth query round and reflect on scaling laws and diminishing returns, possible emergent behaviour, trade-offs among cost, robustness and performance, and the balance between predictable optimisation and sudden but uneven gains. The discussion response must remain under 700 words.

## Submitted inputs and returned outputs

| Function | Submitted input | Returned output | Result |
|---|---|---:|---|
| 1 | `[0.579416, 0.580077]` | `0.00013559457427530258` | Improved on Week 8, below Week 7 incumbent. |
| 2 | `[0.681450, 0.999999]` | `0.6859447566834628` | New best. |
| 3 | `[0.514858, 0.606348, 0.414384]` | `-0.010756463186503352` | Improved on Week 8, below Week 6 incumbent. |
| 4 | `[0.413376, 0.415498, 0.353502, 0.425918]` | `0.678476327092945` | New best. |
| 5 | `[0.000000, 0.175000, 0.999999, 0.999999]` | `1619.284652110411` | New best. |
| 6 | `[0.511157, 0.303311, 0.672180, 0.869815, 0.000000]` | `-0.4479661604289244` | Regressed; lower fourth coordinate was unfavourable. |
| 7 | `[0.049458, 0.461744, 0.256811, 0.249630, 0.356510, 0.751490]` | `1.735559169312152` | New best. |
| 8 | `[0.000000, 0.091221, 0.102042, 0.136824, 0.999999, 0.509188, 0.243672, 0.332028]` | `9.9429468671001` | New best. |

### Copy-ready portal entries

```text
Function 1: 0.579416-0.580077
Function 2: 0.681450-0.999999
Function 3: 0.514858-0.606348-0.414384
Function 4: 0.413376-0.415498-0.353502-0.425918
Function 5: 0.000000-0.175000-0.999999-0.999999
Function 6: 0.511157-0.303311-0.672180-0.869815-0.000000
Function 7: 0.049458-0.461744-0.256811-0.249630-0.356510-0.751490
Function 8: 0.000000-0.091221-0.102042-0.136824-0.999999-0.509188-0.243672-0.332028
```

## Strategy

Week 8 produced new best observations for Functions 4, 5, 7 and 8, while Functions 1, 2, 3 and 6 regressed. The Week 9 policy therefore separates stable trajectories from unstable local surfaces. Functions 4, 5, 7 and 8 receive small continuation steps. Functions 1 and 3 use mirrored points to bracket their incumbents, Function 2 receives a very small one-sided probe because its neighbouring results are inconsistent, and Function 6 changes only one coordinate to obtain a cleaner causal comparison.

The surrogate ensemble remains useful for sensitivity analysis, but repeated empirical directions receive more weight than unsupported model optima. All coordinates were checked for dimensionality, bounds, six-decimal formatting and duplication.

## Reflection (under 700 words)

Scaling-law ideas influenced this round by making me distinguish between functions that still show steady gains and those already showing diminishing or unstable returns. More observations do not guarantee proportional improvement: they mainly reduce uncertainty in the regions actually sampled. Functions 5, 7 and 8 have improved through several measured local steps, so I continued their trajectories. Function 4 also improved, but by a smaller amount, so I reduced the step size. Functions 1, 2, 3 and 6 have inconsistent neighbouring outcomes; for them, I used small bracketing or one-variable probes instead of assuming that a larger model or more data would automatically produce a better query.

Emergent behaviour could appear as a threshold, interaction or narrow peak that was invisible in the initial sparse data. Function 2 is a warning: two extremely close inputs produced very different outputs, which may indicate noise, a sharp response surface or an interaction the current surrogate cannot resolve. I am preparing for such behaviour by preserving raw observations, avoiding irreversible conclusions from one result, comparing multiple surrogate families and selecting queries that discriminate between competing explanations. Boundary optima in Functions 5 and 8 may also reflect interactions rather than simple independent feature effects.

The central trade-off is that each evaluation is expensive, while an overly aggressive query can waste a round. Pure exploitation is economical when improvement is consistent, but it can overfit noise or miss another basin. Wide exploration is more robust globally, but its immediate expected performance is lower. I therefore use function-specific risk levels: conservative continuation for stable improving paths, local bracketing for uncertain peaks and isolated coordinate changes when attribution is unclear. Model complexity is treated similarly. A flexible neural surrogate can represent interactions, but with only 18 points it may be less reliable than a Gaussian Process or a transparent local trend.

I balance predictable optimisation with possible sudden gains by reserving most queries for evidence-supported local moves while retaining controlled novelty in the direction or side being tested. This is similar to managing emergent capability in larger models: improvements can be uneven and averages can hide failures. I therefore judge success not only by the best output, but also by repeatability, distance from known points, agreement among models and the information gained if the query performs poorly. The approach is deliberately adaptive: scale the evidence base, watch for qualitative changes and increase complexity only when the observations justify it.

---

# Week 10 — Transparency and interpretability

## Instructions

Submit one tenth-round query for every function and explain the reasoning behind each choice. The reflection must address reproducibility, assumptions, gaps or biases in the sampled data and one significant limitation of the approach, while remaining under 700 words.

## Submitted inputs and returned outputs

| Function | Submitted input | Returned output | Result |
|---|---|---:|---|
| 1 | `[0.580004, 0.579556]` | `0.00013622277268503162` | New best. |
| 2 | `[0.681925, 0.999999]` | `0.43271609196048777` | Regressed from the Week 9 best. |
| 3 | `[0.516294, 0.605858, 0.419030]` | `-0.01126652220669583` | Below the Week 6 incumbent. |
| 4 | `[0.413552, 0.415596, 0.353404, 0.425836]` | `0.6786583839846192` | New best. |
| 5 | `[0.000000, 0.200000, 0.999999, 0.999999]` | `1620.9657048532827` | New best. |
| 6 | `[0.511157, 0.303311, 0.672180, 0.876627, 0.000000]` | `-0.39232795042604013` | Below the Week 3 incumbent. |
| 7 | `[0.048052, 0.456756, 0.258376, 0.254882, 0.345857, 0.754910]` | `1.7780607366467855` | New best. |
| 8 | `[0.000000, 0.093925, 0.099549, 0.142128, 0.999999, 0.509886, 0.244284, 0.334866]` | `9.9429625536839` | New best, but only a very small gain. |

### Copy-ready portal entries

```text
Function 1: 0.580004-0.579556
Function 2: 0.681925-0.999999
Function 3: 0.516294-0.605858-0.419030
Function 4: 0.413552-0.415596-0.353404-0.425836
Function 5: 0.000000-0.200000-0.999999-0.999999
Function 6: 0.511157-0.303311-0.672180-0.876627-0.000000
Function 7: 0.048052-0.456756-0.258376-0.254882-0.345857-0.754910
Function 8: 0.000000-0.093925-0.099549-0.142128-0.999999-0.509886-0.244284-0.334866
```

## Strategy

The Week 10 policy prioritises auditability. Each query is derived from a short rule that can be recomputed from observed points: interpolation for Functions 1 and 3, fixed-step continuation for Functions 2, 4, 5 and 7, a local quadratic calculation for Function 6, and a reduced continuation step for Function 8. This avoids presenting opaque surrogate predictions as certainty while still using the ensemble as a reasonableness check.

Week 9 produced new incumbents for Functions 2, 4, 5, 7 and 8. Function 1 nearly recovered its Week 7 best, Function 3 remained below its Week 6 best, and Function 6 showed that lowering its fourth coordinate was harmful. These observations determine whether the next point continues, interpolates or reverses the latest move.

## Reflection (under 700 words)

My tenth-round strategy was guided by the most recent empirical pattern for each function rather than by one acquisition rule applied uniformly. Functions 2, 4, 5, 7 and 8 achieved new best outputs in Week 9, so I continued their improving directions, reducing the step for Function 8 because its gains are diminishing. Function 1 was sampled on both sides of its Week 7 incumbent, allowing a small quadratic interpolation inside that bracket. Function 3 moved halfway back toward its Week 6 incumbent after both sides performed worse. For Function 6, the Week 9 decrease in coordinate four caused a large deterioration, while a previous increase caused only a smaller deterioration. I therefore fitted a simple three-point local quadratic along that coordinate and kept the other coordinates fixed.

The process is reasonably transparent because every proposed coordinate has a documented source: a previous incumbent, a displacement vector, an interpolation fraction or a local polynomial calculation. Another researcher could reproduce the strategy if given the initial `.npy` files, all weekly input-output pairs, the maximisation convention, coordinate bounds, rounding rules and the code or formulas used for transformations and candidate checks. They would also need random seeds and model hyperparameters to reproduce the GP, SVM and neural sensitivity analyses exactly. Importantly, the returned observations are recorded separately from model predictions.

A key assumption is local smoothness: nearby inputs are expected, on average, to produce related outputs. This assumption supports interpolation, Gaussian Process kernels and incremental continuation. It may fail when the function is noisy, discontinuous or contains narrow interactions. Function 2 illustrates the risk because very close points have produced sharply different values. A second assumption is that recent improvements identify a useful direction; this may hold only briefly near a curved ridge or local maximum.

The dataset is strongly biased toward exploitation. Later queries cluster around current incumbents, leaving much of each unit hypercube sparsely sampled, especially for the six- and eight-dimensional functions. Boundary values such as `0.000000` and `0.999999` also occur repeatedly. This makes local comparisons efficient but limits conclusions about global optimality and may exaggerate the apparent importance of the coordinates being varied. Correlated multi-coordinate moves in Functions 7 and 8 further make it difficult to attribute improvement to an individual feature.

The most significant limitation is the very small evaluation budget relative to dimensionality. With only 19 observations per function before this round, flexible models can fit the observed data while remaining poorly identified elsewhere. Cross-validation is unstable and uncertainty estimates depend heavily on kernel, noise and regularisation assumptions. I manage this limitation by using small reversible steps, isolating variables where possible, comparing simple trends with multiple surrogate models and treating every query as both an optimisation attempt and an information-gathering experiment. Nevertheless, the resulting strategy is best interpreted as evidence-based local optimisation, not proof that the global maximum has been found.

---

# Week 11 — Cluster-guided refinement

## Instructions

Submit one eleventh-round query for every function. For each point, identify the local cluster being targeted, the similarity or distance cue used, and how the choice builds on the first ten rounds. Reflect on recurring regions, ineffective strategies, the analogy with clustering and the patterns that would appear if the results were plotted.

## Submitted inputs and returned outputs

| Function | Submitted input | Returned output | Result |
|---|---|---:|---|
| 1 | `[0.579900, 0.579650]` | `0.00013621670649687095` | Essentially tied with, but slightly below, Week 10. |
| 2 | `[0.681213, 0.999999]` | `0.4834793480051173` | Below the Week 9 incumbent. |
| 3 | `[0.518300, 0.605100, 0.424800]` | `-0.016348264667780082` | Below the Week 6 incumbent. |
| 4 | `[0.413728, 0.415694, 0.353306, 0.425754]` | `0.6787938819981254` | New best. |
| 5 | `[0.000000, 0.225000, 0.999999, 0.999999]` | `1623.3450577769993` | New best. |
| 6 | `[0.511157, 0.303311, 0.672180, 0.875315, 0.000000]` | `-0.34609936310569855` | Improved on Week 10, below Week 3. |
| 7 | `[0.046646, 0.451768, 0.259941, 0.260134, 0.335204, 0.758330]` | `1.8132680294742978` | New best. |
| 8 | `[0.000000, 0.096629, 0.097056, 0.147432, 0.999999, 0.510584, 0.244896, 0.337704]` | `9.9428659784769` | Regressed from Week 10. |

### Copy-ready portal entries

```text
Function 1: 0.579900-0.579650
Function 2: 0.681213-0.999999
Function 3: 0.518300-0.605100-0.424800
Function 4: 0.413728-0.415694-0.353306-0.425754
Function 5: 0.000000-0.225000-0.999999-0.999999
Function 6: 0.511157-0.303311-0.672180-0.875315-0.000000
Function 7: 0.046646-0.451768-0.259941-0.260134-0.335204-0.758330
Function 8: 0.000000-0.096629-0.097056-0.147432-0.999999-0.510584-0.244896-0.337704
```

## Strategy

The clustering lens treats high-output observations as local groups rather than assuming that every nearby point is equally informative. For Functions 1, 2, 3 and 6, the proposed queries tighten distances around a high-performing medoid or bracket. For Functions 4, 5 and 7, successive improvements form directional ridge-like clusters, so the latest displacement is continued. Function 8 also forms a boundary ridge, but its step is halved because its incremental gain is almost flat.

These are descriptive local clusters, not claims from a statistically stable unsupervised fit. With few adaptive observations and different dimensions, formal `k`-means centroids could be distorted by low-performing global points and arbitrary scaling. Output rank and distance to the incumbent are therefore used together.

## Reflection (under 700 words)

Patterns from the first ten rounds strongly shaped the eleventh submission. I separated the functions into two broad groups. Functions 4, 5, 7 and 8 show sequences of nearby points whose outputs improve along a consistent direction; I treated these as ridge-like clusters and continued them with short steps. Functions 1, 2, 3 and 6 show narrow, noisy or inconsistent neighbourhoods; there I targeted points close to a high-performing medoid or used interval tightening rather than wider extrapolation.

Several recurring promising regions are visible. Function 1 has a compact cluster near both coordinates equal to `0.58`. Function 2 performs best near the upper boundary of its second coordinate and a first coordinate around `0.681`, although neighbouring values are highly variable. Function 3’s best local group is centred near `[0.518, 0.605, 0.424]`. Function 4 has a tight four-dimensional ridge near `[0.414, 0.416, 0.353, 0.426]`. Functions 5 and 8 repeatedly favour boundary coordinates, while Function 7 follows a stable six-dimensional trajectory. Function 6 remains centred near its Week 3 incumbent and has not produced a reliable improving direction.

Some strategies have been less effective. Large exploratory jumps generally underperformed after promising regions were found. For Functions 2 and 6, fitting a smooth local curve to a few points was unreliable because nearby outputs were inconsistent. For Function 3, mirroring and midpoint queries around the incumbent did not improve it. I am responding by reducing step sizes, using medoid-like incumbent points rather than averages distorted by poor samples, and requiring repeated evidence before extending a direction.

This refinement parallels clustering because it separates coherent structure from noise. Distance alone is insufficient: two close Function 2 points can have very different outputs. I therefore combine input-space proximity with output rank. Dense sequences with consistently improving values are interpreted as meaningful local structure; isolated poor results or unstable neighbours are down-weighted rather than allowed to shift the cluster centre substantially. This resembles robust clustering, where medoids or local neighbourhoods can be preferable to means when outliers are present.

If plotted, Functions 4, 5 and 7 would likely show ordered trails moving toward higher outputs. Function 8 would show a similar trail whose vertical improvement is flattening. Functions 1 and 3 would show compact clouds with a central peak, while Functions 2 and 6 would show high local variance despite small input distances. These plots would guide the next iteration by revealing whether the proposed point extends a ridge, tightens a peak or confirms that apparent neighbours belong to different response regimes. I would then shrink steps near flattening clusters and reconsider exploration where no stable grouping emerges.

---

# Week 12 — PCA-inspired directional refinement

## Instructions

Submit one twelfth-round query for each function using the accumulated structure in the search history. Reflect on how the strategy has evolved, which variables or behaviours explain the most variation, which dimensions or behaviours can be simplified, how this round informs the final submission, and how PCA concepts such as retained variance and redundancy apply to the BBO results.

## Proposed inputs

| Function | Proposed input | PCA-style rationale |
|---|---|---|
| 1 | `[0.580056, 0.579509]` | Reverse half of the small Week 10→11 displacement; recent variation is concentrated along this two-coordinate contrast. |
| 2 | `[0.681569, 0.999999]` | Keep the stable boundary coordinate fixed and test only a quarter-step from the Week 9 incumbent toward the poorer Week 10 point. |
| 3 | `[0.517445, 0.605501, 0.423113]` | Move half a step opposite the Week 6→11 displacement, retaining the incumbent neighbourhood while reducing redundant wandering. |
| 4 | `[0.413904, 0.415792, 0.353208, 0.425672]` | Continue the repeated four-coordinate principal direction that has produced successive improvements. |
| 5 | `[0.000000, 0.250000, 0.999999, 0.999999]` | Vary only coordinate two because the other three have remained fixed on the successful boundary face. |
| 6 | `[0.511157, 0.303311, 0.672180, 0.875065, 0.000000]` | Bisect the narrow interval between the Week 3 incumbent and Week 11, treating coordinate four as the dominant local axis. |
| 7 | `[0.045240, 0.446780, 0.261506, 0.265386, 0.324551, 0.761750]` | Continue the stable six-coordinate displacement vector that acts like a leading empirical component. |
| 8 | `[0.000000, 0.092573, 0.100796, 0.139476, 0.999999, 0.509537, 0.243978, 0.333447]` | Reverse halfway from Week 10 toward Week 9 after the continuation direction began to reduce performance. |

### Copy-ready portal entries

```text
Function 1: 0.580056-0.579509
Function 2: 0.681569-0.999999
Function 3: 0.517445-0.605501-0.423113
Function 4: 0.413904-0.415792-0.353208-0.425672
Function 5: 0.000000-0.250000-0.999999-0.999999
Function 6: 0.511157-0.303311-0.672180-0.875065-0.000000
Function 7: 0.045240-0.446780-0.261506-0.265386-0.324551-0.761750
Function 8: 0.000000-0.092573-0.100796-0.139476-0.999999-0.509537-0.243978-0.333447
```

## Returned outputs

| Function | Week 12 output | Incumbent improved? |
|---|---:|---|
| 1 | `0.0001362108320709245` | No |
| 2 | `0.5799427140012658` | No |
| 3 | `-0.02176417684413894` | No |
| 4 | `0.6788828815237555` | Yes |
| 5 | `1626.5956972682993` | Yes |
| 6 | `-0.4066227400332177` | No |
| 7 | `1.8406827164501731` | Yes |
| 8 | `9.9429688307286` | Yes, by a very small margin |

## Strategy

The PCA analogy is applied to the query trajectory rather than claiming a statistically reliable PCA fit from sparse adaptive data. Repeated displacement vectors are treated as empirical principal directions when they explain repeated output improvement. Coordinates that remain fixed at boundaries are treated as locally redundant, while unstable one-dimensional responses are isolated with smaller steps.

Functions 4 and 7 continue strong directional components, and Function 5 retains only its active second coordinate. Functions 1, 3, 6 and 8 use smaller reverse or bisection moves around their incumbents. Function 2 keeps its second coordinate fixed and cautiously probes the first because local output variation remains much larger than geometric distance would suggest.

## Reflection (under 700 words)

My strategy has evolved from broad model-led exploration into a more structured, function-specific process. Early rounds relied mainly on Gaussian Process predictions and acquisition functions. Later observations revealed that the same rule was not appropriate for every function. I now record the incumbent, recent displacement vectors, output changes and whether each move supplied new information. Repeated improvement leads to a controlled continuation; deterioration leads to a smaller step, reversal or interval bisection. This makes the process more systematic and easier to reproduce.

The behaviours that explain the largest variation differ by function. For Function 5, coordinate two is the main observed driver because the other coordinates have remained fixed while the output has risen consistently. Functions 4 and 7 exhibit coordinated multi-variable directions that behave like empirical principal components: movement along these vectors repeatedly explains improvement. For Function 6, most recent local variation comes from coordinate four. Functions 1 and 3 are dominated by distance from a narrow incumbent region rather than by large independent coordinate effects. Function 2 has unusually high output variation among very close inputs, suggesting noise or an unresolved narrow feature. Function 8 follows a coordinated boundary direction, but its gains have flattened and then reversed.

I decide what to retain by looking for repeated association with improvement. Directions supported across several rounds are preserved, while dimensions that remain fixed or movements that repeatedly fail are reduced. This resembles PCA retaining components that explain meaningful variance and discarding redundant directions. However, high variance is not automatically useful: Function 2 varies strongly but unpredictably, so I reduce its step rather than explore more aggressively. Output improvement, stability and interpretability must accompany variance.

This round is designed to prepare the final submission. Functions 4, 5 and 7 test whether their leading directions still produce gains. Functions 1, 3 and 6 tighten around established incumbents. Function 8 tests a smaller reverse move after its latest continuation regressed, while Function 2 cautiously tests the neighbourhood of its best point. If the directional functions improve again, the final round can make one last measured continuation. If they flatten or decline, the final query should shrink toward the best observed point. For unstable functions, the final decision should prioritise robust incumbent neighbourhoods over speculative exploration.

PCA also provides a useful warning about redundancy. Many later observations are close together and therefore add less independent information than their count suggests. Viewing the history through directions, rather than treating every coordinate change separately, helps reveal which experiments repeat the same evidence. The objective is not to reduce dimensions mechanically, but to concentrate the remaining query budget on directions that explain performance while avoiding noise, duplicated movement and unsupported complexity.

---

# Final round — RL-informed exploitation with controlled risk

## Instructions

Submit one final query for each function using the full history. Reflect on the exploration–exploitation trade-off, feedback-driven value updates, the relationship to model-free and model-based learning, and how reinforcement-learning strategies could improve real-world optimisation.

## Proposed final inputs

| Function | Proposed input | Decision rationale |
|---|---|---|
| 1 | `[0.579980, 0.579580]` | Exploit the narrow peak beside the Week 10 incumbent with a non-duplicate micro-step. |
| 2 | `[0.681425, 0.999999]` | Stay close to the Week 9 best point; repeated neighbouring failures make wider exploration poor value in the final round. |
| 3 | `[0.517800, 0.605350, 0.423500]` | Use a small perturbation around the Week 6 incumbent rather than extrapolating from later weaker points. |
| 4 | `[0.414080, 0.415890, 0.353110, 0.425590]` | Continue the stable Week 11→12 improvement direction by one measured step. |
| 5 | `[0.000000, 0.300000, 0.999999, 0.999999]` | Take a moderately larger step along the only repeatedly rewarded coordinate while retaining the successful boundary face. |
| 6 | `[0.511157, 0.303311, 0.672180, 0.874700, 0.000000]` | Return to the best-known neighbourhood and vary only coordinate four by a tiny amount. |
| 7 | `[0.043834, 0.441792, 0.263071, 0.270638, 0.313898, 0.765170]` | Continue the highly consistent six-coordinate policy that improved in every recent round. |
| 8 | `[0.000000, 0.091897, 0.101420, 0.138150, 0.999999, 0.509363, 0.243825, 0.332738]` | Make a half-step continuation from the small Week 10→12 improvement because the response is now nearly flat. |

### Copy-ready portal entries

```text
Function 1: 0.579980-0.579580
Function 2: 0.681425-0.999999
Function 3: 0.517800-0.605350-0.423500
Function 4: 0.414080-0.415890-0.353110-0.425590
Function 5: 0.000000-0.300000-0.999999-0.999999
Function 6: 0.511157-0.303311-0.672180-0.874700-0.000000
Function 7: 0.043834-0.441792-0.263071-0.270638-0.313898-0.765170
Function 8: 0.000000-0.091897-0.101420-0.138150-0.999999-0.509363-0.243825-0.332738
```

## Returned outputs

| Function | Week 13 output | Incumbent improved? |
|---|---:|---|
| 1 | `0.0001362433711915106` | Yes |
| 2 | `0.5386160299431757` | No |
| 3 | `-0.008716282309895589` | No |
| 4 | `0.6789254464625576` | Yes |
| 5 | `1636.5047451170747` | Yes |
| 6 | `-0.3509669016093903` | No |
| 7 | `1.8599092342834425` | Yes |
| 8 | `9.9429615450071` | No |

## Strategy

I treated each function as an independent bandit-like decision problem. A query direction was an action, the returned improvement was its reward and the accumulated history informed its estimated value. Because this is the final round, the policy is deliberately more exploitative than in early weeks. Functions 4, 5 and 7 receive controlled directional continuations because those actions have repeatedly produced positive rewards. Function 8 receives only a half-step because its reward has diminished to almost zero. Functions 1, 2, 3 and 6 receive non-duplicate points close to their incumbents because recent exploratory actions did not improve their estimated value.

## Reflection (under 700 words)

My understanding of exploration and exploitation became more evidence-based as the data set grew. In early rounds, exploration was valuable because the surrogate was uncertain across most of each search space. I therefore used Gaussian Process uncertainty and acquisition functions to test both promising and poorly sampled regions. Once repeated local patterns appeared, the opportunity cost of a wide exploratory query increased. In the final round I exploit the narrow incumbents for Functions 1, 2, 3 and 6, continue clearly rewarded directions for Functions 4, 5 and 7, and use only a very small step for the nearly flat Function 8. The remaining risk is concentrated where several previous rewards justify it.

The returned values acted like rewards that revised my expectations about each action. A successful continuation increased the estimated value of that direction; a failed move reduced it and encouraged reversal, bisection or a return to the incumbent. For example, repeated increases for Functions 5 and 7 strengthened confidence in their current trajectories. In contrast, nearby Function 2 queries produced inconsistent rewards, so I did not assume that proximity guaranteed improvement. This resembles a Q-value update conceptually: new rewards change the expected usefulness of an action, although I did not fit a formal Q-learning model.

The process shares AlphaGo Zero’s iterative learning principle because each new result changes the next policy without revealing the true solution. It is not genuine self-play, since there is no opponent and the observations come from a fixed external function. My approach combines model-free trial and error with model-based planning. The empirical continuation and reversal rules learn directly from rewards, while Gaussian Processes and local response models anticipate outcomes before a query is spent. As the evidence grew, I relied more on observed reward sequences and less on complex global predictions.

RL could strengthen real-world optimisation by treating candidate regions or search directions as arms and allocating evaluations with UCB or Thompson sampling. Contextual bandits could use operating conditions to choose experiments, while an MDP could represent systems in which today’s action changes tomorrow’s state. Reward shaping could combine performance with cost, safety and information gain; constrained policies could prevent unsafe exploration. Experience replay and offline RL could reuse historical evaluations, but only with careful correction for sampling bias and distribution shift.

Overall, the final policy reflects diminishing uncertainty and a finite query budget. I still allow exploration where the expected information or improvement is credible, but I no longer treat uncertainty alone as sufficient reason to take a large risk. That is the main professional lesson: update beliefs transparently from feedback, distinguish stable reward patterns from noise and match the level of exploration to both the remaining budget and the consequences of failure.

---

# Current best observations after Week 13

| Function | Best output | Input | First achieved |
|---|---:|---|---|
| 1 | `0.0001362433711915106` | `[0.579980, 0.579580]` | Week 13 |
| 2 | `0.6859447566834628` | `[0.681450, 0.999999]` | Week 9 |
| 3 | `-0.0028315397301727427` | `[0.517730, 0.605367, 0.423675]` | Week 6 |
| 4 | `0.6789254464625576` | `[0.414080, 0.415890, 0.353110, 0.425590]` | Week 13 |
| 5 | `1636.5047451170747` | `[0.000000, 0.300000, 0.999999, 0.999999]` | Week 13 |
| 6 | `-0.3056501399168364` | `[0.511157, 0.303311, 0.672180, 0.874815, 0.000000]` | Week 3 |
| 7 | `1.8599092342834425` | `[0.043834, 0.441792, 0.263071, 0.270638, 0.313898, 0.765170]` | Week 13 |
| 8 | `9.9429688307286` | `[0.000000, 0.092573, 0.100796, 0.139476, 0.999999, 0.509537, 0.243978, 0.333447]` | Week 12 |

---

# Final project reflection

## Initial codebase

I built the optimisation workflow from scratch in Python rather than starting from a complete public BBO solution. I selected this approach because the project required transparent reasoning for eight functions with different dimensions and behaviours. NumPy handled the supplied `.npy` arrays, while scikit-learn supplied standard Gaussian Process, regression, SVM and neural-network components. Building the orchestration myself made it easier to preserve one model and decision history per function, validate the portal format and explain why each query was chosen. The public repository is available at [github.com/janekao14/capstone](https://github.com/janekao14/capstone).

## Code modification and evolution

The first version loaded each function's initial arrays, standardised its response, fitted an independent Gaussian Process and ranked candidates with Upper Confidence Bound. This gave a principled exploration–exploitation baseline, but the same acquisition settings were not equally suitable for all eight functions.

As new observations arrived, I appended them to the correct function history and refitted the models. I added duplicate checks, bounds validation and six-decimal portal formatting. I then compared UCB with Expected Improvement, expanded candidate generation and considered function-specific kernels and noise assumptions. Linear and logistic regression were used as interpretable local checks rather than global descriptions, since the response surfaces were nonlinear. RBF SVMs provided a way to distinguish relatively strong and weak regions, while small neural-network ensembles and input gradients offered flexible sensitivity checks for the higher-dimensional functions.

The most important change was not a more complicated model; it was making the policy function-specific. Later rounds used local interpolation, bracketing, controlled coordinate changes, output-aware clusters and repeated displacement vectors. Functions 4 and 7 showed stable coordinated directions, Function 5 became effectively a one-dimensional search on its second coordinate, and Function 8 reached a plateau. Functions 2 and 6 remained locally unstable, so I reduced step sizes and returned to strong incumbent neighbourhoods. The final round interpreted query directions as actions and improvements as rewards, using an RL-inspired policy to continue repeatedly rewarded directions while limiting risk elsewhere.

## Final result

The best observed value improved over the best initial observation for all eight functions. In the final weeks, Functions 4, 5 and 7 improved repeatedly. Function 4 reached `0.6789254464625576`; Function 5 increased to `1636.5047451170747`; and Function 7 reached `1.8599092342834425`. The final micro-step for Function 1 also produced its best result, `0.0001362433711915106`. Function 8 peaked in Week 12 at `9.9429688307286`, confirming that its recent trajectory had nearly flattened. The best values for Functions 2, 3 and 6 remained earlier incumbents, showing that additional nearby points did not guarantee improvement.

With a fresh start, I would establish a reproducible experiment log immediately, including seeds, fitted kernel parameters, candidate pools and acquisition scores. I would reserve a fixed portion of the query budget for global space-filling exploration before local exploitation and use repeated evaluations to estimate noise for unstable functions. I would also compare the adaptive policy with simple baselines such as random search and Sobol search so that the benefit of modelling could be measured more directly.

## Trade-offs and decisions

The central trade-off was whether to spend a scarce evaluation on information or immediate performance. Broad exploration reduced the risk of missing another basin but often produced weak short-term scores. Local exploitation improved strong functions but created sampling bias and could trap the search near a local optimum. I responded by changing the balance over time: exploration was stronger when uncertainty dominated, while later rounds required repeated empirical improvement before continuing a direction. I also traded model flexibility against interpretability. Neural networks could represent complex surfaces, but with so few points they risked overfitting; transparent local steps were often more defensible.

## Learning and application

The most important lesson was that optimisation quality depends as much on disciplined feedback as on model choice. A sophisticated surrogate cannot compensate for sparse, biased evidence or an acquisition rule used without regard to function-specific behaviour. In future competitions, I would maintain explicit uncertainty, incumbent and regret-style tracking from the beginning. In real-world ML work, I would also include operational cost, safety and reversibility in the objective rather than optimising predictive performance alone.

I was most surprised that simple directional continuation outperformed increasingly elaborate modelling for several functions once a reliable local pattern emerged. I was equally surprised by how sharply neighbouring Function 2 points could differ. This reinforced the need to distinguish geometric closeness from response reliability. The range of participant strategies also highlighted that strong results can come from different combinations of modelling, visual reasoning and careful heuristics, provided the decisions are documented and updated honestly from evidence.

## Living-document update procedure

Whenever a new round is provided:

1. append the submitted input and returned output to each function’s history;
2. compare each output with the prior incumbent;
3. refit the function-specific surrogate models;
4. record any response transformation and tuned hyperparameters;
5. generate and validate one new query per function;
6. add the activity instructions, strategy and reflection as a new weekly section;
7. update the current-best table;
8. mark unknown outputs as `Pending` rather than treating predictions as observations.

---

# Module 21 — Datasheet and model card

To make the project transparent and reproducible, two repository-level governance documents were created:

- [`docs/DATASHEET.md`](../docs/DATASHEET.md) records the dataset motivation, actual function-specific sample counts, composition, collection process, preprocessing, intended and inappropriate uses, distribution constraints and maintenance procedure.
- [`docs/MODEL_CARD.md`](../docs/MODEL_CARD.md) describes the Adaptive BBO Surrogate Ensemble, its evolution through the RL-informed final policy, completed performance through Week 13, assumptions, limitations, failure modes and ethical considerations.

The repository [`README.md`](../README.md) links both documents and provides a concise project overview, input/output contract, technical approach, structure, status and reproducibility statement. All thirteen weekly queries and their returned outputs are now recorded, and the optimisation phase is complete.

---

# Post-project activity — Successful optimisation strategies

## Instructions

Reflect on the strategies that produced the strongest BBO results, explain why they worked, define what makes an optimisation strategy successful and consider how the lessons transfer to professional ML/AI projects. Compare the approach with one or more peers, identify overlap and suggest improvements. The discussion-board response must remain under 2,000 words.

## Strategy analysis and reflection

The strongest results came from changing the policy as evidence accumulated. Gaussian Process surrogates and Upper Confidence Bound were valuable early because they combined prediction and uncertainty when observations were sparse. Later, three more targeted strategies produced the clearest gains. Controlled directional continuation improved Functions 4 and 7; empirical dimension reduction turned Function 5 into a largely one-dimensional local search; and incumbent tightening produced Function 1's final best value while limiting risk on unstable or plateauing functions.

These results changed the decision rule from model-led global exploration to evidence-led local refinement. A direction was continued only when several observations supported it. Failed movements triggered a smaller step, reversal, interval bisection or return to the incumbent. This made unsuccessful evaluations informative rather than wasted.

Success should not be defined by the leaderboard alone. A strong strategy should also be adaptable, efficient, transparent and reproducible. Useful measures include best observed output, improvement per query, information gained from failures, robustness to noise and whether another researcher can reconstruct the decisions. A fortunate isolated score is less convincing than sustained improvement supported by an auditable process.

In professional ML/AI projects, the same staged approach can guide hyperparameter tuning, scientific experiments, manufacturing optimisation and product testing. Broad Bayesian or space-filling exploration can be followed by local refinement around robust configurations. Real objectives should also include cost, latency, safety, fairness and operational constraints, while promising configurations should be retested under noise and distribution shift.

Peer discussions appeared to show two recurring successful patterns: disciplined Bayesian optimisation and careful local-search heuristics. Both overlap with this project. Space-filling strategies reduce the risk of premature convergence, while ensembles reduce dependence on one surrogate's assumptions. A useful suggestion for aggressive local strategies is to reserve part of the budget for diverse global queries and compare performance with random or Sobol baselines. For complex ensembles, each added model should be evaluated by whether it materially changes the selected query.

The peer comparison also exposes a limitation of my own later strategy: concentrated exploitation produced strong local results but cannot demonstrate that the corresponding regions contain global optima. A stronger future design would combine function-specific local refinement with a pre-allocated exploration budget and repeated measurements for noisy functions.

The copy-ready discussion response is stored in [`outputs/successful-strategies-reflection.md`](successful-strategies-reflection.md). Because no specific peer post was supplied, the peer section deliberately avoids attributing unverified details and includes a prompt to add one participant's name and concrete strategy before posting.

---

# Final repository deliverable

## Instructions

Finalise a public, portfolio-ready GitHub repository containing clear and reproducible code, a Jupyter Notebook, the completed datasheet and model card, an approximately 100-word non-technical explanation, project data or appropriate external access instructions, presentation materials and README links that make every component easy to find.

## Repository refinements

- Added `notebooks/BBO_Capstone_Final.ipynb` with portable data loading, Gaussian Process and UCB methods, query validation, final results and interpretation.
- Replaced personal absolute paths in the analysis script with project-relative paths.
- Added `requirements.txt` and explicit random seeds for reproducibility.
- Added `data/README.md` explaining how authorised course participants can restore the supplied arrays without redistributing restricted materials.
- Added `outputs/final-results.csv` as a machine-readable final performance summary.
- Added an approximately 100-word non-technical summary and a final results table to the main README.
- Added an MIT licence for original code and documentation, with an explicit exclusion for course-provided data, templates, branding and teaching materials.
- Linked the notebook, datasheet, model card, presentation, reflections and final results from the README.

The repository URL for submission is [https://github.com/janekao14/capstone](https://github.com/janekao14/capstone). The local files must be committed and pushed, and the repository visibility should be confirmed as public before the discussion-board link is submitted.
