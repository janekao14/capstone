# Black-Box Optimisation Capstone

This project documents an iterative search for high-performing inputs to eight unknown synthetic black-box functions. The functions range from two to eight dimensions, expose no analytical form or gradient and can be queried only a limited number of times. Every objective is treated as a maximisation problem.

## Non-technical summary

This project explores how to make good decisions when testing an unknown system is expensive. I optimised eight hidden functions by selecting one new input for each function every week and learning from the returned scores. The process began with Bayesian models that balanced promising predictions against uncertain regions. As more evidence became available, the strategy shifted towards careful local experiments around the strongest results. The final best score improved on the initial best score for every function. The project demonstrates that successful optimisation depends not only on sophisticated models, but also on clear records, controlled experiments and the willingness to change strategy when new evidence appears.

## Project goals

- balance exploration of uncertain regions with exploitation of promising observations;
- compare Gaussian Processes, acquisition functions, regression, SVM and neural-surrogate perspectives;
- adapt the query policy as new evaluations arrive;
- make every modelling assumption and query decision auditable.

## Inputs and outputs

Each function receives a continuous vector with coordinates between `0.000000` and `0.999999`. Portal queries use six-decimal, hyphen-separated values:

```text
0.123456-0.654321
```

The portal returns one scalar performance value. Higher is always better, including for negative-valued Functions 3 and 6.

## Technical approach

The strategy began with per-function Gaussian Process surrogates and Upper Confidence Bound acquisition. Later rounds added Expected Improvement, Sobol candidate generation, local linear analysis, RBF SVM classification, small neural-network ensembles and gradient-based sensitivity checks. As the dataset grew, the policy became increasingly function-specific and locally focused. The final policy uses an RL-inspired interpretation of query directions as actions and returned improvements as rewards, favouring controlled continuation or incumbent tightening according to the accumulated feedback.

## Results

| Function | Best initial output | Final best output | First achieved |
|---|---:|---:|---|
| 1 | `7.710875114502849e-16` | `0.0001362433711915106` | Week 13 |
| 2 | `0.6112052157614438` | `0.6859447566834628` | Week 9 |
| 3 | `-0.034835313350078584` | `-0.0028315397301727427` | Week 6 |
| 4 | `-4.025542281908162` | `0.6789254464625576` | Week 13 |
| 5 | `1088.8596181962705` | `1636.5047451170747` | Week 13 |
| 6 | `-0.7142649478202404` | `-0.3056501399168364` | Week 3 |
| 7 | `1.3649683044991994` | `1.8599092342834425` | Week 13 |
| 8 | `9.598482002566342` | `9.9429688307286` | Week 12 |

All functions are maximised. Output scales differ, so values should be compared only within the same function. These results show empirical improvement but do not prove global optimality.

## Run the notebook

```bash
git clone https://github.com/janekao14/capstone.git
cd capstone
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter notebook notebooks/BBO_Capstone_Final.ipynb
```

The notebook runs its final-results and validation sections without the restricted course arrays. To refit the surrogate models, obtain the Module 12 files through the course portal and follow [`data/README.md`](data/README.md). Randomised modelling steps use explicit seeds.

The final notebook calculates its result table directly from the committed [`data/query_history.csv`](data/query_history.csv) and [`data/initial_best.csv`](data/initial_best.csv). Its outputs are saved in the notebook so they render immediately on GitHub. The [`notebooks/rounds/`](notebooks/rounds/) directory contains the reproducible decision record for Rounds 3–13; each notebook limits the history to the observations available at that point to avoid hindsight.

## Documentation

- [Final analysis notebook](notebooks/BBO_Capstone_Final.ipynb) — portable method, query validation and final results
- [Round-by-round notebooks](notebooks/rounds/) — saved calculations for Rounds 3–13
- [Tidy query history](data/query_history.csv) — 104 evaluated queries across thirteen rounds
- [Machine-readable final results](outputs/final-results.csv) — final incumbents and the weeks in which they were achieved
- [Dataset datasheet](docs/DATASHEET.md) — motivation, composition, collection, preprocessing, intended uses and maintenance
- [Optimisation model card](docs/MODEL_CARD.md) — approach, evolution across thirteen rounds, performance, assumptions and limitations
- [Living project documentation](outputs/bbo-capstone-living-documentation.md) — weekly instructions, inputs, outputs, strategies and reflections
- [Final-round submission](outputs/final-round-bbo-submission.md) — copy-ready portal queries and RL-focused reflection
- [Final project reflection](outputs/final-project-reflection.md) — completed discussion-board response covering the codebase, evolution, results, trade-offs and learning
- [Successful strategies reflection](outputs/successful-strategies-reflection.md) — analysis of effective optimisation choices, professional applications and peer approaches
- [Completed course presentation template](outputs/presentation/BBO-capstone-presentation-completed-template.pdf) — submission-ready five-page A4 PDF using the supplied template
- [Capstone approach presentation](outputs/presentation/BBO-capstone-approach-presentation.pdf) — alternative designed five-page summary
- [Editable presentation](outputs/presentation/BBO-capstone-approach-presentation.pptx) — PowerPoint source for future updates

## Repository structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── data/
│   ├── README.md
│   ├── initial_best.csv
│   └── query_history.csv
├── docs/
│   ├── DATASHEET.md
│   └── MODEL_CARD.md
├── notebooks/
│   ├── BBO_Capstone_Final.ipynb
│   └── rounds/
│       ├── round_03_svm_guided_refinement.ipynb
│       └── ... round_13_rl_informed_final_policy.ipynb
├── outputs/
│   ├── bbo-capstone-living-documentation.md
│   ├── final-round-bbo-submission.md
│   ├── final-project-reflection.md
│   ├── successful-strategies-reflection.md
│   ├── final-results.csv
│   └── presentation/
│       ├── BBO-capstone-approach-presentation.pdf
│       ├── BBO-capstone-approach-presentation.pptx
│       └── BBO-capstone-presentation-completed-template.pdf
└── work/
    ├── week4_analysis.py
    └── week4_results.json
```

## Current status

The optimisation phase is complete through Week 13. The final query improved Functions 1, 4, 5 and 7; the incumbents for Functions 2, 3, 6 and 8 remained unchanged. The best observed value improved over the initial incumbent for all eight functions. Because the true optima are unknown, these results demonstrate empirical progress rather than global optimality.

## Reproducibility and responsible use

The project preserves raw returned observations separately from model predictions and documents unsuccessful queries as well as improvements. It is an educational study using synthetic functions and is not intended for safety-critical or causal decision-making. Course-provided arrays are described but not redistributed because their reuse terms were not supplied. The original code and documentation are available under the [MIT License](LICENSE); that licence does not apply to course-provided datasets, templates, branding or teaching materials.

