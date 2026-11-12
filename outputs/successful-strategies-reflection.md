# Successful optimisation strategies: BBO capstone reflection

## Strategies that produced my strongest results

My strongest results came from changing the strategy as evidence accumulated rather than relying on one optimiser throughout the challenge. I began with separate Gaussian Process surrogates and an Upper Confidence Bound acquisition function for each black-box function. This was effective early because it combined predicted performance with uncertainty and provided a rational way to explore when relatively little data were available.

As the challenge progressed, the most successful approach became more local and function-specific. Three strategies were especially valuable:

1. **Controlled directional continuation.** Functions 4 and 7 showed repeated improvement when several coordinates moved together in a consistent direction. Instead of repeatedly searching the entire space, I treated these movements as empirical directions and continued them in small steps. This produced final best values of `0.6789254464625576` for Function 4 and `1.8599092342834425` for Function 7.

2. **Dimension reduction through observed behaviour.** For Function 5, three coordinates remained fixed on successful boundary values while the second coordinate repeatedly drove improvement. Treating the problem as a local one-dimensional search reduced unnecessary variation and produced a final value of `1636.5047451170747`.

3. **Incumbent tightening and step-size reduction.** For narrow, unstable or plateauing functions, I stopped making large jumps and queried close to the strongest observed points. The final micro-step for Function 1 produced its best value, `0.0001362433711915106`. For Function 8, progressively smaller changes showed that the response had effectively plateaued, preventing an unjustified large final move.

These strategies changed my decisions from model-led global exploration to evidence-led local refinement. I continued a direction only when several observations supported it. When an apparent trend failed, I reduced the step, reversed it, bisected an interval or returned to the incumbent. Unsuccessful queries therefore still had value because they changed the next decision.

## What defines a successful strategy?

Outcomes matter, but I would not define success by the leaderboard alone. A successful strategy should also be adaptable, efficient, transparent and reproducible. It should use the evaluation budget carefully, explain why a query was selected and respond appropriately when new evidence conflicts with earlier assumptions.

Efficiency was particularly important because every function evaluation was limited. A complex surrogate was worthwhile only when it improved the decision. Once a stable local direction appeared, a simple controlled step was sometimes more useful than an increasingly complicated model. Conversely, when neighbouring points behaved inconsistently, as they did for Function 2, a successful strategy had to acknowledge uncertainty rather than present a fragile local pattern as fact.

I would therefore judge success using several criteria: the best observed output, the amount of improvement per query, the information gained from failures, robustness to noise and whether another researcher could reproduce the process. A strategy that obtains one strong result by chance is less convincing than one that produces sustained improvement and clearly documents its assumptions.

## Application beyond the capstone

The same principles apply to professional ML and AI projects. Hyperparameter tuning can begin with broad Bayesian or space-filling exploration and then move to local refinement around robust configurations. In manufacturing or scientific experimentation, controlled coordinate changes can reduce the cost of physical trials. In product optimisation, contextual bandits can balance testing new interventions against deploying options already known to work.

However, a professional objective should include more than predictive performance. Cost, latency, safety, fairness, energy consumption and operational constraints may all need to be included in the reward or acquisition function. Repeated evaluations may be required to estimate noise, and apparently optimal configurations should be validated under changing conditions before deployment. The BBO lesson is therefore not simply to maximise a score, but to construct a feedback process that learns efficiently without hiding uncertainty or risk.

## Reflection on peer strategies

In peer discussions, I noticed that successful participants often used either disciplined Bayesian optimisation or carefully designed local-search heuristics. Some emphasised Gaussian Process uncertainty and acquisition functions, while others inspected the returned values closely and adjusted one coordinate or direction at a time. Both approaches overlap with my own progression: Bayesian methods were most useful when coverage was sparse, while transparent local experiments became more valuable after promising regions emerged.

I also found space-filling and ensemble strategies interesting. Participants who preserved more global exploration reduced the chance of becoming trapped in the first promising basin. Those who compared several models were less dependent on the assumptions of a single surrogate. These approaches broadened my view of success: a slightly lower immediate score may be justified if a strategy is more robust, less sensitive to noise or more likely to identify a different basin over a longer budget.

One suggestion I would offer peers using aggressive local exploitation is to reserve part of the evaluation budget for global or diversity-based queries and to compare improvements against a simple random or Sobol baseline. For peers using highly complex ensembles, I would suggest recording whether each added model actually changes the selected query. This would help distinguish useful robustness from complexity that does not affect decisions.

Conversely, peer strategies strengthen my own approach by highlighting the risk of exploitation bias. My later queries became concentrated around strong regions, so I cannot claim that those regions contained the global optima. A more complete strategy would combine my function-specific local refinements with a pre-allocated exploration budget and repeated measurements for noisy functions.

Overall, the most successful strategies were not fixed recipes. They were feedback systems that used models, heuristics and judgement at different stages. Success meant obtaining strong results while remaining willing to revise the policy, explain the trade-offs and learn from both improvements and failures.

> Personalisation before posting: replace “In peer discussions” with the name of a specific participant and one concrete detail from their post if your discussion board requires direct reference to a classmate.
