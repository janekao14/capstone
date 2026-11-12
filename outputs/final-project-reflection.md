# BBO capstone project — final reflection

## Initial codebase

I built the optimisation workflow from scratch in Python rather than starting from a complete public BBO solution. I selected this approach because the project required transparent reasoning for eight functions with different dimensions and behaviours. NumPy handled the supplied `.npy` arrays, while scikit-learn supplied standard Gaussian Process, regression, SVM and neural-network components. Building the orchestration myself made it easier to preserve one model and decision history per function, validate the portal format and explain why each query was chosen.

The public project repository is available at [https://github.com/janekao14/capstone](https://github.com/janekao14/capstone).

## Code modification

The first version loaded each function's initial arrays, standardised its response, fitted an independent Gaussian Process and ranked candidates with Upper Confidence Bound. This gave a principled exploration–exploitation baseline, but the same acquisition settings were not equally suitable for all eight functions.

As new observations arrived, I appended them to the correct function history and refitted the models. I added duplicate checks, bounds validation and six-decimal portal formatting. I then compared UCB with Expected Improvement, expanded candidate generation and considered function-specific kernels and noise assumptions. Linear and logistic regression were used as interpretable local checks rather than global descriptions, since the response surfaces were nonlinear. RBF SVMs provided a way to distinguish relatively strong and weak regions, while small neural-network ensembles and input gradients offered flexible sensitivity checks for the higher-dimensional functions.

The most important change was not a more complicated model; it was making the policy function-specific. Later rounds used local interpolation, bracketing, controlled coordinate changes, output-aware clusters and repeated displacement vectors. Functions 4 and 7 showed stable coordinated directions, Function 5 became effectively a one-dimensional search on its second coordinate, and Function 8 reached a plateau. Functions 2 and 6 remained locally unstable, so I reduced step sizes and returned to strong incumbent neighbourhoods. The final round interpreted query directions as actions and improvements as rewards, using an RL-inspired policy to continue repeatedly rewarded directions while limiting risk elsewhere.

## Final result

The best observed value improved over the best initial observation for all eight functions. In the final weeks, Functions 4, 5 and 7 improved repeatedly. Function 4 reached `0.6789254464625576`; Function 5 increased to `1636.5047451170747`; and Function 7 reached `1.8599092342834425`. The final micro-step for Function 1 also produced its best result, `0.0001362433711915106`. Function 8 peaked in Week 12 at `9.9429688307286`, confirming that its recent trajectory had nearly flattened. The best values for Functions 2, 3 and 6 remained earlier incumbents, showing that additional nearby points did not guarantee improvement.

With a fresh start, I would establish a reproducible experiment log immediately, including seeds, fitted kernel parameters, candidate pools and acquisition scores. I would reserve a fixed portion of the query budget for global space-filling exploration before local exploitation and use repeated evaluations to estimate noise for unstable functions. I would also compare the adaptive policy with simple baselines such as random search and Sobol search so that the benefit of modelling could be measured more directly.

## Trade-offs and decisions

The central trade-off was whether to spend a scarce evaluation on information or immediate performance. Broad exploration reduced the risk of missing another basin but often produced weak short-term scores. Local exploitation improved strong functions but created sampling bias and could trap the search near a local optimum. I responded by changing the balance over time: exploration was stronger when uncertainty dominated, while later rounds required repeated empirical improvement before continuing a direction.

I also traded model flexibility against interpretability. Neural networks could represent complex surfaces, but with so few points they risked overfitting. Gaussian Processes provided uncertainty but depended strongly on kernel and noise assumptions. Transparent local steps were often more defensible once repeated patterns emerged. For the final round, I accepted controlled risk for Functions 4, 5 and 7 because their directions had generated repeated rewards, while I used conservative incumbent-neighbourhood queries for unstable or plateaued functions.

## Learning and application

The most important lesson was that optimisation quality depends as much on disciplined feedback as on model choice. A sophisticated surrogate cannot compensate for sparse, biased evidence or an acquisition rule used without regard to function-specific behaviour. In future competitions, I would maintain explicit uncertainty, incumbent and regret-style tracking from the beginning. In real-world ML work, I would also include operational cost, safety and reversibility in the objective rather than optimising predictive performance alone.

I was most surprised that simple directional continuation outperformed increasingly elaborate modelling for several functions once a reliable local pattern emerged. I was equally surprised by how sharply neighbouring Function 2 points could differ. This reinforced the need to distinguish geometric closeness from response reliability. The range of participant strategies also highlighted that strong results can come from different combinations of modelling, visual reasoning and careful heuristics, provided the decisions are documented and updated honestly from evidence.
