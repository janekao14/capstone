# Final-round BBO submission

## Part 1 — Query inputs

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

| Function | Output | New overall best? |
|---|---:|---|
| 1 | `0.0001362433711915106` | Yes |
| 2 | `0.5386160299431757` | No |
| 3 | `-0.008716282309895589` | No |
| 4 | `0.6789254464625576` | Yes |
| 5 | `1636.5047451170747` | Yes |
| 6 | `-0.3509669016093903` | No |
| 7 | `1.8599092342834425` | Yes |
| 8 | `9.9429615450071` | No |

## Part 2 — Reflection

My understanding of exploration and exploitation became more evidence-based as the data set grew. In early rounds, exploration was valuable because the surrogate was uncertain across most of each search space. I therefore used Gaussian Process uncertainty and acquisition functions to test both promising and poorly sampled regions. Once repeated local patterns appeared, the opportunity cost of a wide exploratory query increased. In the final round I exploit the narrow incumbents for Functions 1, 2, 3 and 6, continue clearly rewarded directions for Functions 4, 5 and 7, and use only a very small step for the nearly flat Function 8. The remaining risk is concentrated where several previous rewards justify it.

The returned values acted like rewards that revised my expectations about each action. A successful continuation increased the estimated value of that direction; a failed move reduced it and encouraged reversal, bisection or a return to the incumbent. For example, repeated increases for Functions 5 and 7 strengthened confidence in their current trajectories. In contrast, nearby Function 2 queries produced inconsistent rewards, so I did not assume that proximity guaranteed improvement. This resembles a Q-value update conceptually: new rewards change the expected usefulness of an action, although I did not fit a formal Q-learning model.

The process shares AlphaGo Zero’s iterative learning principle because each new result changes the next policy without revealing the true solution. It is not genuine self-play, since there is no opponent and the observations come from a fixed external function. My approach combines model-free trial and error with model-based planning. The empirical continuation and reversal rules learn directly from rewards, while Gaussian Processes and local response models anticipate outcomes before a query is spent. As the evidence grew, I relied more on observed reward sequences and less on complex global predictions.

RL could strengthen real-world optimisation by treating candidate regions or search directions as arms and allocating evaluations with UCB or Thompson sampling. Contextual bandits could use operating conditions to choose experiments, while an MDP could represent systems in which today’s action changes tomorrow’s state. Reward shaping could combine performance with cost, safety and information gain; constrained policies could prevent unsafe exploration. Experience replay and offline RL could reuse historical evaluations, but only with careful correction for sampling bias and distribution shift.

Overall, the final policy reflects diminishing uncertainty and a finite query budget. I still allow exploration where the expected information or improvement is credible, but I no longer treat uncertainty alone as sufficient reason to take a large risk. That is the main professional lesson: update beliefs transparently from feedback, distinguish stable reward patterns from noise and match the level of exploration to both the remaining budget and the consequences of failure.
