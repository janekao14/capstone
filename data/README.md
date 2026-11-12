# Data access

The initial `.npy` arrays were supplied through the Emeritus/Imperial College Business School course portal in Module 12. They are not committed to this public repository because redistribution rights were not provided.

To reproduce the modelling workflow:

1. Download the personalised `initial_data` folder from the course portal.
2. Place it at `data/initial_data/`.
3. Preserve the supplied layout:

```text
data/initial_data/
├── function_1/
│   ├── initial_inputs.npy
│   └── initial_outputs.npy
├── ...
└── function_8/
    ├── initial_inputs.npy
    └── initial_outputs.npy
```

The tidy 104-row weekly history is committed as [`query_history.csv`](query_history.csv). The complete queries, returned outputs and decision rationales are documented in [`../outputs/bbo-capstone-living-documentation.md`](../outputs/bbo-capstone-living-documentation.md). Dataset composition, transformations, limitations and intended uses are described in [`../docs/DATASHEET.md`](../docs/DATASHEET.md).
