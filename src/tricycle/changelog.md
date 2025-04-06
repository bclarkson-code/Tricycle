# Changelog

A log of changes to the optimiser (`model.update`) with a corresponding measurement of the time improvement:

Change | Execution time | Percentage of total | Relative speedup
-|-|-|-
baseline | 31.2 | 76.3 | 1
remove loss scaling | 28.96 | 75.2 | 1.077
remove loss scaling + remove manual type conversion | 27.62 | 71.3 | 1.130
put everything in a single array |
