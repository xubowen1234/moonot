# Coupling orientation and support order

`coupling_transpose` creates a target-by-source plan. `coupling_permute`
reindexes rows and columns using complete output-to-input bijections. Both
allocate independent rows; editing the result cannot alter the original plan.

These operations let scientific pipelines reconcile stable support IDs and
reverse alignment direction without solving again. They preserve each entry
exactly and perform no summation, normalization or optimization. A permuted
plan must be paired with equally permuted marginals and costs. Transposing an
unbalanced result swaps the source and target penalties as well as the supports
when interpreting its objective.
