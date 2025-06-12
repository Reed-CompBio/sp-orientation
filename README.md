# Shortest Paths Network Orientation

This implements the ILP solution of network orientation from the paper [Network orientation via shortest paths](https://doi.org/10.1093/bioinformatics/btu043).

## Input

This takes a mixed graph as its interactome, which is a headerless `.tsv` containing
the `source`, `node`, `direction` ∈ {U, D}, and `weight` ∈ [0, ∞).

This also takes in a `sources` file and a `targets` file, both formatted as a newline-separated list of nodes to act as sources and targets for network orientation.
