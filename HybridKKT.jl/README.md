# HybridKKT.jl

Artifact to reproduce the benchmarks presented in
the manuscript ["Condensed-space methods for nonlinear programming on GPUs"](https://arxiv.org/abs/2405.14236).

A `Manifest.toml` file is provided to duplicate the exact package versions we used for the
benchmarks presented in the paper. A `Makefile` is used as a main entry point.

*Important notice:*
This repository is provided only for reproduction purpose. Please use
the following implementation if you want to use the condensed KKT systems in
your own work:

- LiftedKKT has been implemented in [MadNLP](https://github.com/MadNLP/MadNLP.jl/blob/master/src/KKT/Sparse/condensed.jl)
- HyKKT has been implemented in a separate extension: [HybridKKT.jl](https://github.com/MadNLP/HybridKKT.jl)

## Installation
To install all the dependencies, please use:
```shell
make install

```
This command installs MadNLP and all the required dependencies
(including CUDA and cuDSS).

Note that HSL has to be installed independently using
[libHSL](https://licences.stfc.ac.uk/product/libhsl), and then do:

```shell
export LIBHSL="/your/path/to/HSL_jll.jl"
julia --project -e "using Pkg; Pkg.develop(path=ENV[\"LIBHSL\"])"

```

## Tests the installation

You can check the installation has succeeded by running:
```shell
make tests

```

## Reproduce the results
You can reproduce the PGLIB and the COPS benchmarks using:
```shell
make benchmarks

```
