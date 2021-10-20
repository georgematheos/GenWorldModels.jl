# GenWorldModels

GenWorldModels: Open-Universe Probabilistic Modeling with Programmable Inference.

*Warning: This is rapidly evolving, unreleased research software.*

If you use this software or build on this work in your research, please cite our 2021 AABI paper:
```
George Matheos, Alexander K. Lew, Matin Ghavamizadeh, Stuart Russell, Marco Cusumano-Towner, and Vikash Mansinghka. "Transforming Worlds: Automated Involutive MCMC for Open-Universe Probabilistic Models." 3rd Symposium on Advances in Approximate Bayesian Inference, 2021.
```

Feel free to engage via Github, or to email George at georgematheos@berkeley.edu.

## Overview
GenWorldModels is a library extending the [Gen Probabilistic Programming System](https://www.gen.dev).  GenWorldModels adds a domain specific language for writing open-universe models similar to those of the [BLOG probabilistic programming language](https://bayesianlogic.github.io), and a domain specific language for writing custom MCMC inference kernels for these models.  The library implements algorithms to automatically
implement custom inference algorithms efficiently and correctly from code in the inference kernel DSL.

To see examples of models and inference programs implemented using GenWorldModels,
please see the [GenWorldModelsExamples](https://github.com/georgematheos/GenWorldModelsExamples.jl) repository.

GenWorldModels is still in a highly developmental phase;
it is not yet a mature software library intended for release.
There is currently no user-facing documentation.
GenWorldModels does not currently interoperate with the official
version of Gen; it instead uses [a custom fork of Gen](https://github.com/georgematheos/Gen/tree/genworldmodels-main) containing experimental features
and some architectural changes, many of which we hope to someday incorporate into Gen.

## Related Repositories
The code for the GenWorldModels project is split into several repositories modularizing different features:
- GenWorldModels (this repository) contains the source code and tests for GenWorldModels.
- [GenWorldModelsExamples](https://github.com/georgematheos/GenWorldModelsExamples.jl) contains example models and inference programs using GenWorldModels.
  To see how examples of how models and inference programs are implemented using GenWorldModels, please
  see this repository.
- [This Fork of Gen](https://github.com/georgematheos/Gen/tree/genworldmodels-main) is the probabilistic programming system GenWorldModels plugs into.
- [GenTraceKernelDSL](https://github.com/probcomp/GenTraceKernelDSL.jl/tree/genworldmodels-main) contains a DSL for writing inference programs for Gen Models.
  A branch of this is used, with the addition of additional syntactic macros, to implement
  the GenWorldModels inference DSL.