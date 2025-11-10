# 3D-Prover: Diversity Driven Theorem Proving with Determinantal Point Processes (NeurIPS 2025)
This is the official repository for the NeurIPS 2025 paper *3D-Prover: Diversity Driven Theorem Proving with Determinantal Point Processes*

We have organised our code into 2 directories, `reprover` and `internlm`, to replicate the results in the paper.

The `reprover` directory contains the majority of our code and experiments, including details for 
setting up the environment, replicating the transition model results and all ReProver based experiments.

The `internlm` directory contains more recent code and experiments for InternLM2.5-Step-Prover, which we have separated as 
it uses Lean 4, with a newer version of LeanDojo (2.2.0) and a different model (InternLM).

We recommend to start with the README in the `reprover` directory.
