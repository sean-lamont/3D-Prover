# Supplementary Material for 3D-Prover (InternLM results)

## Replicating InternLM2.5-Step-Prover 
- Setup the environment as done in ReProver, except now install LeanDojo version 2.2.0, and vllm for the InternLM model.

- We use miniF2F for Lean 4, as before run from the `data/LeanDojo` directory:
`python scripts/process_miniF2F.py`.

- Run a single bestfs experiment with the InternLM model, using `python -m experiments.end_to_end.end_to_end_experiment --config-name=end_to_end/internlm_bestfs` to obtain
trace files to train the transition model.

- Train the transition model with `python -m experiments.lightning_runner --config-name=end_to_end/train/error_pred/minif2f_itnernlm`, setting the `trace_files` parameter to be the traces obtained from the previous run.

- Save the best checkpoint to a single file with `python scripts/zero_to_fp32.py --checkpoint_path {path_to_checkpoint}`.

- To run InternLM with 3D-Prover (BestFS), run `python -m experiments.end_to_end.end_to_end_experiment --config-name=end_to_end/3dprover_internlm`
 
- To run InternLM with 3D-Prover (Critic Guided Search), run `python -m experiments.end_to_end.end_to_end_experiment --config-name=end_to_end/3dprover_internlm_critic`
