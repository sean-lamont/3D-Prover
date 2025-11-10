# Supplementary Material for 3D-Prover (ReProver Results)

## Environment Setup
Set up conda/virtual environment with basic requirements:

     conda create -n '3dprover_env' python=3.10 ipython numpy

     conda install torch (2.0.1 in our case)

Ensure MongoDB is installed and running on the default port.

Ensure Weights and Biases is installed and configured.

Install additional requirements:
`pip install -r requirements.txt`

## Data Preparation
We use the Lean 3 miniF2F benchmark from LeanDojo v1.6.0. 

This needs to be downloaded and traced, by running from the `data/LeanDojo` directory:
`python scripts/process_miniF2F.py`.

For our system, we had a bug and had to modify the LeanDojo source code, specifically the `data_extraction/trace.py` file
in the trace function,
after the line `TracedRepo.load_from_disk(cached_path, build_deps)`, we add: 

    ## temporary fix for mathlib3 -> mathlib
    if 'mathlib3' in traced_repo.dependencies:
        traced_repo.dependencies['mathlib'] = traced_repo.dependencies.pop('mathlib3')

To replicate the large scale experiment over the LeanDojo Novel Premises Benchmark, 
run `process_data.sh` to download and trace the LeanDojo Novel Premises benchmark.

## Tactic Generation model
We assume a pretrained tactic generator is available.
Where we have `pretrained_tactic_generator_path` in our configs, this should
be replaced with a HuggingFace model path to the pretrained tactic generator.

We use ReProver trained on the LeanDojo Lean 3 novel premises training split,
available at (link removed for review anonymity, however we can share the model upon request)

The retrieval model for ReProver will also need to index the miniF2F corpus.
This can be done by running the `experiments.models.end_to_end.tactic_models.retrieval.index` script,
with the retrieval model checkpoint (also available upon request) and the miniF2F corpus as arguments.

## Experiments
Our experiments are set up using Hydra as a configuration manager, 
with all the configurations stored in the `config/end_to_end` directory. 

The `train` directory contains the configurations for training the transition models and autoencoder.

The `tac_model` config directory specifies the details of the tactic filtering and generation models.

The `env_config` directory contains the configurations for the environment setup, 
including the split of the miniF2F dataset used for each experiment.

The `search_config` directory contains the configurations for the search strategies used in the experiments
(just BestFS for our experiments).

Proof search experiments inherit from the `tac_model`, `env_config`, and `search_config` configurations,
to define the details of the experiments.

Each experiment will set up a Weights and Biases run, with details given by the `logging_config`
in each experiment configuration.

Each experiment will also create a directory under `runs/`,
which will contain checkpoints, logs, and other details of the experiment, with the name given by
the `exp_config` details.


## Initial run to obtain transition data
To obtain the initial transition data, taken from a vanilla run of ReProver over miniF2F-valid, we run:
`python -m experiments.end_to_end.end_to_end_experiment --config-name=end_to_end/vanilla_reprover`

This will create a directory under `runs/` matching the details in the config, 
which will contain the trace files under `traces/0`, which we will call the `trace_dir`,
which is used by the configs in the next step.

## Train Transition Models
The transition models are trained using the trace data obtained in the previous step,
where the path to the trace data is specified in the config file under `trace_dir`.

The first run will process the transition data from the traces into a MongoDB database, 
which will be used by subsequent runs to train the transition models.

All Tokens Model:
`python -m experiments.lightning_runner --config-name=end_to_end/train/error_pred/minif2f_all_tokens`

Combined Model:
`python -m experiments.lightning_runner --config-name=end_to_end/train/error_pred/minif2f`

Separate Model:
`python -m experiments.lightning_runner --config-name=end_to_end/train/error_pred/minif2f_single_vec`

No Tactic Model
`python -m experiments.lightning_runner --config-name=end_to_end/train/error_pred/minif2f_no_tac`

## Train Autoencoder
The autoencoder for ablations can be run with:
`python -m experiments.lightning_runner --config-name=end_to_end/train/tac_autoencoder/run`

## Save checkpoints to single file
If using deepspeed for training, as we have, run the zero_to_fp32.py script in the checkpoint
directories generated from the above runs to save these model checkpoints to a single file,
and use the resulting file paths for loading the transition model or autoencoder in the next step.

## Run end-to-end experiments
Evaluating the models on live proofs can be done with:
`python -m experiments.end_to_end.end_to_end_experiment --config-name=end_to_end/{experiment_name}`

Where `{experiment_name}` is the name of the experiment configuration file in the `config/end_to_end/` directory.

The `3d_prover` experiment contains details for the full 3D-Prover model as outlined in the paper.

The `3d_prover_autoenc` configuration contains details for the 3D-Prover model with the autoencoder as the transition model.

The `bestfs_topk` configuration contains details for the Top-K and Random baselines.

The `3d_prover_leandojo` configuration contains details for the 3D-Prover model over the LeanDojo Novel Premises benchmark.

## Sweeps
We used wandb sweeps to simplify the experiment process. 
The sweep configuration files are in the `config/sweeps/` directory, which can be modified as needed to 
run the desired experiments.

## Ablation statistics
Statistics for the ablation studies can be obtained
with `ablation_stats.ipynb` file, using the directory to the 
traces obtained from the desired experiment(s).


# Additional Material

## Transition Model Predictions
We include a csv file with the predictions of the combined transition model for the test set. 

The file is zipped in the folder `transition_predictions/combined_predictions.zip`


