"""

Helper functions for environment setup used in end-to-end experiments.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from environments.LeanDojo.get_lean_theorems import _get_theorems
from environments.LeanDojo.leandojo_env import LeanDojoEnv
from experiments.end_to_end.common import zip_strict


def get_thm_name(env, thm):
    if env == 'leandojo':
        return str(thm.full_name)
    else:
        raise NotImplementedError


def get_env(cfg):
    if cfg == 'leandojo':
        return LeanDojoEnv
    else:
        raise NotImplementedError

def get_lean_thms(config, prev_theorems):
    repo, theorems, positions = _get_theorems(config)

    # Remove proven theorems if resuming
    final_theorems = []
    final_positions = []

    for i, theorem in enumerate(theorems):
        if theorem.full_name in prev_theorems:
            continue
        else:
            final_theorems.append(theorem)
            final_positions.append(positions[i])

    theorems = final_theorems
    positions = final_positions

    theorems = list(zip_strict([repo] * len(theorems), theorems, positions))

    return theorems


def get_theorems(cfg, prev_theorems):
    if cfg.env == 'leandojo':
        return get_lean_thms(cfg, prev_theorems)
    else:
        raise NotImplementedError
