from __future__ import division, absolute_import, print_function

import ray
from models.end_to_end.search_models.bestfs import BestFS
from models.end_to_end.search_models.critic_guided import CriticGuidedSearch
from models.end_to_end.search_models.goal_models.internlm_critic import InternLMCritic


def get_search_model(config, device):
    """

    Initialise search model based on configuration.

    """

    if config.search == 'bestfs':
        return BestFS()
    elif config.search == 'internlm':
        goal_model = ray.remote(num_gpus=config.gpu_per_process, num_cpus=config.cpu_per_process)(InternLMCritic).remote(config, device=device)

        return CriticGuidedSearch(goal_model)
    else:
        raise NotImplementedError(f'Search approach {config.search} not implemented')
