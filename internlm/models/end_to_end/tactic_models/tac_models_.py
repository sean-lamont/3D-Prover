from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import time
import warnings

import ray
import torch
from loguru import logger

from models.end_to_end.tactic_models.diversity_model.batched_model import BatchedDiversityModel

warnings.filterwarnings('ignore')

from experiments.end_to_end.common import Context
from models.end_to_end.tactic_models.internlm_gen import InternLMGenerator
# from experiments.end_to_end.proof_node import *
from abc import abstractmethod


# todo tidy
class TacModel:
    @abstractmethod
    def get_tactics(self, goals, premises):
        return


class TacWrapper(TacModel):
    def __init__(self, tac_model):
        super().__init__()
        self.tac_model = tac_model

    def get_tactics(self, goal, premises):
        tactics = ray.get(self.tac_model.get_tactics.remote(goal, premises))

        return tactics


class TopKTacGenerator(TacModel):
    def __init__(self, tac_model: TacModel, num_filtered, random=False):
        super().__init__()
        self.tac_model = tac_model
        self.num_filtered = num_filtered
        self.random = random

    def get_tactics(self, goal, premises):
        _, theorem, _ = premises
        tactics = self.tac_model.get_tactics(goal, premises)

        goal.data['original_tacs'] = tactics

        if self.random:
            return random.sample(tactics, self.num_filtered)
        else:
            # tactics are expected to be sorted here
            return tactics[:self.num_filtered]


class DiversityTacGenerator(TacModel):
    def __init__(self, tac_model: TacModel, filter_model, num_filtered, temperature=1., scale=1e5, p=0.9):
        super().__init__()
        self.tac_model = tac_model
        self.filter_model = filter_model
        self.num_filtered = num_filtered
        self.temperature = temperature
        self.scale = scale
        self.p = p

    def get_tactics(self, goal, premises):
        _, theorem, _ = premises

        t0 = time.monotonic()
        tactics = self.tac_model.get_tactics(goal, premises)
        # logger.warning(f"Time to get tactics: {time.monotonic() - t0}")

        goal.data['original_tacs'] = tactics
        goal.data['tac_gen_time'] = time.monotonic() - t0

        # state = goal.data['augmented_state'] if hasattr(goal, 'data') and 'augmented_state' in goal.data else goal.goal
        state = goal.goal
        # filter with filter_model

        t0 = time.monotonic()
        inds, sim_matrix = ray.get(self.filter_model.filter_tacs.remote(tactics, self.num_filtered,
                                                                        state=state, theorem=theorem.full_name,
                                                                        temperature=self.temperature, scale=self.scale,
                                                                        p=self.p))
        goal.data['filter_time'] = time.monotonic() - t0
        # logger.warning(f"Time to filter tactics: {time.monotonic() - t0}")

        # goal.data['similarity_scores'] = sim_matrix

        return [tactics[i] for i in sorted(inds[0])]


# wrapper to add the retrieval augmented state to the goal node, and to call tac model with Ray
class InternLMWrapper(TacModel):
    def __init__(self, tac_model):
        super().__init__()
        self.tac_model = tac_model

    def get_tactics(self, goal, premises):
        path, theorem, position = premises

        state = f"---\nNAME: {theorem.full_name}\n\n---\nPROOF_BEFORE: {get_prev_tactics(goal)}\n\n---\nSTATE_BEFORE: {goal.goal}\n\n---\nTACTIC: "

        tactics = ray.get(self.tac_model.get_tactics.remote(state, premises))

        return tactics

# returns tactics leading to goal in the tree (where multiple paths lead to goal, takes only the first)
def get_prev_tactics(goal):
    if not goal.in_edges:
        return ''
    else:
        return get_prev_tactics(goal.in_edges[0].src) + goal.in_edges[0].tactic


class InternLMTacModel(TacModel):
    def __init__(self, config, num_sampled_tactics=64):
        super().__init__()
        if hasattr(config, 'ckpt_path') and config.ckpt_path:
            tac_gen = InternLMGenerator.load(
                config.ckpt_path, device='cuda', freeze=True
            )
        else:
            tac_gen = InternLMGenerator(config.config)  # .to('cuda')
            # tac_gen.freeze()

        self.tac_model = tac_gen
        self.num_sampled_tactics = num_sampled_tactics

    def get_tactics(self, goal, premises):
        tactics, _ = self.tac_model.generate(
            state=goal,
            num_samples=self.num_sampled_tactics,
            # retriever_args=Context(path=path, theorem_full_name=theorem.full_name, theorem_pos=position,
            #                        state=goal),
            retriever_args=None)

        return tactics


def get_model_dict(prefix, state_dict):
    return {k[len(prefix) + 1:]: v for k, v in state_dict.items()
            if k.startswith(prefix)}


def load_pretrained_encoders(self, encoder_premise, encoder_goal):
    ckpt_dir = self.config.pretrain_ckpt
    ckpt = torch.load(ckpt_dir)['state_dict']
    encoder_premise.load_state_dict(get_model_dict('embedding_model_premise', ckpt))
    encoder_goal.load_state_dict(get_model_dict('embedding_model_goal', ckpt))


def get_tac_model(config, device):
    if config.model == 'batched_diversity_model':

        if config.distributed:
            tac_model = ray.remote(num_gpus=config.gpu_per_process, num_cpus=config.cpu_per_process)(
                InternLMTacModel).remote(
                config=config, num_sampled_tactics=config.num_sampled_tactics)

            filter_model = ray.remote(num_gpus=config.gpu_per_diversity, num_cpus=config.cpu_per_diversity)(
                BatchedDiversityModel).remote(config.diversity_config, device=device)

            tac_model = InternLMWrapper(tac_model)

            return DiversityTacGenerator(tac_model=tac_model, filter_model=filter_model,
                                         num_filtered=config.diversity_config.num_filtered,
                                         temperature=config.diversity_config.temperature if hasattr(
                                             config.diversity_config, 'temperature') else 1.,
                                         scale=config.diversity_config.scale if hasattr(
                                             config.diversity_config, 'scale') else 1,
                                         p=config.diversity_config.p if hasattr(
                                             config.diversity_config, 'p') else 0.9)

        else:
            raise NotImplementedError

    if config.model == 'internlm':
        if config.distributed:
            tac_model = ray.remote(num_gpus=config.gpu_per_process, num_cpus=config.cpu_per_process)(
                InternLMTacModel).remote(
                config=config, num_sampled_tactics=config.num_sampled_tactics)
            return InternLMWrapper(tac_model)

        else:
            raise NotImplementedError

    if config.model == 'topk_internlm':

        if config.distributed:
            tac_model = ray.remote(num_gpus=config.gpu_per_process, num_cpus=config.cpu_per_process)(
                InternLMTacModel).remote(
                config=config, num_sampled_tactics=config.num_sampled_tactics)

            tac_model = InternLMWrapper(tac_model)

            return TopKTacGenerator(tac_model=tac_model,
                                    num_filtered=config.diversity_config.num_filtered,
                                    random=config.diversity_config.random)


        else:
            raise NotImplementedError

    else:
        raise NotImplementedError
