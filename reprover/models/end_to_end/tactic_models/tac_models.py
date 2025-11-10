from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import warnings

import ray

from models.end_to_end.tactic_models.diversity_model.model import DiversityModel

warnings.filterwarnings('ignore')

from experiments.end_to_end.common import Context
from models.end_to_end.tactic_models.generator.model import RetrievalAugmentedGenerator
from experiments.end_to_end.proof_node import *


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

        # t0 = time.monotonic()
        tactics = self.tac_model.get_tactics(goal, premises)
        # logger.warning(f"Time to get tactics: {time.monotonic() - t0}")

        goal.data['original_tacs'] = tactics

        state = goal.data['augmented_state'] if hasattr(goal, 'data') and 'augmented_state' in goal.data else goal.goal

        # filter with filter_model
        # t0 = time.monotonic()
        inds, sim_matrix = ray.get(self.filter_model.filter_tacs.remote(tactics, self.num_filtered,
                                                                        state=state, theorem=theorem.full_name,
                                                                        temperature=self.temperature, scale=self.scale,
                                                                        p=self.p))
        # logger.warning(f"Time to filter tactics: {time.monotonic() - t0}")

        goal.data['similarity_scores'] = sim_matrix

        return [tactics[i] for i in sorted(inds[0])]


# wrapper to add the retrieval augmented state to the goal node
class ReProverWrapper(TacModel):
    def __init__(self, tac_model, retriever=False):
        super().__init__()
        self.tac_model = tac_model
        self.retriever = retriever

    def get_tactics(self, goal, premises):
        tactics, new_state = ray.get(self.tac_model.get_tactics.remote(goal.goal, premises))

        # save retrieved data to node for retrieval models
        if self.retriever:
            if hasattr(goal, 'data'):
                goal.data['augmented_state'] = new_state
            else:
                goal.data = {'augmented_state': new_state}

        return tactics


class ReProverTacGen(TacModel):
    def __init__(self, tac_model, num_sampled_tactics=64):
        super().__init__()
        self.tac_model = tac_model
        self.num_sampled_tactics = num_sampled_tactics

    def get_tactics(self, goal, premises):
        path, theorem, position = premises

        tactics, new_state = self.tac_model.generate(
            state=goal,
            num_samples=self.num_sampled_tactics,
            retriever_args=Context(path=path, theorem_full_name=theorem.full_name, theorem_pos=position,
                                   state=goal)
        )

        return tactics, new_state

def get_tac_model(config, device):
    if config.model == 'topk':

        if hasattr(config, 'ckpt_path') and config.ckpt_path:
            tac_gen = RetrievalAugmentedGenerator.load(
                config.ckpt_path, device=device, freeze=True
            )

        else:
            tac_gen = RetrievalAugmentedGenerator(config.config).to(device)
            tac_gen.freeze()

        if tac_gen.retriever is not None:
            assert config.config.indexed_corpus_path is not None
            tac_gen.retriever.load_corpus(config.config.indexed_corpus_path)

            # check if corpus is up to date, otherwise recompute
            if tac_gen.retriever.embeddings_staled:
                tac_gen.retriever.reindex_corpus(batch_size=2)

        if config.distributed:
            tac_model = ray.remote(num_gpus=config.gpu_per_process, num_cpus=config.cpu_per_process)(
                ReProverTacGen).remote(
                tac_model=tac_gen, num_sampled_tactics=config.num_sampled_tactics)

            tac_model = ReProverWrapper(tac_model, retriever=tac_gen.retriever is not None)

            return TopKTacGenerator(tac_model=tac_model,
                                    num_filtered=config.diversity_config.num_filtered,
                                    random=config.diversity_config.random)
        else:
            raise NotImplementedError

    if config.model == 'diversity':

        if hasattr(config, 'ckpt_path') and config.ckpt_path:
            tac_gen = RetrievalAugmentedGenerator.load(
                config.ckpt_path, device=device, freeze=True
            )

        else:
            tac_gen = RetrievalAugmentedGenerator(config.config).to(device)
            tac_gen.freeze()

        if tac_gen.retriever is not None:
            assert config.config.indexed_corpus_path is not None
            tac_gen.retriever.load_corpus(config.config.indexed_corpus_path)

            # check if corpus is up to date, otherwise recompute
            if tac_gen.retriever.embeddings_staled:
                tac_gen.retriever.reindex_corpus(batch_size=2)

        if config.distributed:
            tac_model = ray.remote(num_gpus=config.gpu_per_process, num_cpus=config.cpu_per_process)(
                ReProverTacGen).remote(
                tac_model=tac_gen, num_sampled_tactics=config.num_sampled_tactics)

            filter_model = ray.remote(num_gpus=config.gpu_per_diversity, num_cpus=config.cpu_per_diversity)(
                DiversityModel).remote(config.diversity_config, device=device)

            tac_model = ReProverWrapper(tac_model, retriever=tac_gen.retriever is not None)

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

    if config.model == 'reprover':

        if hasattr(config, 'ckpt_path') and config.ckpt_path:
            tac_gen = RetrievalAugmentedGenerator.load(
                config.ckpt_path, device=device, freeze=True
            )

        else:
            tac_gen = RetrievalAugmentedGenerator(config.config).to(device)
            tac_gen.freeze()

        if tac_gen.retriever is not None:
            assert config.config.indexed_corpus_path is not None
            tac_gen.retriever.load_corpus(config.config.indexed_corpus_path)

            # check if corpus is up to date, otherwise recompute
            if tac_gen.retriever.embeddings_staled:
                tac_gen.retriever.reindex_corpus(batch_size=2)

        if config.distributed:
            tac_model = ray.remote(num_gpus=config.gpu_per_process, num_cpus=config.cpu_per_process)(
                ReProverTacGen).remote(
                tac_model=tac_gen, num_sampled_tactics=config.num_sampled_tactics)
            return ReProverWrapper(tac_model, retriever=tac_gen.retriever is not None)

        else:
            return ReProverTacGen(tac_model=tac_gen, num_sampled_tactics=config.num_sampled_tactics)
    else:
        raise NotImplementedError
