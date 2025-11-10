from __future__ import division, absolute_import, print_function

from models.end_to_end.search_models.bestfs import BestFS
from models.end_to_end.search_models.bfs import BFS


def get_search_model(config, device):
    """

    Initialise search model based on configuration.

    """

    if config.search == 'bestfs':
        return BestFS()
    elif config.search == 'bfs':
        return BFS()
    else:
        raise NotImplementedError(f'Search approach {config.search} not implemented')
